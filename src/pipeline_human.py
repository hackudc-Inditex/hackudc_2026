"""
PIPELINE CON HUMAN-IN-THE-LOOP.

Extiende pipeline_train.py con anotaciones humanas:
  - Modo --generate-candidates: genera top 40 por bundle para la herramienta de anotación
  - Modo normal: carga anotaciones humanas, las añade al training set, reentrena y predice

Uso:
  # Paso 1: Generar candidatos para anotar
  python src/pipeline_human.py --generate-candidates

  # Paso 2: Anotar con la herramienta web (annotation/server.py)

  # Paso 3: Reentrenar con anotaciones humanas
  python src/pipeline_human.py --epochs 50

  Opciones:
    --generate-candidates  Generar annotation/candidates.json y salir
    --no-yolo              Saltar YOLO, usar multi-crop por zonas
    --model vitb32         Usar ViT-B-32 (más rápido, menos preciso)
    --epochs 50            Número de épocas de entrenamiento
    --skip-train           Saltar entrenamiento (usar modelo guardado)
"""

import csv
import re
import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter
import random
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLES_HIRES_DIR = BASE_DIR / "images" / "bundles_hires"
BUNDLES_DIR = BASE_DIR / "images" / "bundles"
PRODUCTS_DIR = BASE_DIR / "images" / "products"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
CROPS_DIR = BASE_DIR / "crops"
MODELS_DIR = BASE_DIR / "models"

ANNOTATION_DIR = BASE_DIR / "annotation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parse args
USE_YOLO = "--no-yolo" not in sys.argv
SKIP_TRAIN = "--skip-train" in sys.argv
GENERATE_CANDIDATES = "--generate-candidates" in sys.argv
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_DIM = 768
for i, arg in enumerate(sys.argv):
    if arg == "--model" and i + 1 < len(sys.argv):
        name = sys.argv[i + 1]
        if name == "vitb32":
            CLIP_MODEL_NAME = "ViT-B-32"
            CLIP_DIM = 512

NUM_EPOCHS = 50
for i, arg in enumerate(sys.argv):
    if arg == "--epochs" and i + 1 < len(sys.argv):
        NUM_EPOCHS = int(sys.argv[i + 1])

# YOLO → catalog category mapping
YOLO_TO_CATALOG = {
    "short_sleeved_shirt": ["T-SHIRT", "POLO SHIRT", "BABY T-SHIRT", "TOPS AND OTHERS"],
    "long_sleeved_shirt": ["SHIRT", "SWEATER", "CARDIGAN", "SWEATSHIRT", "OVERSHIRT",
                           "BABY SHIRT", "BABY SWEATER", "BABY CARDIGAN"],
    "short_sleeved_outwear": ["BLAZER", "WAISTCOAT"],
    "long_sleeved_outwear": ["WIND-JACKET", "COAT", "ANORAK", "BLAZER", "OVERSHIRT",
                              "TRENCH RAINCOAT", "BABY JACKET/COAT", "BABY WIND-JACKET"],
    "vest": ["WAISTCOAT", "BODYSUIT", "TOPS AND OTHERS"],
    "sling": ["TOPS AND OTHERS", "BODYSUIT", "DRESS"],
    "shorts": ["BERMUDA", "SHORTS", "BABY BERMUDAS"],
    "trousers": ["TROUSERS", "LEGGINGS", "BABY TROUSERS", "BABY LEGGINGS"],
    "skirt": ["SKIRT", "BABY SKIRT"],
    "short_sleeved_dress": ["DRESS", "BABY DRESS", "OVERALL"],
    "long_sleeved_dress": ["DRESS", "BABY DRESS", "OVERALL"],
    "vest_dress": ["DRESS", "BABY DRESS"],
    "sling_dress": ["DRESS", "BABY DRESS"],
}

EXTRA_GROUPS = {
    "footwear": [
        "SHOES", "FLAT SHOES", "SANDAL", "HEELED SHOES", "MOCCASINS", "SPORT SHOES",
        "RUNNING SHOES", "TRAINERS", "ANKLE BOOT", "FLAT ANKLE BOOT", "HEELED ANKLE BOOT",
        "FLAT BOOT", "HEELED BOOT", "HIGH TOPS", "BOOT", "RAIN BOOT", "ATHLETIC FOOTWEAR",
        "SPORTY SANDAL",
    ],
    "bags": ["HAND BAG-RUCKSACK", "PURSE WALLET"],
    "headwear": ["HAT", "GLASSES", "BABY BONNET"],
    "accessories": [
        "BELT", "IMIT JEWELLER", "SCARF", "SOCKS", "TIE", "ACCESSORIES",
        "GLOVES", "SHAWL/FOULARD", "PANTY/UNDERPANT", "STOCKINGS-TIGHTS", "BABY SOCKS",
    ],
}

ALL_GROUPS = {}
ALL_GROUPS.update(YOLO_TO_CATALOG)
ALL_GROUPS.update(EXTRA_GROUPS)

# Reverse: catalog category → group name
CAT_TO_GROUP = {}
for group_name, cats in ALL_GROUPS.items():
    for cat in cats:
        if cat not in CAT_TO_GROUP:
            CAT_TO_GROUP[cat] = group_name

# Multi-crop zones (fallback when no YOLO)
ZONE_CROPS = {
    "upper": [(0.05, 0.50, 0.05, 0.95), (0.10, 0.45, 0.10, 0.90)],
    "lower": [(0.45, 0.85, 0.05, 0.95), (0.40, 0.80, 0.10, 0.90)],
    "feet":  [(0.78, 1.00, 0.05, 0.95)],
    "full":  [(0.05, 0.90, 0.05, 0.95)],
    "head":  [(0.00, 0.18, 0.15, 0.85)],
}

ZONE_TO_CATALOG = {
    "upper": ["T-SHIRT", "SHIRT", "SWEATER", "BLAZER", "WIND-JACKET", "SWEATSHIRT",
              "CARDIGAN", "COAT", "ANORAK", "WAISTCOAT", "OVERSHIRT", "TOPS AND OTHERS",
              "POLO SHIRT", "BODYSUIT", "TRENCH RAINCOAT",
              "BABY T-SHIRT", "BABY SWEATER", "BABY SHIRT", "BABY JACKET/COAT",
              "BABY WIND-JACKET", "BABY CARDIGAN"],
    "lower": ["TROUSERS", "SKIRT", "BERMUDA", "SHORTS", "LEGGINGS",
              "BABY TROUSERS", "BABY BERMUDAS", "BABY SKIRT", "BABY LEGGINGS"],
    "feet":  EXTRA_GROUPS["footwear"],
    "full":  ["DRESS", "OVERALL", "BABY DRESS", "BABY OUTFIT", "SWIMSUIT", "BIB OVERALL"],
    "head":  EXTRA_GROUPS["headwear"],
}


# ============================================================
# UTILITIES
# ============================================================
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    for pattern in [r'/(\d{8,15}(?:-\d+)?)-[pe]', r'/(T\d{8,15}(?:-\d+)?)-[pe]',
                    r'/(M\d{8,15}(?:-\d+)?)-[pe]']:
        match = re.search(pattern, str(url))
        if match:
            return match.group(1)
    return None


def extract_sku_core(url):
    sku = extract_sku(url)
    if not sku:
        return None
    core = re.sub(r'^[A-Za-z]+', '', sku)
    core = re.sub(r'-\d+$', '', core)
    return core


def extract_ts(url):
    match = re.search(r'ts=(\d+)', str(url))
    return int(match.group(1)) if match else None


def get_bundle_path(bid):
    hires = BUNDLES_HIRES_DIR / f"{bid}.jpg"
    return hires if hires.exists() else BUNDLES_DIR / f"{bid}.jpg"


def crop_zone(img, y1_pct, y2_pct, x1_pct, x2_pct):
    w, h = img.size
    return img.crop((int(w * x1_pct), int(h * y1_pct), int(w * x2_pct), int(h * y2_pct)))


def crop_bbox(img, bbox, padding=0.1):
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * padding)
    y1 = max(0, y1 - bh * padding)
    x2 = min(w, x2 + bw * padding)
    y2 = min(h, y2 + bh * padding)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


# ============================================================
# CLIP
# ============================================================
def load_clip():
    print(f"Cargando CLIP {CLIP_MODEL_NAME} en {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained="openai", device=DEVICE
    )
    model.eval()
    if DEVICE == "cuda":
        model = model.half()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return model, preprocess, tokenizer


def embed_pil(model, preprocess, img):
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy().flatten()


def embed_batch_from_dir(model, preprocess, image_dir, ids, batch_size=64):
    all_embs = []
    for i in tqdm(range(0, len(ids), batch_size), desc="Embeddings"):
        batch_ids = ids[i:i + batch_size]
        images = []
        for img_id in batch_ids:
            try:
                img = preprocess(Image.open(image_dir / f"{img_id}.jpg").convert("RGB"))
            except Exception:
                img = preprocess(Image.new("RGB", (224, 224)))
            images.append(img)
        batch = torch.stack(images).to(DEVICE)
        if DEVICE == "cuda":
            batch = batch.half()
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().float().numpy())
    return np.vstack(all_embs)


def embed_pil_batch(model, preprocess, pil_images, batch_size=64):
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[i:i + batch_size]
        tensors = [preprocess(img) for img in batch_imgs]
        batch = torch.stack(tensors).to(DEVICE)
        if DEVICE == "cuda":
            batch = batch.half()
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().float().numpy())
    return np.vstack(all_embs)


# ============================================================
# YOLO
# ============================================================
def load_yolo_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    print("Cargando YOLOv8 DeepFashion2...")
    path = hf_hub_download("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt")
    return YOLO(path)


def detect_garments(yolo_model, image_path, conf=0.15):
    results = yolo_model(str(image_path), verbose=False, conf=conf)
    detections = []
    if results and results[0].boxes is not None:
        r = results[0]
        for i in range(len(r.boxes)):
            label = r.names[int(r.boxes.cls[i])]
            bbox = r.boxes.xyxy[i].cpu().numpy()
            conf_score = float(r.boxes.conf[i])
            detections.append({"label": label, "bbox": bbox, "conf": conf_score})
    return detections


# ============================================================
# PROJECTION HEAD (the model we train)
# ============================================================
class ProjectionHead(nn.Module):
    """Maps bundle crop embeddings to product embedding space."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x) + x, dim=-1)  # residual + L2 norm


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning: (crop_emb, positive_product_emb, hard_negatives)."""

    def __init__(self, crop_embs, pos_product_embs, all_product_embs, neg_indices_per_sample,
                 num_negatives=31):
        self.crop_embs = crop_embs
        self.pos_product_embs = pos_product_embs
        self.all_product_embs = all_product_embs
        self.neg_indices = neg_indices_per_sample
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.crop_embs)

    def __getitem__(self, idx):
        crop = torch.FloatTensor(self.crop_embs[idx])
        pos = torch.FloatTensor(self.pos_product_embs[idx])

        neg_pool = self.neg_indices[idx]
        if len(neg_pool) > self.num_negatives:
            chosen = random.sample(list(neg_pool), self.num_negatives)
        else:
            chosen = list(neg_pool)
            while len(chosen) < self.num_negatives:
                chosen.append(random.choice(chosen))

        negs = torch.FloatTensor(self.all_product_embs[chosen])

        return crop, pos, negs


def train_projection(train_data, val_data, product_embeddings, dim, num_epochs=30, lr=1e-3):
    """Train projection head with InfoNCE contrastive loss."""

    model = ProjectionHead(dim).to(DEVICE)

    crop_embs_train, pos_embs_train, neg_indices_train = train_data
    crop_embs_val, pos_embs_val, neg_indices_val = val_data

    train_dataset = ContrastiveDataset(
        crop_embs_train, pos_embs_train, product_embeddings, neg_indices_train
    )
    val_dataset = ContrastiveDataset(
        crop_embs_val, pos_embs_val, product_embeddings, neg_indices_val
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    temperature = 0.07

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        for crop, pos, negs in train_loader:
            crop, pos, negs = crop.to(DEVICE), pos.to(DEVICE), negs.to(DEVICE)

            projected = model(crop)  # (B, D)

            # Similarities
            pos_sim = (projected * pos).sum(-1, keepdim=True)  # (B, 1)
            neg_sim = torch.bmm(negs, projected.unsqueeze(-1)).squeeze(-1)  # (B, N)

            logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature  # (B, 1+N)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for crop, pos, negs in val_loader:
                crop, pos, negs = crop.to(DEVICE), pos.to(DEVICE), negs.to(DEVICE)
                projected = model(crop)
                pos_sim = (projected * pos).sum(-1, keepdim=True)
                neg_sim = torch.bmm(negs, projected.unsqueeze(-1)).squeeze(-1)
                logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)
                loss = F.cross_entropy(logits, labels)
                val_losses.append(loss.item())

                # Accuracy: is the positive ranked first?
                correct += (logits.argmax(dim=-1) == 0).sum().item()
                total += logits.size(0)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total if total > 0 else 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{num_epochs}: "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}  lr={scheduler.get_last_lr()[0]:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  PIPELINE CON ENTRENAMIENTO")
    print(f"  CLIP: {CLIP_MODEL_NAME} (dim={CLIP_DIM})")
    print(f"  YOLO: {'ON' if USE_YOLO else 'OFF'}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    # =========================================================
    # PHASE 1: DATA LOADING
    # =========================================================
    print("\n[FASE 1] Cargando datos...")
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train_pairs = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test_rows = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url_map = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}
    product_url_map = {p["product_asset_id"]: p["product_image_url"] for p in products}
    product_ids_all = [p["product_asset_id"] for p in products]

    # SKU indexes
    sku_to_products = defaultdict(list)
    sku_prefix_to_products = {n: defaultdict(list) for n in [4, 5, 6, 7, 8]}
    for p in products:
        sku = extract_sku(p["product_image_url"])
        if sku:
            sku_to_products[sku].append(p["product_asset_id"])
            for n in sku_prefix_to_products:
                if len(sku) >= n:
                    sku_prefix_to_products[n][sku[:n]].append(p["product_asset_id"])

    # Section constraints
    cat_sections = defaultdict(set)
    for row in train_pairs:
        sec = bundle_section.get(row["bundle_asset_id"])
        desc = product_desc.get(row["product_asset_id"])
        if sec and desc:
            cat_sections[desc].add(sec)

    # Training bundle → products
    train_bundle_products = defaultdict(set)
    for row in train_pairs:
        train_bundle_products[row["bundle_asset_id"]].add(row["product_asset_id"])

    # --- HUMAN LABELS: Load and merge ---
    human_bids = set()
    human_labels_path = ANNOTATION_DIR / "human_labels.csv"
    if human_labels_path.exists() and not GENERATE_CANDIDATES:
        human_rows = load_csv(human_labels_path)
        human_count = 0
        for row in human_rows:
            bid = row["bundle_asset_id"]
            pid = row["product_asset_id"]
            if pid:  # skip empty
                train_bundle_products[bid].add(pid)
                human_bids.add(bid)
                human_count += 1
        print(f"  [HUMAN] Cargadas {human_count} anotaciones para {len(human_bids)} bundles")
    elif not GENERATE_CANDIDATES:
        print(f"  [HUMAN] No hay anotaciones en {human_labels_path}")

    # Co-occurrence
    cooccurrence = defaultdict(Counter)
    for bid, pids in train_bundle_products.items():
        for p1 in pids:
            for p2 in pids:
                if p1 != p2:
                    cooccurrence[p1][p2] += 1

    # Popularity per section
    section_product_freq = defaultdict(Counter)
    for row in train_pairs:
        sec = bundle_section.get(row["bundle_asset_id"])
        if sec:
            section_product_freq[sec][row["product_asset_id"]] += 1

    # Timestamps
    bundle_ts = {}
    for b in bundles:
        ts = extract_ts(b["bundle_image_url"])
        if ts:
            bundle_ts[b["bundle_asset_id"]] = ts
    product_ts = {}
    for p in products:
        ts = extract_ts(p["product_image_url"])
        if ts:
            product_ts[p["product_asset_id"]] = ts

    # Train/val split: human-labeled bundles ALWAYS go to training
    original_bids = [bid for bid in train_bundle_products.keys() if bid not in human_bids]
    random.seed(42)
    random.shuffle(original_bids)
    split = int(0.8 * len(original_bids))
    train_bids = original_bids[:split] + list(human_bids)
    val_bids = original_bids[split:]
    print(f"  Train bundles: {len(train_bids)} ({len(human_bids)} human), Val bundles: {len(val_bids)}")
    print(f"  Productos: {len(product_ids_all)}")

    # =========================================================
    # PHASE 2: CLIP EMBEDDINGS
    # =========================================================
    print("\n[FASE 2] Embeddings CLIP...")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = CLIP_MODEL_NAME.replace("-", "").replace("/", "").lower()

    # Product embeddings
    prod_emb_path = EMBEDDINGS_DIR / f"products_{model_tag}.npy"
    prod_ids_path = EMBEDDINGS_DIR / f"product_ids_{model_tag}.json"

    if prod_emb_path.exists() and prod_ids_path.exists():
        print(f"  Cargando embeddings productos de {prod_emb_path.name}...")
        product_embeddings = np.load(prod_emb_path)
        with open(prod_ids_path) as f:
            product_ids = json.load(f)
    else:
        product_ids = product_ids_all
        print(f"  Generando embeddings de {len(product_ids)} productos...")
        product_embeddings = embed_batch_from_dir(
            clip_model, clip_preprocess, PRODUCTS_DIR, product_ids
        )
        np.save(prod_emb_path, product_embeddings)
        with open(prod_ids_path, "w") as f:
            json.dump(product_ids, f)

    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    print(f"  {len(product_ids)} productos, dim={product_embeddings.shape[1]}")

    # Bundle embeddings (ALL training bundles + human-labeled test bundles)
    all_bundle_bids = list(train_bundle_products.keys())
    cache_suffix = "_human" if human_bids else ""
    bundle_emb_path = EMBEDDINGS_DIR / f"bundles_{model_tag}{cache_suffix}.npy"
    bundle_ids_path = EMBEDDINGS_DIR / f"bundle_ids_{model_tag}{cache_suffix}.json"

    if bundle_emb_path.exists() and bundle_ids_path.exists():
        print(f"  Cargando embeddings bundles...")
        all_bundle_embs = np.load(bundle_emb_path)
        with open(bundle_ids_path) as f:
            all_bundle_bids_ordered = json.load(f)
        # Check if all needed bundles are in cache
        cached_set = set(all_bundle_bids_ordered)
        missing = [bid for bid in all_bundle_bids if bid not in cached_set]
        if missing:
            print(f"  {len(missing)} bundles nuevos no cacheados, regenerando...")
            os.remove(bundle_emb_path)
            os.remove(bundle_ids_path)
    else:
        all_bundle_bids_ordered = all_bundle_bids
        print(f"  Generando embeddings de {len(all_bundle_bids)} bundles...")
        all_embs = []
        for i in tqdm(range(0, len(all_bundle_bids_ordered), 64), desc="Bundles"):
            batch_ids = all_bundle_bids_ordered[i:i + 64]
            images = []
            for bid in batch_ids:
                try:
                    img = clip_preprocess(
                        Image.open(get_bundle_path(bid)).convert("RGB")
                    )
                except Exception:
                    img = clip_preprocess(Image.new("RGB", (224, 224)))
                images.append(img)
            batch = torch.stack(images).to(DEVICE)
            if DEVICE == "cuda":
                batch = batch.half()
            with torch.no_grad():
                feats = clip_model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().float().numpy())
        all_bundle_embs = np.vstack(all_embs)
        np.save(bundle_emb_path, all_bundle_embs)
        with open(bundle_ids_path, "w") as f:
            json.dump(all_bundle_bids_ordered, f)

    bid_to_emb_idx = {bid: i for i, bid in enumerate(all_bundle_bids_ordered)}

    # =========================================================
    # PHASE 3: YOLO DETECTION + CROP EMBEDDINGS
    # =========================================================
    print("\n[FASE 3] Detección de prendas y crop embeddings...")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    crops_cache_path = EMBEDDINGS_DIR / f"crops_data_{model_tag}{cache_suffix}.pkl"

    if crops_cache_path.exists():
        print("  Cargando crops cacheados...")
        with open(crops_cache_path, "rb") as f:
            crops_data = pickle.load(f)
    else:
        yolo_model = None
        if USE_YOLO:
            try:
                yolo_model = load_yolo_model()
            except Exception as e:
                print(f"  YOLO no disponible: {e}")

        # crops_data[bid] = list of {"label": str, "embedding": np.array, "bbox": ...}
        crops_data = {}

        for bid in tqdm(all_bundle_bids_ordered, desc="Detecting crops"):
            img_path = get_bundle_path(bid)
            try:
                bundle_img = Image.open(img_path).convert("RGB")
            except Exception:
                crops_data[bid] = []
                continue

            crops_for_bundle = []

            if yolo_model:
                detections = detect_garments(yolo_model, img_path)
                for det in detections:
                    cropped = crop_bbox(bundle_img, det["bbox"], padding=0.08)
                    emb = embed_pil(clip_model, clip_preprocess, cropped)
                    crops_for_bundle.append({
                        "label": det["label"],
                        "embedding": emb,
                        "conf": det["conf"],
                    })
            else:
                # Multi-crop fallback
                for zone_name, zone_coords_list in ZONE_CROPS.items():
                    for coords in zone_coords_list:
                        cropped = crop_zone(bundle_img, *coords)
                        emb = embed_pil(clip_model, clip_preprocess, cropped)
                        crops_for_bundle.append({
                            "label": zone_name,
                            "embedding": emb,
                            "conf": 0.5,
                        })

            # Also add full image as a "crop"
            full_emb = embed_pil(clip_model, clip_preprocess, bundle_img)
            crops_for_bundle.append({
                "label": "_full_",
                "embedding": full_emb,
                "conf": 1.0,
            })

            crops_data[bid] = crops_for_bundle

        print(f"  Guardando crops cache...")
        with open(crops_cache_path, "wb") as f:
            pickle.dump(crops_data, f)

    # =========================================================
    # PHASE 4: CREATE TRAINING PAIRS FOR PROJECTION HEAD
    # =========================================================
    print("\n[FASE 4] Creando pares de entrenamiento...")

    # Pre-compute negative pools by (category, section) → list of product indices
    # This avoids iterating 28K products for each training pair
    print("  Pre-computing negative pools...")
    neg_pool_cache = {}  # (category, section) → [product_indices]
    section_pool_cache = {}  # section → [product_indices]

    for sec in ["1", "2", "3"]:
        sec_indices = []
        for i, pid in enumerate(product_ids):
            desc = product_desc.get(pid, "")
            if desc not in cat_sections or sec in cat_sections[desc]:
                sec_indices.append(i)
        section_pool_cache[sec] = sec_indices

        # Per-category pools
        cat_indices = defaultdict(list)
        for i in sec_indices:
            desc = product_desc.get(product_ids[i], "")
            cat_indices[desc].append(i)
        for cat, indices in cat_indices.items():
            neg_pool_cache[(cat, sec)] = indices

    def create_pairs(bids_list):
        """Create (crop_emb, product_emb, hard_negative_indices) pairs."""
        crop_embs_list = []
        pos_embs_list = []
        neg_indices_list = []

        for bid in bids_list:
            sec = bundle_section.get(bid, "1")
            matched_products = train_bundle_products.get(bid, set())

            if bid not in crops_data or not crops_data[bid]:
                continue

            for pid in matched_products:
                if pid not in pid_to_idx:
                    continue

                product_cat = product_desc.get(pid, "")
                product_emb_idx = pid_to_idx[pid]
                product_emb = product_embeddings[product_emb_idx]

                # Find the best matching crop for this product
                best_crop_emb = None
                best_sim = -1

                for crop_info in crops_data[bid]:
                    crop_label = crop_info["label"]
                    compatible = False
                    if crop_label == "_full_":
                        compatible = True
                    elif USE_YOLO and crop_label in YOLO_TO_CATALOG:
                        compatible = product_cat in YOLO_TO_CATALOG[crop_label]
                    elif not USE_YOLO and crop_label in ZONE_TO_CATALOG:
                        compatible = product_cat in ZONE_TO_CATALOG[crop_label]

                    if compatible:
                        sim = np.dot(crop_info["embedding"], product_emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_crop_emb = crop_info["embedding"]

                if best_crop_emb is None:
                    for crop_info in crops_data[bid]:
                        if crop_info["label"] == "_full_":
                            best_crop_emb = crop_info["embedding"]
                            break

                if best_crop_emb is None:
                    continue

                # Hard negatives from pre-computed pools
                key = (product_cat, sec)
                hard_neg_indices = [i for i in neg_pool_cache.get(key, [])
                                    if product_ids[i] != pid]

                if len(hard_neg_indices) < 10:
                    # Fallback: all products in section
                    hard_neg_indices = [i for i in section_pool_cache.get(sec, [])
                                        if product_ids[i] != pid][:500]

                if not hard_neg_indices:
                    continue

                crop_embs_list.append(best_crop_emb)
                pos_embs_list.append(product_emb)
                neg_indices_list.append(hard_neg_indices)

        return (
            np.array(crop_embs_list) if crop_embs_list else np.zeros((0, CLIP_DIM)),
            np.array(pos_embs_list) if pos_embs_list else np.zeros((0, CLIP_DIM)),
            neg_indices_list,
        )

    print("  Creando pares de train...")
    train_data = create_pairs(train_bids)
    print(f"    → {len(train_data[0])} pares de entrenamiento")

    print("  Creando pares de validación...")
    val_data = create_pairs(val_bids)
    print(f"    → {len(val_data[0])} pares de validación")

    # =========================================================
    # PHASE 5: TRAIN PROJECTION HEAD
    # =========================================================
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_suffix = "_human" if human_bids else ""
    model_save_path = MODELS_DIR / f"projection_{model_tag}{model_suffix}.pt"

    if SKIP_TRAIN and model_save_path.exists():
        print(f"\n[FASE 5] Cargando modelo guardado de {model_save_path}...")
        projection = ProjectionHead(CLIP_DIM).to(DEVICE)
        projection.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        projection.eval()
    else:
        lr = 3e-4 if human_bids else 5e-4
        print(f"\n[FASE 5] Entrenando projection head ({NUM_EPOCHS} epochs, lr={lr})...")
        if len(train_data[0]) == 0:
            print("  ERROR: No hay pares de entrenamiento!")
            projection = None
        else:
            projection = train_projection(
                train_data, val_data, product_embeddings, CLIP_DIM,
                num_epochs=NUM_EPOCHS, lr=lr
            )
            torch.save(projection.state_dict(), model_save_path)
            print(f"  Modelo guardado en {model_save_path}")

    # =========================================================
    # PHASE 6: VALIDATE ON HELD-OUT BUNDLES
    # =========================================================
    print(f"\n[FASE 6] Validación en {len(val_bids)} bundles...")

    # Build FAISS indices per group × section
    group_indices = {}
    for group_name, catalog_cats in ALL_GROUPS.items():
        cats_set = set(catalog_cats)
        for sec in ["1", "2", "3"]:
            idxs, pids = [], []
            for pid in product_ids:
                desc = product_desc.get(pid, "")
                if desc not in cats_set:
                    continue
                if desc in cat_sections and sec not in cat_sections[desc]:
                    continue
                if pid in pid_to_idx:
                    idxs.append(pid_to_idx[pid])
                    pids.append(pid)
            if idxs:
                embs = product_embeddings[idxs].copy()
                faiss.normalize_L2(embs)
                index = faiss.IndexFlatIP(embs.shape[1])
                index.add(embs)
                group_indices[(group_name, sec)] = (index, pids)

    # Section-wide indices
    section_indices = {}
    section_pids_map = {}
    for sec in ["1", "2", "3"]:
        idxs, pids = [], []
        for pid in product_ids:
            desc = product_desc.get(pid, "")
            if desc in cat_sections and sec not in cat_sections[desc]:
                continue
            if pid in pid_to_idx:
                idxs.append(pid_to_idx[pid])
                pids.append(pid)
        embs = product_embeddings[idxs].copy()
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        section_indices[sec] = index
        section_pids_map[sec] = pids

    # Bundle-to-bundle FAISS (per section, only train bundles for val)
    train_section_b2b = {}
    for sec in ["1", "2", "3"]:
        idxs = []
        bids_sec = []
        for i, bid in enumerate(all_bundle_bids_ordered):
            if bid in set(train_bids) and bundle_section.get(bid) == sec:
                idxs.append(i)
                bids_sec.append(bid)
        if idxs:
            embs = all_bundle_embs[idxs].copy()
            faiss.normalize_L2(embs)
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs)
            train_section_b2b[sec] = (index, bids_sec)

    def predict_bundle(bid, b2b_indices, use_projection=True, top_k=15):
        """Predict products for a single bundle. Returns sorted list of (score, pid)."""
        sec = bundle_section.get(bid, "1")
        burl = bundle_url_map.get(bid, "")
        bsku = extract_sku(burl)
        bts = bundle_ts.get(bid)

        candidates = {}

        # --- SIGNAL 1: SKU exact ---
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                candidates[pid] = candidates.get(pid, 0) + 200

        # --- SIGNAL 2: SKU prefix ---
        prefix_scores = {8: 80, 7: 60, 6: 40, 5: 25, 4: 15}
        if bsku:
            for plen, pscore in prefix_scores.items():
                if len(bsku) >= plen:
                    prefix = bsku[:plen]
                    for pid in sku_prefix_to_products[plen].get(prefix, []):
                        desc = product_desc.get(pid, "")
                        if desc in cat_sections and sec not in cat_sections[desc]:
                            continue
                        if pid not in candidates:
                            candidates[pid] = pscore

        # --- SIGNAL 3: Bundle-to-bundle ---
        bundle_emb = None
        if bid in bid_to_emb_idx:
            bundle_emb = all_bundle_embs[bid_to_emb_idx[bid]].reshape(1, -1).copy()
        elif bid in crops_data:
            for c in crops_data[bid]:
                if c["label"] == "_full_":
                    bundle_emb = c["embedding"].reshape(1, -1).copy()
                    break

        if bundle_emb is not None and sec in b2b_indices:
            faiss.normalize_L2(bundle_emb)
            b2b_index, b2b_bids = b2b_indices[sec]
            k = min(20, b2b_index.ntotal)
            sim_scores, sim_idx = b2b_index.search(bundle_emb, k)
            for j in range(k):
                similar_bid = b2b_bids[sim_idx[0][j]]
                if similar_bid == bid:
                    continue
                sim = float(sim_scores[0][j])
                for pid in train_bundle_products.get(similar_bid, set()):
                    bonus = sim * 35
                    candidates[pid] = candidates.get(pid, 0) + bonus
                    # Co-occurrence
                    for copid, count in cooccurrence[pid].most_common(3):
                        if copid in pid_to_idx:
                            candidates[copid] = candidates.get(copid, 0) + count * sim * 3

        # --- SIGNAL 4: Trained visual matching (projection head) ---
        if bid in crops_data and projection is not None and use_projection:
            for crop_info in crops_data[bid]:
                crop_label = crop_info["label"]
                if crop_label == "_full_":
                    continue

                crop_emb = torch.FloatTensor(crop_info["embedding"]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    projected = projection(crop_emb).cpu().numpy()
                faiss.normalize_L2(projected)

                # Determine which group to search
                if USE_YOLO and crop_label in YOLO_TO_CATALOG:
                    groups_to_search = [crop_label]
                elif not USE_YOLO and crop_label in ZONE_TO_CATALOG:
                    groups_to_search = [crop_label]
                else:
                    groups_to_search = []

                for group in groups_to_search:
                    key = (group, sec)
                    if key in group_indices:
                        index, gpids = group_indices[key]
                        search_k = min(10, index.ntotal)
                        scores, indices = index.search(projected, search_k)
                        for j2 in range(search_k):
                            pid = gpids[indices[0][j2]]
                            clip_score = float(scores[0][j2])
                            conf = crop_info.get("conf", 0.5)
                            bonus = clip_score * (1 + conf) * 20
                            candidates[pid] = candidates.get(pid, 0) + bonus

        # --- SIGNAL 5: Raw CLIP matching (no projection) as complement ---
        if bid in crops_data:
            for crop_info in crops_data[bid]:
                crop_label = crop_info["label"]
                if crop_label == "_full_":
                    continue

                crop_emb = crop_info["embedding"].reshape(1, -1).copy()
                faiss.normalize_L2(crop_emb)

                if USE_YOLO and crop_label in YOLO_TO_CATALOG:
                    groups_to_search = [crop_label]
                elif not USE_YOLO and crop_label in ZONE_TO_CATALOG:
                    groups_to_search = [crop_label]
                else:
                    groups_to_search = []

                for group in groups_to_search:
                    key = (group, sec)
                    if key in group_indices:
                        index, gpids = group_indices[key]
                        search_k = min(5, index.ntotal)
                        scores, indices = index.search(crop_emb, search_k)
                        for j2 in range(search_k):
                            pid = gpids[indices[0][j2]]
                            clip_score = float(scores[0][j2])
                            bonus = clip_score * 8  # lower weight than projected
                            candidates[pid] = candidates.get(pid, 0) + bonus

        # --- SIGNAL 6: Popularity ---
        for pid, freq in section_product_freq[sec].most_common(30):
            candidates[pid] = candidates.get(pid, 0) + freq * 0.3

        # --- SIGNAL 7: Timestamp proximity ---
        if bts:
            for pid in list(candidates.keys()):
                if pid in product_ts:
                    diff_days = abs(bts - product_ts[pid]) / (1000 * 86400)
                    if diff_days < 7:
                        candidates[pid] *= 1.25
                    elif diff_days < 30:
                        candidates[pid] *= 1.12
                    elif diff_days < 90:
                        candidates[pid] *= 1.04
                    elif diff_days > 365:
                        candidates[pid] *= 0.7

        # --- SIGNAL 8: Section fallback ---
        if len(candidates) < 15 and bundle_emb is not None and sec in section_indices:
            scores, indices = section_indices[sec].search(bundle_emb, 50)
            spids = section_pids_map[sec]
            for j in range(min(50, len(spids))):
                pid = spids[indices[0][j]]
                if pid not in candidates:
                    candidates[pid] = float(scores[0][j]) * 1.5
                if len(candidates) >= 20:
                    break

        # Sort and return top_k
        sorted_c = sorted(candidates.items(), key=lambda x: -x[1])
        return sorted_c[:top_k]

    # =========================================================
    # GENERATE CANDIDATES MODE (--generate-candidates)
    # =========================================================
    if GENERATE_CANDIDATES:
        print("\n[GENERATE] Generando candidatos para anotación...")
        test_bids = list(set(row["bundle_asset_id"] for row in test_rows))

        # Generate crops for test bundles if needed
        yolo_model = None
        if USE_YOLO:
            try:
                yolo_model = load_yolo_model()
            except Exception:
                yolo_model = None

        for bid in tqdm(test_bids, desc="Test crops"):
            if bid in crops_data:
                continue
            img_path = get_bundle_path(bid)
            try:
                bundle_img = Image.open(img_path).convert("RGB")
            except Exception:
                crops_data[bid] = []
                continue
            crops_for_bundle = []
            if yolo_model:
                detections = detect_garments(yolo_model, img_path)
                for det in detections:
                    cropped = crop_bbox(bundle_img, det["bbox"], padding=0.08)
                    emb = embed_pil(clip_model, clip_preprocess, cropped)
                    crops_for_bundle.append({"label": det["label"], "embedding": emb, "conf": det["conf"]})
            else:
                for zone_name, zone_coords_list in ZONE_CROPS.items():
                    for coords in zone_coords_list:
                        cropped = crop_zone(bundle_img, *coords)
                        emb = embed_pil(clip_model, clip_preprocess, cropped)
                        crops_for_bundle.append({"label": zone_name, "embedding": emb, "conf": 0.5})
            full_emb = embed_pil(clip_model, clip_preprocess, bundle_img)
            crops_for_bundle.append({"label": "_full_", "embedding": full_emb, "conf": 1.0})
            crops_data[bid] = crops_for_bundle

        if yolo_model:
            del yolo_model
            torch.cuda.empty_cache()

        # Test bundle embeddings for b2b
        for bid in test_bids:
            if bid not in bid_to_emb_idx:
                if bid in crops_data:
                    for c in crops_data[bid]:
                        if c["label"] == "_full_":
                            idx = len(all_bundle_bids_ordered)
                            all_bundle_bids_ordered.append(bid)
                            all_bundle_embs = np.vstack([all_bundle_embs, c["embedding"].reshape(1, -1)])
                            bid_to_emb_idx[bid] = idx
                            break

        # Build b2b indices with all training bundles
        all_section_b2b = {}
        for sec in ["1", "2", "3"]:
            idxs, bids_sec = [], []
            for i, bid in enumerate(all_bundle_bids_ordered):
                if bundle_section.get(bid) == sec:
                    idxs.append(i)
                    bids_sec.append(bid)
            if idxs:
                embs = all_bundle_embs[idxs].copy()
                faiss.normalize_L2(embs)
                index = faiss.IndexFlatIP(embs.shape[1])
                index.add(embs)
                all_section_b2b[sec] = (index, bids_sec)

        # Generate top 40 candidates per test bundle
        output = {"generated_at": "", "bundles": {}}
        for bid in tqdm(test_bids, desc="Generating candidates"):
            sec = bundle_section.get(bid, "1")
            burl = bundle_url_map.get(bid, "")
            predicted = predict_bundle(bid, all_section_b2b, use_projection=(projection is not None), top_k=40)
            cands = []
            for rank, (pid, score) in enumerate(predicted, 1):
                cands.append({
                    "product_id": pid,
                    "product_url": product_url_map.get(pid, ""),
                    "description": product_desc.get(pid, ""),
                    "score": round(score, 2),
                    "rank": rank,
                })
            output["bundles"][bid] = {
                "section": sec,
                "bundle_url": burl,
                "candidates": cands,
            }

        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ANNOTATION_DIR / "candidates.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print(f"  CANDIDATOS GENERADOS: {out_path}")
        print(f"  Bundles: {len(test_bids)}")
        print(f"  Candidatos/bundle: 40")
        print(f"  Ahora ejecuta: python annotation/server.py")
        print(f"{'=' * 60}")
        return

    # Validation
    val_hits = 0
    val_total = 0
    for bid in tqdm(val_bids, desc="Validating"):
        true_products = train_bundle_products.get(bid, set())
        predicted = predict_bundle(bid, train_section_b2b, use_projection=(projection is not None))
        predicted_pids = set(pid for pid, score in predicted)
        hits = len(predicted_pids & true_products)
        val_hits += hits
        val_total += len(true_products)

    val_recall = val_hits / val_total if val_total > 0 else 0
    print(f"\n  VALIDACIÓN: recall@15 = {val_recall:.4f} ({val_hits}/{val_total})")
    print(f"  (Esto es en datos de entrenamiento held-out, no en test real)")

    # =========================================================
    # PHASE 7: INFERENCE ON TEST
    # =========================================================
    print(f"\n[FASE 7] Inferencia en test...")

    # Rebuild b2b indices using ALL training bundles
    all_section_b2b = {}
    for sec in ["1", "2", "3"]:
        idxs = []
        bids_sec = []
        for i, bid in enumerate(all_bundle_bids_ordered):
            if bundle_section.get(bid) == sec:
                idxs.append(i)
                bids_sec.append(bid)
        if idxs:
            embs = all_bundle_embs[idxs].copy()
            faiss.normalize_L2(embs)
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs)
            all_section_b2b[sec] = (index, bids_sec)

    test_bids = list(set(row["bundle_asset_id"] for row in test_rows))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    # Generate test bundle crops if needed
    yolo_model = None
    if USE_YOLO:
        try:
            yolo_model = load_yolo_model()
        except Exception:
            yolo_model = None

    test_crops_need = [bid for bid in test_bids if bid not in crops_data]
    if test_crops_need:
        print(f"  Generando crops para {len(test_crops_need)} test bundles...")
        for bid in tqdm(test_crops_need, desc="Test crops"):
            img_path = get_bundle_path(bid)
            try:
                bundle_img = Image.open(img_path).convert("RGB")
            except Exception:
                crops_data[bid] = []
                continue

            crops_for_bundle = []

            if yolo_model:
                detections = detect_garments(yolo_model, img_path)
                for det in detections:
                    cropped = crop_bbox(bundle_img, det["bbox"], padding=0.08)
                    emb = embed_pil(clip_model, clip_preprocess, cropped)
                    crops_for_bundle.append({
                        "label": det["label"],
                        "embedding": emb,
                        "conf": det["conf"],
                    })
            else:
                for zone_name, zone_coords_list in ZONE_CROPS.items():
                    for coords in zone_coords_list:
                        cropped = crop_zone(bundle_img, *coords)
                        emb = embed_pil(clip_model, clip_preprocess, cropped)
                        crops_for_bundle.append({
                            "label": zone_name,
                            "embedding": emb,
                            "conf": 0.5,
                        })

            full_emb = embed_pil(clip_model, clip_preprocess, bundle_img)
            crops_for_bundle.append({
                "label": "_full_",
                "embedding": full_emb,
                "conf": 1.0,
            })

            crops_data[bid] = crops_for_bundle

    # Generate test bundle full-image embeddings (for b2b)
    test_embs = {}
    for bid in test_bids:
        if bid in bid_to_emb_idx:
            test_embs[bid] = all_bundle_embs[bid_to_emb_idx[bid]]
        elif bid in crops_data:
            for c in crops_data[bid]:
                if c["label"] == "_full_":
                    test_embs[bid] = c["embedding"]
                    break
        if bid not in test_embs:
            try:
                img = Image.open(get_bundle_path(bid)).convert("RGB")
                test_embs[bid] = embed_pil(clip_model, clip_preprocess, img)
            except Exception:
                test_embs[bid] = np.zeros(CLIP_DIM)

    # Add test embeddings to the lookup
    for bid in test_bids:
        if bid not in bid_to_emb_idx and bid in test_embs:
            idx = len(all_bundle_bids_ordered)
            all_bundle_bids_ordered.append(bid)
            all_bundle_embs = np.vstack([all_bundle_embs, test_embs[bid].reshape(1, -1)])
            bid_to_emb_idx[bid] = idx

    # Predict
    results = []
    for bid in tqdm(test_bids, desc="PREDICTING"):
        predicted = predict_bundle(bid, all_section_b2b, use_projection=(projection is not None))
        for pid, score in predicted:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

        # Fill remaining
        current = len([r for r in results if r["bundle_asset_id"] == bid])
        if current < 15:
            sec = bundle_section.get(bid, "1")
            seen = set(pid for pid, _ in predicted)
            spids = section_pids_map.get(sec, product_ids)
            for pid in spids:
                if pid not in seen:
                    results.append({"bundle_asset_id": bid, "product_asset_id": pid})
                    seen.add(pid)
                    current += 1
                    if current >= 15:
                        break

    # Save
    out_path = SUBMISSIONS_DIR / f"submission_human_{model_tag}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print(f"  SUBMISSION: {out_path}")
    print(f"  Filas: {len(results)}")
    print(f"  Bundles: {len(test_bids)}")
    print(f"  Media productos/bundle: {len(results) / len(test_bids):.1f}")
    print(f"  Val recall@15: {val_recall:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
