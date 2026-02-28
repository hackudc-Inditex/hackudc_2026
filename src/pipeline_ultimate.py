"""
PIPELINE ULTIMATE - Maximizar score en CPU.

Estrategia combinada con scoring acumulativo:
1. SKU exacto (máxima prioridad)
2. SKU prefijos múltiples (8, 6, 4 chars) con scores decrecientes
3. Bundle-to-bundle: buscar looks similares en training → usar sus productos
4. Co-ocurrencia: si detectamos producto X, los que van con X suben
5. Popularidad: productos frecuentes en la sección tienen bonus
6. CLIP zero-shot categorías: detectar qué categorías hay en el bundle
7. YOLO + CLIP per-garment matching
8. Timestamp proximity como señal adicional

Uso:
  python src/pipeline_ultimate.py [--no-yolo] [--fast]
"""

import csv
import re
import json
import sys
import numpy as np
import torch
import open_clip
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLES_HIRES_DIR = BASE_DIR / "images" / "bundles_hires"
BUNDLES_DIR = BASE_DIR / "images" / "bundles"
PRODUCTS_DIR = BASE_DIR / "images" / "products"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_YOLO = "--no-yolo" not in sys.argv
FAST_MODE = "--fast" in sys.argv  # skip slow visual parts

# DeepFashion2 YOLO categories → catalog categories
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

# Extra category groups for full-image matching
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


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    """Extract raw SKU from Zara URL."""
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    if match:
        return match.group(1)
    match = re.search(r'/(T\d{8,15}(?:-\d+)?)-[pe]', str(url))
    if match:
        return match.group(1)
    match = re.search(r'/(M\d{8,15}(?:-\d+)?)-[pe]', str(url))
    if match:
        return match.group(1)
    return None


def extract_sku_core(url):
    """Extract numeric-only core SKU (strip T/M prefix and -NNN suffix)."""
    sku = extract_sku(url)
    if not sku:
        return None
    # Remove alpha prefix
    core = re.sub(r'^[A-Za-z]+', '', sku)
    # Remove -NNN color suffix
    core = re.sub(r'-\d+$', '', core)
    return core


def extract_ts(url):
    match = re.search(r'ts=(\d+)', str(url))
    return int(match.group(1)) if match else None


def get_bundle_path(bid):
    hires = BUNDLES_HIRES_DIR / f"{bid}.jpg"
    return hires if hires.exists() else BUNDLES_DIR / f"{bid}.jpg"


# ============================================================
# MODELS
# ============================================================
def load_clip():
    model_name = "ViT-B-32"
    print(f"Cargando CLIP {model_name} en {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device=DEVICE
    )
    model.eval()
    if DEVICE == "cuda":
        model = model.half()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def embed_pil(model, preprocess, img):
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy().flatten()


def embed_batch(model, preprocess, image_dir, ids, batch_size=64):
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


def embed_text(model, tokenizer, texts):
    """Encode text descriptions with CLIP."""
    tokens = tokenizer(texts).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def load_yolo_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    print("Cargando YOLOv8 DeepFashion2...")
    path = hf_hub_download("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt")
    return YOLO(path)


def detect_garments(yolo_model, image_path):
    results = yolo_model(str(image_path), verbose=False, conf=0.15)
    detections = []
    if results and results[0].boxes is not None:
        r = results[0]
        for i in range(len(r.boxes)):
            label = r.names[int(r.boxes.cls[i])]
            bbox = r.boxes.xyxy[i].cpu().numpy()
            conf = float(r.boxes.conf[i])
            detections.append((label, bbox, conf))
    return detections


def crop_detection(img, bbox, padding=0.1):
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * padding)
    y1 = max(0, y1 - bh * padding)
    x2 = min(w, x2 + bw * padding)
    y2 = min(h, y2 + bh * padding)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  PIPELINE ULTIMATE")
    print(f"  YOLO: {'ON' if USE_YOLO else 'OFF'}")
    print(f"  Mode: {'FAST' if FAST_MODE else 'FULL'}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # --- Load data ---
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}
    product_url = {p["product_asset_id"]: p["product_image_url"] for p in products}
    product_ids_all = [p["product_asset_id"] for p in products]

    # --- SKU indexes ---
    # Raw SKU → products
    sku_raw_to_products = defaultdict(list)
    # Core SKU → products (without color suffix)
    sku_core_to_products = defaultdict(list)
    # Various prefix lengths → products
    sku_prefix_to_products = {n: defaultdict(list) for n in [4, 5, 6, 7, 8]}

    for p in products:
        raw = extract_sku(p["product_image_url"])
        core = extract_sku_core(p["product_image_url"])
        if raw:
            sku_raw_to_products[raw].append(p["product_asset_id"])
            for n in sku_prefix_to_products:
                if len(raw) >= n:
                    sku_prefix_to_products[n][raw[:n]].append(p["product_asset_id"])
        if core:
            sku_core_to_products[core].append(p["product_asset_id"])

    # --- Section data ---
    cat_sections = defaultdict(set)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        desc = product_desc.get(row["product_asset_id"])
        if sec and desc:
            cat_sections[desc].add(sec)

    # Products valid per section (from training observation)
    section_valid_products = defaultdict(set)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        if sec:
            section_valid_products[sec].add(row["product_asset_id"])

    # --- Training bundle→products ---
    train_bundle_products = defaultdict(set)
    for row in train:
        train_bundle_products[row["bundle_asset_id"]].add(row["product_asset_id"])
    train_bids = list(train_bundle_products.keys())

    # --- Co-occurrence graph ---
    print("\nCo-ocurrencia...")
    cooccurrence = defaultdict(Counter)
    for bid, pids in train_bundle_products.items():
        for p1 in pids:
            for p2 in pids:
                if p1 != p2:
                    cooccurrence[p1][p2] += 1

    # Category co-occurrence
    cat_cooccurrence = defaultdict(Counter)
    for bid, pids in train_bundle_products.items():
        cats = set(product_desc.get(p, "") for p in pids)
        for c1 in cats:
            for c2 in cats:
                if c1 != c2:
                    cat_cooccurrence[c1][c2] += 1

    # --- Product popularity per section ---
    print("Popularidad por sección...")
    section_product_freq = defaultdict(Counter)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        if sec:
            section_product_freq[sec][row["product_asset_id"]] += 1

    # Category popularity per section
    section_cat_freq = defaultdict(Counter)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        desc = product_desc.get(row["product_asset_id"], "")
        if sec and desc:
            section_cat_freq[sec][desc] += 1

    # Average products per category per bundle per section
    section_cat_per_bundle = defaultdict(lambda: defaultdict(list))
    for bid, pids in train_bundle_products.items():
        sec = bundle_section.get(bid)
        if not sec:
            continue
        cat_count = Counter()
        for p in pids:
            cat_count[product_desc.get(p, "")] += 1
        for cat, cnt in cat_count.items():
            section_cat_per_bundle[sec][cat].append(cnt)

    # --- Timestamp data ---
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

    # ============================================================
    # LOAD MODELS
    # ============================================================
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    yolo_model = None
    if USE_YOLO:
        try:
            yolo_model = load_yolo_model()
        except Exception as e:
            print(f"YOLO no disponible: {e}")
            yolo_model = None

    # ============================================================
    # PRODUCT EMBEDDINGS
    # ============================================================
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = EMBEDDINGS_DIR / "products_vitb32.npy"
    ids_path = EMBEDDINGS_DIR / "product_ids_vitb32.json"

    # Also check for old naming
    if not emb_path.exists():
        for alt_name in ["product_embeddings.npy"]:
            alt = EMBEDDINGS_DIR / alt_name
            if alt.exists():
                emb_path = alt
                ids_path = EMBEDDINGS_DIR / "product_ids.json"
                break

    if emb_path.exists() and ids_path.exists():
        print(f"Cargando embeddings de {emb_path.name}...")
        product_embeddings = np.load(emb_path)
        with open(ids_path) as f:
            product_ids = json.load(f)
    else:
        product_ids = product_ids_all
        print(f"Generando embeddings de {len(product_ids)} productos...")
        product_embeddings = embed_batch(clip_model, clip_preprocess, PRODUCTS_DIR, product_ids)
        save_path = EMBEDDINGS_DIR / "products_vitb32.npy"
        save_ids = EMBEDDINGS_DIR / "product_ids_vitb32.json"
        np.save(save_path, product_embeddings)
        with open(save_ids, "w") as f:
            json.dump(product_ids, f)

    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    print(f"  {len(product_ids)} productos, dim={product_embeddings.shape[1]}")

    # ============================================================
    # TRAINING BUNDLE EMBEDDINGS (for bundle-to-bundle matching)
    # ============================================================
    train_emb_path = EMBEDDINGS_DIR / "train_bundles_vitb32.npy"
    train_ids_path = EMBEDDINGS_DIR / "train_bundles_ids_vitb32.json"

    if train_emb_path.exists() and train_ids_path.exists():
        print("Cargando embeddings de bundles training...")
        train_bundle_embs = np.load(train_emb_path)
        with open(train_ids_path) as f:
            train_bids_ordered = json.load(f)
    else:
        print(f"Generando embeddings de {len(train_bids)} bundles training...")
        train_bids_ordered = train_bids
        all_embs = []
        for i in tqdm(range(0, len(train_bids_ordered), 64), desc="Train bundles"):
            batch_ids = train_bids_ordered[i:i + 64]
            images = []
            for bid in batch_ids:
                img_path = get_bundle_path(bid)
                try:
                    img = clip_preprocess(Image.open(img_path).convert("RGB"))
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
        train_bundle_embs = np.vstack(all_embs)
        np.save(train_emb_path, train_bundle_embs)
        with open(train_ids_path, "w") as f:
            json.dump(train_bids_ordered, f)

    # FAISS index for training bundles (per section for better matching)
    print("Índices FAISS de bundles training por sección...")
    train_section_indices = {}  # section → (index, [bids])
    for sec in ["1", "2", "3"]:
        idxs = []
        bids_sec = []
        for i, bid in enumerate(train_bids_ordered):
            if bundle_section.get(bid) == sec:
                idxs.append(i)
                bids_sec.append(bid)
        if idxs:
            embs = train_bundle_embs[idxs].copy()
            faiss.normalize_L2(embs)
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs)
            train_section_indices[sec] = (index, bids_sec)
            print(f"  Sección {sec}: {len(bids_sec)} bundles")

    # ============================================================
    # PRODUCT FAISS INDICES (per group × section)
    # ============================================================
    print("Índices FAISS por categoría × sección...")
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

    # ============================================================
    # CLIP TEXT EMBEDDINGS for zero-shot category detection
    # ============================================================
    print("Embeddings de texto para clasificación zero-shot...")
    all_categories = sorted(set(p["product_description"] for p in products))
    cat_prompts = [f"a photo of a person wearing {cat.lower()}" for cat in all_categories]
    cat_text_embs = embed_text(clip_model, clip_tokenizer, cat_prompts)
    faiss.normalize_L2(cat_text_embs)
    cat_to_group = {}
    for cat in all_categories:
        for group_name, group_cats in ALL_GROUPS.items():
            if cat in group_cats:
                cat_to_group[cat] = group_name
                break

    # ============================================================
    # PROCESS TEST BUNDLES
    # ============================================================
    test_bids = list(set(row["bundle_asset_id"] for row in test))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Procesando {len(test_bids)} bundles...")
    print(f"{'=' * 60}\n")

    results = []
    stats = Counter()

    for bid in tqdm(test_bids, desc="ULTIMATE"):
        sec = bundle_section.get(bid, "1")
        burl = bundle_url.get(bid, "")
        bsku_raw = extract_sku(burl)
        bsku_core = extract_sku_core(burl)
        bts = bundle_ts.get(bid)

        candidates = {}  # pid → score

        # Load bundle image
        img_path = get_bundle_path(bid)
        try:
            bundle_img = Image.open(img_path).convert("RGB")
        except Exception:
            bundle_img = Image.new("RGB", (224, 224))

        # ===== SIGNAL 1: SKU EXACT MATCH (score 200) =====
        if bsku_raw and bsku_raw in sku_raw_to_products:
            for pid in sku_raw_to_products[bsku_raw]:
                candidates[pid] = candidates.get(pid, 0) + 200
                stats["sku_exact"] += 1

        # SKU core match (without color suffix)
        if bsku_core and bsku_core in sku_core_to_products:
            for pid in sku_core_to_products[bsku_core]:
                if pid not in candidates:
                    candidates[pid] = candidates.get(pid, 0) + 150
                    stats["sku_core"] += 1

        # ===== SIGNAL 2: SKU PREFIX MATCHES (decreasing scores) =====
        prefix_scores = {8: 80, 7: 60, 6: 40, 5: 25, 4: 15}
        if bsku_raw:
            for prefix_len, score in prefix_scores.items():
                if len(bsku_raw) >= prefix_len:
                    prefix = bsku_raw[:prefix_len]
                    for pid in sku_prefix_to_products[prefix_len].get(prefix, []):
                        # Only if same section
                        desc = product_desc.get(pid, "")
                        if desc in cat_sections and sec not in cat_sections[desc]:
                            continue
                        if pid not in candidates:
                            # Timestamp proximity bonus
                            ts_bonus = 0
                            if bts and pid in product_ts:
                                diff_days = abs(bts - product_ts[pid]) / (1000 * 86400)
                                if diff_days < 7:
                                    ts_bonus = 10
                                elif diff_days < 30:
                                    ts_bonus = 5
                                elif diff_days < 90:
                                    ts_bonus = 2
                            candidates[pid] = score + ts_bonus
                            stats[f"sku_prefix_{prefix_len}"] += 1

        # ===== SIGNAL 3: BUNDLE-TO-BUNDLE MATCHING =====
        bundle_emb = embed_pil(clip_model, clip_preprocess, bundle_img).reshape(1, -1)
        faiss.normalize_L2(bundle_emb)

        if sec in train_section_indices:
            t_index, t_bids = train_section_indices[sec]
            k = min(20, t_index.ntotal)
            sim_scores, sim_indices = t_index.search(bundle_emb, k)

            for j in range(k):
                similar_bid = t_bids[sim_indices[0][j]]
                sim_score = float(sim_scores[0][j])

                # Higher weight for more similar bundles
                for pid in train_bundle_products[similar_bid]:
                    bonus = sim_score * 30  # Scale: 0.5-1.0 → 15-30
                    candidates[pid] = candidates.get(pid, 0) + bonus
                    stats["b2b"] += 1

                    # Co-occurrence boost: products that appear with this product
                    for copid, count in cooccurrence[pid].most_common(3):
                        if copid in pid_to_idx:
                            cobonus = count * sim_score * 5
                            candidates[copid] = candidates.get(copid, 0) + cobonus

        # ===== SIGNAL 4: PRODUCT POPULARITY BONUS =====
        for pid, freq in section_product_freq[sec].most_common(50):
            popularity_bonus = freq * 0.5  # Frequent products get small bonus
            candidates[pid] = candidates.get(pid, 0) + popularity_bonus

        # ===== SIGNAL 5: YOLO DETECTION + CLIP PER GARMENT =====
        if yolo_model and not FAST_MODE:
            detections = detect_garments(yolo_model, img_path)
            detected_groups = set()

            for label, bbox, yolo_conf in detections:
                detected_groups.add(label)
                crop = crop_detection(bundle_img, bbox, padding=0.08)
                crop_emb = embed_pil(clip_model, clip_preprocess, crop).reshape(1, -1)
                faiss.normalize_L2(crop_emb)

                key = (label, sec)
                if key in group_indices:
                    index, gpids = group_indices[key]
                    search_k = min(8, index.ntotal)
                    scores, indices = index.search(crop_emb, search_k)
                    for j2 in range(search_k):
                        pid = gpids[indices[0][j2]]
                        clip_score = float(scores[0][j2])
                        bonus = clip_score * (1 + yolo_conf) * 15
                        candidates[pid] = candidates.get(pid, 0) + bonus
                        stats["yolo_clip"] += 1

        # ===== SIGNAL 6: CLIP ZERO-SHOT CATEGORY + FULL IMAGE =====
        if not FAST_MODE:
            # Detect what categories are likely in this bundle via text matching
            cat_scores = bundle_emb @ cat_text_embs.T
            top_cat_indices = np.argsort(cat_scores[0])[::-1][:8]
            detected_cats = [all_categories[i] for i in top_cat_indices]

            for cat in detected_cats:
                group = cat_to_group.get(cat)
                if not group:
                    continue
                key = (group, sec)
                if key in group_indices:
                    index, gpids = group_indices[key]
                    search_k = min(5, index.ntotal)
                    scores, indices = index.search(bundle_emb, search_k)
                    for j2 in range(search_k):
                        pid = gpids[indices[0][j2]]
                        clip_score = float(scores[0][j2])
                        candidates[pid] = candidates.get(pid, 0) + clip_score * 8

            # Full image against accessories (always likely to have some)
            for extra_group in EXTRA_GROUPS:
                key = (extra_group, sec)
                if key in group_indices:
                    index, gpids = group_indices[key]
                    search_k = min(3, index.ntotal)
                    scores, indices = index.search(bundle_emb, search_k)
                    for j2 in range(search_k):
                        pid = gpids[indices[0][j2]]
                        candidates[pid] = candidates.get(pid, 0) + float(scores[0][j2]) * 5

        # ===== SIGNAL 7: TIMESTAMP PROXIMITY RERANKING =====
        if bts:
            for pid in list(candidates.keys()):
                if pid in product_ts:
                    diff_days = abs(bts - product_ts[pid]) / (1000 * 86400)
                    if diff_days < 7:
                        candidates[pid] *= 1.3
                    elif diff_days < 30:
                        candidates[pid] *= 1.15
                    elif diff_days < 90:
                        candidates[pid] *= 1.05
                    elif diff_days > 365:
                        candidates[pid] *= 0.7

        # ===== FILL REMAINING WITH SECTION SEARCH =====
        if len(candidates) < 15 and sec in section_indices:
            scores, indices = section_indices[sec].search(bundle_emb, 50)
            spids = section_pids_map[sec]
            for j in range(50):
                pid = spids[indices[0][j]]
                if pid not in candidates:
                    candidates[pid] = float(scores[0][j]) * 2
                if len(candidates) >= 15:
                    break

        # ===== TOP 15 =====
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        for pid, score in sorted_candidates[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

        # Fill if < 15
        if len(sorted_candidates) < 15:
            remaining = 15 - len(sorted_candidates)
            spids = section_pids_map.get(sec, product_ids)
            for pid in spids:
                if pid not in candidates:
                    results.append({"bundle_asset_id": bid, "product_asset_id": pid})
                    remaining -= 1
                    if remaining <= 0:
                        break

    # ============================================================
    # SAVE
    # ============================================================
    suffix = "_fast" if FAST_MODE else ""
    suffix += "_noyolo" if not USE_YOLO else ""
    out_path = SUBMISSIONS_DIR / f"submission_ultimate{suffix}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print(f"SUBMISSION: {out_path}")
    print(f"  Filas: {len(results)}")
    print(f"  Bundles: {len(test_bids)}")
    print(f"  Media productos/bundle: {len(results) / len(test_bids):.1f}")
    print(f"\nSeñales usadas:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
