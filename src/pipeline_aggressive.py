"""
PIPELINE AGRESIVO - Todo lo que tenemos para ganar.

Estrategia:
1. SKU exacto + hermanos (ya da 21%)
2. Bundle-to-bundle: encontrar looks similares en training → robar sus productos
3. YOLO segmentación + CLIP matching por prenda
4. Co-ocurrencia: si detectamos producto X, buscar qué va con X
5. Todo combinado con scoring inteligente

Uso:
  pip install ultralytics huggingface_hub open-clip-torch faiss-cpu fashion-clip
  python src/download_hires.py   (si no lo hiciste)
  python src/pipeline_aggressive.py
"""

import csv
import re
import json
import numpy as np
import torch
import open_clip
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLES_HIRES_DIR = BASE_DIR / "images" / "bundles_hires"
BUNDLES_DIR = BASE_DIR / "images" / "bundles"
PRODUCTS_DIR = BASE_DIR / "images" / "products"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("Sin GPU, usando CPU")

YOLO_TO_CATALOG = {
    "short_sleeved_shirt": ["T-SHIRT", "POLO SHIRT", "BABY T-SHIRT", "TOPS AND OTHERS"],
    "long_sleeved_shirt": ["SHIRT", "SWEATER", "CARDIGAN", "SWEATSHIRT", "BABY SHIRT", "BABY SWEATER", "BABY CARDIGAN"],
    "short_sleeved_outwear": ["BLAZER", "WAISTCOAT", "BABY WAISTCOAT"],
    "long_sleeved_outwear": ["WIND-JACKET", "COAT", "ANORAK", "BLAZER", "OVERSHIRT", "TRENCH RAINCOAT", "BABY JACKET/COAT", "BABY WIND-JACKET"],
    "vest": ["WAISTCOAT", "BODYSUIT", "TOPS AND OTHERS"],
    "sling": ["TOPS AND OTHERS", "BODYSUIT", "DRESS"],
    "shorts": ["BERMUDA", "SHORTS", "BABY BERMUDAS"],
    "trousers": ["TROUSERS", "LEGGINGS", "BABY TROUSERS", "BABY LEGGINGS"],
    "skirt": ["SKIRT", "BABY SKIRT"],
    "short_sleeved_dress": ["DRESS", "BABY DRESS"],
    "long_sleeved_dress": ["DRESS", "BABY DRESS"],
    "vest_dress": ["DRESS", "BABY DRESS"],
    "sling_dress": ["DRESS", "BABY DRESS"],
}

EXTRA_CATEGORIES = {
    "footwear": [
        "SHOES", "FLAT SHOES", "SANDAL", "HEELED SHOES", "MOCCASINS", "SPORT SHOES",
        "RUNNING SHOES", "TRAINERS", "ANKLE BOOT", "FLAT ANKLE BOOT", "HEELED ANKLE BOOT",
        "FLAT BOOT", "HEELED BOOT", "HIGH TOPS", "BOOT", "RAIN BOOT", "ATHLETIC FOOTWEAR",
        "SPORTY SANDAL",
    ],
    "bags": ["HAND BAG-RUCKSACK", "PURSE WALLET"],
    "headwear": ["HAT", "GLASSES", "BABY BONNET"],
    "small_accessories": [
        "BELT", "IMIT JEWELLER", "SCARF", "SOCKS", "TIE", "ACCESSORIES",
        "GLOVES", "SHAWL/FOULARD", "PANTY/UNDERPANT", "STOCKINGS-TIGHTS", "BABY SOCKS",
    ],
}

ALL_GROUPS = {}
ALL_GROUPS.update(YOLO_TO_CATALOG)
ALL_GROUPS.update(EXTRA_CATEGORIES)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    return match.group(1) if match else None


def get_bundle_path(bid):
    hires = BUNDLES_HIRES_DIR / f"{bid}.jpg"
    return hires if hires.exists() else BUNDLES_DIR / f"{bid}.jpg"


# ============================================================
# MODELOS
# ============================================================
def load_yolo():
    print("Cargando YOLOv8...")
    path = hf_hub_download("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt")
    return YOLO(path)


def load_clip_model():
    print(f"Cargando CLIP ViT-L-14 en {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=DEVICE
    )
    model.eval()
    if DEVICE == "cuda":
        model = model.half()
    return model, preprocess


def embed_pil(model, preprocess, img):
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy().flatten()


def embed_batch(model, preprocess, image_dir, ids, batch_size=32):
    all_embs = []
    for i in tqdm(range(0, len(ids), batch_size), desc="Embeddings"):
        batch_ids = ids[i:i+batch_size]
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
    print("  PIPELINE AGRESIVO - A por el 44%+")
    print("=" * 60)

    # --- Datos ---
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}
    product_ids = [p["product_asset_id"] for p in products]

    # SKU
    sku_to_products = defaultdict(list)
    for p in products:
        sku = extract_sku(p["product_image_url"])
        if sku:
            sku_to_products[sku].append(p["product_asset_id"])

    # Secciones
    cat_sections = defaultdict(set)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        desc = product_desc.get(row["product_asset_id"])
        if sec and desc:
            cat_sections[desc].add(sec)

    # Training: bundle → productos
    train_bundle_products = defaultdict(set)
    for row in train:
        train_bundle_products[row["bundle_asset_id"]].add(row["product_asset_id"])
    train_bids = list(train_bundle_products.keys())

    # ============================================================
    # CO-OCURRENCIA: qué productos aparecen juntos
    # ============================================================
    print("\nConstruyendo grafo de co-ocurrencia...")
    cooccurrence = defaultdict(Counter)
    for bid, pids in train_bundle_products.items():
        for p1 in pids:
            for p2 in pids:
                if p1 != p2:
                    cooccurrence[p1][p2] += 1

    # También co-ocurrencia por categoría
    cat_cooccurrence = defaultdict(Counter)
    for bid, pids in train_bundle_products.items():
        cats = [product_desc.get(p, "") for p in pids]
        for i, c1 in enumerate(cats):
            for j, c2 in enumerate(cats):
                if i != j:
                    cat_cooccurrence[c1][c2] += 1

    # ============================================================
    # CARGAR MODELOS
    # ============================================================
    yolo_model = load_yolo()
    clip_model, clip_preprocess = load_clip_model()

    # ============================================================
    # EMBEDDINGS DE PRODUCTOS
    # ============================================================
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = EMBEDDINGS_DIR / "products_vitl14.npy"
    ids_path = EMBEDDINGS_DIR / "product_ids_vitl14.json"

    # También aceptar los del baseline.py
    if not emb_path.exists() and (EMBEDDINGS_DIR / "product_embeddings.npy").exists():
        emb_path = EMBEDDINGS_DIR / "product_embeddings.npy"
        ids_path = EMBEDDINGS_DIR / "product_ids.json"

    if emb_path.exists() and ids_path.exists():
        print(f"Cargando embeddings de {emb_path.name}...")
        product_embeddings = np.load(emb_path)
        with open(ids_path) as f:
            product_ids = json.load(f)
    else:
        print("Generando embeddings de productos (ViT-L-14)...")
        product_embeddings = embed_batch(clip_model, clip_preprocess, PRODUCTS_DIR, product_ids)
        np.save(emb_path, product_embeddings)
        with open(ids_path, "w") as f:
            json.dump(product_ids, f)

    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    print(f"  {len(product_ids)} productos, dim={product_embeddings.shape[1]}")

    # ============================================================
    # EMBEDDINGS DE BUNDLES DE TRAINING (para bundle-to-bundle)
    # ============================================================
    train_emb_path = EMBEDDINGS_DIR / "train_bundles_emb.npy"
    train_ids_path = EMBEDDINGS_DIR / "train_bundles_ids.json"

    if train_emb_path.exists() and train_ids_path.exists():
        print("Cargando embeddings de bundles training...")
        train_bundle_embs = np.load(train_emb_path)
        with open(train_ids_path) as f:
            train_bids_ordered = json.load(f)
    else:
        print("Generando embeddings de bundles training...")
        train_bids_ordered = train_bids
        # Usar hires si disponible, sino 224px
        all_embs = []
        for i in tqdm(range(0, len(train_bids_ordered), 32), desc="Train bundles"):
            batch_ids = train_bids_ordered[i:i+32]
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

    # Índice FAISS de bundles training
    print("Índice FAISS de bundles training...")
    train_bundle_embs_norm = train_bundle_embs.copy()
    faiss.normalize_L2(train_bundle_embs_norm)
    train_bundle_index = faiss.IndexFlatIP(train_bundle_embs_norm.shape[1])
    train_bundle_index.add(train_bundle_embs_norm)

    # ============================================================
    # ÍNDICES FAISS POR CATEGORÍA × SECCIÓN
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

    # Índice general por sección
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
    # PROCESAR TEST
    # ============================================================
    test_bids = list(set(row["bundle_asset_id"] for row in test))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Procesando {len(test_bids)} bundles...")
    print(f"{'='*60}\n")

    results = []

    for bid in tqdm(test_bids, desc="AGRESIVO"):
        sec = bundle_section.get(bid, "1")
        burl = bundle_url.get(bid, "")
        bsku = extract_sku(burl)

        candidates = {}  # pid → score (acumulativo)

        img_path = get_bundle_path(bid)
        try:
            bundle_img = Image.open(img_path).convert("RGB")
        except Exception:
            bundle_img = Image.new("RGB", (224, 224))

        # ===== 1. SKU EXACTO (score 100) =====
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                candidates[pid] = candidates.get(pid, 0) + 100

        # ===== 2. SKU HERMANOS (score 30) =====
        if bsku:
            prefix = bsku[:5]
            for sku, pids in sku_to_products.items():
                if sku[:5] == prefix and sku != bsku:
                    for pid in pids:
                        desc = product_desc.get(pid, "")
                        if desc not in cat_sections or sec in cat_sections[desc]:
                            candidates[pid] = candidates.get(pid, 0) + 30

        # ===== 3. BUNDLE-TO-BUNDLE (looks similares del training) =====
        bundle_emb = embed_pil(clip_model, clip_preprocess, bundle_img).reshape(1, -1)
        faiss.normalize_L2(bundle_emb)

        sim_scores, sim_indices = train_bundle_index.search(bundle_emb, 15)
        for j in range(15):
            similar_bid = train_bids_ordered[sim_indices[0][j]]
            sim_score = float(sim_scores[0][j])

            # Solo si es la misma sección
            if bundle_section.get(similar_bid) != sec:
                continue

            # Los productos de este bundle similar son candidatos
            for pid in train_bundle_products[similar_bid]:
                bonus = sim_score * 25
                candidates[pid] = candidates.get(pid, 0) + bonus

                # Co-ocurrencia: productos que aparecen con este
                for copid, count in cooccurrence[pid].most_common(5):
                    if copid in pid_to_idx:
                        candidates[copid] = candidates.get(copid, 0) + count * sim_score * 3

        # ===== 4. YOLO + CLIP POR PRENDA =====
        detections = detect_garments(yolo_model, img_path)

        for label, bbox, yolo_conf in detections:
            crop = crop_detection(bundle_img, bbox, padding=0.08)
            crop_emb = embed_pil(clip_model, clip_preprocess, crop).reshape(1, -1)
            faiss.normalize_L2(crop_emb)

            key = (label, sec)
            if key in group_indices:
                index, gpids = group_indices[key]
                k = min(10, index.ntotal)
                scores, indices = index.search(crop_emb, k)
                for j in range(k):
                    pid = gpids[indices[0][j]]
                    clip_score = float(scores[0][j])
                    bonus = clip_score * (1 + yolo_conf) * 20
                    candidates[pid] = candidates.get(pid, 0) + bonus

        # ===== 5. ACCESORIOS (imagen completa) =====
        full_emb = bundle_emb  # ya lo tenemos

        for extra_group in EXTRA_CATEGORIES:
            key = (extra_group, sec)
            if key in group_indices:
                index, gpids = group_indices[key]
                k = min(5, index.ntotal)
                scores, indices = index.search(full_emb, k)
                for j in range(k):
                    pid = gpids[indices[0][j]]
                    candidates[pid] = candidates.get(pid, 0) + float(scores[0][j]) * 10

        # ===== 6. FALLBACK SECCIÓN =====
        if len(candidates) < 15 and sec in section_indices:
            scores, indices = section_indices[sec].search(full_emb, 30)
            spids = section_pids_map[sec]
            for j in range(30):
                pid = spids[indices[0][j]]
                if pid not in candidates:
                    candidates[pid] = float(scores[0][j]) * 2

        # ===== TOP 15 =====
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        for pid, score in sorted_candidates[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

        # Rellenar si < 15
        if len(sorted_candidates) < 15:
            remaining = 15 - len(sorted_candidates)
            spids = section_pids_map.get(sec, [])
            for pid in spids:
                if pid not in candidates:
                    results.append({"bundle_asset_id": bid, "product_asset_id": pid})
                    remaining -= 1
                    if remaining <= 0:
                        break

    # Guardar
    out_path = SUBMISSIONS_DIR / "submission_aggressive.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"SUBMISSION: {out_path} ({len(results)} filas)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
