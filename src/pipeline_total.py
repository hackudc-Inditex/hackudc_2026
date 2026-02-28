"""
PIPELINE TOTAL: Segmentación + Matching por prenda + Metadatos.
El enfoque definitivo: detectar cada prenda en el bundle, recortarla,
y buscar el producto más parecido dentro de su categoría y sección.

Uso: python src/pipeline_total.py

Requiere: pip install transformers timm supervision
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
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLES_DIR = BASE_DIR / "images" / "bundles"
PRODUCTS_DIR = BASE_DIR / "images" / "products"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Categorías de ropa agrupadas por zona del cuerpo
ZONE_CATEGORIES = {
    "head": ["HAT", "GLASSES", "BABY BONNET"],
    "upper": [
        "T-SHIRT", "SHIRT", "SWEATER", "BLAZER", "WIND-JACKET", "SWEATSHIRT",
        "CARDIGAN", "COAT", "ANORAK", "WAISTCOAT", "OVERSHIRT", "TOPS AND OTHERS",
        "POLO SHIRT", "BODYSUIT", "BABY T-SHIRT", "BABY SWEATER", "BABY SHIRT",
        "BABY JACKET/COAT", "BABY WIND-JACKET", "BABY CARDIGAN", "TRENCH RAINCOAT",
    ],
    "full": [
        "DRESS", "OVERALL", "BABY DRESS", "BABY OUTFIT", "SWIMSUIT", "BIB OVERALL",
    ],
    "lower": [
        "TROUSERS", "SKIRT", "BERMUDA", "SHORTS", "LEGGINGS", "BABY TROUSERS",
        "BABY BERMUDAS", "BABY SKIRT", "BABY LEGGINGS",
    ],
    "feet": [
        "SHOES", "FLAT SHOES", "SANDAL", "HEELED SHOES", "MOCCASINS", "SPORT SHOES",
        "RUNNING SHOES", "TRAINERS", "ANKLE BOOT", "FLAT ANKLE BOOT", "HEELED ANKLE BOOT",
        "FLAT BOOT", "HEELED BOOT", "HIGH TOPS", "BOOT", "RAIN BOOT", "ATHLETIC FOOTWEAR",
        "SPORTY SANDAL",
    ],
    "accessories": [
        "HAND BAG-RUCKSACK", "BELT", "IMIT JEWELLER", "SCARF", "SOCKS", "TIE",
        "ACCESSORIES", "GLOVES", "SHAWL/FOULARD", "PANTY/UNDERPANT", "STOCKINGS-TIGHTS",
        "BABY SOCKS", "PURSE WALLET",
    ],
}

# Regiones de crop para cada zona (y_start%, y_end%, x_start%, x_end%)
ZONE_CROPS = {
    "head":        [(0.00, 0.18, 0.15, 0.85)],
    "upper":       [(0.10, 0.50, 0.05, 0.95), (0.08, 0.45, 0.10, 0.90)],
    "full":        [(0.08, 0.80, 0.05, 0.95)],
    "lower":       [(0.45, 0.82, 0.05, 0.95), (0.40, 0.78, 0.10, 0.90)],
    "feet":        [(0.78, 1.00, 0.05, 0.95)],
    "accessories": [(0.25, 0.60, 0.00, 0.30), (0.25, 0.60, 0.70, 1.00), (0.35, 0.55, 0.10, 0.90)],
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    return match.group(1) if match else None


def extract_ts(url):
    match = re.search(r'ts=(\d+)', str(url))
    return int(match.group(1)) if match else None


def crop_zone(img, y1_pct, y2_pct, x1_pct, x2_pct):
    """Recorta una zona de la imagen por porcentajes."""
    w, h = img.size
    x1, x2 = int(w * x1_pct), int(w * x2_pct)
    y1, y2 = int(h * y1_pct), int(h * y2_pct)
    return img.crop((x1, y1, x2, y2))


def embed_single(model, preprocess, img):
    """Genera embedding de una sola imagen PIL."""
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy().flatten()


def embed_batch(model, preprocess, image_dir, ids, batch_size=64):
    """Genera embeddings en batch."""
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


def main():
    print("=== PIPELINE TOTAL ===\n")

    # --- Cargar datos ---
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}
    product_urls = {p["product_asset_id"]: p["product_image_url"] for p in products}

    # SKU mapping
    sku_to_products = defaultdict(list)
    for p in products:
        sku = extract_sku(p["product_image_url"])
        if sku:
            sku_to_products[sku].append(p["product_asset_id"])

    # Categorías válidas por sección
    cat_sections = defaultdict(set)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        desc = product_desc.get(row["product_asset_id"])
        if sec and desc:
            cat_sections[desc].add(sec)

    # Mapear cada producto a su zona
    product_zone = {}
    for p in products:
        desc = p["product_description"]
        for zone, cats in ZONE_CATEGORIES.items():
            if desc in cats:
                product_zone[p["product_asset_id"]] = zone
                break

    # --- Cargar modelo CLIP ---
    print(f"Cargando modelo en {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=DEVICE
    )
    model.eval()
    if DEVICE == "cuda":
        model = model.half()

    # --- Generar embeddings de productos (o cargar) ---
    emb_path = EMBEDDINGS_DIR / "products_vitb32.npy"
    ids_path = EMBEDDINGS_DIR / "product_ids_vitb32.json"
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    if emb_path.exists() and ids_path.exists():
        print("Cargando embeddings de productos...")
        product_embeddings = np.load(emb_path)
        with open(ids_path) as f:
            product_ids = json.load(f)
    else:
        product_ids = [p["product_asset_id"] for p in products]
        print("Generando embeddings de productos...")
        product_embeddings = embed_batch(model, preprocess, PRODUCTS_DIR, product_ids)
        np.save(emb_path, product_embeddings)
        with open(ids_path, "w") as f:
            json.dump(product_ids, f)

    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}

    # --- Construir índices FAISS por sección + zona ---
    print("Construyendo índices por sección × zona...")
    zone_section_indices = {}  # (zone, section) → (index, pids)

    for zone in ZONE_CATEGORIES:
        for sec in ["1", "2", "3"]:
            key = (zone, sec)
            zone_cats = set(ZONE_CATEGORIES[zone])
            idxs = []
            pids = []
            for pid in product_ids:
                desc = product_desc.get(pid, "")
                if desc not in zone_cats:
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
                zone_section_indices[key] = (index, pids)

    print(f"  Índices creados: {len(zone_section_indices)}")
    for key, (idx, pids) in sorted(zone_section_indices.items()):
        if len(pids) > 100:
            print(f"    {key[0]:15s} sec {key[1]}: {len(pids)} productos")

    # --- Procesar bundles de test ---
    test_bids = list(set(row["bundle_asset_id"] for row in test))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    print(f"\nProcesando {len(test_bids)} bundles...")
    results = []

    for bid in tqdm(test_bids, desc="Bundles"):
        sec = bundle_section.get(bid, "1")
        burl = bundle_url.get(bid, "")
        bsku = extract_sku(burl)

        # Cargar imagen del bundle
        try:
            bundle_img = Image.open(BUNDLES_DIR / f"{bid}.jpg").convert("RGB")
        except Exception:
            bundle_img = Image.new("RGB", (224, 224))

        candidates = []  # lista de (score, pid)
        seen = set()

        # --- SEÑAL 1: SKU exacto (score altísimo) ---
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                candidates.append((10.0, pid))
                seen.add(pid)

        # --- SEÑAL 2: Hermanos SKU (score alto) ---
        if bsku:
            prefix = bsku[:5]
            for sku, pids in sku_to_products.items():
                if sku[:5] == prefix and sku != bsku:
                    for pid in pids:
                        if pid not in seen:
                            desc = product_desc.get(pid, "")
                            if desc not in cat_sections or sec in cat_sections[desc]:
                                candidates.append((5.0, pid))
                                seen.add(pid)

        # --- SEÑAL 3: Multi-crop por zona ---
        for zone, crops in ZONE_CROPS.items():
            key = (zone, sec)
            if key not in zone_section_indices:
                continue
            index, zone_pids = zone_section_indices[key]

            # Probar cada crop de esta zona, quedarse con el mejor
            best_scores = {}
            for crop_coords in crops:
                cropped = crop_zone(bundle_img, *crop_coords)
                crop_emb = embed_single(model, preprocess, cropped)
                crop_emb = crop_emb.reshape(1, -1)
                faiss.normalize_L2(crop_emb)

                scores, indices = index.search(crop_emb, 5)
                for j in range(5):
                    pid = zone_pids[indices[0][j]]
                    s = float(scores[0][j])
                    if pid not in best_scores or s > best_scores[pid]:
                        best_scores[pid] = s

            for pid, s in best_scores.items():
                if pid not in seen:
                    candidates.append((s, pid))
                    seen.add(pid)

        # --- SEÑAL 4: Imagen completa como fallback ---
        full_emb = embed_single(model, preprocess, bundle_img).reshape(1, -1)
        faiss.normalize_L2(full_emb)

        # Buscar en todos los productos de la sección
        for zone in ZONE_CATEGORIES:
            key = (zone, sec)
            if key not in zone_section_indices:
                continue
            index, zone_pids = zone_section_indices[key]
            scores, indices = index.search(full_emb, 3)
            for j in range(3):
                pid = zone_pids[indices[0][j]]
                if pid not in seen:
                    s = float(scores[0][j]) * 0.5  # penalizar vs crop
                    candidates.append((s, pid))
                    seen.add(pid)

        # --- Ordenar por score y tomar top 15 ---
        candidates.sort(key=lambda x: -x[0])
        for score, pid in candidates[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

        # Rellenar si quedan slots
        remaining = 15 - min(len(candidates), 15)
        if remaining > 0:
            for zone in ZONE_CATEGORIES:
                key = (zone, sec)
                if key not in zone_section_indices:
                    continue
                _, zone_pids = zone_section_indices[key]
                for pid in zone_pids:
                    if pid not in seen:
                        results.append({"bundle_asset_id": bid, "product_asset_id": pid})
                        seen.add(pid)
                        remaining -= 1
                        if remaining <= 0:
                            break
                if remaining <= 0:
                    break

    # --- Guardar ---
    out_path = SUBMISSIONS_DIR / "submission_pipeline_total.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSubmission guardada en {out_path} ({len(results)} filas)")
    print(f"Media productos/bundle: {len(results)/len(test_bids):.1f}")


if __name__ == "__main__":
    main()
