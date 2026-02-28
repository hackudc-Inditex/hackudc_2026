"""
PIPELINE FINAL - La versión definitiva.
YOLOv8 (detección de prendas) + CLIP (matching visual) + SKU + filtros.

Uso:
  pip install ultralytics huggingface_hub open-clip-torch faiss-cpu
  python src/pipeline_final.py

Si CUDA no funciona, instalar PyTorch con CUDA:
  pip uninstall torch torchvision -y
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLES_DIR = BASE_DIR / "images" / "bundles"
PRODUCTS_DIR = BASE_DIR / "images" / "products"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
CROPS_DIR = BASE_DIR / "crops"

# Forzar CUDA si está disponible
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    DEVICE = "cpu"
    print("AVISO: CUDA no disponible, usando CPU (será más lento)")
    print("Para usar GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

# Mapeo de categorías YOLOv8 DeepFashion2 → nuestras categorías del catálogo
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

# Categorías que YOLO no detecta pero existen en el catálogo
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


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    return match.group(1) if match else None


# ============================================================
# PASO 1: DETECCIÓN DE PRENDAS CON YOLO
# ============================================================
def load_yolo():
    print("\nCargando YOLOv8 DeepFashion2...")
    path = hf_hub_download("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt")
    model = YOLO(path)
    return model


def detect_garments(yolo_model, image_path, conf_threshold=0.3):
    """Detecta prendas en una imagen. Devuelve lista de (label, bbox, confidence)."""
    results = yolo_model(str(image_path), verbose=False, conf=conf_threshold)
    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                label = result.names[int(result.boxes.cls[i])]
                bbox = result.boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                conf = float(result.boxes.conf[i])
                detections.append((label, bbox, conf))
    return detections


def crop_detection(img, bbox, padding=0.1):
    """Recorta una detección con padding extra."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    # Añadir padding
    x1 = max(0, x1 - bw * padding)
    y1 = max(0, y1 - bh * padding)
    x2 = min(w, x2 + bw * padding)
    y2 = min(h, y2 + bh * padding)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


# ============================================================
# PASO 2: EMBEDDINGS CLIP
# ============================================================
def load_clip():
    print(f"\nCargando CLIP ViT-B-32 en {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=DEVICE
    )
    model.eval()
    if DEVICE == "cuda":
        model = model.half()
    return model, preprocess


def embed_pil(model, preprocess, img):
    """Embedding de una imagen PIL."""
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy().flatten()


def embed_batch(model, preprocess, image_dir, ids, batch_size=64):
    """Embeddings en batch."""
    all_embs = []
    for i in tqdm(range(0, len(ids), batch_size), desc="Embeddings productos"):
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


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  PIPELINE FINAL - Detección + Matching + Metadatos")
    print("=" * 60)

    # --- Cargar datos ---
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}

    product_ids = [p["product_asset_id"] for p in products]
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}

    # SKU mapping
    sku_to_products = defaultdict(list)
    for p in products:
        sku = extract_sku(p["product_image_url"])
        if sku:
            sku_to_products[sku].append(p["product_asset_id"])

    # Categorías válidas por sección (del training)
    cat_sections = defaultdict(set)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        desc = product_desc.get(row["product_asset_id"])
        if sec and desc:
            cat_sections[desc].add(sec)

    # --- Cargar modelos ---
    yolo_model = load_yolo()
    clip_model, clip_preprocess = load_clip()

    # --- Embeddings de productos (generar o cargar) ---
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = EMBEDDINGS_DIR / "products_vitb32.npy"
    ids_path = EMBEDDINGS_DIR / "product_ids_vitb32.json"

    if emb_path.exists() and ids_path.exists():
        print("\nCargando embeddings de productos existentes...")
        product_embeddings = np.load(emb_path)
        with open(ids_path) as f:
            product_ids = json.load(f)
        pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    else:
        print("\nGenerando embeddings de productos...")
        product_embeddings = embed_batch(clip_model, clip_preprocess, PRODUCTS_DIR, product_ids)
        np.save(emb_path, product_embeddings)
        with open(ids_path, "w") as f:
            json.dump(product_ids, f)
        print(f"Guardados {len(product_ids)} embeddings")

    # --- Construir índices FAISS por grupo de categoría × sección ---
    print("\nConstruyendo índices FAISS por categoría × sección...")

    # Juntar YOLO_TO_CATALOG y EXTRA_CATEGORIES
    all_category_groups = {}
    all_category_groups.update(YOLO_TO_CATALOG)
    all_category_groups.update(EXTRA_CATEGORIES)

    # Índice por (grupo, sección)
    group_indices = {}
    for group_name, catalog_cats in all_category_groups.items():
        catalog_cats_set = set(catalog_cats)
        for sec in ["1", "2", "3"]:
            idxs = []
            pids = []
            for pid in product_ids:
                desc = product_desc.get(pid, "")
                if desc not in catalog_cats_set:
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

    # También índice general por sección (fallback)
    section_indices = {}
    section_pids_map = {}
    for sec in ["1", "2", "3"]:
        idxs = []
        pids = []
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
        print(f"  Sección {sec}: {len(pids)} productos")

    # --- Procesar bundles de test ---
    test_bids = list(set(row["bundle_asset_id"] for row in test))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Procesando {len(test_bids)} bundles de test...")
    print(f"{'='*60}\n")

    results = []

    for bid in tqdm(test_bids, desc="Pipeline"):
        sec = bundle_section.get(bid, "1")
        burl = bundle_url.get(bid, "")
        bsku = extract_sku(burl)

        candidates = []  # (score, pid)
        seen = set()

        # Cargar imagen original (no la de 224x224, la descargada)
        img_path = BUNDLES_DIR / f"{bid}.jpg"
        try:
            bundle_img = Image.open(img_path).convert("RGB")
        except Exception:
            bundle_img = Image.new("RGB", (224, 224))

        # ========== SEÑAL 1: SKU exacto (score 100) ==========
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                candidates.append((100.0, pid))
                seen.add(pid)

        # ========== SEÑAL 2: SKU hermanos (score 50) ==========
        if bsku:
            prefix = bsku[:5]
            for sku, pids in sku_to_products.items():
                if sku[:5] == prefix and sku != bsku:
                    for pid in pids:
                        if pid not in seen:
                            desc = product_desc.get(pid, "")
                            if desc not in cat_sections or sec in cat_sections[desc]:
                                candidates.append((50.0, pid))
                                seen.add(pid)

        # ========== SEÑAL 3: YOLO detección + CLIP matching ==========
        detections = detect_garments(yolo_model, img_path)

        for label, bbox, yolo_conf in detections:
            # Recortar la prenda detectada
            crop = crop_detection(bundle_img, bbox, padding=0.05)

            # Embedding del recorte
            crop_emb = embed_pil(clip_model, clip_preprocess, crop)
            crop_emb = crop_emb.reshape(1, -1)
            faiss.normalize_L2(crop_emb)

            # Buscar en el índice de la categoría correspondiente
            key = (label, sec)
            if key in group_indices:
                index, gpids = group_indices[key]
                k = min(5, index.ntotal)
                scores, indices = index.search(crop_emb, k)
                for j in range(k):
                    pid = gpids[indices[0][j]]
                    if pid not in seen:
                        # Score = similitud CLIP × confianza YOLO
                        clip_score = float(scores[0][j])
                        combined = clip_score * (1 + yolo_conf) * 10
                        candidates.append((combined, pid))
                        seen.add(pid)

        # ========== SEÑAL 4: Buscar categorías que YOLO no detecta ==========
        # Zapatos, bolsos, gorros, accesorios - YOLO DeepFashion2 no los detecta
        # Usar imagen completa para estos
        full_emb = embed_pil(clip_model, clip_preprocess, bundle_img)
        full_emb = full_emb.reshape(1, -1)
        faiss.normalize_L2(full_emb)

        for extra_group in EXTRA_CATEGORIES:
            key = (extra_group, sec)
            if key in group_indices:
                index, gpids = group_indices[key]
                k = min(3, index.ntotal)
                scores, indices = index.search(full_emb, k)
                for j in range(k):
                    pid = gpids[indices[0][j]]
                    if pid not in seen:
                        clip_score = float(scores[0][j])
                        candidates.append((clip_score * 5, pid))
                        seen.add(pid)

        # ========== SEÑAL 5: Fallback con índice general de sección ==========
        if len(candidates) < 15 and sec in section_indices:
            index = section_indices[sec]
            spids = section_pids_map[sec]
            scores, indices = index.search(full_emb, 30)
            for j in range(30):
                if len(candidates) >= 20:
                    break
                pid = spids[indices[0][j]]
                if pid not in seen:
                    candidates.append((float(scores[0][j]), pid))
                    seen.add(pid)

        # ========== Ordenar por score, tomar top 15 ==========
        candidates.sort(key=lambda x: -x[0])
        for score, pid in candidates[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

    # --- Guardar ---
    out_path = SUBMISSIONS_DIR / "submission_final.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"SUBMISSION GUARDADA: {out_path}")
    print(f"Total: {len(results)} filas ({len(results)/len(test_bids):.1f} productos/bundle)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
