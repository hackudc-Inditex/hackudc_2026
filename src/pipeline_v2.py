"""
PIPELINE V2 - Mejoras sobre pipeline_final:
  - Usa imágenes de bundles en ALTA RESOLUCIÓN para YOLO
  - Usa ViT-L-14 si los embeddings están disponibles
  - Mejor scoring y más detecciones

Uso:
  python src/download_hires.py    (primero, descargar bundles en alta res)
  python src/pipeline_v2.py
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
BUNDLES_HIRES_DIR = BASE_DIR / "images" / "bundles_hires"
BUNDLES_DIR = BASE_DIR / "images" / "bundles"  # fallback a 224px
PRODUCTS_DIR = BASE_DIR / "images" / "products"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
else:
    DEVICE = "cpu"
    print("AVISO: Sin GPU, usando CPU")

# Mapeo YOLO DeepFashion2 → categorías catálogo
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


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    return match.group(1) if match else None


def get_bundle_image_path(bid):
    """Usa alta resolución si existe, sino fallback a 224px."""
    hires = BUNDLES_HIRES_DIR / f"{bid}.jpg"
    if hires.exists():
        return hires
    return BUNDLES_DIR / f"{bid}.jpg"


def load_yolo():
    print("\nCargando YOLOv8 DeepFashion2...")
    path = hf_hub_download("Bingsu/adetailer", "deepfashion2_yolov8s-seg.pt")
    model = YOLO(path)
    return model


def detect_garments(yolo_model, image_path, conf_threshold=0.25):
    results = yolo_model(str(image_path), verbose=False, conf=conf_threshold)
    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                label = result.names[int(result.boxes.cls[i])]
                bbox = result.boxes.xyxy[i].cpu().numpy()
                conf = float(result.boxes.conf[i])
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


def load_clip():
    # Detectar qué embeddings tenemos para usar el modelo correcto
    if (EMBEDDINGS_DIR / "product_embeddings.npy").exists():
        model_name = "ViT-L-14"
        print(f"\nCargando CLIP {model_name} (embeddings ViT-L-14 encontrados)")
    else:
        model_name = "ViT-B-32"
        print(f"\nCargando CLIP {model_name}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device=DEVICE
    )
    model.eval()
    if DEVICE == "cuda":
        model = model.half()
    return model, preprocess, model_name


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


def main():
    print("=" * 60)
    print("  PIPELINE V2 - Alta resolución + mejor matching")
    print("=" * 60)

    # --- Verificar imágenes hires ---
    if not BUNDLES_HIRES_DIR.exists() or len(list(BUNDLES_HIRES_DIR.glob("*.jpg"))) == 0:
        print("\nAVISO: No hay imágenes de alta resolución.")
        print("Ejecuta primero: python src/download_hires.py")
        print("Continuando con imágenes de 224px (resultados serán peores)...\n")

    # --- Cargar datos ---
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}

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

    # --- Cargar modelos ---
    yolo_model = load_yolo()
    clip_model, clip_preprocess, clip_name = load_clip()

    # --- Embeddings de productos ---
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Buscar embeddings existentes
    if clip_name == "ViT-L-14" and (EMBEDDINGS_DIR / "product_embeddings.npy").exists():
        emb_path = EMBEDDINGS_DIR / "product_embeddings.npy"
        ids_path = EMBEDDINGS_DIR / "product_ids.json"
    elif (EMBEDDINGS_DIR / "products_vitb32.npy").exists():
        emb_path = EMBEDDINGS_DIR / "products_vitb32.npy"
        ids_path = EMBEDDINGS_DIR / "product_ids_vitb32.json"
    else:
        # Generar
        product_ids = [p["product_asset_id"] for p in products]
        tag = "vitl14" if clip_name == "ViT-L-14" else "vitb32"
        emb_path = EMBEDDINGS_DIR / f"products_{tag}.npy"
        ids_path = EMBEDDINGS_DIR / f"product_ids_{tag}.json"
        print("Generando embeddings de productos...")
        product_embeddings = embed_batch(clip_model, clip_preprocess, PRODUCTS_DIR, product_ids)
        np.save(emb_path, product_embeddings)
        with open(ids_path, "w") as f:
            json.dump(product_ids, f)

    print(f"Cargando embeddings de {emb_path.name}...")
    product_embeddings = np.load(emb_path)
    with open(ids_path) as f:
        product_ids = json.load(f)
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    print(f"  {len(product_ids)} productos, dim={product_embeddings.shape[1]}")

    # --- Índices FAISS por categoría × sección ---
    print("Construyendo índices FAISS...")
    all_groups = {}
    all_groups.update(YOLO_TO_CATALOG)
    all_groups.update(EXTRA_CATEGORIES)

    group_indices = {}
    for group_name, catalog_cats in all_groups.items():
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

    # --- Procesar test ---
    test_bids = list(set(row["bundle_asset_id"] for row in test))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    print(f"\nProcesando {len(test_bids)} bundles...\n")
    results = []
    yolo_detection_count = 0

    for bid in tqdm(test_bids, desc="Pipeline V2"):
        sec = bundle_section.get(bid, "1")
        burl = bundle_url.get(bid, "")
        bsku = extract_sku(burl)

        candidates = []
        seen = set()

        # Cargar imagen
        img_path = get_bundle_image_path(bid)
        try:
            bundle_img = Image.open(img_path).convert("RGB")
        except Exception:
            bundle_img = Image.new("RGB", (224, 224))

        # ===== SEÑAL 1: SKU exacto =====
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                candidates.append((100.0, pid))
                seen.add(pid)

        # ===== SEÑAL 2: SKU hermanos =====
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

        # ===== SEÑAL 3: YOLO detección + CLIP matching =====
        detections = detect_garments(yolo_model, img_path, conf_threshold=0.2)
        yolo_detection_count += len(detections)

        for label, bbox, yolo_conf in detections:
            crop = crop_detection(bundle_img, bbox, padding=0.08)
            crop_emb = embed_pil(clip_model, clip_preprocess, crop).reshape(1, -1)
            faiss.normalize_L2(crop_emb)

            # Buscar en categoría específica
            key = (label, sec)
            if key in group_indices:
                index, gpids = group_indices[key]
                k = min(8, index.ntotal)
                scores, indices = index.search(crop_emb, k)
                for j in range(k):
                    pid = gpids[indices[0][j]]
                    if pid not in seen:
                        clip_score = float(scores[0][j])
                        combined = clip_score * (1 + yolo_conf) * 15
                        candidates.append((combined, pid))
                        seen.add(pid)

        # ===== SEÑAL 4: Accesorios (YOLO no detecta) =====
        full_emb = embed_pil(clip_model, clip_preprocess, bundle_img).reshape(1, -1)
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
                        candidates.append((float(scores[0][j]) * 8, pid))
                        seen.add(pid)

        # ===== SEÑAL 5: Fallback sección completa =====
        if len(candidates) < 15 and sec in section_indices:
            scores, indices = section_indices[sec].search(full_emb, 30)
            spids = section_pids_map[sec]
            for j in range(30):
                if len(candidates) >= 20:
                    break
                pid = spids[indices[0][j]]
                if pid not in seen:
                    candidates.append((float(scores[0][j]) * 2, pid))
                    seen.add(pid)

        # Top 15
        candidates.sort(key=lambda x: -x[0])
        for score, pid in candidates[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

    # Guardar
    out_path = SUBMISSIONS_DIR / "submission_pipeline_v2.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"YOLO detecciones totales: {yolo_detection_count} (media {yolo_detection_count/len(test_bids):.1f}/bundle)")
    print(f"Submission: {out_path} ({len(results)} filas)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
