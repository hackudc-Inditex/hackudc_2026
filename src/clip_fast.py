"""
Pipeline rápido: CLIP ViT-B/32 (modelo ligero) + SKU exacto + filtros.
Mucho más rápido que ViT-L/14. Para ver resultados ya.
Uso: python src/clip_fast.py
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

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    return match.group(1) if match else None


def extract_ts(url):
    match = re.search(r'ts=(\d+)', str(url))
    return int(match.group(1)) if match else None


def load_model():
    print(f"Cargando {MODEL_NAME} en {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model = model.half()
    return model, preprocess


def embed_images(model, preprocess, image_dir, ids):
    all_embeddings = []
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc=f"Embeddings {image_dir.name}"):
        batch_ids = ids[i:i + BATCH_SIZE]
        images = []
        for img_id in batch_ids:
            img_path = image_dir / f"{img_id}.jpg"
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
                images.append(img)
            except Exception:
                images.append(preprocess(Image.new("RGB", (224, 224))))

        batch = torch.stack(images).to(DEVICE)
        if DEVICE == "cuda":
            batch = batch.half()
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().float().numpy())

    return np.vstack(all_embeddings)


def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}
    product_urls = {p["product_asset_id"]: p["product_image_url"] for p in products}

    product_ids = [p["product_asset_id"] for p in products]

    # SKU → productos
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

    # Cargar modelo
    model, preprocess = load_model()

    # Generar o cargar embeddings de productos
    emb_path = EMBEDDINGS_DIR / "products_vitb32.npy"
    ids_path = EMBEDDINGS_DIR / "product_ids_vitb32.json"

    if emb_path.exists() and ids_path.exists():
        print("Cargando embeddings existentes...")
        product_embeddings = np.load(emb_path)
        with open(ids_path) as f:
            product_ids = json.load(f)
    else:
        print("Generando embeddings de productos...")
        product_embeddings = embed_images(model, preprocess, PRODUCTS_DIR, product_ids)
        np.save(emb_path, product_embeddings)
        with open(ids_path, "w") as f:
            json.dump(product_ids, f)
        print(f"Guardados {len(product_ids)} embeddings")

    # Crear mapa pid → índice
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}

    # Índice FAISS por sección
    print("Construyendo índices FAISS por sección...")
    section_indices = {}
    section_pids = {}

    for sec in ["1", "2", "3"]:
        sec_product_indices = []
        sec_product_ids = []
        for pid in product_ids:
            desc = product_desc.get(pid, "")
            # Incluir si categoría válida para sección o categoría desconocida
            if desc not in cat_sections or sec in cat_sections[desc]:
                idx = pid_to_idx[pid]
                sec_product_indices.append(idx)
                sec_product_ids.append(pid)

        sec_embeddings = product_embeddings[sec_product_indices].copy()
        faiss.normalize_L2(sec_embeddings)
        index = faiss.IndexFlatIP(sec_embeddings.shape[1])
        index.add(sec_embeddings)
        section_indices[sec] = index
        section_pids[sec] = sec_product_ids
        print(f"  Sección {sec}: {len(sec_product_ids)} productos")

    # Procesar test bundles
    test_bids = list(set(row["bundle_asset_id"] for row in test))
    print(f"\nProcesando {len(test_bids)} bundles de test...")

    # Generar embeddings de bundles
    bundle_embeddings = embed_images(model, preprocess, BUNDLES_DIR, test_bids)
    faiss.normalize_L2(bundle_embeddings)

    results = []
    for i, bid in enumerate(test_bids):
        sec = bundle_section.get(bid, "1")
        bsku = extract_sku(bundle_url.get(bid, ""))

        selected = []
        selected_set = set()

        # SLOT 1: SKU exacto
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                if pid not in selected_set:
                    selected.append(pid)
                    selected_set.add(pid)

        # SLOTS 2-15: CLIP top similares de la sección
        if sec in section_indices:
            query = bundle_embeddings[i:i+1]
            scores, indices = section_indices[sec].search(query, 30)
            for j in range(30):
                if len(selected) >= 15:
                    break
                pid = section_pids[sec][indices[0][j]]
                if pid not in selected_set:
                    selected.append(pid)
                    selected_set.add(pid)

        for pid in selected[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

    # Guardar
    out_path = SUBMISSIONS_DIR / "submission_clip_fast.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSubmission guardada en {out_path} ({len(results)} filas)")


if __name__ == "__main__":
    main()
