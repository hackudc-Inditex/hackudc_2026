"""
Test rápido con bundles de TRAINING (donde sabemos la respuesta).
100 bundles, productos filtrados por sección.
Uso: python src/clip_mini.py
"""

import csv
import re
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

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BUNDLES = 100


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def embed_images(model, preprocess, image_dir, ids):
    all_embeddings = []
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Embeddings"):
        batch_ids = ids[i:i + BATCH_SIZE]
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
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().float().numpy())
    return np.vstack(all_embeddings)


def main():
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}

    # Ground truth: bundle → set de productos correctos
    bundle_gt = defaultdict(set)
    for row in train:
        bundle_gt[row["bundle_asset_id"]].add(row["product_asset_id"])

    # Coger 100 bundles de training (los que tienen ground truth)
    eval_bids = list(bundle_gt.keys())[:MAX_BUNDLES]
    print(f"Evaluando {len(eval_bids)} bundles de training")

    # Solo usar como candidatos: los productos que aparecen en training
    # + una muestra aleatoria del resto para simular ruido (2000 extra)
    import random
    random.seed(42)
    train_pids = set()
    for pids in bundle_gt.values():
        train_pids.update(pids)
    other_pids = [p["product_asset_id"] for p in products if p["product_asset_id"] not in train_pids]
    extra = random.sample(other_pids, min(2000, len(other_pids)))
    all_pids = list(train_pids) + extra
    print(f"Productos candidatos: {len(all_pids)} ({len(train_pids)} de training + {len(extra)} extra)")

    # Cargar modelo
    print(f"Cargando {MODEL_NAME} en {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model = model.half()

    # Embeddings de productos
    print("Generando embeddings de productos (esto tarda un rato)...")
    prod_embeddings = embed_images(model, preprocess, PRODUCTS_DIR, all_pids)
    faiss.normalize_L2(prod_embeddings)
    index = faiss.IndexFlatIP(prod_embeddings.shape[1])
    index.add(prod_embeddings)

    # Embeddings de bundles
    print("Generando embeddings de bundles...")
    bundle_embeddings = embed_images(model, preprocess, BUNDLES_DIR, eval_bids)
    faiss.normalize_L2(bundle_embeddings)

    # Buscar top-15
    scores, indices = index.search(bundle_embeddings, 15)

    # Evaluar
    total_correct = 0
    total_gt = 0
    total_predicted = 0

    print(f"\n{'='*70}")
    for i, bid in enumerate(eval_bids):
        gt = bundle_gt[bid]
        predicted = set(all_pids[indices[i][j]] for j in range(15))
        hits = predicted & gt

        total_correct += len(hits)
        total_gt += len(gt)
        total_predicted += 15

        if i < 10:  # mostrar detalle de los primeros 10
            print(f"\nBundle: {bid} (sección {bundle_section.get(bid, '?')})")
            print(f"  Productos reales ({len(gt)}): {', '.join(product_desc.get(p,'?') for p in gt)}")
            print(f"  CLIP top-5:")
            for j in range(5):
                pid = all_pids[indices[i][j]]
                desc = product_desc.get(pid, "?")
                hit = " ✓ ACIERTO!" if pid in gt else ""
                print(f"    {j+1}. {desc} (score={scores[i][j]:.3f}){hit}")
            print(f"  Aciertos en top-15: {len(hits)}/{len(gt)}")

    print(f"\n{'='*70}")
    print(f"RESUMEN ({len(eval_bids)} bundles)")
    print(f"{'='*70}")
    print(f"Productos reales total: {total_gt}")
    print(f"Aciertos en top-15: {total_correct}")
    print(f"Recall: {100*total_correct/total_gt:.1f}%")
    print(f"Precisión: {100*total_correct/total_predicted:.1f}%")


if __name__ == "__main__":
    main()
