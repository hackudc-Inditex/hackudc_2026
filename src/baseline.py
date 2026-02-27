"""
Pipeline baseline: genera embeddings CLIP de productos y bundles, busca top-15 y genera CSV.
Uso: python src/baseline.py
"""

import json
import numpy as np
import pandas as pd
import torch
import open_clip
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLES_DIR = BASE_DIR / "images" / "bundles"
PRODUCTS_DIR = BASE_DIR / "images" / "products"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Cargando {MODEL_NAME} en {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_pretrained(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model = model.half()
    return model, preprocess


def embed_images(model, preprocess, image_dir: Path, ids: list[str]) -> np.ndarray:
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


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def search_bundles(model, preprocess, bundle_ids: list[str], product_ids: list[str], index: faiss.IndexFlatIP, top_k: int = 15):
    results = []
    bundle_embeddings = embed_images(model, preprocess, BUNDLES_DIR, bundle_ids)
    faiss.normalize_L2(bundle_embeddings)
    scores, indices = index.search(bundle_embeddings, top_k)

    for i, bundle_id in enumerate(bundle_ids):
        for j in range(top_k):
            if indices[i][j] >= 0:
                results.append({
                    "bundle_asset_id": bundle_id,
                    "product_asset_id": product_ids[indices[i][j]],
                    "score": float(scores[i][j])
                })
    return results


def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess = load_model()

    # Cargar IDs de productos
    products_df = pd.read_csv(BASE_DIR / "product_dataset.csv")
    product_ids = products_df["product_asset_id"].tolist()

    # Generar o cargar embeddings de productos
    emb_path = EMBEDDINGS_DIR / "product_embeddings.npy"
    ids_path = EMBEDDINGS_DIR / "product_ids.json"

    if emb_path.exists() and ids_path.exists():
        print("Cargando embeddings de productos existentes...")
        product_embeddings = np.load(emb_path)
        with open(ids_path) as f:
            product_ids = json.load(f)
    else:
        print("Generando embeddings de productos...")
        product_embeddings = embed_images(model, preprocess, PRODUCTS_DIR, product_ids)
        np.save(emb_path, product_embeddings)
        with open(ids_path, "w") as f:
            json.dump(product_ids, f)
        print(f"Guardados {len(product_ids)} embeddings en {emb_path}")

    # Construir índice FAISS
    print("Construyendo índice FAISS...")
    index = build_faiss_index(product_embeddings.copy())

    # Cargar test bundles
    test_df = pd.read_csv(BASE_DIR / "bundles_product_match_test.csv")
    test_bundle_ids = test_df["bundle_asset_id"].unique().tolist()

    # Buscar
    print(f"Buscando matches para {len(test_bundle_ids)} bundles de test...")
    results = search_bundles(model, preprocess, test_bundle_ids, product_ids, index, top_k=15)

    # Generar CSV de submission
    submission = pd.DataFrame(results)[["bundle_asset_id", "product_asset_id"]]
    out_path = SUBMISSIONS_DIR / "submission_v1_baseline.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission guardada en {out_path} ({len(submission)} filas)")


if __name__ == "__main__":
    main()
