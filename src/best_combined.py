"""
Combina TODO: SKU exacto + hermanos + CLIP visual + filtros sección/timestamp.
Usa embeddings ya generados por clip_fast.py o baseline.py.
Uso: python src/best_combined.py
"""

import csv
import re
import json
import numpy as np
import faiss
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
SUBMISSIONS_DIR = BASE_DIR / "submissions"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    return match.group(1) if match else None


def extract_ts(url):
    match = re.search(r'ts=(\d+)', str(url))
    return int(match.group(1)) if match else None


def main():
    # Buscar embeddings disponibles (preferir ViT-L-14, luego ViT-B-32)
    if (EMBEDDINGS_DIR / "product_embeddings.npy").exists():
        emb_path = EMBEDDINGS_DIR / "product_embeddings.npy"
        ids_path = EMBEDDINGS_DIR / "product_ids.json"
        print("Usando embeddings ViT-L-14")
    elif (EMBEDDINGS_DIR / "products_vitb32.npy").exists():
        emb_path = EMBEDDINGS_DIR / "products_vitb32.npy"
        ids_path = EMBEDDINGS_DIR / "product_ids_vitb32.json"
        print("Usando embeddings ViT-B-32")
    else:
        print("ERROR: No hay embeddings. Ejecuta primero baseline.py o clip_fast.py")
        return

    # Cargar embeddings
    print("Cargando embeddings...")
    product_embeddings = np.load(emb_path)
    with open(ids_path) as f:
        product_ids = json.load(f)
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}

    # Cargar datos
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}
    product_urls = {p["product_asset_id"]: p["product_image_url"] for p in products}

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

    # Construir índice FAISS por sección (filtrado por categoría + timestamp no se puede en FAISS, lo hacemos manual)
    print("Construyendo índices FAISS por sección...")
    section_indices = {}
    section_pids = {}

    for sec in ["1", "2", "3"]:
        sec_idxs = []
        sec_pids = []
        for pid in product_ids:
            desc = product_desc.get(pid, "")
            if desc not in cat_sections or sec in cat_sections[desc]:
                if pid in pid_to_idx:
                    sec_idxs.append(pid_to_idx[pid])
                    sec_pids.append(pid)

        sec_embs = product_embeddings[sec_idxs].copy()
        faiss.normalize_L2(sec_embs)
        index = faiss.IndexFlatIP(sec_embs.shape[1])
        index.add(sec_embs)
        section_indices[sec] = index
        section_pids[sec] = sec_pids
        print(f"  Sección {sec}: {len(sec_pids)} productos")

    # Cargar embeddings de bundles (si existen) o buscar en el archivo de productos
    # Los bundles de test necesitan embeddings - buscar archivo
    bundle_emb_path = EMBEDDINGS_DIR / "bundle_test_embeddings.npy"
    bundle_ids_path = EMBEDDINGS_DIR / "bundle_test_ids.json"

    test_bids = list(set(row["bundle_asset_id"] for row in test))

    if bundle_emb_path.exists() and bundle_ids_path.exists():
        print("Cargando embeddings de bundles...")
        bundle_embeddings = np.load(bundle_emb_path)
        with open(bundle_ids_path) as f:
            saved_bids = json.load(f)
        bid_to_emb = {bid: bundle_embeddings[i] for i, bid in enumerate(saved_bids)}
    else:
        # Generar embeddings de bundles
        print("Generando embeddings de bundles (necesita modelo)...")
        import torch
        import open_clip
        from PIL import Image
        from tqdm import tqdm

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        BUNDLES_DIR = BASE_DIR / "images" / "bundles"

        # Detectar qué modelo usar
        dim = product_embeddings.shape[1]
        if dim == 768:
            model_name, pretrained = "ViT-L-14", "openai"
        else:
            model_name, pretrained = "ViT-B-32", "openai"

        print(f"Cargando {model_name} en {DEVICE}...")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=DEVICE)
        model.eval()
        if DEVICE == "cuda":
            model = model.half()

        all_embs = []
        BS = 32
        for i in tqdm(range(0, len(test_bids), BS), desc="Bundles"):
            batch_ids = test_bids[i:i+BS]
            images = []
            for bid in batch_ids:
                try:
                    img = preprocess(Image.open(BUNDLES_DIR / f"{bid}.jpg").convert("RGB"))
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

        bundle_embeddings = np.vstack(all_embs)
        np.save(bundle_emb_path, bundle_embeddings)
        with open(bundle_ids_path, "w") as f:
            json.dump(test_bids, f)
        bid_to_emb = {bid: bundle_embeddings[i] for i, bid in enumerate(test_bids)}
        del model
        print("Embeddings de bundles guardados")

    # === GENERAR SUBMISSION ===
    print(f"\nGenerando submission para {len(test_bids)} bundles...")
    days_90_ms = 90 * 86400 * 1000
    results = []

    for bid in test_bids:
        sec = bundle_section.get(bid, "1")
        burl = bundle_url.get(bid, "")
        bsku = extract_sku(burl)
        bts = extract_ts(burl)

        selected = []
        selected_set = set()

        # --- CAPA 1: SKU exacto (confianza máxima) ---
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                selected.append(pid)
                selected_set.add(pid)

        # --- CAPA 2: CLIP top matches de la sección ---
        if bid in bid_to_emb and sec in section_indices:
            query = bid_to_emb[bid].reshape(1, -1).copy()
            faiss.normalize_L2(query)
            scores, indices = section_indices[sec].search(query, 50)
            for j in range(50):
                if len(selected) >= 12:  # dejar 3 slots para hermanos
                    break
                pid = section_pids[sec][indices[0][j]]
                if pid not in selected_set:
                    selected.append(pid)
                    selected_set.add(pid)

        # --- CAPA 3: Hermanos SKU (prefijo 5, filtrados por sección) ---
        if bsku:
            prefix = bsku[:5]
            for sku, pids in sku_to_products.items():
                if sku[:5] == prefix and sku != bsku:
                    for pid in pids:
                        if pid not in selected_set:
                            desc = product_desc.get(pid, "")
                            if desc not in cat_sections or sec in cat_sections[desc]:
                                selected.append(pid)
                                selected_set.add(pid)
                        if len(selected) >= 15:
                            break
                if len(selected) >= 15:
                    break

        # --- CAPA 4: Si quedan slots, más CLIP ---
        if len(selected) < 15 and bid in bid_to_emb and sec in section_indices:
            query = bid_to_emb[bid].reshape(1, -1).copy()
            faiss.normalize_L2(query)
            scores, indices = section_indices[sec].search(query, 100)
            for j in range(100):
                if len(selected) >= 15:
                    break
                pid = section_pids[sec][indices[0][j]]
                if pid not in selected_set:
                    selected.append(pid)
                    selected_set.add(pid)

        for pid in selected[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

    # Guardar
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / "submission_best_combined.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Submission guardada en {out_path} ({len(results)} filas)")
    print(f"Media productos/bundle: {len(results)/len(test_bids):.1f}")


if __name__ == "__main__":
    main()
