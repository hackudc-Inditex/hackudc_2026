"""
Combina todos los filtros de datos descubiertos + relleno aleatorio hasta 15.
Sin visión artificial. Solo explotación de metadatos.
Uso: python src/combined_filters.py
"""

import csv
import re
import random
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
random.seed(42)


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
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    train = load_csv(BASE_DIR / "bundles_product_match_train.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    # Mapas básicos
    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}

    # --- FILTRO 1: Categorías válidas por sección (del training) ---
    cat_sections = defaultdict(set)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        desc = product_desc.get(row["product_asset_id"])
        if sec and desc:
            cat_sections[desc].add(sec)

    # --- Preparar datos de productos ---
    product_data = []
    sku_to_products = defaultdict(list)

    for p in products:
        pid = p["product_asset_id"]
        sku = extract_sku(p["product_image_url"])
        ts = extract_ts(p["product_image_url"])
        desc = p["product_description"]
        product_data.append({"pid": pid, "sku": sku, "ts": ts, "desc": desc})
        if sku:
            sku_to_products[sku].append(pid)

    # --- Procesar cada bundle de test ---
    test_bids = list(set(row["bundle_asset_id"] for row in test))
    days_90_ms = 90 * 86400 * 1000

    results = []
    stats = {"url_exact": 0, "url_prefix": 0, "filtered": 0, "random_fill": 0}

    for bid in test_bids:
        sec = bundle_section.get(bid)
        burl = bundle_url.get(bid, "")
        bsku = extract_sku(burl)
        bts = extract_ts(burl)

        selected = []  # lista ordenada por confianza
        selected_set = set()

        # --- CAPA 1: Match exacto por SKU (máxima confianza) ---
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                if pid not in selected_set:
                    selected.append(pid)
                    selected_set.add(pid)
                    stats["url_exact"] += 1

        # --- CAPA 2: Match por prefijo SKU 5 dígitos (hermanos) ---
        if bsku:
            prefix = bsku[:5]
            siblings = []
            for sku, pids in sku_to_products.items():
                if sku[:5] == prefix and sku != bsku:
                    siblings.extend(pids)
            # Filtrar hermanos por sección
            for pid in siblings:
                if pid not in selected_set:
                    desc = product_desc.get(pid, "")
                    # Solo incluir si la categoría es válida para esta sección
                    if desc not in cat_sections or sec in cat_sections[desc]:
                        selected.append(pid)
                        selected_set.add(pid)
                        stats["url_prefix"] += 1
                if len(selected) >= 15:
                    break

        # --- CAPA 3: Rellenar con productos filtrados por sección + timestamp ---
        if len(selected) < 15:
            candidates = []
            for p in product_data:
                if p["pid"] in selected_set:
                    continue
                # Filtro sección: categoría válida para esta sección
                desc = p["desc"]
                if desc in cat_sections and sec not in cat_sections[desc]:
                    continue
                # Filtro timestamp: ±90 días
                if bts and p["ts"] and abs(p["ts"] - bts) > days_90_ms:
                    continue
                candidates.append(p["pid"])

            # Elegir aleatoriamente de los candidatos filtrados
            needed = 15 - len(selected)
            if len(candidates) > needed:
                random_picks = random.sample(candidates, needed)
            else:
                random_picks = candidates

            for pid in random_picks:
                selected.append(pid)
                selected_set.add(pid)
                stats["random_fill"] += 1

        # Guardar resultados (max 15)
        for pid in selected[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

    # --- Guardar submission ---
    out_dir = BASE_DIR / "submissions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "submission_combined_filters.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"=== ESTADÍSTICAS ===")
    print(f"Bundles procesados: {len(test_bids)}")
    print(f"Productos por URL exacta: {stats['url_exact']}")
    print(f"Productos por SKU hermano: {stats['url_prefix']}")
    print(f"Relleno aleatorio (filtrado): {stats['random_fill']}")
    print(f"Total predicciones: {len(results)}")
    print(f"Media productos/bundle: {len(results)/len(test_bids):.1f}")
    print(f"\nSubmission guardada en {out_path}")


if __name__ == "__main__":
    main()
