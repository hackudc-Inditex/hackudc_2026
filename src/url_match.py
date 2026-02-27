"""
Match de productos por código SKU extraído de las URLs de Zara.
Uso: python src/url_match.py
"""

import csv
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent


def extract_sku(url):
    """Extrae el número SKU de una URL de Zara (el número antes de -p o -e1)."""
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    if match:
        return match.group(1)
    return None


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    # Mapear SKU → lista de product_asset_id
    sku_to_products = defaultdict(list)
    for p in products:
        sku = extract_sku(p["product_image_url"])
        if sku:
            sku_to_products[sku].append(p["product_asset_id"])

    # Mapear bundle_asset_id → URL
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}

    # Para cada bundle de test, buscar matches por SKU
    test_bundle_ids = list(set(row["bundle_asset_id"] for row in test))

    results = []
    bundles_con_match = 0
    bundles_sin_match = 0

    for bid in test_bundle_ids:
        url = bundle_url.get(bid, "")
        bundle_sku = extract_sku(url)

        matched_products = set()

        if bundle_sku:
            # 1. Match exacto: mismo SKU
            if bundle_sku in sku_to_products:
                for pid in sku_to_products[bundle_sku]:
                    matched_products.add(pid)

            # 2. Match por prefijo 5 dígitos (productos "hermanos")
            prefix = bundle_sku[:5]
            for sku, pids in sku_to_products.items():
                if sku[:5] == prefix:
                    for pid in pids:
                        matched_products.add(pid)

        if matched_products:
            bundles_con_match += 1
            for pid in list(matched_products)[:15]:
                results.append({"bundle_asset_id": bid, "product_asset_id": pid})
        else:
            bundles_sin_match += 1

    # Guardar submission
    out_dir = BASE_DIR / "submissions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "submission_url_match.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Bundles con match por URL: {bundles_con_match}/{len(test_bundle_ids)}")
    print(f"Bundles sin match: {bundles_sin_match}/{len(test_bundle_ids)}")
    print(f"Total predicciones: {len(results)}")
    print(f"Submission guardada en {out_path}")


if __name__ == "__main__":
    main()
