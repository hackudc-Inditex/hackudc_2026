"""
Solo matches exactos por SKU de URL. Máxima confianza, sin relleno.
Uso: python src/url_exact_only.py
"""

import csv
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_sku(url):
    match = re.search(r'/(\d{8,15}(?:-\d+)?)-[pe]', str(url))
    return match.group(1) if match else None


def main():
    bundles = load_csv(BASE_DIR / "bundles_dataset.csv")
    products = load_csv(BASE_DIR / "product_dataset.csv")
    test = load_csv(BASE_DIR / "bundles_product_match_test.csv")

    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}

    # SKU → productos (solo exacto)
    sku_to_products = defaultdict(list)
    for p in products:
        sku = extract_sku(p["product_image_url"])
        if sku:
            sku_to_products[sku].append(p["product_asset_id"])

    test_bids = list(set(row["bundle_asset_id"] for row in test))

    results = []
    bundles_con = 0
    bundles_sin = 0

    for bid in test_bids:
        bsku = extract_sku(bundle_url.get(bid, ""))
        if bsku and bsku in sku_to_products:
            bundles_con += 1
            for pid in sku_to_products[bsku]:
                results.append({"bundle_asset_id": bid, "product_asset_id": pid})
        else:
            bundles_sin += 1

    out_dir = BASE_DIR / "submissions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "submission_url_exact_only.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Bundles con match exacto: {bundles_con}/{len(test_bids)}")
    print(f"Bundles sin match: {bundles_sin}/{len(test_bids)}")
    print(f"Total predicciones: {len(results)}")
    print(f"Submission guardada en {out_path}")


if __name__ == "__main__":
    main()
