"""
SKU exacto PRIMERO, luego rellena hasta 15 con candidatos filtrados aleatorios.
Sirve para confirmar si rellenar ayuda o perjudica el score.
Uso: python src/exact_first_then_fill.py
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

    bundle_section = {b["bundle_asset_id"]: b["bundle_id_section"] for b in bundles}
    bundle_url = {b["bundle_asset_id"]: b["bundle_image_url"] for b in bundles}
    product_desc = {p["product_asset_id"]: p["product_description"] for p in products}

    # Categorías válidas por sección
    cat_sections = defaultdict(set)
    for row in train:
        sec = bundle_section.get(row["bundle_asset_id"])
        desc = product_desc.get(row["product_asset_id"])
        if sec and desc:
            cat_sections[desc].add(sec)

    # SKU → productos
    sku_to_products = defaultdict(list)
    product_data = []
    for p in products:
        sku = extract_sku(p["product_image_url"])
        ts = extract_ts(p["product_image_url"])
        if sku:
            sku_to_products[sku].append(p["product_asset_id"])
        product_data.append({"pid": p["product_asset_id"], "sku": sku, "ts": ts, "desc": p["product_description"]})

    test_bids = list(set(row["bundle_asset_id"] for row in test))
    days_90_ms = 90 * 86400 * 1000

    results = []

    for bid in test_bids:
        sec = bundle_section.get(bid)
        burl = bundle_url.get(bid, "")
        bsku = extract_sku(burl)
        bts = extract_ts(burl)

        selected = []
        selected_set = set()

        # PASO 1: SKU exacto PRIMERO
        if bsku and bsku in sku_to_products:
            for pid in sku_to_products[bsku]:
                selected.append(pid)
                selected_set.add(pid)

        # PASO 2: Rellenar hasta 15 con aleatorios filtrados (sección + timestamp)
        if len(selected) < 15:
            candidates = []
            for p in product_data:
                if p["pid"] in selected_set:
                    continue
                desc = p["desc"]
                if desc in cat_sections and sec not in cat_sections[desc]:
                    continue
                if bts and p["ts"] and abs(p["ts"] - bts) > days_90_ms:
                    continue
                candidates.append(p["pid"])

            needed = 15 - len(selected)
            if len(candidates) > needed:
                picks = random.sample(candidates, needed)
            else:
                picks = candidates

            for pid in picks:
                selected.append(pid)
                selected_set.add(pid)

        for pid in selected[:15]:
            results.append({"bundle_asset_id": bid, "product_asset_id": pid})

    out_dir = BASE_DIR / "submissions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "submission_exact_first_fill.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Total predicciones: {len(results)}")
    print(f"Media productos/bundle: {len(results)/len(test_bids):.1f}")
    print(f"Submission guardada en {out_path}")


if __name__ == "__main__":
    main()
