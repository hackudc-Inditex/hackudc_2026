"""
URL Pattern Analysis for Bundle-Product Matching
Research script - no submission files created.
"""

import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
from collections import Counter, defaultdict
import numpy as np

# ─────────────────────────────────────────
# 1. Load all CSVs
# ─────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

bundles  = pd.read_csv("/home/jorge/Escritorio/hackudc/bundles_dataset.csv")
products = pd.read_csv("/home/jorge/Escritorio/hackudc/product_dataset.csv")
train    = pd.read_csv("/home/jorge/Escritorio/hackudc/bundles_product_match_train.csv")

print(f"Bundles  : {len(bundles):,} rows")
print(f"Products : {len(products):,} rows")
print(f"Train    : {len(train):,} rows")

# ─────────────────────────────────────────
# 2. URL helper functions
# ─────────────────────────────────────────

def parse_url(url: str):
    """Return dict with useful URL components."""
    if not isinstance(url, str):
        return {}
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    path = parsed.path                       # e.g. /assets/public/xxxx/yyyy/.../SKU-p/SKU-p.jpg
    segments = [s for s in path.split("/") if s]
    # The last directory before the filename is typically  "SKU-suffix"
    # e.g. "02878302711-p"  or  "T9243982705-p"  or  "09942310250-200-p"
    # The SKU lives in segment[-2] (or the filename stem)
    sku_segment = segments[-2] if len(segments) >= 2 else ""
    # Strip trailing suffix (-p, -e1, -200-p …)
    sku_raw = re.sub(r'[-_](p|e\d+)$', '', sku_segment)   # e.g. "02878302711" or "09942310250-200"
    sku_core = re.split(r'[-_]', sku_raw)[0]               # first token only
    ts_val = int(qs["ts"][0]) if "ts" in qs else None
    # Hash segments: the 4×4-char hex blocks after /public/
    try:
        pub_idx = segments.index("public")
        hash_segs = segments[pub_idx + 1: pub_idx + 5]    # 4 hash segments
    except ValueError:
        hash_segs = []
    return {
        "path": path,
        "segments": segments,
        "sku_segment": sku_segment,
        "sku_raw": sku_raw,
        "sku_core": sku_core,
        "ts": ts_val,
        "hash_segs": hash_segs,
        "hash_path": "/".join(hash_segs),
    }


def sku_prefix(sku: str, n: int) -> str:
    """First n characters of the SKU core."""
    return sku[:n] if len(sku) >= n else sku


# Parse all bundle and product URLs
bundles["_parsed"]  = bundles["bundle_image_url"].apply(parse_url)
products["_parsed"] = products["product_image_url"].apply(parse_url)

# Flatten into columns
for col in ["sku_segment", "sku_raw", "sku_core", "ts", "hash_path"]:
    bundles[col]  = bundles["_parsed"].apply(lambda d: d.get(col))
    products[col] = products["_parsed"].apply(lambda d: d.get(col))

# ─────────────────────────────────────────
# 3. Merge training pairs with URL data
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("MERGING TRAINING PAIRS WITH URL DATA")
print("=" * 70)

merged = train.merge(
    bundles[["bundle_asset_id", "bundle_image_url", "sku_segment", "sku_raw",
             "sku_core", "ts", "hash_path"]],
    on="bundle_asset_id", how="left"
).merge(
    products[["product_asset_id", "product_image_url",
              "sku_segment", "sku_raw", "sku_core", "ts", "hash_path"]],
    on="product_asset_id", how="left",
    suffixes=("_b", "_p")
)

print(f"Merged training pairs: {len(merged):,}")
print(f"  Missing bundle URL : {merged['bundle_image_url'].isna().sum()}")
print(f"  Missing product URL: {merged['product_image_url'].isna().sum()}")

# ─────────────────────────────────────────
# 4. Exact SKU segment match
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("4. EXACT SKU SEGMENT MATCH  (e.g. '02878302711-p' vs '02878302711-e1')")
print("=" * 70)

# The bundle suffix is -p, the product suffix is -e1; the SKU_RAW should match
merged["sku_raw_match"] = merged["sku_raw_b"] == merged["sku_raw_p"]
merged["sku_core_match"] = merged["sku_core_b"] == merged["sku_core_p"]

n = len(merged)
print(f"  sku_raw exact match : {merged['sku_raw_match'].sum():5d} / {n}  "
      f"({100*merged['sku_raw_match'].mean():.2f}%)")
print(f"  sku_core exact match: {merged['sku_core_match'].sum():5d} / {n}  "
      f"({100*merged['sku_core_match'].mean():.2f}%)")

# Show a few non-matching examples
non_match = merged[~merged["sku_raw_match"]].head(10)
if len(non_match):
    print("\n  Sample NON-matching sku_raw pairs:")
    for _, row in non_match.iterrows():
        print(f"    bundle sku_raw={row['sku_raw_b']!r:30s}  "
              f"product sku_raw={row['sku_raw_p']!r}")

# ─────────────────────────────────────────
# 5. Path structure analysis
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("5. URL PATH STRUCTURE ANALYSIS")
print("=" * 70)

def extract_path_dir(url):
    """Return the /photos/... directory part (everything before the last component)."""
    if not isinstance(url, str):
        return ""
    parsed = urlparse(url)
    path = parsed.path
    # Remove filename, keep directory
    parts = path.rsplit("/", 1)
    return parts[0] if len(parts) == 2 else path

merged["dir_b"] = merged["bundle_image_url"].apply(extract_path_dir)
merged["dir_p"] = merged["product_image_url"].apply(extract_path_dir)
merged["same_dir"] = merged["dir_b"] == merged["dir_p"]

print(f"  Same directory path : {merged['same_dir'].sum():5d} / {n}  "
      f"({100*merged['same_dir'].mean():.2f}%)")

# Break down by path depth / structure
def path_template(url):
    """Return path with hash segments blanked out."""
    if not isinstance(url, str):
        return ""
    parsed = urlparse(url)
    segments = parsed.path.split("/")
    # Replace 4-char hex blocks (hash segments) with '<hash>'
    cleaned = [re.sub(r'^[0-9a-f]{4}$', '<hash>', s) for s in segments]
    return "/".join(cleaned)

merged["template_b"] = merged["bundle_image_url"].apply(path_template)
merged["template_p"] = merged["product_image_url"].apply(path_template)
merged["same_template"] = merged["template_b"] == merged["template_p"]
print(f"  Same URL template   : {merged['same_template'].sum():5d} / {n}  "
      f"({100*merged['same_template'].mean():.2f}%)")

# Show the most common templates in bundles vs products
print("\n  Top 5 bundle URL templates:")
for tmpl, cnt in Counter(merged["template_b"]).most_common(5):
    print(f"    {cnt:5d}  {tmpl}")

print("\n  Top 5 product URL templates:")
for tmpl, cnt in Counter(merged["template_p"]).most_common(5):
    print(f"    {cnt:5d}  {tmpl}")

# ─────────────────────────────────────────
# 6. Timestamp difference analysis
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("6. TIMESTAMP ('ts') DIFFERENCE ANALYSIS")
print("=" * 70)

merged["ts_b_num"] = pd.to_numeric(merged["ts_b"], errors="coerce")
merged["ts_p_num"] = pd.to_numeric(merged["ts_p"], errors="coerce")
merged["ts_diff_ms"] = (merged["ts_p_num"] - merged["ts_b_num"]).abs()
merged["ts_diff_days"] = merged["ts_diff_ms"] / (1000 * 60 * 60 * 24)

has_ts = merged["ts_diff_ms"].notna()
print(f"  Pairs with both timestamps   : {has_ts.sum():,} / {n}")
ts_sub = merged[has_ts]["ts_diff_days"]
print(f"  Timestamp diff (days) stats:")
print(f"    min    : {ts_sub.min():.2f}")
print(f"    median : {ts_sub.median():.2f}")
print(f"    mean   : {ts_sub.mean():.2f}")
print(f"    max    : {ts_sub.max():.2f}")
print(f"    std    : {ts_sub.std():.2f}")

# Distribution buckets
buckets = [0, 1, 7, 30, 90, 180, 365, float("inf")]
labels  = ["<1d", "1d-1w", "1w-1mo", "1mo-3mo", "3mo-6mo", "6mo-1yr", ">1yr"]
for lo, hi, lbl in zip(buckets, buckets[1:], labels):
    cnt = ((ts_sub >= lo) & (ts_sub < hi)).sum()
    print(f"    {lbl:10s}: {cnt:5d}  ({100*cnt/len(ts_sub):.1f}%)")

# Is timestamp difference correlated with match quality? (all pairs are matches)
# Let's instead compare ts values between bundle and product
merged["ts_b_sec"] = merged["ts_b_num"] / 1000
merged["ts_p_sec"] = merged["ts_p_num"] / 1000
# Exact ts match
merged["ts_exact"] = merged["ts_b_num"] == merged["ts_p_num"]
print(f"\n  Exact timestamp match        : {merged['ts_exact'].sum():5d} / {n}  "
      f"({100*merged['ts_exact'].mean():.2f}%)")

# ─────────────────────────────────────────
# 7. Hash segment analysis
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("7. HASH SEGMENT ANALYSIS (the 4x4-hex blocks in URL)")
print("=" * 70)

merged["hash_path_match"] = merged["hash_path_b"] == merged["hash_path_p"]
print(f"  Identical hash path : {merged['hash_path_match'].sum():5d} / {n}  "
      f"({100*merged['hash_path_match'].mean():.2f}%)")

# Check individual hash segment positions
for seg_idx in range(4):
    def get_seg(url, idx=seg_idx):
        d = parse_url(url)
        segs = d.get("hash_segs", [])
        return segs[idx] if idx < len(segs) else ""
    merged[f"hash_{seg_idx}_b"] = merged["bundle_image_url"].apply(get_seg)
    merged[f"hash_{seg_idx}_p"] = merged["product_image_url"].apply(get_seg)
    match_col = f"hash_{seg_idx}_match"
    merged[match_col] = merged[f"hash_{seg_idx}_b"] == merged[f"hash_{seg_idx}_p"]
    pct = 100 * merged[match_col].mean()
    print(f"  Hash segment [{seg_idx}] match : {merged[match_col].sum():5d} / {n}  ({pct:.2f}%)")

# ─────────────────────────────────────────
# 8. SKU prefix length sweep (4–10 chars)
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("8. SKU PREFIX LENGTH SWEEP (hit rate on training pairs)")
print("=" * 70)

# For each prefix length, check how many training pairs share that prefix
# Also compute precision: among all (bundle, product) pairs sharing the prefix,
# what fraction are true matches?

# Build product lookup by sku_core prefix
print(f"\n  {'Prefix len':>10}  {'Train hit rate':>15}  {'Description'}")
print(f"  {'-'*10}  {'-'*15}  {'-'*40}")

for plen in range(4, 11):
    merged[f"prefix_b_{plen}"] = merged["sku_core_b"].apply(
        lambda s: sku_prefix(str(s), plen) if isinstance(s, str) else "")
    merged[f"prefix_p_{plen}"] = merged["sku_core_p"].apply(
        lambda s: sku_prefix(str(s), plen) if isinstance(s, str) else "")
    col = f"prefix_match_{plen}"
    merged[col] = (merged[f"prefix_b_{plen}"] == merged[f"prefix_p_{plen}"]) & \
                  (merged[f"prefix_b_{plen}"] != "")
    hit_rate = merged[col].mean()
    print(f"  {plen:>10d}  {hit_rate:>14.2%}  "
          f"(matched {merged[col].sum()} / {n} pairs)")

# ─────────────────────────────────────────
# 9. Combined signals
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("9. COMBINED SIGNAL ANALYSIS")
print("=" * 70)

# How many pairs match on BOTH sku_raw AND a timestamp within 7 days?
close_ts = merged["ts_diff_days"] <= 7
both = merged["sku_raw_match"] & close_ts
print(f"  sku_raw match AND ts within 7 days  : {both.sum():5d} / {n}  ({100*both.mean():.2f}%)")

close_ts30 = merged["ts_diff_days"] <= 30
both30 = merged["sku_raw_match"] & close_ts30
print(f"  sku_raw match AND ts within 30 days : {both30.sum():5d} / {n}  ({100*both30.mean():.2f}%)")

# Pairs where sku_raw does NOT match – what else might link them?
no_sku = merged[~merged["sku_raw_match"]]
print(f"\n  Non-SKU-matching pairs              : {len(no_sku):5d}")
if len(no_sku):
    ts_close_no_sku = (no_sku["ts_diff_days"] <= 30).sum()
    print(f"  ...of which ts within 30 days       : {ts_close_no_sku:5d}  "
          f"({100*ts_close_no_sku/len(no_sku):.2f}%)")
    print("\n  Sample non-SKU-matching pairs (bundle_sku_raw | product_sku_raw | ts_diff_days):")
    for _, row in no_sku.head(15).iterrows():
        print(f"    {str(row['sku_raw_b']):20s} | {str(row['sku_raw_p']):20s} | "
              f"{row['ts_diff_days']:.1f}d")

# ─────────────────────────────────────────
# 10. Suffix / image variant analysis
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("10. URL SUFFIX / IMAGE VARIANT ANALYSIS")
print("=" * 70)

def get_suffix(url):
    """Return the trailing variant string, e.g. '-p', '-e1', '-200-p'."""
    if not isinstance(url, str):
        return ""
    parsed = urlparse(url)
    path = parsed.path
    segments = [s for s in path.split("/") if s]
    last_dir = segments[-2] if len(segments) >= 2 else ""
    # The part after the numeric/alpha SKU
    m = re.match(r'^([A-Za-z0-9]+)([-_].+)$', last_dir)
    return m.group(2) if m else ""

bundle_suffixes  = bundles["bundle_image_url"].apply(get_suffix)
product_suffixes = products["product_image_url"].apply(get_suffix)
print("  Bundle URL suffixes (top 10):")
for sfx, cnt in Counter(bundle_suffixes).most_common(10):
    print(f"    {cnt:6d}  {sfx!r}")
print("  Product URL suffixes (top 10):")
for sfx, cnt in Counter(product_suffixes).most_common(10):
    print(f"    {cnt:6d}  {sfx!r}")

# ─────────────────────────────────────────
# 11. Candidate retrieval precision estimate
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("11. CANDIDATE RETRIEVAL PRECISION (sku_raw prefix matching)")
print("=" * 70)

# Build product index by sku_raw
prod_sku_index = defaultdict(list)
for _, row in products.iterrows():
    key = row["sku_raw"]
    if isinstance(key, str) and key:
        prod_sku_index[key].append(row["product_asset_id"])

# For each bundle in training, how many products share its sku_raw?
merged["n_products_same_sku_raw"] = merged["sku_raw_b"].apply(
    lambda s: len(prod_sku_index.get(s, [])) if isinstance(s, str) else 0)

print(f"  Distribution of #products sharing same sku_raw as bundle:")
vc = merged["n_products_same_sku_raw"].value_counts().sort_index()
for k, v in vc.items():
    if k <= 10:
        print(f"    {k:3d} products sharing sku_raw : {v:5d} bundle-product pairs")
print(f"    ... (values > 10 omitted)")

# How often does the sku_raw match bring exactly 1 candidate? (perfect retrieval)
exact_one = (merged["n_products_same_sku_raw"] == 1) & merged["sku_raw_match"]
print(f"\n  sku_raw match -> exactly 1 product candidate: "
      f"{exact_one.sum()} / {n} ({100*exact_one.mean():.2f}%)")

# ─────────────────────────────────────────
# 12. Numeric SKU vs alpha-prefixed SKU
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("12. NUMERIC vs ALPHA-PREFIXED SKU PATTERNS")
print("=" * 70)

def sku_type(s):
    if not isinstance(s, str) or not s:
        return "empty"
    if s[0].isdigit():
        return "numeric"
    return "alpha-prefix"

merged["sku_type_b"] = merged["sku_core_b"].apply(sku_type)
merged["sku_type_p"] = merged["sku_core_p"].apply(sku_type)

print("  Bundle SKU types:")
for t, cnt in Counter(merged["sku_type_b"]).most_common():
    print(f"    {cnt:6d}  {t}")
print("  Product SKU types:")
for t, cnt in Counter(merged["sku_type_p"]).most_common():
    print(f"    {cnt:6d}  {t}")

# Match rates broken down by SKU type combination
for bt in ["numeric", "alpha-prefix"]:
    for pt in ["numeric", "alpha-prefix"]:
        sub = merged[(merged["sku_type_b"] == bt) & (merged["sku_type_p"] == pt)]
        if len(sub):
            hit = sub["sku_raw_match"].mean()
            print(f"  bundle={bt:13s} product={pt:13s} -> "
                  f"{len(sub):5d} pairs, sku_raw match rate {hit:.2%}")

# ─────────────────────────────────────────
# 13. Full-URL exact match
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("13. FULL URL EXACT / NEAR MATCH")
print("=" * 70)

# Strip ts parameter for comparison
def url_no_ts(url):
    if not isinstance(url, str):
        return ""
    return re.sub(r'\?ts=\d+', '', url)

merged["url_no_ts_b"] = merged["bundle_image_url"].apply(url_no_ts)
merged["url_no_ts_p"] = merged["product_image_url"].apply(url_no_ts)
merged["url_no_ts_match"] = merged["url_no_ts_b"] == merged["url_no_ts_p"]
print(f"  URL match (ignoring ts)  : {merged['url_no_ts_match'].sum():5d} / {n}  "
      f"({100*merged['url_no_ts_match'].mean():.2f}%)")

# Replace -p suffix with -e1 in bundle URL and compare with product URL (ignoring hash blocks)
def normalize_bundle_url(url):
    """Replace -p image variant with -e1 to see if it becomes a product URL."""
    if not isinstance(url, str):
        return ""
    # Replace /SKU-p/SKU-p.jpg -> /SKU-e1/SKU-e1.jpg
    return re.sub(r'/([\w-]+)-p/([\w-]+)-p\.jpg', r'/\1-e1/\1-e1.jpg', url)

merged["bundle_as_product_url"] = merged["bundle_image_url"].apply(normalize_bundle_url)
merged["url_variant_match"] = merged["bundle_as_product_url"].apply(url_no_ts) == \
                               merged["url_no_ts_p"]
print(f"  URL match after -p -> -e1: {merged['url_variant_match'].sum():5d} / {n}  "
      f"({100*merged['url_variant_match'].mean():.2f}%)")

# ─────────────────────────────────────────
# 14. Summary table
# ─────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

summary = [
    ("sku_raw exact match",               merged["sku_raw_match"].mean()),
    ("sku_core exact match",              merged["sku_core_match"].mean()),
    ("same directory path",               merged["same_dir"].mean()),
    ("same URL template",                 merged["same_template"].mean()),
    ("identical hash path",               merged["hash_path_match"].mean()),
    ("hash seg[0] match",                 merged["hash_0_match"].mean()),
    ("hash seg[1] match",                 merged["hash_1_match"].mean()),
    ("hash seg[2] match",                 merged["hash_2_match"].mean()),
    ("hash seg[3] match",                 merged["hash_3_match"].mean()),
    ("ts exact match",                    merged["ts_exact"].mean()),
    ("URL match (no ts)",                 merged["url_no_ts_match"].mean()),
    ("URL match (-p -> -e1)",             merged["url_variant_match"].mean()),
    ("sku_raw match & ts <=7d",           both.mean()),
    ("sku_raw match & ts <=30d",          both30.mean()),
]
for label, val in summary:
    bar = "#" * int(val * 40)
    print(f"  {label:35s}: {val:6.2%}  {bar}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
