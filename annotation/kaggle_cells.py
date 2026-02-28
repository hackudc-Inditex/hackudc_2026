###############################################################################
# CELDA 1: GENERAR CANDIDATES.JSON
# Pega esto en una nueva celda al final de tu notebook de Kaggle
# (después de que todo haya terminado de ejecutar)
###############################################################################

import json
from collections import Counter

# Accesorios que raramente se ven en las fotos
ACCESSORY_WORDS = {"bracelet", "earring", "necklace", "ring", "brooch", "pin",
                   "keychain", "cufflink", "anklet", "hairclip", "hairband",
                   "pulsera", "pendiente", "anillo", "collar", "broche"}
MAX_PER_CATEGORY = 4  # Max candidatos del mismo tipo (ej. max 4 camisetas)

def get_category(desc):
    """Extrae categoría base de la descripción (primera palabra clave)."""
    d = desc.upper().strip()
    # Agrupar variantes comunes
    for key in ["T-SHIRT", "SHIRT", "TOP", "BLOUSE", "POLO"]:
        if key in d: return "TOPS"
    for key in ["TROUSER", "PANT", "JEAN", "CHINO"]:
        if key in d: return "BOTTOMS"
    for key in ["JACKET", "COAT", "BLAZER", "PARKA", "OVERSHIRT"]:
        if key in d: return "OUTERWEAR"
    for key in ["SHOE", "SNEAKER", "BOOT", "SANDAL", "LOAFER", "TRAINER"]:
        if key in d: return "SHOES"
    for key in ["DRESS"]:
        if key in d: return "DRESS"
    for key in ["SKIRT"]:
        if key in d: return "SKIRT"
    for key in ["BAG", "TOTE", "BACKPACK", "CROSSBODY"]:
        if key in d: return "BAGS"
    for key in ["SCARF", "HAT", "CAP", "BELT", "GLOVE", "SUNGLASSES"]:
        if key in d: return "ACCESSORIES"
    return d.split()[0] if d else "OTHER"

def is_small_accessory(desc):
    d = desc.lower()
    return any(w in d for w in ACCESSORY_WORDS)

output = {"bundles": {}}

for bid in tqdm(test_bids, desc="Generando candidatos"):
    sec = bundle_section.get(bid, "1")
    burl = bundle_url_map.get(bid, "")

    # Obtener predicciones del modelo
    predicted = predict_bundle(bid, all_section_b2b, use_projection=(projection is not None))
    predicted_pids = set(pid for pid, _ in predicted)

    # Ampliar candidatos con FAISS directo
    if len(predicted) < 80 and sec in section_indices:
        bundle_emb = None
        if bid in bid_to_emb_idx:
            bundle_emb = all_bundle_embs[bid_to_emb_idx[bid]].reshape(1, -1).copy()
        elif bid in crops_data:
            for c in crops_data[bid]:
                if c["label"] == "_full_":
                    bundle_emb = c["embedding"].reshape(1, -1).copy()
                    break
        if bundle_emb is not None:
            faiss.normalize_L2(bundle_emb)
            scores, indices = section_indices[sec].search(bundle_emb, 200)
            spids = section_pids_map[sec]
            for j in range(min(200, len(spids))):
                pid = spids[indices[0][j]]
                if pid not in predicted_pids:
                    predicted.append((pid, float(scores[0][j]) * 1.5))
                    predicted_pids.add(pid)
                if len(predicted) >= 80:
                    break

    # Filtrar y diversificar
    cat_count = Counter()
    diverse_cands = []
    accessory_cands = []

    for pid, score in predicted:
        desc = product_desc.get(pid, "")
        cat = get_category(desc)

        # SKU matches (score >= 90) siempre pasan
        if score >= 90:
            diverse_cands.append((pid, score))
            cat_count[cat] += 1
            continue

        # Accesorios pequeños: apartar (solo si score bajo)
        if is_small_accessory(desc) and score < 20:
            accessory_cands.append((pid, score))
            continue

        # Limitar por categoría
        if cat_count[cat] >= MAX_PER_CATEGORY:
            continue

        diverse_cands.append((pid, score))
        cat_count[cat] += 1

        if len(diverse_cands) >= 35:
            break

    # Rellenar con accesorios si faltan candidatos
    if len(diverse_cands) < 35:
        for pid, score in accessory_cands:
            diverse_cands.append((pid, score))
            if len(diverse_cands) >= 40:
                break

    # Construir lista final
    cands = []
    for rank, (pid, score) in enumerate(diverse_cands[:40], 1):
        cands.append({
            "product_id": pid,
            "product_url": product_url_map.get(pid, ""),
            "description": product_desc.get(pid, ""),
            "score": round(score, 2),
            "rank": rank,
        })

    output["bundles"][bid] = {
        "section": sec,
        "bundle_url": burl,
        "candidates": cands,
    }

# Guardar
out_path = SUBMISSIONS_DIR / "candidates.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

from IPython.display import FileLink
display(FileLink(str(out_path)))
print(f"\n==> Descarga candidates.json ({len(output['bundles'])} bundles x 40 candidatos)")
print("==> Haz clic en el enlace de arriba para descargar")


###############################################################################
# CELDA 2: REENTRENAR CON ANOTACIONES HUMANAS
# Pega esto DESPUÉS de anotar y subir human_labels.csv + human_negatives.csv
# (sube ambos archivos al notebook con File > Upload)
###############################################################################

# === SUBIR human_labels.csv Y human_negatives.csv ANTES DE EJECUTAR ===

import csv

def find_file(name):
    for p in [WORK_DIR / name, Path(f"/kaggle/input/human-labels/{name}"), Path(f"/kaggle/working/{name}")]:
        if p.exists():
            return p
    return None

human_labels_path = find_file("human_labels.csv")
human_negatives_path = find_file("human_negatives.csv")

if human_labels_path is None:
    print("ERROR: No se encuentra human_labels.csv")
    print("Súbelo al notebook o como dataset de Kaggle")
else:
    print(f"Positivos: {human_labels_path}")
    if human_negatives_path:
        print(f"Negativos: {human_negatives_path}")

    # Cargar anotaciones humanas (positivos)
    human_rows = load_csv(human_labels_path)
    human_bids = set()
    human_count = 0
    for row in human_rows:
        bid = row["bundle_asset_id"]
        pid = row["product_asset_id"]
        if pid:
            train_bundle_products[bid].add(pid)
            human_bids.add(bid)
            human_count += 1
    print(f"Cargados {human_count} positivos para {len(human_bids)} bundles")

    # Cargar negativos explícitos (lo que NO marcaron los anotadores)
    human_negatives = {}  # bid -> set of negative pids
    if human_negatives_path:
        neg_rows = load_csv(human_negatives_path)
        neg_count = 0
        for row in neg_rows:
            bid = row["bundle_asset_id"]
            pid = row["product_asset_id"]
            if pid:
                if bid not in human_negatives:
                    human_negatives[bid] = set()
                human_negatives[bid].add(pid)
                neg_count += 1
        print(f"Cargados {neg_count} negativos explícitos para {len(human_negatives)} bundles")

    # Split: bundles humanos SIEMPRE a training
    original_bids = [bid for bid in list(train_bundle_products.keys()) if bid not in human_bids]
    random.seed(42)
    random.shuffle(original_bids)
    split = int(0.8 * len(original_bids))
    train_bids = original_bids[:split] + list(human_bids)
    val_bids = original_bids[split:]
    print(f"Train: {len(train_bids)} ({len(human_bids)} human), Val: {len(val_bids)}")

    # Generar crops para los test bundles que anotamos
    test_crops_need = [bid for bid in human_bids if bid not in crops_data]
    if test_crops_need:
        yolo_model = load_yolo_model()
        for bid in tqdm(test_crops_need, desc="Human bundle crops"):
            img_path = get_bundle_path(bid)
            try:
                bundle_img = Image.open(img_path).convert("RGB")
            except Exception:
                crops_data[bid] = []
                continue
            crops_for_bundle = []
            detections = detect_garments(yolo_model, img_path)
            for det in detections:
                cropped = crop_bbox(bundle_img, det["bbox"], padding=0.08)
                emb = embed_pil(clip_model, clip_preprocess, cropped)
                crops_for_bundle.append({"label": det["label"], "embedding": emb, "conf": det["conf"]})
            full_emb = embed_pil(clip_model, clip_preprocess, bundle_img)
            crops_for_bundle.append({"label": "_full_", "embedding": full_emb, "conf": 1.0})
            crops_data[bid] = crops_for_bundle
        del yolo_model
        torch.cuda.empty_cache()
        gc.collect()

    # Recrear pares de entrenamiento CON negativos explícitos
    print("Recreando pares de entrenamiento...")

    # Monkey-patch create_pairs para inyectar hard negatives humanos
    _original_create_pairs = create_pairs

    def create_pairs_with_negatives(bids):
        anchor_embs, pos_embs, neg_embs, pos_pids, neg_pids = _original_create_pairs(bids)

        # Añadir pares con negativos explícitos humanos
        extra_anchors, extra_pos, extra_neg, extra_ppids, extra_npids = [], [], [], [], []
        for bid in bids:
            if bid not in human_negatives or bid not in crops_data:
                continue
            neg_pid_set = human_negatives[bid]
            pos_pid_set = train_bundle_products.get(bid, set())
            if not pos_pid_set:
                continue
            for crop in crops_data[bid]:
                if crop["label"] == "_full_":
                    continue
                crop_emb = crop["embedding"]
                for ppid in pos_pid_set:
                    if ppid not in pid_to_emb_idx:
                        continue
                    pos_emb = product_embeddings[pid_to_emb_idx[ppid]]
                    # Para cada positivo, emparejar con los negativos humanos
                    for npid in neg_pid_set:
                        if npid not in pid_to_emb_idx:
                            continue
                        neg_emb = product_embeddings[pid_to_emb_idx[npid]]
                        extra_anchors.append(crop_emb)
                        extra_pos.append(pos_emb)
                        extra_neg.append(neg_emb)
                        extra_ppids.append(ppid)
                        extra_npids.append(npid)

        if extra_anchors:
            anchor_embs = np.concatenate([anchor_embs, np.array(extra_anchors)])
            pos_embs = np.concatenate([pos_embs, np.array(extra_pos)])
            neg_embs = np.concatenate([neg_embs, np.array(extra_neg)])
            pos_pids = pos_pids + extra_ppids
            neg_pids = neg_pids + extra_npids
            print(f"  + {len(extra_anchors)} hard negatives humanos añadidos")

        return anchor_embs, pos_embs, neg_embs, pos_pids, neg_pids

    train_data = create_pairs_with_negatives(train_bids)
    print(f"  -> {len(train_data[0])} pares de entrenamiento total")
    val_data = _original_create_pairs(val_bids)
    print(f"  -> {len(val_data[0])} pares de validación")

    # Reentrenar con más epochs
    NUM_EPOCHS = 50
    print(f"\nReentrenando projection head ({NUM_EPOCHS} epochs)...")
    projection = train_projection(
        train_data, val_data, product_embeddings, CLIP_DIM,
        num_epochs=NUM_EPOCHS, lr=3e-4
    )
    print("Reentrenamiento completado!")


###############################################################################
# CELDA 3: GENERAR SUBMISSION FINAL
# Pega esto después de que el reentrenamiento termine
###############################################################################

# Reconstruir índices FAISS con la nueva projection
print("Reconstruyendo índices FAISS con projection reentrenada...")
section_indices = {}
section_pids_map = {}

for sec in ["1", "2", "3"]:
    pids_in_sec = [pid for pid, s in product_section.items() if s == sec]
    if not pids_in_sec:
        continue
    embs = []
    for pid in pids_in_sec:
        if pid in pid_to_emb_idx:
            emb = product_embeddings[pid_to_emb_idx[pid]]
        else:
            continue
        embs.append(emb)

    if not embs:
        continue

    emb_matrix = np.array(embs, dtype=np.float32)

    # Aplicar projection
    if projection is not None:
        projection.eval()
        with torch.no_grad():
            t = torch.from_numpy(emb_matrix).to(DEVICE)
            emb_matrix = projection(t).cpu().numpy()

    faiss.normalize_L2(emb_matrix)
    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)
    section_indices[sec] = index
    section_pids_map[sec] = pids_in_sec

print(f"Índices reconstruidos: {list(section_indices.keys())}")

# Regenerar predicciones
rows = []
for bid in tqdm(test_bids, desc="Predicción final"):
    predicted = predict_bundle(bid, all_section_b2b, use_projection=True)
    pids = [pid for pid, _ in predicted[:15]]
    while len(pids) < 15:
        pids.append("")
    rows.append({"bundle_asset_id": bid, "product_asset_id": " ".join(pids)})

# Guardar submission
sub_path = SUBMISSIONS_DIR / "submission_human_vitl14.csv"
with open(sub_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
    writer.writeheader()
    writer.writerows(rows)

from IPython.display import FileLink
display(FileLink(str(sub_path)))
print(f"\n==> Submission guardado: {sub_path}")
print(f"==> {len(rows)} bundles, descarga y envía a la competición")
