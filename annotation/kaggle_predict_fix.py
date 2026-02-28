###############################################################################
# CELDA FIX: PREDICT_BUNDLE MEJORADO
# Pega esto ANTES de la celda de predicción (antes de "PREDICTING")
# Sobreescribe la función predict_bundle con bugs corregidos
###############################################################################

def predict_bundle(bid, b2b_indices, use_projection=True):
    """Predict products for a single bundle - VERSION MEJORADA."""
    sec = bundle_section.get(bid, "1")
    burl = bundle_url_map.get(bid, "")
    bsku = extract_sku(burl)
    bts = bundle_ts.get(bid)

    candidates = {}

    # --- SIGNAL 1: SKU exact (sin cambios, funciona bien) ---
    if bsku and bsku in sku_to_products:
        for pid in sku_to_products[bsku]:
            candidates[pid] = candidates.get(pid, 0) + 200

    # --- SIGNAL 2: SKU prefix --- FIX: ACUMULA en vez de ignorar
    prefix_scores = {8: 80, 7: 60, 6: 40, 5: 25, 4: 15}
    if bsku:
        for plen, pscore in prefix_scores.items():
            if len(bsku) >= plen:
                prefix = bsku[:plen]
                for pid in sku_prefix_to_products[plen].get(prefix, []):
                    desc = product_desc.get(pid, "")
                    if desc in cat_sections and sec not in cat_sections[desc]:
                        continue
                    # FIX: siempre sumar, usar max del prefix score
                    old = candidates.get(pid, 0)
                    candidates[pid] = max(old, pscore) if old < 100 else old + pscore

    # --- SIGNAL 3: Bundle-to-bundle --- FIX: top-40 en vez de top-20
    bundle_emb = None
    if bid in bid_to_emb_idx:
        bundle_emb = all_bundle_embs[bid_to_emb_idx[bid]].reshape(1, -1).copy().astype(np.float32)
    elif bid in crops_data:
        for c in crops_data[bid]:
            if c["label"] == "_full_":
                bundle_emb = c["embedding"].reshape(1, -1).copy().astype(np.float32)
                break

    if bundle_emb is not None and sec in b2b_indices:
        faiss.normalize_L2(bundle_emb)
        b2b_index, b2b_bids = b2b_indices[sec]
        k = min(40, b2b_index.ntotal)  # FIX: 40 en vez de 20
        sim_scores, sim_idx = b2b_index.search(bundle_emb, k)
        for j in range(k):
            similar_bid = b2b_bids[sim_idx[0][j]]
            if similar_bid == bid:
                continue
            sim = float(sim_scores[0][j])
            if sim < 0.3:  # FIX: ignorar bundles poco similares
                continue
            for pid in train_bundle_products.get(similar_bid, set()):
                bonus = sim * 35
                candidates[pid] = candidates.get(pid, 0) + bonus
                # Co-occurrence con peso aumentado
                for copid, count in cooccurrence[pid].most_common(5):  # FIX: top-5 en vez de top-3
                    if copid in pid_to_idx:
                        candidates[copid] = candidates.get(copid, 0) + count * sim * 5  # FIX: *5 en vez de *3

    # --- SIGNAL 4: Trained visual matching (projection head) ---
    if bid in crops_data and projection is not None and use_projection:
        for crop_info in crops_data[bid]:
            crop_label = crop_info["label"]
            if crop_label == "_full_":
                continue

            crop_emb = torch.FloatTensor(crop_info["embedding"]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                projected = projection(crop_emb).cpu().numpy().astype(np.float32)
            faiss.normalize_L2(projected)

            if USE_YOLO and crop_label in YOLO_TO_CATALOG:
                groups_to_search = [crop_label]
            elif not USE_YOLO and crop_label in ZONE_TO_CATALOG:
                groups_to_search = [crop_label]
            else:
                groups_to_search = []

            for group in groups_to_search:
                key = (group, sec)
                if key in group_indices:
                    index, gpids = group_indices[key]
                    search_k = min(15, index.ntotal)  # FIX: 15 en vez de 10
                    scores, indices = index.search(projected, search_k)
                    for j2 in range(search_k):
                        pid = gpids[indices[0][j2]]
                        clip_score = float(scores[0][j2])
                        conf = crop_info.get("conf", 0.5)
                        bonus = clip_score * (1 + conf) * 20
                        candidates[pid] = candidates.get(pid, 0) + bonus

    # --- SIGNAL 5: Raw CLIP matching --- FIX: top-15 en vez de top-5
    if bid in crops_data:
        for crop_info in crops_data[bid]:
            crop_label = crop_info["label"]
            if crop_label == "_full_":
                continue

            crop_emb = crop_info["embedding"].reshape(1, -1).copy().astype(np.float32)
            faiss.normalize_L2(crop_emb)

            if USE_YOLO and crop_label in YOLO_TO_CATALOG:
                groups_to_search = [crop_label]
            elif not USE_YOLO and crop_label in ZONE_TO_CATALOG:
                groups_to_search = [crop_label]
            else:
                groups_to_search = []

            for group in groups_to_search:
                key = (group, sec)
                if key in group_indices:
                    index, gpids = group_indices[key]
                    search_k = min(15, index.ntotal)  # FIX: 15 en vez de 5
                    scores, indices = index.search(crop_emb, search_k)
                    for j2 in range(search_k):
                        pid = gpids[indices[0][j2]]
                        clip_score = float(scores[0][j2])
                        bonus = clip_score * 10  # FIX: 10 en vez de 8
                        candidates[pid] = candidates.get(pid, 0) + bonus

    # --- SIGNAL 6: Popularity --- FIX: peso x2.0 en vez de x0.3, top-50
    for pid, freq in section_product_freq[sec].most_common(50):
        candidates[pid] = candidates.get(pid, 0) + freq * 2.0  # FIX: 2.0 vs 0.3

    # --- SIGNAL 7: Timestamp --- FIX: aditivo + multiplicativo
    if bts:
        for pid in list(candidates.keys()):
            if pid in product_ts:
                diff_days = abs(bts - product_ts[pid]) / (1000 * 86400)
                if diff_days < 7:
                    candidates[pid] = candidates[pid] * 1.2 + 5  # FIX: aditivo
                elif diff_days < 30:
                    candidates[pid] = candidates[pid] * 1.1 + 3
                elif diff_days < 90:
                    candidates[pid] = candidates[pid] * 1.05 + 1
                elif diff_days > 365:
                    candidates[pid] *= 0.7

    # --- SIGNAL 8: Section fallback + FULL IMAGE search --- FIX: siempre buscar
    if bundle_emb is not None and sec in section_indices:
        scores, indices = section_indices[sec].search(bundle_emb, 80)  # FIX: 80 en vez de 50
        spids = section_pids_map[sec]
        for j in range(min(80, len(spids))):
            pid = spids[indices[0][j]]
            base_score = float(scores[0][j]) * 2.5  # FIX: peso 2.5 en vez de 1.5
            if pid not in candidates:
                candidates[pid] = base_score
            elif candidates[pid] < 50:  # FIX: reforzar los débiles
                candidates[pid] += base_score * 0.5

    # Sort and return top 15
    sorted_c = sorted(candidates.items(), key=lambda x: -x[1])
    return sorted_c[:15]

print("predict_bundle MEJORADO cargado!")
