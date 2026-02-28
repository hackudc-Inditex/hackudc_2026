#!/usr/bin/env python3
"""
Servidor de anotación colaborativa para bundle-product matching.
Uso: python annotation/server.py [--port 8080]
"""

import http.server
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import parse_qs
import csv

PORT = 8080
for i, arg in enumerate(sys.argv):
    if arg == "--port" and i + 1 < len(sys.argv):
        PORT = int(sys.argv[i + 1])

BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_PATH = BASE_DIR / "annotations.json"
CANDIDATES_PATH = BASE_DIR / "candidates.json"
HUMAN_LABELS_PATH = BASE_DIR / "human_labels.csv"
HUMAN_NEGATIVES_PATH = BASE_DIR / "human_negatives.csv"

file_lock = threading.Lock()


def load_annotations():
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH) as f:
            return json.load(f)
    return {}


def save_annotations(data):
    fd, tmppath = tempfile.mkstemp(dir=BASE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmppath, ANNOTATIONS_PATH)
    except Exception:
        os.unlink(tmppath)
        raise


def export_labels(annotations, rejected=None):
    if rejected is None:
        rejected = set()

    # Load candidates to know which bundles were reviewed
    all_candidates = {}
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH) as f:
            cdata = json.load(f)
        all_candidates = cdata.get("bundles", {})

    positives = []
    # 1. Human annotations (positive)
    for bundle_id, products in annotations.items():
        for product_id, info in products.items():
            if info.get("votes"):
                positives.append({"bundle_asset_id": bundle_id, "product_asset_id": product_id})

    # 2. Auto-confirmed DISABLED - was adding circular data (model predicting itself)
    # Only manual annotations from annotations.json are exported
    pos_set = {(r["bundle_asset_id"], r["product_asset_id"]) for r in positives}

    # Write positives
    positives.sort(key=lambda r: (r["bundle_asset_id"], r["product_asset_id"]))
    with open(HUMAN_LABELS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(positives)

    # 3. Generate negatives: for annotated bundles, candidates NOT marked = negatives
    annotated_bundles = set(annotations.keys())
    negatives = []
    for bundle_id in annotated_bundles:
        if bundle_id not in all_candidates:
            continue
        for cand in all_candidates[bundle_id].get("candidates", []):
            pid = cand["product_id"]
            pair = (bundle_id, pid)
            if pair not in pos_set:
                negatives.append({"bundle_asset_id": bundle_id, "product_asset_id": pid})

    # Add rejected auto-confirmed as negatives too
    for pair in rejected:
        if len(pair) == 2:
            negatives.append({"bundle_asset_id": pair[0], "product_asset_id": pair[1]})

    negatives.sort(key=lambda r: (r["bundle_asset_id"], r["product_asset_id"]))
    with open(HUMAN_NEGATIVES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bundle_asset_id", "product_asset_id"])
        writer.writeheader()
        writer.writerows(negatives)

    return len(positives)


class AnnotationHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
            return super().do_GET()
        elif self.path == "/annotations.json":
            with file_lock:
                data = load_annotations()
            self._json_response(data)
        elif self.path == "/api/progress":
            with file_lock:
                annotations = load_annotations()
            total = 0
            if CANDIDATES_PATH.exists():
                with open(CANDIDATES_PATH) as f:
                    candidates = json.load(f)
                total = len(candidates.get("bundles", {}))
            annotated = len(annotations)
            complete = sum(1 for b in annotations.values()
                          if any(len(info.get("votes", [])) >= 2
                                 for info in b.values()))
            self._json_response({
                "total": total,
                "annotated": annotated,
                "complete": complete,
            })
        elif self.path == "/candidates.json":
            return super().do_GET()
        elif self.path.startswith("/images/"):
            # Serve local images as fallback
            project_dir = BASE_DIR.parent
            self.path = self.path  # keep as-is
            old_dir = self.directory
            self.directory = str(project_dir)
            super().do_GET()
            self.directory = old_dir
        else:
            super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/api/annotate":
            try:
                data = json.loads(body)
                bundle_id = data["bundle_id"]
                product_id = data["product_id"]
                annotator = data["annotator"]
                action = data.get("action", "add")

                with file_lock:
                    annotations = load_annotations()
                    if bundle_id not in annotations:
                        annotations[bundle_id] = {}
                    if product_id not in annotations[bundle_id]:
                        annotations[bundle_id][product_id] = {"votes": []}

                    votes = annotations[bundle_id][product_id]["votes"]
                    if action == "add" and annotator not in votes:
                        votes.append(annotator)
                    elif action == "remove" and annotator in votes:
                        votes.remove(annotator)

                    # Clean up empty entries
                    if not votes:
                        del annotations[bundle_id][product_id]
                    if not annotations[bundle_id]:
                        del annotations[bundle_id]

                    save_annotations(annotations)

                self._json_response({"ok": True})

            except (json.JSONDecodeError, KeyError) as e:
                self._json_response({"error": str(e)}, status=400)

        elif self.path == "/api/export":
            rejected = set()
            try:
                data = json.loads(body) if body else {}
                for key in data.get("rejected", []):
                    # key format: "bundle_id|product_id"
                    rejected.add(tuple(key.split("|", 1)))
            except Exception:
                pass
            with file_lock:
                annotations = load_annotations()
                count = export_labels(annotations, rejected)
            self._json_response({
                "ok": True,
                "exported": count,
                "path": str(HUMAN_LABELS_PATH),
            })

        else:
            self._json_response({"error": "not found"}, status=404)

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        try:
            msg = str(args[0]) if args else ""
            if "/api/" in msg:
                return
        except Exception:
            pass
        super().log_message(format, *args)


def main():
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), AnnotationHandler)
    ip = "0.0.0.0"
    print(f"{'=' * 50}")
    print(f"  Servidor de anotación")
    print(f"  http://localhost:{PORT}")
    print(f"  http://{ip}:{PORT} (red local)")
    print(f"{'=' * 50}")
    print(f"  Candidates: {'OK' if CANDIDATES_PATH.exists() else 'FALTA - ejecuta pipeline_human.py --generate-candidates'}")
    print(f"  Annotations: {ANNOTATIONS_PATH}")
    print(f"{'=' * 50}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
        # Auto-export on shutdown
        with file_lock:
            annotations = load_annotations()
            if annotations:
                count = export_labels(annotations)
                print(f"Exportadas {count} anotaciones a {HUMAN_LABELS_PATH}")


if __name__ == "__main__":
    main()
