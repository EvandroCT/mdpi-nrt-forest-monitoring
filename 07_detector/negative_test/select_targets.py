# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Negative test, step 1: the certified logging-free targets that no fold ever trained on.

Every entry of annotations.json's oversample_pool is an interpreter-certified logging-free
target observation of a logging-free site (the target and the 20 observations before it were
read free of new selective logging). The cross-site negatives of the downstream model are drawn
from this pool by HLSClassifierDataset at seed 0; the candidates that the draw does not reach
were never a training target of any fold. This script rebuilds that draw exactly as training
does and writes the undrawn candidates to targets.csv (aoi_id, date as YYYYMMDD, k, view date
of the site's own DETER alert), sorted.

Usage: uv run python 07_detector/negative_test/select_targets.py    (CPU; about a minute)
Env: DATA_DIR (default ./data), ZARRS, ANNOTATIONS, VDJSON (as in 07_detector/sweep_kernels.py);
     OUT_DIR (default DATA_DIR/runs/negative_test)."""
import csv, json, os, sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from dataset_classifier import load_annotations, HLSClassifierDataset   # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ZARRS = Path(os.environ["ZARRS"]) if os.environ.get("ZARRS") else DATA_DIR / "labelled-dataset"
ANNOTATIONS = Path(os.environ["ANNOTATIONS"]) if os.environ.get("ANNOTATIONS") else DATA_DIR / "annotations.json"
VDJSON = Path(os.environ["VDJSON"]) if os.environ.get("VDJSON") else DATA_DIR / "view_dates.json"
OUT_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else DATA_DIR / "runs" / "negative_test"

if __name__ == "__main__":
    activity, ambiguous, pool, first_obs = load_annotations(ANNOTATIONS)
    vd = json.loads(VDJSON.read_text(encoding="utf-8"))
    ds = HLSClassifierDataset(ZARRS, activity, ambiguous, first_obs, {a: date.fromisoformat(s) for a, s in vd.items()},
                              oversample_pool=pool, variants=["raw"], seed=0)   # the training draw
    drawn = {(s["data_roi"], int(s["target_idx"])) for s in ds._samples if s["fold_roi"] == "__cross__"}
    rows = []
    for roi, d, k in pool:
        res = ds._resolve(roi, d, k)
        assert res is not None, f"logging-free candidate {roi} {d} k={k} does not resolve to an observation"
        if (roi, res[0]) not in drawn:
            rows.append((roi, d.strftime("%Y%m%d"), k, vd[roi]))
    rows.sort()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "targets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["aoi_id", "date", "k", "view"]); w.writerows(rows)
    print(f"logging-free candidates: {len(pool)} | drawn as training negatives (seed 0): {len(drawn)} | "
          f"never drawn (targets): {len(rows)} over {len({r[0] for r in rows})} sites -> {OUT_DIR / 'targets.csv'}")
