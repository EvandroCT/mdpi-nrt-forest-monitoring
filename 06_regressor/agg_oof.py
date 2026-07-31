# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Pool the leave-one-out regression metrics from the trained per-fold checkpoints.
Each fold's best.pt stores val_preds/val_tgts (held-out ROI's own obs) — same source the
detection plots use. OOF = concat over the 20 folds. Baselines: predict-0 (MSE=mean(tgt^2)),
predict-mean (MSE=var(tgt)); R2-like = 1 - OOF_MSE/predict-mean_MSE."""
import os, re, numpy as np, torch
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
RUNS = DATA_DIR / "runs"
for CK in ("best", "last"):
    paths = sorted(RUNS.glob(f"regressor-*-loo*/checkpoints/{CK}.pt"))
    byvar = defaultdict(list)
    for p in paths:
        # match on the path COMPONENT (branch dir name), not str(p): on Windows str(p) uses "\\"
        # separators and a "/"-anchored regex would silently match nothing.
        m = re.fullmatch(r"regressor-([A-Za-z]+)-loo(\d+_\w+)", p.parent.parent.name)
        if m:
            byvar[m.group(1)].append((m.group(2), p))
    print(f"\n===== {CK}.pt =====")
    for var in sorted(byvar):
        allpr, alltg, macro = [], [], []
        for roi, p in sorted(byvar[var]):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            pr = np.asarray(ck["val_preds"], dtype=float).ravel()
            tg = np.asarray(ck["val_tgts"], dtype=float).ravel()
            macro.append(float(((pr - tg) ** 2).mean()))
            allpr.append(pr); alltg.append(tg)
        pr = np.concatenate(allpr); tg = np.concatenate(alltg)
        oof = float(((pr - tg) ** 2).mean()); mae = float(np.abs(pr - tg).mean())
        pos, neg = tg > 0, tg == 0
        msep = float(((pr[pos] - tg[pos]) ** 2).mean()) if pos.any() else float("nan")
        msen = float(((pr[neg] - tg[neg]) ** 2).mean()) if neg.any() else float("nan")
        base0 = float((tg ** 2).mean()); bm = float(((tg - tg.mean()) ** 2).mean())
        macro = np.array(macro)
        print(f"  {var:10s} folds {len(byvar[var]):2d} | OOF MSE {oof:.4f} MAE {mae:.4f} | "
              f"pos {msep:.4f} neg {msen:.4f} | macro {macro.mean():.4f}±{macro.std():.4f} | "
              f"base0 {base0:.4f} predmean {bm:.4f} R2 {1 - oof / bm:+.3f} | "
              f"n {len(tg)} ({int(pos.sum())}/{int(neg.sum())}) | pr[{pr.min():.2f},{pr.max():.2f}]")
    if CK == "best" and "raw" in byvar:   # staleness check vs figure (11302 RMSE ~0.039)
        print("  -- raw per-fold MSE (sorted) --")
        rows = []
        for roi, p in byvar["raw"]:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            pr = np.asarray(ck["val_preds"], dtype=float).ravel(); tg = np.asarray(ck["val_tgts"], dtype=float).ravel()
            rows.append((roi, float(((pr - tg) ** 2).mean()), len(tg)))
        for roi, m, n in sorted(rows, key=lambda x: x[1]):
            print(f"     {roi:14s} MSE {m:.4f} RMSE {m**0.5:.3f} n{n}")
