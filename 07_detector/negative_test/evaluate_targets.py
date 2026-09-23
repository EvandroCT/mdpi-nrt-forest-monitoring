# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Negative test, step 3: read the scores with the reference detector (the paper's Table 3).

For each target and each of the 20 fold models (yhat_targets.json, score_targets.py): the score
at the target, and whether the reference rule confirms an alert AT the target -- the edge
filter's look-ahead ends there (edge response at t-f+1 >= T_E) or a sustained run of K smoothed
scores >= T_G ends there (t-K+1..t). The filters are sweep_kernels.py's own, run over the scored
stretch (79 observations before the target, the target, 2 after). A target counts as alerted
when a majority of the 20 models alert at it. Sites are split by whether they also contributed
other logging-free candidates to training (drawn at seed 0; oversample_pool minus the targets).

Writes targets_results.json (per target, plus the pooled summary) and
negative_test_groups.csv (Table 3; the score shares above 0.5 and T_G are quoted in the text).

Usage: uv run python 07_detector/negative_test/evaluate_targets.py    (CPU; seconds)
Env: DATA_DIR (default ./data), ANNOTATIONS; OUT_DIR (default DATA_DIR/runs/negative_test,
     where the two previous steps write)."""
import csv, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sweep_kernels as sk                                     # noqa: E402  (the detector's own filters)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ANNOTATIONS = Path(os.environ["ANNOTATIONS"]) if os.environ.get("ANNOTATIONS") else DATA_DIR / "annotations.json"
OUT_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else DATA_DIR / "runs" / "negative_test"
RULE = dict(f=2, p=2, TE=0.86, L=5, sigma=2.4, TG=0.65, K=5)     # reference operating point (sweep_kernels.py, best_combo_confirmation)
GROUPS = [("no_training_sample", "Sites with no training sample"), ("fed_training", "Sites that also fed training")]


def alert_at(P, t):
    """True if the reference rule confirms an alert at index t of the score series P."""
    er = sk.filt(P, *sk.edge_kernel(RULE["f"], RULE["p"]))
    gs = sk.filt(P, *sk.gauss_kernel(RULE["L"], RULE["sigma"])) >= RULE["TG"]
    edge = t - RULE["f"] + 1 >= 0 and er[t - RULE["f"] + 1] >= RULE["TE"]
    gauss = t - RULE["K"] + 1 >= 0 and bool(gs[t - RULE["K"] + 1:t + 1].all())
    return edge or gauss


def stats(sub, n_folds):
    y = np.array([v for r in sub for v in r["yhat_at_target"]])
    n_alert = sum(r["folds_alert_at_target"] for r in sub)
    return dict(sites=len({r["site"] for r in sub}), targets=len(sub),
                alerted_majority=sum(r["folds_alert_at_target"] > n_folds / 2 for r in sub),
                alerted_any=sum(r["folds_alert_at_target"] > 0 for r in sub),
                readings_alert=f"{n_alert}/{n_folds * len(sub)}", readings_pct=round(100 * n_alert / (n_folds * len(sub)), 1),
                yhat_median=round(float(np.median(y)), 3), yhat_p90=round(float(np.percentile(y, 90)), 3),
                share_ge_05=round(100 * float(np.mean(y >= 0.5)), 1), share_ge_TG=round(100 * float(np.mean(y >= RULE["TG"])), 1))


if __name__ == "__main__":
    Y = json.loads((OUT_DIR / "yhat_targets.json").read_text(encoding="utf-8")); folds = Y["folds"]; nf = len(folds)
    targets = {(r["aoi_id"], r["date"], int(r["k"])) for r in csv.DictReader(open(OUT_DIR / "targets.csv", encoding="utf-8"))}
    pool = [(r, d.replace("-", ""), int(k)) for r, d, k in json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["oversample_pool"]]
    fed = {r for r, d, k in pool if (r, d, k) not in targets}      # sites with at least one drawn (training) candidate
    rows = []
    for sid in Y["meta"]:
        aoi, d8, k = sid.rsplit("_", 2)
        dates = [d for d, _ in Y["yhat"][sid][folds[0]]]
        t = [i for i, d in enumerate(dates) if d == f"{d8[:4]}-{d8[4:6]}-{d8[6:]}"][int(k) - 1]
        yt, n_alert = [], 0
        for f in folds:
            P = np.array([p for _, p in Y["yhat"][sid][f]], float)
            yt.append(float(P[t])); n_alert += alert_at(P, t)
        rows.append(dict(target=sid, site=aoi, date=dates[t], k=int(k), group=GROUPS[aoi in fed][0],
                         n_obs_before=t, n_obs_after=len(dates) - 1 - t,
                         yhat_at_target=yt, median_yhat_at_target=float(np.median(yt)), max_yhat_at_target=float(max(yt)),
                         folds_alert_at_target=int(n_alert)))
    table = [dict(group=label, **stats([r for r in rows if r["group"] == key], nf)) for key, label in GROUPS]
    table.append(dict(group="All", **stats(rows, nf)))
    summary = table[-1]
    (OUT_DIR / "targets_results.json").write_text(json.dumps(dict(rule=RULE, n_folds=nf, rows=rows, summary=summary), indent=1), encoding="utf-8")
    with open(OUT_DIR / "negative_test_groups.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n"); w.writeheader(); w.writerows(table)
    print(f"rule f={RULE['f']} p={RULE['p']} TE={RULE['TE']} | L={RULE['L']} sigma={RULE['sigma']} TG={RULE['TG']} K={RULE['K']}; "
          f"{nf} folds; {len(rows)} targets over {len({r['site'] for r in rows})} sites")
    print(f"{'group':<30} {'sites':>5} {'targets':>7} {'maj.':>4} {'any':>3} {'readings':>14} {'Yhat med/p90':>13} {'>=0.5':>6} {'>=TG':>6}")
    for g in table:
        print(f"{g['group']:<30} {g['sites']:>5} {g['targets']:>7} {g['alerted_majority']:>4} {g['alerted_any']:>3} "
              f"{g['readings_alert'] + ' (' + str(g['readings_pct']) + '%)':>14} {g['yhat_median']:>6.3f}/{g['yhat_p90']:.3f} "
              f"{g['share_ge_05']:>5.1f}% {g['share_ge_TG']:>5.1f}%")
    print(f"wrote {OUT_DIR / 'targets_results.json'} and {OUT_DIR / 'negative_test_groups.csv'}")
