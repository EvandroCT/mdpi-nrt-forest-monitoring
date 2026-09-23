# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Nested leave-one-MU-out (LOMO) evaluation of the detector grid search.

The seven detector parameters are selected by the grid search of sweep_kernels.py over the same
20 MUs on which the detector is evaluated. Here the SAME grid, SAME lexicographic objective and
SAME tie-break (enumeration order, as implemented in sweep_kernels.process) are run 20 times,
each time on 19 MUs, and the winning configuration of each fold is applied, unchanged, to the
held-out MU. Only the detector is nested; the out-of-fold regressor scores
(dense_predictions.json) are the same in every fold.

Everything is delegated to sweep_kernels.py: this file never re-implements the filters, the
metrics or the ranking. It only (i) restricts the MU list seen by sweep_kernels.process,
(ii) applies a chosen configuration to one MU with the same filter functions, and
(iii) aggregates with a cluster (per-MU) bootstrap.

Usage: uv run python 07_detector/nested_sweep.py    (CPU only; about 8 minutes with 21 workers)
Env: DATA_DIR and the other inputs as in sweep_kernels.py; DENSE (default
     DATA_DIR/dense_predictions.json); OUT_DIR (default DATA_DIR/runs/nested_sweep);
     NBOOT (default 10000); SEED (default 0); WORKERS (default min(21, cpu-2)).
Writes nested_sweep_results.json and nested_sweep_summary.md; make_nested_tables.py turns the
first into the paper's Table 5 and Table S3.
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_kernels as sk                                   # noqa: E402  (grid, filters, ranking)

DENSE_PATH = Path(os.environ.get("DENSE", sk.CLF / "dense_predictions.json"))
OUT_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else sk.DATA_DIR / "runs" / "nested_sweep"
NBOOT = int(os.environ.get("NBOOT", 10000)); SEED = int(os.environ.get("SEED", 0))
PARAMS = ["f", "p", "TE", "L", "sigma", "TG", "K"]
POINTS = {"confirmation": "best_combo_confirmation", "detection": "best_combo"}   # the two reported operating points

ROI_INFO = {}; ROIS = []


def py(cfg):                                                 # numpy scalars -> plain Python (clean repr / JSON)
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}


def _init(roi_info, rois):
    global ROI_INFO, ROIS
    ROI_INFO, ROIS = roi_info, rois
    sk._init_worker(roi_info, rois)


def run_grid(rois):
    """Full grid search of sweep_kernels.process over exactly `rois` (unchanged objective/tie-break)."""
    sk.ROIS = list(rois)                                      # process() reads this global
    r = sk.process(("nested", "raw", DENSE_PATH))
    assert "error" not in r, r
    return r


def fold(k):
    t0 = time.time()
    r = run_grid([a for a in ROIS if a != k])
    return dict(held_out=k, n_train=r["n_roi"], secs=round(time.time() - t0, 1),
                cfg={pt: py(r[key]["cfg"]) for pt, key in POINTS.items()})


def apply_cfg(cfg, roi, dense):
    """Detection and confirmation lag (days vs t_on) of `cfg` on one MU, with sweep_kernels' own
    filters: combo = min(edge, gauss); confirmation = min(edge anchor+f-1, gauss anchor+K-1)."""
    diso, cdr = ROI_INFO[roi]
    P = np.nan_to_num(np.array([dense[roi].get(d, np.nan) for d in diso]), nan=0.0)
    e = sk.filt(P, *sk.edge_kernel(cfg["f"], cfg["p"])) >= cfg["TE"]
    g = sk.sustained(sk.filt(P, *sk.gauss_kernel(cfg["L"], cfg["sigma"])) >= cfg["TG"], cfg["K"])
    det = min(sk.first_day(e, cdr), sk.first_day(g, cdr))
    conf = min(sk.first_conf(e, cdr, cfg["f"]), sk.first_conf(g, cdr, cfg["K"]))
    fire = e | g
    return dict(detection=None if not np.isfinite(det) else float(det),
                confirmation=None if not np.isfinite(conf) else float(conf),
                pre_onset_firing_obs=int((fire & (cdr < 0)).sum()),     # how many pre-onset observations fire at all
                n_obs=int(len(cdr)), n_pre_onset_obs=int((cdr < 0).sum()))


def summarise(days):
    """miss / early-FP / on-time / median / mean, exactly as sweep_kernels.metrics (tau = 0)."""
    d = np.array([np.inf if v is None else v for v in days], float)
    return sk.metrics(d, 0)


def bootstrap(days, rng, nboot):
    """Cluster bootstrap over MUs (resample the 20 MUs with replacement) of the on-time median
    and mean lag, the early-FP count and the on-time count. Percentile 95% intervals."""
    d = np.array([np.inf if v is None else v for v in days], float); n = len(d)
    idx = rng.integers(0, n, size=(nboot, n)); s = d[idx]
    fin = np.isfinite(s); fp = (fin & (s < 0)).sum(1); ok = fin & (s >= 0)
    with np.errstate(all="ignore"):
        med = np.array([np.median(row[m]) if m.any() else np.nan for row, m in zip(s, ok)])
        mean = np.array([np.mean(row[m]) if m.any() else np.nan for row, m in zip(s, ok)])
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return dict(median_lag_ci95=q(med), mean_lag_ci95=q(mean), fp_ci95=q(fp), on_time_ci95=q(ok.sum(1)),
                nboot=nboot, seed=SEED)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    roi_info, rois = sk.build_roi_info()
    _init(roi_info, rois)
    dense = {a: dict(v) for a, v in json.loads(DENSE_PATH.read_text(encoding="utf-8")).items()}
    rois = [a for a in rois if a in dense]; N = len(rois)
    print(f"MUs={N} | dense={DENSE_PATH} | grid: edge {len(sk.EDGE_CFG)} x gauss {len(sk.GAU_CFG)} = {sk.N_COMBO:,} combos/fold", flush=True)

    # 1) in-sample reference: the grid over all N MUs (must reproduce the published operating points)
    workers = int(os.environ.get("WORKERS", min(N + 1, max(1, os.cpu_count() - 2))))
    with Pool(workers, initializer=_init, initargs=(roi_info, rois)) as pool:
        full = pool.apply_async(run_grid, (rois,))
        folds = pool.map(fold, rois)                       # 2) one grid per held-out MU (N folds in parallel)
        full = full.get()
    insample_cfg = {pt: py(full[key]["cfg"]) for pt, key in POINTS.items()}
    print(f"in-sample grid ({N} MUs): " + " | ".join(f"{pt}: {c}" for pt, c in insample_cfg.items()), flush=True)
    for fd in folds:
        print(f"fold -{fd['held_out']}: {fd['secs']:6.1f}s  conf-cfg={fd['cfg']['confirmation']}", flush=True)

    # 3) apply theta_(-k) to MU k; compare with the in-sample configuration on the same MU
    rng = np.random.default_rng(SEED)
    report = dict(n_mu=N, dense=DENSE_PATH.name, grid_combos_per_fold=sk.N_COMBO, insample_cfg=insample_cfg,
                  operating_points={}, folds=folds, secs_total=None)
    for pt in POINTS:
        per_mu = []
        for fd in folds:
            k = fd["held_out"]
            nested = apply_cfg(fd["cfg"][pt], k, dense)
            insample = apply_cfg(insample_cfg[pt], k, dense)
            per_mu.append(dict(mu=k, cfg=fd["cfg"][pt], same_as_insample=fd["cfg"][pt] == insample_cfg[pt],
                               nested=nested, insample=insample))
        agg = {}
        for which in ("nested", "insample"):
            for lag in ("confirmation", "detection"):
                days = [m[which][lag] for m in per_mu]
                agg[f"{which}_{lag}"] = dict(**summarise(days), bootstrap=bootstrap(days, rng, NBOOT),
                                            pre_onset_firing_obs_total=int(sum(m[which]["pre_onset_firing_obs"] for m in per_mu)))
        # stability of theta_(-k): how many folds reproduce the in-sample configuration, and which parameters move
        varied = {p: sorted({fd["cfg"][pt][p] for fd in folds}) for p in PARAMS}
        stability = dict(folds_identical_to_insample=int(sum(m["same_as_insample"] for m in per_mu)),
                         distinct_cfgs=len({json.dumps(fd["cfg"][pt], sort_keys=True) for fd in folds}),
                         values_per_param=varied,
                         params_that_vary=[p for p in PARAMS if len(varied[p]) > 1])
        report["operating_points"][pt] = dict(per_mu=per_mu, aggregate=agg, stability=stability)
    report["secs_total"] = round(time.time() - T0, 1)
    (OUT_DIR / "nested_sweep_results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    # 4) human-readable summary
    def fmt(a):
        b = a["bootstrap"]
        return (f"FP={a['fp']} miss={a['miss']} on-time={a['det']}/{N} median={a['median']:+.1f} "
                f"[{b['median_lag_ci95'][0]:+.1f},{b['median_lag_ci95'][1]:+.1f}] mean={a['mean']:+.1f} "
                f"[{b['mean_lag_ci95'][0]:+.1f},{b['mean_lag_ci95'][1]:+.1f}]  pre-onset firing obs={a['pre_onset_firing_obs_total']}")
    lines = [f"# Nested LOMO of the detector grid search  ({N} MUs, {sk.N_COMBO:,} combos/fold, {report['secs_total']} s)", ""]
    for pt in POINTS:
        op = report["operating_points"][pt]; a = op["aggregate"]; st = op["stability"]
        lines += [f"## Operating point: {pt}  (in-sample cfg {insample_cfg[pt]})", "",
                  f"- in-sample, {pt} lag:  {fmt(a['insample_' + pt])}",
                  f"- nested,    {pt} lag:  {fmt(a['nested_' + pt])}",
                  f"- in-sample, detection lag: {fmt(a['insample_detection'])}" if pt == "confirmation" else
                  f"- in-sample, confirmation lag: {fmt(a['insample_confirmation'])}",
                  f"- nested,    detection lag: {fmt(a['nested_detection'])}" if pt == "confirmation" else
                  f"- nested,    confirmation lag: {fmt(a['nested_confirmation'])}",
                  f"- stability: {st['folds_identical_to_insample']}/{N} folds pick the in-sample cfg; "
                  f"{st['distinct_cfgs']} distinct cfgs; varying params: {st['params_that_vary'] or 'none'}; "
                  f"values: { {p: st['values_per_param'][p] for p in st['params_that_vary']} }", "",
                  "| MU | theta_(-k) | same | nested det | nested conf | in-sample det | in-sample conf | pre-onset firing obs (nested) |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in op["per_mu"]:
            c = m["cfg"]; cs = f"f{c['f']} p{c['p']} TE{c['TE']} ∨ L{c['L']} σ{c['sigma']} TG{c['TG']} K{c['K']}"
            g = lambda v: "miss" if v is None else f"{v:+.0f}"
            lines.append(f"| {m['mu']} | {cs} | {'=' if m['same_as_insample'] else 'differs'} | {g(m['nested']['detection'])} | "
                         f"{g(m['nested']['confirmation'])} | {g(m['insample']['detection'])} | {g(m['insample']['confirmation'])} | "
                         f"{m['nested']['pre_onset_firing_obs']}/{m['nested']['n_pre_onset_obs']} |")
        lines.append("")
    (OUT_DIR / "nested_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"=== done in {report['secs_total']} s -> {OUT_DIR}/nested_sweep_results.json, nested_sweep_summary.md", flush=True)
