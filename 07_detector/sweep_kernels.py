# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Exhaustive kernel sweep for the detection-LAG grid search (paper scope: NRT response time
to selective logging; 20 common ROIs; no accuracy claims).

Detector arms (padding 0-left / mirror-right for both):
  EDGE (look-ahead):  resp[i] = mean(P[i..i+f-1]) - mean(P[i-p..i-1]) >= TE
                      kernel = [-1/p]*p + [1/f]*f, anchor=p  (f=5,p=10 == original)
                      f in 1..5, p in 1..20, TE in 0.30..0.99 step 0.01
  GAUSS (centered):   smoothed[i] >= TG sustained for >= K consecutive obs
                      L in odd 5..21, sigma in {0.8,1.2,1.6,1.8,2.0,2.4,2.8,3.2} (drop sigma>=L/2)
                      TG in 0.40..0.99 step 0.01, K in 1..5
  COMBO = edge OR gauss; first detection = earliest fire.

Metrics vs t0 (= first in-window seed): miss (never fires) / early-FP (lag < -tau) /
median lag (lag >= -tau). tau=0 strict (primary) + sensitivity tau in {7,15,30}.
Ranking (lexicographic, on the confirmation lags): min FP -> min miss -> min median lag -> min mean lag.

What is searched here is the DETECTOR -- the edge/persistence kernel pair -- over scores the
downstream model produced. DATASETS below is the single reported configuration (the "raw" input variant). Runs the full combo cross-product for whatever DATASETS contains.
Outputs sweep_kernels_results.json (best combo, fp-frontier, best per-arm, original-kernel check)
and prints a comparison summary."""
import json, os, sys, time
from datetime import date
from pathlib import Path
import numpy as np
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy>=2.0 renamed trapz -> trapezoid
from numpy.lib.stride_tricks import sliding_window_view
from multiprocessing import Pool
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from dataset_classifier import load_annotations, HLSClassifierDataset

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ZARRS = Path(os.environ["ZARRS"]) if os.environ.get("ZARRS") else DATA_DIR / "labelled-dataset"
CLF = Path(os.environ["CLF"]) if os.environ.get("CLF") else DATA_DIR
ANNOTATIONS = Path(os.environ["ANNOTATIONS"]) if os.environ.get("ANNOTATIONS") else DATA_DIR / "annotations.json"
VDJSON = Path(os.environ["VDJSON"]) if os.environ.get("VDJSON") else DATA_DIR / "view_dates.json"
EXCL = {"11874_curr", "11984_curr", "11310_curr"}
OUT = CLF / "sweep_kernels_results.json"

DATASETS = [   # (label, variant, dense-P json) -- reported downstream model only, see docstring
    ("reference", "raw", CLF / "dense_predictions.json"),
]
FS = list(range(1, 6)); PS = list(range(1, 21)); TES = [round(x, 2) for x in np.arange(0.30, 0.991, 0.01)]   # f in 1..K_max=5 (== KS); p in 1..20; edge constrained to p>=f; the COMBO restricted to f<=K (edge fires before gauss)
LS = list(range(5, 22, 2)); SIGS = [0.8, 1.2, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2]
TGS = [round(x, 2) for x in np.arange(0.40, 0.991, 0.01)]; KS = list(range(1, 6))
TAUS = [0, 7, 15, 30]


def filt(P, kernel, anchor):                      # correlation, left 0-pad / right mirror
    P = np.nan_to_num(np.asarray(P, float), nan=0.0); n = len(P); K = len(kernel)
    lo = anchor; hi = K - 1 - anchor
    if n < K:                                     # series shorter than kernel -> safe edge pad
        pp = np.pad(P, (lo, hi), mode="edge")
    else:
        pp = np.pad(P, (lo, hi), mode="reflect")
    pp[:lo] = 0.0
    return sliding_window_view(pp, K) @ kernel    # length n


def edge_kernel(f, p):
    return np.concatenate([np.full(p, -1.0 / p), np.full(f, 1.0 / f)]), p   # anchor=p


def gauss_kernel(L, sig):
    a = (L - 1) // 2; w = np.exp(-0.5 * ((np.arange(L) - a) / sig) ** 2)
    return w / w.sum(), a


def sustained(mask, k):                           # True at i iff mask[i:i+k] all True (vectorized)
    m = np.asarray(mask, bool); n = len(m)
    if k <= 1:
        return m
    out = np.zeros(n, bool)
    if n >= k:
        out[:n - k + 1] = np.convolve(m.astype(int), np.ones(k, int), "valid") == k
    return out


def first_day(fire, cdr):
    return cdr[int(np.argmax(fire))] if fire.any() else np.inf


def first_conf(fire, cdr, k):                     # CONFIRMATION day = k-th obs of the sustained run (anchor + k-1); == anchor for k=1
    return cdr[min(int(np.argmax(fire)) + k - 1, len(cdr) - 1)] if fire.any() else np.inf


def metrics(days, tau):                           # days: (na,) first-fire day or inf
    fin = np.isfinite(days)
    miss = int((~fin).sum()); fp = int((fin & (days < -tau)).sum())
    correct = fin & (days >= -tau)
    med = float(np.median(days[correct])) if correct.any() else float("nan")
    mean = float(np.mean(days[correct])) if correct.any() else float("nan")
    return dict(miss=miss, fp=fp, det=int(correct.sum()), median=med, mean=mean)


def build_roi_info():
    """Load seeds + dataset and compute the per-ROI (dates, cdr) window once; variant/version
    independent. Heavy (HLSClassifierDataset opens every ROI's zarr) -- kept OUT of module scope
    and called only under the __main__ guard, so multiprocessing.Pool under spawn (Windows/macOS,
    where each worker re-imports this module) doesn't silently re-run the whole build per worker."""
    ps, amb, pool, refs = load_annotations(ANNOTATIONS)
    vds = {a: date.fromisoformat(s) for a, s in json.loads(VDJSON.read_text(encoding="utf-8")).items()}
    _ds = HLSClassifierDataset(ZARRS, ps, amb, refs, vds, oversample_pool=pool, variants=["raw"], seed=0)
    roi_info = {}
    for roi in _ds.activity_rois:
        if roi in EXCL:
            continue
        seq = _ds._soft_seq(roi, ps[roi], amb.get(roi, []), refs.get(roi), vds[roi], neg_wait=0)
        _, _, _, dates, _, _ = _ds._open(roi)
        change = next((dates[int(t)] for t, v, _ in seq if v >= 1.0), min(d for d, _k in ps[roi]))
        cdr = np.array([(dates[int(t)] - change).days for t, _, _ in seq], float)
        o = np.argsort(cdr)
        roi_info[roi] = (np.array([dates[int(t)].isoformat() for t, _, _ in seq], object)[o], cdr[o])
    return roi_info, sorted(roi_info)


EDGE = [(f, p) for f in FS for p in PS if p >= f]   # p>=f (edge past window >= look-ahead; long-baseline, like the original f5/p10); p in f..20
GAU = [(L, s) for L in LS for s in SIGS if s < L / 2]
EDGE_CFG = [(f, p, te) for (f, p) in EDGE for te in TES]
GAU_CFG = [(L, s, tg, k) for (L, s) in GAU for tg in TGS for k in KS]
N_COMBO = len(EDGE_CFG) * len(GAU_CFG)

ROI_INFO = {}   # populated in __main__ (parent) via build_roi_info(), and in each Pool worker via _init_worker
ROIS = []


def _init_worker(roi_info, rois):
    global ROI_INFO, ROIS
    ROI_INFO = roi_info
    ROIS = rois


def process(args):
    version, variant, path = args
    try:
        DENSE = {a: dict(v) for a, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except FileNotFoundError:
        return dict(version=version, variant=variant, error="dense P json missing")
    rois = [a for a in ROIS if a in DENSE]
    Pa = {a: np.nan_to_num(np.array([DENSE[a].get(d, np.nan) for d in ROI_INFO[a][0]]), nan=0.0) for a in rois}
    cdra = {a: ROI_INFO[a][1] for a in rois}
    na = len(rois)
    t0 = time.time()
    # precompute responses
    eresp = {(f, p): {a: filt(Pa[a], *edge_kernel(f, p)) for a in rois} for (f, p) in EDGE}
    gresp = {(L, s): {a: filt(Pa[a], *gauss_kernel(L, s)) for a in rois} for (L, s) in GAU}
    # first-fire matrices (days; inf = no fire)
    EFF = np.empty((len(EDGE_CFG), na)); EFF_conf = np.empty((len(EDGE_CFG), na))
    for ci, (f, p, te) in enumerate(EDGE_CFG):
        for ai, a in enumerate(rois):
            ef = eresp[(f, p)][a] >= te; cdr = cdra[a]
            EFF[ci, ai] = first_day(ef, cdr); EFF_conf[ci, ai] = first_conf(ef, cdr, f)   # + edge CONFIRMATION = anchor + (f-1) obs (last obs of the look-ahead column)
    GFF = np.empty((len(GAU_CFG), na)); GFF_conf = np.empty((len(GAU_CFG), na))
    for ci, (L, s, tg, k) in enumerate(GAU_CFG):
        for ai, a in enumerate(rois):
            gf = sustained(gresp[(L, s)][a] >= tg, k); cdr = cdra[a]
            GFF[ci, ai] = first_day(gf, cdr); GFF_conf[ci, ai] = first_conf(gf, cdr, k)   # + gauss CONFIRMATION (anchor + K-1 obs)

    # best per-arm (strict)
    def best_arm(FFM, cfgs, names):
        rows = [metrics(FFM[i], 0) for i in range(len(cfgs))]
        fpv = [r["fp"] for r in rows]; mss = [r["miss"] for r in rows]
        mdv = [r["median"] if r["median"] == r["median"] else 1e9 for r in rows]
        mnv = [r["mean"] if r["mean"] == r["mean"] else 1e9 for r in rows]
        bi = int(np.lexsort((mnv, mdv, mss, fpv))[0])    # FP -> miss -> median -> mean (tie-break)
        return dict(cfg=dict(zip(names, cfgs[bi])), **rows[bi])
    best_edge = best_arm(EFF, EDGE_CFG, ["f", "p", "TE"])
    best_gauss = best_arm(GFF, GAU_CFG, ["L", "sigma", "TG", "K"])

    # combo cross (vectorized per edge-cfg): combo days = min(edge, gauss)
    INF = np.inf
    best = None; best_conf = None; best_lm = None; fp_front = {}       # best=visualization, best_conf=confirmation (alert-publishable); fp_front: fp -> (miss, median, mean, ecfg, gcfg)
    NCELL = 21 * 21                                  # FP×miss -> (min median, then min mean, edge cfg, gauss cfg)
    tmed = np.full(NCELL, np.inf); tmean = np.full(NCELL, np.inf); tec = np.full(NCELL, -1, int); tgc = np.full(NCELL, -1, int)
    GK = np.array([g[3] for g in GAU_CFG])           # gauss persistence K per cfg (for the f<=K combo constraint)
    for ei in range(len(EDGE_CFG)):
        gbad = GK < EDGE_CFG[ei][0]                  # f<=K: drop pairs where edge look-ahead f exceeds gauss persistence K (edge must be able to fire before gauss)
        combo = np.minimum(EFF[ei][None, :], GFF); combo[gbad] = np.inf   # (n_gcfg, na)
        fin = np.isfinite(combo)
        miss = (~fin).sum(1); fp = (fin & (combo < 0)).sum(1)
        corr = fin & (combo >= 0)
        cc = np.where(corr, combo, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(cc, axis=1)          # nan where no correct
            mean = np.nanmean(cc, axis=1)
        med_safe = np.where(np.isnan(med), 1e9, med)
        mean_safe = np.where(np.isnan(mean), 1e9, mean)
        # global best by (fp, miss, median, mean) -- FP -> coverage (min miss) -> median -> mean (tie-break)
        gi = int(np.lexsort((mean_safe, med_safe, miss, fp))[0])
        cand = (int(fp[gi]), int(miss[gi]), float(med_safe[gi]), float(mean_safe[gi]), ei, gi)
        if best is None or cand[:4] < best[:4]:
            best = cand
        # CONFIRMATION combo = min(edge conf [anchor+f-1, last look-ahead obs], gauss conf [anchor+K-1]) -- earliest alert-publishable date
        comboc = np.minimum(EFF_conf[ei][None, :], GFF_conf); comboc[gbad] = np.inf
        finc = np.isfinite(comboc); missc = (~finc).sum(1); fpc = (finc & (comboc < 0)).sum(1)
        ccc = np.where(finc & (comboc >= 0), comboc, np.nan)
        with np.errstate(all="ignore"):
            medc = np.nanmedian(ccc, axis=1); meanc = np.nanmean(ccc, axis=1)
        medc_safe = np.where(np.isnan(medc), 1e9, medc); meanc_safe = np.where(np.isnan(meanc), 1e9, meanc)
        gic = int(np.lexsort((meanc_safe, medc_safe, missc, fpc))[0])
        candc = (int(fpc[gic]), int(missc[gic]), float(medc_safe[gic]), float(meanc_safe[gic]), ei, gic)
        if best_conf is None or candc[:4] < best_conf[:4]:
            best_conf = candc
        # min-MISS at FP=0 with median<20 (relax median, favor coverage); mean tie-break
        m20 = (fp == 0) & np.isfinite(med) & (med < 20)
        if m20.any():
            idx = np.where(m20)[0]; j = int(idx[np.lexsort((mean[idx], med[idx], miss[idx]))[0]])
            cand2 = (int(miss[j]), float(med[j]), float(mean[j]), ei, j)
            if best_lm is None or (cand2[0], cand2[1], cand2[2]) < (best_lm[0], best_lm[1], best_lm[2]):
                best_lm = cand2
        # fp-frontier (FP budget 0..6): per FP, max coverage (min miss) -> min median -> min mean
        for fpv in range(0, 7):
            sel = np.where(fp == fpv)[0]
            if len(sel) == 0:
                continue
            j = int(sel[np.lexsort((mean_safe[sel], med_safe[sel], miss[sel]))[0]])
            if (fpv not in fp_front) or (int(miss[j]), med_safe[j], mean_safe[j]) < (fp_front[fpv][0], fp_front[fpv][1], fp_front[fpv][2]):
                fp_front[fpv] = (int(miss[j]), float(med_safe[j]), float(mean_safe[j]), ei, int(j))
        # FP×miss -> min-(median,mean) table (compact; enables ANY re-ranking as a pure read, no re-run)
        key = fp * 21 + miss
        order = np.lexsort((mean_safe, med_safe, key)); ks = key[order]
        fm = np.empty(len(ks), bool); fm[0] = True; fm[1:] = ks[1:] != ks[:-1]
        for gc in order[fm]:
            c = int(key[gc])
            if (med_safe[gc], mean_safe[gc]) < (tmed[c], tmean[c]):
                tmed[c] = med_safe[gc]; tmean[c] = mean_safe[gc]; tec[c] = ei; tgc[c] = int(gc)

    def combo_detail(ei, gi):                        # recompute VISUALIZATION + CONFIRMATION combo days + metrics + area
        combo = np.minimum(EFF[ei], GFF[gi])
        comboc = np.minimum(EFF_conf[ei], GFF_conf[gi])   # confirmation = min(edge anchor+f-1, gauss anchor+K-1)
        f, p, te = EDGE_CFG[ei]; L, s, tg, k = GAU_CFG[gi]
        cfg = dict(f=f, p=p, TE=te, L=L, sigma=s, TG=tg, K=k)
        per_tau = {str(t): metrics(combo, t) for t in TAUS}
        conf_tau = {str(t): metrics(comboc, t) for t in TAUS}
        per_roi = {a: (None if not np.isfinite(combo[i]) else float(combo[i])) for i, a in enumerate(rois)}
        areas = {}                                   # green hatched firing-band area = integral of (gauss-TG) over sustained firing, per ROI (P*day)
        for a in rois:
            g = gresp[(L, s)][a]; gf = sustained(g >= tg, k)
            areas[a] = float(_trapz(np.where(gf, np.maximum(g - tg, 0.0), 0.0), cdra[a]))
        av = list(areas.values())
        return dict(cfg=cfg, metrics_strict=per_tau["0"], sensitivity=per_tau, per_roi=per_roi,
                    confirmation=dict(metrics_strict=conf_tau["0"], sensitivity=conf_tau),
                    hatch_area=dict(total=round(float(sum(av)), 2), mean=round(float(np.mean(av)), 2),
                                    per_roi={a: round(v, 2) for a, v in areas.items()}))

    fpmiss = []
    for c in range(NCELL):
        if np.isfinite(tmed[c]):
            fpv, mv = divmod(c, 21)
            fpmiss.append(dict(fp=fpv, miss=mv, det=na - fpv - mv, median=round(float(tmed[c]), 1),
                               mean=round(float(tmean[c]), 1),
                               edge=dict(zip(["f", "p", "TE"], EDGE_CFG[tec[c]])),
                               gauss=dict(zip(["L", "sigma", "TG", "K"], GAU_CFG[tgc[c]]))))
    bi_fp, bi_miss, bi_med, bi_mean, bei, bgi = best
    result = dict(version=version, variant=variant, n_roi=na, secs=round(time.time() - t0, 1),
                  fp_miss_table=fpmiss,
                  best_combo=combo_detail(bei, bgi),
                  best_combo_confirmation=combo_detail(best_conf[4], best_conf[5]),
                  best_edge_alone=best_edge, best_gauss_alone=best_gauss,
                  best_lowmiss_fp0=(combo_detail(best_lm[3], best_lm[4]) if best_lm else None),
                  fp_frontier={str(k): dict(miss=v[0], median=v[1], mean=v[2],
                                            edge=dict(zip(["f", "p", "TE"], EDGE_CFG[v[3]])),
                                            gauss=dict(zip(["L", "sigma", "TG", "K"], GAU_CFG[v[4]])))
                               for k, v in sorted(fp_front.items())})
    return result


if __name__ == "__main__":
    ROI_INFO, ROIS = build_roi_info()
    N = len(ROIS)
    print(f"ROIs={N} | edge cfgs={len(EDGE_CFG)} (kernels {len(EDGE)}) | gauss cfgs={len(GAU_CFG)} "
          f"(kernels {len(GAU)}) | combos/set={N_COMBO:,} | x{len(DATASETS)} sets = {N_COMBO*len(DATASETS):,}",
          flush=True)
    t0 = time.time()
    with Pool(min(len(DATASETS), max(1, os.cpu_count() - 2)),
              initializer=_init_worker, initargs=(ROI_INFO, ROIS)) as pool_:
        results = pool_.map(process, DATASETS)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n=== done in {time.time()-t0:.0f}s -> {OUT} ===\n")
    def _row(r, key, by):                            # by: 'vis'|'conf' -> miss/FP/det of the RANKING date; both lags always shown
        c = r[key]; cf = c["cfg"]; m = c["metrics_strict"]; mc = c["confirmation"]["metrics_strict"]
        mm = m if by == "vis" else mc
        kerns = f"f{cf['f']} p{cf['p']} TE{cf['TE']} | L{cf['L']} σ{cf['sigma']} TG{cf['TG']} K{cf['K']}"
        print(f"{r['version']:<7} {r['variant']:<10} | {kerns:<40} | {mm['miss']:>4} {mm['fp']:>3} {mm['det']:>3} | {m['median']:>+5.0f} {m['mean']:>+6.0f} | {mc['median']:>+5.0f} {mc['mean']:>+6.0f} | {c['hatch_area']['mean']:>6.1f}")
    _hdr = f"{'set':<10} {'var':<10} | {'combo (f,p,TE | L,σ,TG,K)':<40} | {'miss':>4} {'FP':>3} {'det':>3} | {'vMed':>5} {'vMean':>6} | {'cMed':>5} {'cMean':>6} | {'hatchA':>6}"
    print("=== BEST by DETECTION date (first detectable)  ·  v=detection, c=confirmation (publishable alert) ===")
    print(_hdr)
    for r in results:
        if "error" in r:
            print(f"{r['version']:<7} {r['variant']:<10} | {r['error']}"); continue
        _row(r, "best_combo", "vis")
    print("\n=== BEST by CONFIRMATION date (publishable alert date) ===")
    print(_hdr)
    for r in results:
        if "error" not in r:
            _row(r, "best_combo_confirmation", "conf")
    print(f"\n=== FEWEST misses @ FP=0 with median<20 (relaxes the median, favours coverage) ===")
    print(f"{'set':<10} {'var':<10} | {'kernel (edge ∨ gauss)':<40} | {'miss':>4} {'det':>3} {'medLag':>6}")
    for r in results:
        if "error" in r:
            continue
        c = r.get("best_lowmiss_fp0")
        if c is None:
            print(f"{r['version']:<10} {r['variant']:<10} | (no FP=0 combo with median<20)"); continue
        cf = c["cfg"]; m = c["metrics_strict"]
        kerns = f"f{cf['f']} p{cf['p']} TE{cf['TE']} | L{cf['L']} σ{cf['sigma']} TG{cf['TG']} K{cf['K']}"
        print(f"{r['version']:<7} {r['variant']:<10} | {kerns:<40} | {m['miss']:>4} {m['det']:>3} {m['median']:>+6.0f}")
    # ---- self-check: a fixed kernel pair on the published dense predictions, as a regression
    # guard for the filters (expected 5 misses / 1 FP / +9 d median) ----
    DV = {a: dict(v) for a, v in json.loads((CLF / "dense_predictions.json").read_text(encoding="utf-8")).items()}
    vdays = []
    for a in [x for x in ROIS if x in DV]:
        diso, cdr = ROI_INFO[a]
        P = np.nan_to_num(np.array([DV[a].get(d, np.nan) for d in diso]), nan=0.0)
        e = filt(P, *edge_kernel(5, 10)); g = filt(P, *gauss_kernel(11, 1.8))
        vdays.append(first_day((e >= 0.8) | sustained(g >= 0.8, 1), cdr))
    vm = metrics(np.array(vdays), 0)
    print(f"\n[self-check] fixed kernel pair (f5 p10 TE0.8 ∨ L11 σ1.8 TG0.8 K1): "
          f"miss={vm['miss']} FP={vm['fp']} det={vm['det']} medLag={vm['median']:+.0f}  (expected 5/1/+9)")
