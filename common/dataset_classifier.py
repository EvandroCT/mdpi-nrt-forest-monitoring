# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""HLSClassifierDataset — soft-label regression over per-ROI monitoring windows.

Soft label v(t) (per ROI, obs in time order over the observations interval, t>=L-1). Walking
the series forward from first_observation:
  * 0 on the pre-disturbance baseline, before the onset (ambiguous observations there are
    INERT — they only count once logging activity has been seen);
  * 1.0 at every logging-activity observation, onset and increment alike;
  * afterwards each next obs: v -= STEP (STEP=0.05); an ambiguous obs instead does v += STEP,
    reading as weak evidence of ongoing disturbance; clamp [0,1].
POSITIVES = obs with v>0 (target = soft v). NEGATIVES = obs with v=0, stratified by TIME x CLOUD
using ONLY non-empty cells (RELAXED: do NOT force a cloud substratum where the ROI has no obs —
middle cloud [10,30)/[30,60) is rare/bimodal). Quota = N_pos per ROI (1:1), drawn ROUND-ROBIN
uniformly across non-empty (time-bin x cloud-substratum) cells; candidates = within-ROI v=0 obs
(K=max(1,round(N_pos/4)) equal-count time bins x 4 cloud substrata) + a DISJOINT slice of the
logging-free candidate pool (adds mid-cloud diversity where it exists). That pool is split
disjointly per ROI => no LOOCV leak.

Per target, 4 input variants (target slot): raw / composite (on-the-fly) and recon / masked (caches).
LOOCV by `fold_roi`. Loss = MSE against the soft target.

Negative sampling (within-ROI and candidate-pool draws alike) is fully deterministic given `seed`
(default seed=0, round-robin over non-empty time x cloud-substratum cells) — it is not a
hand-picked or curated selection."""
from __future__ import annotations
from collections import defaultdict
from datetime import date, datetime
import json
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

_MB = 0b10001111
_L = 80
_SIDE = 64
_NEG_WAIT = 10
_STEP = 0.05
_SUB_EDGES = [0.0, 0.10, 0.30, 0.60, 1.01]   # last bin [0.60, 1.0] inclusive (100% cloud)
_NSUB = 4


def _np_date(t64):
    return datetime.utcfromtimestamp((t64 - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")).date()


def _parse_date(s: str) -> date:
    s = s.strip()
    if "-" in s:
        return date.fromisoformat(s)
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


# load_annotations/_parse_date live here because the training and cache-building scripts
# both need this loader.
def load_annotations(
    path: str | Path,
) -> tuple[dict[str, list[tuple[date, int]]], dict[str, list[date]],
           list[tuple[str, date, int]], dict[str, date]]:
    """Parse the interpreter annotations the soft labels are derived from.

    The annotation protocol records only two verdicts per observation: LOGGING
    ACTIVITY, when a new selective-logging event is unambiguously discernible in the
    canopy, and AMBIGUOUS, when the logging state cannot be confidently resolved.
    Everything left unmarked is taken to be undisturbed.

    Returns (logging_activity, ambiguous_events, oversample_pool, first_observation):
      logging_activity:  {roi: [(date, k), ...]} — every observation showing logging
                         activity. The earliest is the onset, against which all detection
                         timing is measured; the rest are increments. `k` is 1-based and
                         picks among several observations sharing one date (two
                         platforms, or two tiles) -- it is not the kind of activity.
      ambiguous_events:  {roi: [date, ...]} — never counted as logging activity. Before
                         the onset they bound the undisturbed region; after it they read
                         as weak evidence of ongoing disturbance and raise the soft label.
      oversample_pool:   [(roi, date, k), ...] — the LOGGING-FREE CANDIDATES: candidate
                         target observations from the logging-free sites, from which the
                         negative samples are drawn. The key name is historical; this is
                         the candidate pool, larger than the subset actually drawn.
      first_observation: {roi: date} — the first acquisition inside the unit's monitoring
                         period; the anchor every window below is measured from.

    Four terms, used with these meanings throughout the repository and the record:

        monitoring_period       [ start_period .............................. end_period ]
        observations interval        [ first_observation ... last_observation ]

        start_period <= first_observation <= last_observation <= end_period

      * end_period is the DETER alert's view date, the one primary datum here; it is what
        view_dates.json holds.
      * start_period is end_period minus one calendar year.
      * first_observation is the first acquisition at or after start_period. Acquisitions
        are irregular, so it lands a few days inside the period rather than on its edge
        (1-8 days across the 20 monitoring units).
      * last_observation is the last acquisition at or before end_period (0-6 days early).

    Only end_period and first_observation are stored: the other two are derivable, and
    duplicating derived values into a published artefact only invites them to drift.
    """
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    logging_activity: dict[str, list[tuple[date, int]]] = {}
    for roi, entries in cfg.get("logging_activity", {}).items():
        obs = sorted({(_parse_date(d), int(k)) for d, k in entries})
        if obs:
            logging_activity[roi] = obs
    ambiguous_events: dict[str, list[date]] = {
        roi: sorted({_parse_date(d) for d in dates})
        for roi, dates in cfg.get("ambiguous_events", {}).items()
    }
    pool_raw = cfg.get("oversample_pool", [])
    oversample_pool = [(r, _parse_date(d), int(k)) for r, d, k in pool_raw]
    first_observation = {roi: _parse_date(d)
                       for roi, d in cfg.get("first_observation", {}).items()}
    return logging_activity, ambiguous_events, oversample_pool, first_observation


def substratum(frac: float) -> int:
    for i in range(_NSUB):
        if _SUB_EDGES[i] <= frac < _SUB_EDGES[i + 1]:
            return i
    return _NSUB - 1


class HLSClassifierDataset(Dataset):
    def __init__(self, zarr_dir, logging_activity, ambiguous_events, first_observation, view_dates,
                 oversample_pool=(), variants=("raw",), exclude_rois=(),
                 recon_cache=None,
                 masked_cache=None,
                 lookback=_L, crop_size=_SIDE, mask_bits=_MB, seed=0):
        super().__init__()
        self.zarr_dir = Path(zarr_dir); self.L = int(lookback); self.side = int(crop_size)
        self.mb = int(mask_bits); self.variants = list(variants)
        # recon_cache/masked_cache default to DATA_DIR-relative (self.zarr_dir.parent, since
        # every caller sets zarr_dir=DATA_DIR/"labelled-dataset"), not the cwd -- a bare
        # cwd-relative default silently breaks recon/masked variants whenever the caller's cwd
        # isn't the repo root (build_caches.py writes the caches to this same DATA_DIR-relative
        # location; see RECON/MASKED there).
        self.recon_cache = Path(recon_cache) if recon_cache is not None else self.zarr_dir.parent / "pred_cache_recon"
        self.masked_cache = Path(masked_cache) if masked_cache is not None else self.zarr_dir.parent / "pred_cache_masked"
        self._zc = {}
        rng = np.random.default_rng(seed)

        activity_rois = sorted(r for r in logging_activity if r in view_dates and r not in set(exclude_rois))

        # ---- soft labels: positives (v>0) + within-ROI negatives (t, substratum) ----
        pos_samples = []; within_neg = {}; npos = {}
        for roi in activity_rois:
            seq = self._soft_seq(roi, logging_activity[roi], ambiguous_events.get(roi, []),
                                 first_observation.get(roi), view_dates[roi])
            wn = []; np_roi = 0
            for t, v, frac in seq:
                if v > 0:
                    pos_samples.append(dict(fold_roi=roi, data_roi=roi, target_idx=t, target=float(v)))
                    np_roi += 1
                else:
                    wn.append((t, substratum(frac)))
            within_neg[roi] = sorted(wn); npos[roi] = np_roi

        # ---- logging-free candidates / cross-ROI extras, resolved to (roi, t, substratum) ----
        cross_neg = []
        for croi, D, k in oversample_pool:
            res = self._resolve(croi, D, k)
            if res is not None:
                t, frac = res
                cross_neg.append((croi, t, substratum(frac)))
        rng.shuffle(cross_neg)
        v1_slice = {r: [] for r in activity_rois}
        for i, item in enumerate(cross_neg):
            v1_slice[activity_rois[i % len(activity_rois)]].append(item)   # round-robin, disjoint

        # ---- negatives: round-robin over NON-EMPTY (time-bin x cloud) cells, quota = N_pos ----
        neg_samples = []; self._counts = {}
        for roi in activity_rois:
            quota = npos[roi]
            wn = within_neg[roi]
            K = max(1, int(round(npos[roi] / 4)))
            ts = np.asarray([t for t, _ in wn], dtype=np.int64); sub_of = {t: s for t, s in wn}
            bin_of = {}
            if len(ts):
                for bi, chunk in enumerate(np.array_split(np.sort(ts), K)):
                    for t in chunk:
                        bin_of[int(t)] = bi
            cells = defaultdict(list)
            for t, s in wn:
                cells[(bin_of[t], s)].append((roi, t))
            for (croi, ct, cs) in v1_slice[roi]:
                cells[("X", cs)].append((croi, ct))
            for c in cells:
                rng.shuffle(cells[c])
            keys = sorted(cells, key=lambda k: (str(k[0]), k[1]))
            ptr = {c: 0 for c in keys}; c = dict(npos=npos[roi], within=0, cross=0, per_sub=[0, 0, 0, 0])
            drawn = 0
            while drawn < quota and any(ptr[c2] < len(cells[c2]) for c2 in keys):
                for key in keys:
                    if drawn >= quota:
                        break
                    if ptr[key] < len(cells[key]):
                        droi, t = cells[key][ptr[key]]; ptr[key] += 1; drawn += 1
                        is_cross = key[0] == "X"
                        # cross-ROI negs are TRAINING-ONLY (never in any val) -> clean per-ROI eval
                        neg_samples.append(dict(fold_roi="__cross__" if is_cross else roi,
                                                data_roi=droi, target_idx=int(t), target=0.0))
                        c["per_sub"][key[1]] += 1
                        c["cross" if is_cross else "within"] += 1
            self._counts[roi] = c

        # ---- expand by variants ----
        self._samples = []
        for b in pos_samples + neg_samples:
            for vr in self.variants:
                s = dict(b); s["variant"] = vr
                s["lookback_idx"] = np.arange(s["target_idx"] - (self.L - 1), s["target_idx"], dtype=np.int64)
                self._samples.append(s)
        self.activity_rois = activity_rois
        self.n_pos_base = len(pos_samples); self.n_neg_base = len(neg_samples)

    # ---- soft-label sequence -------------------------------------------------
    def _soft_seq(self, roi, activity, ambiguous_dates, period_start, vdate, neg_wait=_NEG_WAIT):
        refl, fm, times, dates, _, _ = self._open(roi)
        T = len(times)
        if period_start is not None:
            post = [t for t in range(self.L - 1, T) if dates[t] >= period_start]
            es = dates[post[neg_wait]] if len(post) > neg_wait else (dates[post[0]] if post else date.min)
        else:
            es = date.min
        win = [t for t in range(T) if t >= self.L - 1 and es <= dates[t] <= vdate]
        activity_t = set()
        for D, k in activity:
            idx = sorted([t for t in win if dates[t] == D], key=lambda t: times[t])
            if 1 <= k <= len(idx):
                activity_t.add(idx[k - 1])
        ambiguous_set = set(ambiguous_dates)
        out = []; v = 0.0; seen = False
        for t in win:
            if t in activity_t:
                v = 1.0; seen = True
            elif seen:
                v = min(1.0, v + _STEP) if dates[t] in ambiguous_set else max(0.0, v - _STEP)
            out.append((t, v, self._frac(refl, fm, t)))
        return out

    def _resolve(self, roi, D, k):
        try:
            refl, fm, times, dates, _, _ = self._open(roi)
        except Exception:
            return None
        idx = sorted([t for t in range(len(times)) if dates[t] == D], key=lambda t: times[t])
        if 1 <= k <= len(idx):
            t = idx[k - 1]
            if t >= self.L - 1:
                return t, self._frac(refl, fm, t)
        return None

    # ---- io / helpers --------------------------------------------------------
    def _open(self, roi):
        if roi in self._zc:
            return self._zc[roi]
        z = xr.open_zarr(self.zarr_dir / f"{roi}.zarr", consolidated=False)
        yt = min(self.side, int(z.sizes["y"])); xt = min(self.side, int(z.sizes["x"]))
        refl = z["reflectance"].isel(y=slice(0, yt), x=slice(0, xt)).values.astype(np.float32)
        fmv = z["fmask"].isel(y=slice(0, yt), x=slice(0, xt)).values
        fm = np.where(np.isfinite(fmv), fmv, 255).astype(np.uint8)
        times = z["time"].values
        dates = np.array([_np_date(t) for t in times], dtype=object)
        self._zc[roi] = (refl, fm, times, dates, yt, xt)
        return self._zc[roi]

    def _frac(self, refl, fm, t):
        return float((((fm[t] & self.mb) != 0) | ~np.isfinite(refl[t, 0])).mean())

    def __len__(self):
        return len(self._samples)

    def fold_indices(self, held_out_roi):
        val = [i for i, s in enumerate(self._samples) if s["fold_roi"] == held_out_roi]
        train = [i for i, s in enumerate(self._samples) if s["fold_roi"] != held_out_roi]
        return train, val

    def __getitem__(self, idx):
        s = self._samples[idx]
        roi = s["data_roi"]; t = int(s["target_idx"]); variant = s["variant"]
        refl_full, fm_full, times, dates, yt, xt = self._open(roi)
        L = self.L; side = self.side
        tidx = np.concatenate([s["lookback_idx"], [t]]).astype(np.int64)
        base = refl_full[tidx].copy(); fmu = fm_full[tidx]
        clear = ((fmu & self.mb) == 0) & np.isfinite(base[:, 0])
        for k in range(L):
            bad = (fmu[k] & self.mb) != 0
            if bad.any():
                base[k][:, bad] = np.nan
        if variant == "composite":
            C = base.shape[1]; comp = np.full((C, yt, xt), np.nan, np.float32); fill = np.zeros((yt, xt), bool)
            for k in range(L - 1, -1, -1):
                tk = clear[k] & ~fill
                if tk.any():
                    comp[:, tk] = base[k][:, tk]; fill |= tk
            base[L - 1] = comp
        elif variant == "recon":
            base[L - 1] = np.load(self.recon_cache / roi / f"{t}.npy")[:, :yt, :xt]
        elif variant == "masked":
            base[L - 1] = np.load(self.masked_cache / roi / f"{t}.npy")[:, :yt, :xt]
        inv = np.isnan(base[:, 0]).astype(np.float32)
        arr = np.concatenate([base, inv[:, None]], axis=1).astype(np.float32)
        dts = ((times[t] - times[tidx]) / np.timedelta64(1, "D")).astype(np.float32)
        if yt < side or xt < side:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, side - yt), (0, side - xt)), constant_values=0.0)
        return {"inputs": torch.from_numpy(arr), "dates": torch.from_numpy(dts),
                "target": torch.tensor(s["target"], dtype=torch.float32),
                "meta": {"fold_roi": s["fold_roi"], "data_roi": roi, "target_idx": t,
                         "variant": variant, "target": s["target"]}}


def collate(batch):
    return {"inputs": torch.stack([b["inputs"] for b in batch]),
            "dates": torch.stack([b["dates"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]),
            "meta": [b["meta"] for b in batch]}
