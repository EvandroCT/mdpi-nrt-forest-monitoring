# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Precompute the FM target-slot caches for the downstream model's sample set (recon + masked).

Enumerates every unique (data_roi, target_idx) that the dataset emits as a base
sample (positives v>0 + negatives v=0, within-ROI + cross-ROI v1-pool) and runs the
FROZEN foundation trunk to produce, per target:

  recon  : FM reconstruction of the VISIBLE obs — natural cloud mask, NO fwd offset,
           target at its real date t. -> <DATA_DIR>/pred_cache_recon/<roi>/<t>.npy
  masked : offset-+1, 100%-masked target slot, DT_FWD=32 (same convention as the
           earlier masked-cache builder). -> <DATA_DIR>/pred_cache_masked/<roi>/<t>.npy

Each is a (6, yt, xt) float32. Needs a GPU. Usage:
  python build_caches.py --which both
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_pretraining"))
import xarray as xr
from dataset_classifier import load_annotations, HLSClassifierDataset
from model import TimeFirstAutoencoder

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ZARRS = DATA_DIR / "labelled-dataset"
VDJSON = DATA_DIR / "view_dates.json"
ANNOTATIONS = DATA_DIR / "annotations.json"
CKPT = DATA_DIR / "checkpoints" / "foundation" / "best.pt"
RECON = DATA_DIR / "pred_cache_recon"
MASKED = DATA_DIR / "pred_cache_masked"
MB = 0b10001111; L = 80; SIDE = 64; DEV = "cuda"; DT_FWD = 32.0


def load_trunk():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False); a = ck["args"]
    t = TimeFirstAutoencoder(in_channels=a["in_channels"], out_channels=a["out_channels"],
        embed_dim=a["embed_dim"], num_heads=a["num_heads"], temporal_depth=a["temporal_depth"],
        spatial_depth=a["spatial_depth"], seq_len=a["lookback"], crop_size=a["crop_size"],
        window=a["window"], mlp_ratio=a["mlp_ratio"], dropout=a["dropout"],
        identity_bypass=a["identity_bypass"], spatial_attn=a["spatial_attn"]).to(DEV)
    t.load_state_dict(ck["model"], strict=True); t.eval(); return t


def _pad(arr, yt, xt):
    if yt < SIDE or xt < SIDE:
        arr = np.pad(arr, ((0, 0), (0, 0), (0, SIDE - yt), (0, SIDE - xt)), constant_values=0.0)
    return arr


def _fwd(trunk, arr, dts, yt, xt):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pr = trunk(torch.from_numpy(arr).unsqueeze(0).to(DEV),
                   torch.from_numpy(dts).unsqueeze(0).to(DEV)).float().cpu().numpy()[0]
    return pr[:, :yt, :xt].astype(np.float32)


def gen_recon(trunk, refl, fm, times, t, yt, xt):
    tidx = np.concatenate([np.arange(t - (L - 1), t), [t]]).astype(np.int64)
    r = refl[tidx].copy(); fmu = fm[tidx]
    for k in range(L):
        bad = (fmu[k] & MB) != 0
        if bad.any(): r[k][:, bad] = np.nan
    inv = np.isnan(r[:, 0]).astype(np.float32)          # target keeps NATURAL mask
    arr = _pad(np.concatenate([r, inv[:, None]], axis=1).astype(np.float32), yt, xt)
    dts = ((times[t] - times[tidx]) / np.timedelta64(1, "D")).astype(np.float32)
    return _fwd(trunk, arr, dts, yt, xt)


def gen_masked(trunk, refl, fm, times, t, yt, xt, T):
    if t + 1 < T:
        tidx = np.concatenate([np.arange(t - (L - 2), t + 1), [t + 1]]).astype(np.int64)
    else:
        tidx = np.concatenate([np.arange(t - (L - 1), t), [t]]).astype(np.int64)
    r = refl[tidx].copy(); fmu = fm[tidx]
    for k in range(L):
        bad = (fmu[k] & MB) != 0
        if bad.any(): r[k][:, bad] = np.nan
    r[L - 1] = np.nan                                    # 100% mask the predicted slot
    inv = np.isnan(r[:, 0]).astype(np.float32)
    arr = _pad(np.concatenate([r, inv[:, None]], axis=1).astype(np.float32), yt, xt)
    tg = times[tidx[-1]]; dts = ((tg - times[tidx]) / np.timedelta64(1, "D")).astype(np.float32)
    dts[:-1] += DT_FWD
    return _fwd(trunk, arr, dts, yt, xt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["recon", "masked", "both"], default="both")
    args = ap.parse_args()
    import json
    from datetime import date
    ps, amb, pool, refs = load_annotations(ANNOTATIONS)
    vds = {a: date.fromisoformat(s) for a, s in json.loads(VDJSON.read_text(encoding="utf-8")).items()}
    excl = set(a for a in os.environ.get("EXCLUDE", "").split(",") if a)
    ds = HLSClassifierDataset(ZARRS, ps, amb, refs, vds, oversample_pool=pool, variants=["raw"], exclude_rois=excl)
    if excl:
        print(f"[cache] exclude_rois={sorted(excl)} (match training sampling)", flush=True)
    targets = {}
    for s in ds._samples:
        targets.setdefault(s["data_roi"], set()).add(int(s["target_idx"]))
    n_total = sum(len(v) for v in targets.values())
    print(f"[cache] {n_total} unique targets across {len(targets)} ROIs; which={args.which}", flush=True)

    trunk = load_trunk()
    do_recon = args.which in ("recon", "both"); do_masked = args.which in ("masked", "both")
    t0 = time.time(); n = 0
    for roi, tset in sorted(targets.items()):
        z = xr.open_zarr(ZARRS / f"{roi}.zarr", consolidated=False)
        yt = min(SIDE, int(z.sizes["y"])); xt = min(SIDE, int(z.sizes["x"]))
        refl = z["reflectance"].isel(y=slice(0, yt), x=slice(0, xt)).values.astype(np.float32)
        fmv = z["fmask"].isel(y=slice(0, yt), x=slice(0, xt)).values
        fm = np.where(np.isfinite(fmv), fmv, 255).astype(np.uint8)
        times = z["time"].values; T = len(times)
        if do_recon: (RECON / roi).mkdir(parents=True, exist_ok=True)
        if do_masked: (MASKED / roi).mkdir(parents=True, exist_ok=True)
        for t in sorted(tset):
            if do_recon:
                p = RECON / roi / f"{t}.npy"
                if not p.exists(): np.save(p, gen_recon(trunk, refl, fm, times, t, yt, xt))
            if do_masked:
                p = MASKED / roi / f"{t}.npy"
                if not p.exists(): np.save(p, gen_masked(trunk, refl, fm, times, t, yt, xt, T))
            n += 1
        print(f"[cache] {roi}: {len(tset)} ({n}/{n_total}, {time.time()-t0:.0f}s)", flush=True)
    print(f"[cache] DONE {n} targets in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
