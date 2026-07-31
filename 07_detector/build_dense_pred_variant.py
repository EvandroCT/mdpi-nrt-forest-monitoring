# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Dense leak-free OOF P over the monitoring window, per input variant.

For each positive ROI, load its LOO model ({PREFIX}-{VARIANT}-loo{roi}/checkpoints/best.pt)
and run it over EVERY obs in [ref (NEG_WAIT=0) .. VIEW_DATE], building the variant's
target-slot input EXACTLY as dataset_classifier.__getitem__ does. Saves
{roi: [[date_iso, P], ...]} -> dense_predictions.json for the raw variant (the reported
detector, and the file 07_detector/sweep_kernels.py reads), or
dense_predictions_{VARIANT}.json for the comparison variants.

Variants: raw, composite, all (-> raw target slot, on-the-fly); recon, masked (-> the
FROZEN foundation trunk reconstructs/inpaints the target slot ON-THE-FLY for EVERY window
obs, replicating build_caches.gen_recon/gen_masked -- the training cache only covered
training targets). 'all' = the all-variant model (VARIANTS[all]=raw,recon,masked,composite)
run on the raw input, the natural choice over real observations.

Only the raw variant is the downstream model reported in the paper; the rest exist for the
Discussion-section comparison and are not part of the published record.

Env: VARIANT (raw|composite|all|recon|masked), PREFIX (run-directory namespace; must match
what the cross-validation was trained under). GPU; small batch."""
import json, os, sys, time
from datetime import date, datetime
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_pretraining"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import xarray as xr
from classifier_head import ClassificationHead
from model import TimeFirstAutoencoder
from dataset_classifier import load_annotations

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ZARRS = Path(os.environ["ZARRS"]) if os.environ.get("ZARRS") else DATA_DIR / "labelled-dataset"
RUNS = Path(os.environ["RUNS"]) if os.environ.get("RUNS") else DATA_DIR / "runs"
CLF = Path(os.environ["CLF"]) if os.environ.get("CLF") else DATA_DIR
ANNOTATIONS = Path(os.environ["ANNOTATIONS"]) if os.environ.get("ANNOTATIONS") else DATA_DIR / "annotations.json"
VDJSON = Path(os.environ["VDJSON"]) if os.environ.get("VDJSON") else DATA_DIR / "view_dates.json"
VARIANT = os.environ.get("VARIANT", "raw"); PREFIX = os.environ.get("PREFIX", "regressor")
# raw is the reported detector, so its output takes the unqualified name that
# 07_detector/sweep_kernels.py reads and that ships in the published record.
_OUT_NAME = "dense_predictions.json" if VARIANT == "raw" else f"dense_predictions_{VARIANT}.json"
OUT = Path(os.environ["OUT"]) if os.environ.get("OUT") else CLF / _OUT_NAME
MB = 0b10001111; L = 80; SIDE = 64; NEG_WAIT = 0; DEV = "cuda"; BATCH = 4; DT_FWD = 32.0
CKPT_FM = Path(os.environ["FM_CKPT"]) if os.environ.get("FM_CKPT") else DATA_DIR / "checkpoints" / "foundation" / "best.pt"   # frozen foundation trunk (recon/masked slots)

assert VARIANT in ("raw", "composite", "all", "recon", "masked"), f"unknown VARIANT {VARIANT}"
NEEDS_FM = VARIANT in ("recon", "masked")   # 'all' uses the raw input; only recon/masked need the FM


def _npdate(t64):
    return datetime.utcfromtimestamp((t64 - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")).date()


def _pad(arr, yt, xt):
    if yt < SIDE or xt < SIDE:
        arr = np.pad(arr, ((0, 0), (0, 0), (0, SIDE - yt), (0, SIDE - xt)), constant_values=0.0)
    return arr


def load_fm_trunk():
    """Frozen foundation autoencoder used to reconstruct/inpaint the target slot."""
    ck = torch.load(CKPT_FM, map_location="cpu", weights_only=False); a = ck["args"]
    t = TimeFirstAutoencoder(in_channels=a["in_channels"], out_channels=a["out_channels"],
        embed_dim=a["embed_dim"], num_heads=a["num_heads"], temporal_depth=a["temporal_depth"],
        spatial_depth=a["spatial_depth"], seq_len=a["lookback"], crop_size=a["crop_size"],
        window=a["window"], mlp_ratio=a["mlp_ratio"], dropout=a["dropout"],
        identity_bypass=a["identity_bypass"], spatial_attn=a["spatial_attn"]).to(DEV)
    t.load_state_dict(ck["model"], strict=True); t.eval(); return t


def _fm_fwd(trunk, arr, dts, yt, xt):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pr = trunk(torch.from_numpy(arr).unsqueeze(0).to(DEV),
                   torch.from_numpy(dts).unsqueeze(0).to(DEV)).float().cpu().numpy()[0]
    return pr[:, :yt, :xt].astype(np.float32)


def gen_recon(trunk, refl, fm, times, t, yt, xt):   # FM reconstruction of the visible obs at t (natural mask, no fwd offset)
    tidx = np.concatenate([np.arange(t - (L - 1), t), [t]]).astype(np.int64)
    r = refl[tidx].copy(); fmu = fm[tidx]
    for k in range(L):
        bad = (fmu[k] & MB) != 0
        if bad.any(): r[k][:, bad] = np.nan
    inv = np.isnan(r[:, 0]).astype(np.float32)          # target keeps NATURAL mask
    arr = _pad(np.concatenate([r, inv[:, None]], axis=1).astype(np.float32), yt, xt)
    dts = ((times[t] - times[tidx]) / np.timedelta64(1, "D")).astype(np.float32)
    return _fm_fwd(trunk, arr, dts, yt, xt)


def gen_masked(trunk, refl, fm, times, t, yt, xt, T):   # offset-+1, 100%-masked slot, DT_FWD=32 (== build_caches)
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
    return _fm_fwd(trunk, arr, dts, yt, xt)


def build_input(refl, fm, times, t, yt, xt, T=None, fmtrunk=None):
    """Replicates HLSClassifierDataset.__getitem__: same context/inv/dts (uses times[t]);
    only the target slot base[L-1] differs per variant. raw/all keep the real obs."""
    tidx = np.concatenate([np.arange(t - (L - 1), t), [t]]).astype(np.int64)
    base = refl[tidx].copy(); fmu = fm[tidx]
    clear = ((fmu & MB) == 0) & np.isfinite(base[:, 0])      # BEFORE masking (matches dataset)
    for k in range(L):
        bad = (fmu[k] & MB) != 0
        if bad.any():
            base[k][:, bad] = np.nan
    if VARIANT == "composite":
        C = base.shape[1]; comp = np.full((C, yt, xt), np.nan, np.float32); fill = np.zeros((yt, xt), bool)
        for k in range(L - 1, -1, -1):
            tk = clear[k] & ~fill
            if tk.any():
                comp[:, tk] = base[k][:, tk]; fill |= tk
        base[L - 1] = comp
    elif VARIANT == "recon":
        base[L - 1] = gen_recon(fmtrunk, refl, fm, times, t, yt, xt)
    elif VARIANT == "masked":
        base[L - 1] = gen_masked(fmtrunk, refl, fm, times, t, yt, xt, T)
    # raw / all: target slot stays the real (naturally-masked) obs at t
    inv = np.isnan(base[:, 0]).astype(np.float32)
    arr = np.concatenate([base, inv[:, None]], axis=1).astype(np.float32)
    dts = ((times[t] - times[tidx]) / np.timedelta64(1, "D")).astype(np.float32)
    if yt < SIDE or xt < SIDE:
        arr = np.pad(arr, ((0, 0), (0, 0), (0, SIDE - yt), (0, SIDE - xt)), constant_values=0.0)
    return arr, dts


ps, amb, pool, refs = load_annotations(ANNOTATIONS)
vds = {a: date.fromisoformat(s) for a, s in json.loads(VDJSON.read_text(encoding="utf-8")).items()}
ROIS = sorted(a for a in ps if a in vds)
FMTRUNK = load_fm_trunk() if NEEDS_FM else None   # frozen FM, loaded once (recon/masked only)

out = {}; t0 = time.time(); miss = []
for roi in ROIS:
    p = RUNS / f"{PREFIX}-{VARIANT}-loo{roi}" / "checkpoints" / "best.pt"
    try:
        ck = torch.load(p, map_location=DEV, weights_only=False)
    except Exception as e:
        miss.append(roi); print(f"  SKIP {roi}: no ckpt ({p})", flush=True); continue
    trunk = TimeFirstAutoencoder(**ck["trunk_arch"]).to(DEV); trunk.load_state_dict(ck["trunk"]); trunk.eval()
    hh = int(ck["args"].get("head_hidden", 256))
    head = ClassificationHead(embed_dim=ck["embed_dim"], hidden=hh).to(DEV); head.load_state_dict(ck["head"]); head.eval()
    z = xr.open_zarr(ZARRS / f"{roi}.zarr", consolidated=False); T = int(z.sizes["time"])
    yt = min(SIDE, int(z.sizes["y"])); xt = min(SIDE, int(z.sizes["x"]))
    refl = z["reflectance"].isel(y=slice(0, yt), x=slice(0, xt)).values.astype(np.float32)
    fmv = z["fmask"].isel(y=slice(0, yt), x=slice(0, xt)).values
    fm = np.where(np.isfinite(fmv), fmv, 255).astype(np.uint8)
    times = z["time"].values; dates = np.array([_npdate(x) for x in times], object)
    ref = refs.get(roi); vdate = vds[roi]
    post = [t for t in range(L - 1, T) if ref is not None and dates[t] >= ref]
    es = dates[post[NEG_WAIT]] if len(post) > NEG_WAIT else (dates[post[0]] if post else date.min)
    win = [t for t in range(T) if t >= L - 1 and es <= dates[t] <= vdate]
    recs = []
    for i in range(0, len(win), BATCH):
        chunk = win[i:i + BATCH]
        arrs, dtss = zip(*[build_input(refl, fm, times, t, yt, xt, T, FMTRUNK) for t in chunk])
        X = torch.from_numpy(np.stack(arrs)).to(DEV); D = torch.from_numpy(np.stack(dtss)).to(DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = torch.sigmoid(head(trunk.forward_features(X, D).float())).float().cpu().numpy()
        recs.extend([dates[t].isoformat(), float(pp)] for t, pp in zip(chunk, P))
    out[roi] = recs
    del trunk, head; torch.cuda.empty_cache()
    print(f"{roi}: {len(win)} obs ({time.time()-t0:.0f}s)", flush=True)

OUT.write_text(json.dumps(out), encoding="utf-8")
print(f"wrote {OUT} ({len(out)} ROIs, {sum(len(v) for v in out.values())} obs); missing ckpt: {miss}")
