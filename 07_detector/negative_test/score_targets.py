# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Negative test, step 2: score each target with the 20 leave-one-MU-out models.

For every target of targets.csv (select_targets.py) and every fold checkpoint
(RUNS/regressor-raw-loo<MU>/checkpoints/best.pt, as unpacked from the record's
regressor-checkpoints), the downstream score is computed at every observation from the 79
before the target to the 2 after it -- the persistence filter is centred, so reading it at the
target needs the two observations that follow. The input is built exactly as
build_dense_pred_variant.py builds it for the reported (raw) variant. No target was a training
target of any fold, so every fold is out-of-sample for every target.

Writes yhat_targets.json: {"meta": {target: {aoi, target_idx, n_obs}}, "folds": [MU, ...],
"yhat": {target: {MU: [[date, score], ...]}}}, target = <aoi_id>_<YYYYMMDD>_<k>.

Usage: uv run python 07_detector/negative_test/score_targets.py    (CUDA GPU; about an hour on one H100)
Env: DATA_DIR (default ./data), ZARRS, RUNS (default DATA_DIR/runs); OUT_DIR (default
     DATA_DIR/runs/negative_test, where select_targets.py writes targets.csv)."""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "04_pretraining"))
sys.path.insert(0, str(ROOT / "common"))
from classifier_head import ClassificationHead                 # noqa: E402
from model import TimeFirstAutoencoder                         # noqa: E402

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ZARRS = Path(os.environ["ZARRS"]) if os.environ.get("ZARRS") else DATA_DIR / "labelled-dataset"
RUNS = Path(os.environ["RUNS"]) if os.environ.get("RUNS") else DATA_DIR / "runs"
OUT_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else DATA_DIR / "runs" / "negative_test"
MB = 0b10001111; L = 80; SIDE = 64; DEV = "cuda"; BATCH = 4
BEFORE, AFTER = L - 1, 2                                         # observations scored around each target


def _npdate(t64):
    return t64.astype("datetime64[D]").item()                   # UTC calendar date of a datetime64


def build_input(refl, fm, times, t, yt, xt):
    """build_dense_pred_variant.build_input for the raw variant: the real (naturally masked)
    observation in the target slot, the L-1 previous observations as context."""
    tidx = np.concatenate([np.arange(t - (L - 1), t), [t]]).astype(np.int64)
    base = refl[tidx].copy(); fmu = fm[tidx]
    for k in range(L):
        bad = (fmu[k] & MB) != 0
        if bad.any():
            base[k][:, bad] = np.nan
    inv = np.isnan(base[:, 0]).astype(np.float32)
    arr = np.concatenate([base, inv[:, None]], axis=1).astype(np.float32)
    dts = ((times[t] - times[tidx]) / np.timedelta64(1, "D")).astype(np.float32)
    if yt < SIDE or xt < SIDE:
        arr = np.pad(arr, ((0, 0), (0, 0), (0, SIDE - yt), (0, SIDE - xt)), constant_values=0.0)
    return arr, dts


def load_site(aoi):
    z = xr.open_zarr(ZARRS / f"{aoi}.zarr", consolidated=False)
    yt = min(SIDE, int(z.sizes["y"])); xt = min(SIDE, int(z.sizes["x"]))
    refl = z["reflectance"].isel(y=slice(0, yt), x=slice(0, xt)).values.astype(np.float32)
    fmv = z["fmask"].isel(y=slice(0, yt), x=slice(0, xt)).values
    fm = np.where(np.isfinite(fmv), fmv, 255).astype(np.uint8)
    times = z["time"].values
    return dict(refl=refl, fm=fm, times=times, dates=np.array([_npdate(x) for x in times], object), yt=yt, xt=xt)


def resolve(site, d8, k):
    """Index of the k-th observation (1-based, in acquisition-time order) dated d8 -- as
    HLSClassifierDataset._resolve does."""
    same = sorted((t for t, d in enumerate(site["dates"]) if d.strftime("%Y%m%d") == d8), key=lambda t: site["times"][t])
    assert 1 <= k <= len(same), f"no observation {k} on {d8}"
    return same[k - 1]


if __name__ == "__main__":
    targets = list(csv.DictReader(open(OUT_DIR / "targets.csv", encoding="utf-8")))
    sites = {r["aoi_id"]: load_site(r["aoi_id"]) for r in targets}
    stretches = []
    for r in targets:
        c = sites[r["aoi_id"]]; j = resolve(c, r["date"], int(r["k"]))
        win = [t for t in range(j - BEFORE, j + AFTER + 1) if L - 1 <= t < len(c["dates"])]
        stretches.append(dict(id=f"{r['aoi_id']}_{r['date']}_{r['k']}", aoi=r["aoi_id"], target_idx=j, win=win))
    folds = sorted(p.name[len("regressor-raw-loo"):] for p in RUNS.glob("regressor-raw-loo*")
                   if (p / "checkpoints" / "best.pt").is_file())
    assert len(folds) == 20, f"expected the 20 fold checkpoints under {RUNS}, found {len(folds)}"
    print(f"targets: {len(stretches)} over {len(sites)} sites | folds: {len(folds)}", flush=True)
    out = {s["id"]: {} for s in stretches}; t0 = time.time()
    for fi, f in enumerate(folds, 1):
        ck = torch.load(RUNS / f"regressor-raw-loo{f}" / "checkpoints" / "best.pt", map_location=DEV, weights_only=False)
        trunk = TimeFirstAutoencoder(**ck["trunk_arch"]).to(DEV); trunk.load_state_dict(ck["trunk"]); trunk.eval()
        head = ClassificationHead(embed_dim=ck["embed_dim"], hidden=int(ck["args"].get("head_hidden", 256))).to(DEV)
        head.load_state_dict(ck["head"]); head.eval()
        for s in stretches:
            c = sites[s["aoi"]]; recs = []
            for i in range(0, len(s["win"]), BATCH):
                chunk = s["win"][i:i + BATCH]
                arrs, dtss = zip(*[build_input(c["refl"], c["fm"], c["times"], t, c["yt"], c["xt"]) for t in chunk])
                X = torch.from_numpy(np.stack(arrs)).to(DEV); D = torch.from_numpy(np.stack(dtss)).to(DEV)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    P = torch.sigmoid(head(trunk.forward_features(X, D).float())).float().cpu().numpy()
                recs.extend([c["dates"][t].isoformat(), float(p)] for t, p in zip(chunk, P))
            out[s["id"]][f] = recs
        del trunk, head; torch.cuda.empty_cache()
        print(f"fold {fi}/{len(folds)} {f} ({time.time() - t0:.0f} s)", flush=True)
    meta = {s["id"]: dict(aoi=s["aoi"], target_idx=s["target_idx"], n_obs=len(s["win"])) for s in stretches}
    (OUT_DIR / "yhat_targets.json").write_text(json.dumps({"meta": meta, "folds": folds, "yhat": out}), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'yhat_targets.json'}", flush=True)
