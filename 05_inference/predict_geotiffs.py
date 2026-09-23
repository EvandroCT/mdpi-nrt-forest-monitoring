# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""predict_geotiffs.py — run the masked-pixel autoencoder over per-cluster zarrs
and emit georeferenced GeoTIFFs of the reconstructed reflectance.

For every zarr under ``--input`` (or just one if ``--input`` points at a single
``.zarr``) and every target time in the requested date range that has at least
``L - 1`` past frames, the script:

  1. Tiles the ROI into ``crop_size`` windows via ``enumerate_crops``.
  2. Builds one model-input sample per crop using the actual target-time
     Fmask as the cloud mask (so the model fills cloudy pixels and the
     hard identity bypass passes clear pixels through unchanged).
  3. Runs the model on every crop, stitches the predictions back to
     full ROI extent (averaging overlapping pixels at the seams between
     adjacent crops, which can happen when ROI dims aren't multiples of
     ``crop_size``).
  4. Writes a multi-band ``float32`` GeoTIFF carrying the ROI's CRS and
     pixel-to-world transform — drop into QGIS, GDAL, etc. without
     reprojection.

Use the checkpoint's saved ``args`` to reconstruct the model architecture
verbatim, so this script works against any TimeFirstAutoencoder variant
(per-cluster baseline, ablations, future runs).

Example:

    python predict_geotiffs.py \
        --input ./data/earthaccess_clustered_zarrs \
        --checkpoint ./data/checkpoints/foundation/best.pt \
        --start-date 2025-09-01 --end-date 2025-10-31 \
        --output-dir ./data/predictions
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_pretraining"))
from model import TimeFirstAutoencoder    # noqa: E402
from make_indices_vrts import make_combined_vrt_for_tif  # noqa: E402

_DEFAULT_MASK_BITS = 0b00001111             # cirrus | cloud | adj_cloud | shadow
_HLSS30_CODE = 1                            # platform code used elsewhere
_NO_AERO = False                            # set by load_model() from the checkpoint's own args


def enumerate_prediction_tiles(
    H: int, W: int, crop_size: int,
) -> list[tuple[int, int, int, int]]:
    """Non-overlapping tiles covering (H, W) exactly once.

    Tile starts step by `crop_size` from the origin; the right/bottom tile
    in each axis may be smaller than `crop_size` if the dim isn't a multiple
    (those tiles get zero-padded to `crop_size` before the model call, then
    only their actual (y_take, x_take) region is written back).

    This is different from training-time `enumerate_crops`, which produces
    uniformly-spaced (and sometimes overlapping) tiles to extract more
    training samples per ROI. Prediction needs every pixel produced exactly
    once.

    Returns list of (top, left, y_take, x_take).
    """
    tops = list(range(0, max(H, 1), crop_size))
    lefts = list(range(0, max(W, 1), crop_size))
    return [
        (t, l, min(crop_size, H - t), min(crop_size, W - l))
        for t in tops for l in lefts
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", type=Path, required=True,
                   help="A directory of *.zarr stores OR a single *.zarr path.")
    p.add_argument("--reliable-list", type=Path, default=None,
                   help="Optional text file with one FID per line. When set and --input is a "
                        "directory, only the FIDs listed are processed (each FID's zarr is "
                        "looked up as <input>/<FID>.zarr).")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to best.pt / last.pt / ep_NNN.pt produced by train.py.")
    p.add_argument("--start-date", type=str, required=True,
                   help="ISO date (inclusive). Only targets on or after this date are predicted.")
    p.add_argument("--end-date", type=str, required=True,
                   help="ISO date (inclusive).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output root; produces <output-dir>/<zarr_stem>/<date>.tif.")
    p.add_argument("--mask-bits", type=int, default=_DEFAULT_MASK_BITS,
                   help="Fmask bits that mark 'to be filled' pixels (default cirrus|cloud|adj_cloud|shadow).")
    p.add_argument("--platforms", choices=["all", "HLSS30"], default="all",
                   help="Filter by platform. 'all' predicts every date in range; 'HLSS30' "
                        "matches training-time eligibility (platform==1).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-crops", type=int, default=8,
                   help="Batch size when running crops through the model.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-emit GeoTIFFs even when the output file already exists.")
    p.add_argument("--no-vrt", action="store_true",
                   help="Skip writing the sidecar RGB VRT (true-colour, fixed 0..0.3 stretch).")
    p.add_argument("--no-vari", action="store_true",
                   help="Skip writing the sidecar VARI visualization TIF (RdYlGn colormap).")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, stop after this many (zarr, date) pairs — useful for smoke tests.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------


def load_model(ckpt_path: Path, device: str) -> tuple[TimeFirstAutoencoder, dict]:
    """Reconstruct the model from the saved args, load state dict, return both."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = dict(ck["args"])
    model = TimeFirstAutoencoder(
        in_channels   = int(args.get("in_channels",   8)),
        out_channels  = int(args.get("out_channels",  6)),
        embed_dim     = int(args.get("embed_dim",    96)),
        num_heads     = int(args.get("num_heads",     3)),
        temporal_depth= int(args.get("temporal_depth", 3)),
        spatial_depth = int(args.get("spatial_depth",  2)),
        seq_len       = int(args.get("lookback", args.get("seq_len", 70))),
        crop_size     = int(args.get("crop_size", 64)),
        window        = int(args.get("window", 8)),
        mlp_ratio     = float(args.get("mlp_ratio", 4.0)),
        dropout       = float(args.get("dropout", 0.0)),
        identity_bypass = args.get("identity_bypass", "hard"),
    )
    # state_dict was saved unwrapped (see train.py's save block).
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    global _NO_AERO
    _NO_AERO = bool(args.get("no_aero_channel", False)) or int(args.get("in_channels", 8)) == 7
    print(f"[setup] aerosol channel = {'excluded' if _NO_AERO else 'included'} "
          f"(checkpoint in_channels={int(args.get('in_channels', 8))})", flush=True)
    return model, args


# ---------------------------------------------------------------------------
# Per-crop sample assembly (no synthetic mask — uses target's real Fmask)
# ---------------------------------------------------------------------------


def build_inference_sample(
    ds: xr.Dataset,
    top: int, left: int, target_t: int,
    L: int, crop_size: int, mask_bits: int,
    scale_factor: float, fill_value,
    y_take: int, x_take: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct (inputs, dates) for one (tile, target time).

    Matches HLSGeoZarrDataset.__getitem__ exactly EXCEPT (a) the synthetic
    mask is replaced by the target-frame's own Fmask — so cloudy pixels at
    the target time get NaN'd in the input (model fills them) and clear
    pixels are passed through unchanged via the hard identity bypass; and
    (b) `y_take`/`x_take` are caller-supplied (the prediction tiler decides
    the tile size; this function just reads that rectangle).
    """
    s = crop_size
    y_slice = slice(top, top + y_take)
    x_slice = slice(left, left + x_take)

    start = target_t - L + 1
    input_slice = slice(start, target_t + 1)

    refl = ds["reflectance"]
    fmask = ds["fmask"]
    in_da = refl.isel(time=input_slice, y=y_slice, x=x_slice)
    fm_da = fmask.isel(time=input_slice, y=y_slice, x=x_slice)

    # Materialise reflectance with scale_factor and fill→NaN.
    raw = in_da.values
    in_refl = raw.astype(np.float32)
    if fill_value is not None:
        in_refl[raw == fill_value] = np.nan
    in_refl *= float(scale_factor)

    fm_float = fm_da.values
    fm_uint = np.where(np.isfinite(fm_float), fm_float, 255).astype(np.uint8)
    aero = ((fm_uint >> 6) & 0b11).astype(np.float32) / 3.0

    # Past slots: NaN out wherever mask_bits triggers (cirrus/cloud/adj_cloud/shadow).
    for k in range(L - 1):
        bad = (fm_uint[k] & mask_bits) != 0
        if bad.any():
            in_refl[k][:, bad] = np.nan

    # Target slot: use the target's OWN Fmask to drive the model's bypass.
    target_fm = fm_uint[L - 1]
    M = (target_fm & mask_bits) != 0
    if M.any():
        in_refl[L - 1][:, M] = np.nan

    # Invalid-mask channel (computed BEFORE NaN→0; stays finite).
    is_invalid = np.isnan(in_refl[:, 0]).astype(np.float32)

    # Checkpoints trained with --no-aero-channel (in_channels=7: 6 reflectance +
    # invalid-mask), the published one included, take no aerosol plane -- mirrors
    # HLSGeoZarrDataset.__getitem__'s include_aero_channel toggle (dataset.py).
    parts = [in_refl]
    if not _NO_AERO:
        parts.append(aero[:, None, :, :])
    parts.append(is_invalid[:, None, :, :])
    in_arr = np.concatenate(parts, axis=1).astype(np.float32)

    # Pad to crop_size if ROI is smaller than the crop.
    if y_take < s or x_take < s:
        pad_y = s - y_take
        pad_x = s - x_take
        in_arr = np.pad(
            in_arr,
            ((0, 0), (0, 0), (0, pad_y), (0, pad_x)),
            mode="constant", constant_values=0.0,
        )

    # Date encoding: delta-days from target, dates[-1] = 0.
    lb_times = ds["time"].isel(time=input_slice).values
    target_time = ds["time"].isel(time=target_t).values
    dates = ((target_time - lb_times) / np.timedelta64(1, "D")).astype(np.float32)

    return in_arr, dates


# ---------------------------------------------------------------------------
# Per-zarr, per-date prediction
# ---------------------------------------------------------------------------


_BAND_META: list[dict[str, str]] = [
    {"name": "Blue",   "common_name": "blue",   "center_wavelength_nm": "490",
     "description": "HLS harmonized Blue (L30 B02 / S30 B02)"},
    {"name": "Green",  "common_name": "green",  "center_wavelength_nm": "560",
     "description": "HLS harmonized Green (L30 B03 / S30 B03)"},
    {"name": "Red",    "common_name": "red",    "center_wavelength_nm": "665",
     "description": "HLS harmonized Red (L30 B04 / S30 B04)"},
    {"name": "NIR",    "common_name": "nir",    "center_wavelength_nm": "865",
     "description": "HLS harmonized NIR (L30 B05 / S30 B8A)"},
    {"name": "SWIR1",  "common_name": "swir16", "center_wavelength_nm": "1610",
     "description": "HLS harmonized SWIR1 (L30 B06 / S30 B11)"},
    {"name": "SWIR2",  "common_name": "swir22", "center_wavelength_nm": "2200",
     "description": "HLS harmonized SWIR2 (L30 B07 / S30 B12)"},
    {"name": "cloud_mask", "common_name": "alpha", "center_wavelength_nm": "",
     "description": "Cloud mask from target-time Fmask (1=clear/observed, 0=cloudy/inpainted)"},
]


def write_geotiff_atomic(
    path: Path, data: np.ndarray, transform: Affine, crs: rasterio.crs.CRS,
    alpha_band_index: int | None = None,
) -> None:
    """Write a multi-band float32 GeoTIFF atomically with rich per-band metadata.

    The first six bands carry HLS-aligned spectral metadata (band name,
    common name, center wavelength). The alpha band (when present) is
    described as the cloud mask and tagged with `colorinterp=Alpha`, so
    QGIS and any GDAL-aware viewer render it as transparency.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bands, H, W = data.shape
    profile = dict(
        driver="GTiff", height=H, width=W, count=bands, dtype="float32",
        compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256,
        transform=transform, crs=crs,
    )
    with rasterio.open(tmp, "w", **profile) as dst:
        # Set colorinterp BEFORE writing data so GDAL persists the EXTRASAMPLES
        # tag in the TIFF header (setting after write is silently dropped on
        # multi-band float GTiffs in rasterio).
        if alpha_band_index is not None:
            cis = [rasterio.enums.ColorInterp.gray] + [rasterio.enums.ColorInterp.undefined] * (bands - 1)
            cis[alpha_band_index - 1] = rasterio.enums.ColorInterp.alpha
            dst.colorinterp = tuple(cis)
        dst.write(data.astype(np.float32))
        for i in range(bands):
            if i >= len(_BAND_META):
                continue
            meta = _BAND_META[i]
            dst.set_band_description(i + 1, meta["name"])
            # Per-band tags surface in QGIS's Properties → Metadata pane and in
            # gdalinfo's "Metadata" section per band.
            dst.update_tags(
                i + 1,
                NAME=meta["name"],
                COMMON_NAME=meta["common_name"],
                CENTER_WAVELENGTH_NM=meta["center_wavelength_nm"],
                DESCRIPTION=meta["description"],
            )
    os.replace(tmp, path)


def predict_zarr_date(
    ds: xr.Dataset, model: torch.nn.Module, args: dict,
    target_t: int, device: str, mask_bits: int, batch_crops: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on every non-overlapping tile and stitch back.

    Returns:
        recon : (C_out, H, W) reconstructed reflectance
        alpha : (H, W) float32 — 1.0 where the pixel was observed (clear),
                0.0 where the model had to inpaint (cloudy per `mask_bits`).
                Identical to the inverse of the cloud-mask used to drive
                the hard identity bypass during inference.
    """
    H = int(ds["reflectance"].sizes["y"])
    W = int(ds["reflectance"].sizes["x"])
    L = int(args.get("lookback", args.get("seq_len", 70)))
    s = int(args.get("crop_size", 64))
    C_out = int(args.get("out_channels", 6))

    refl_attrs = ds["reflectance"].attrs
    scale_factor = float(refl_attrs.get("scale_factor", 1.0))
    fill_value = refl_attrs.get("_FillValue", None)

    # Alpha band = 1 - cloud_mask at target slot. Derived once from full Fmask.
    target_fm_full = ds["fmask"].isel(time=target_t).values
    target_fm_uint = np.where(np.isfinite(target_fm_full), target_fm_full, 255).astype(np.uint8)
    cloud_mask = ((target_fm_uint & mask_bits) != 0)
    alpha = (~cloud_mask).astype(np.float32)                     # (H, W)

    tiles = enumerate_prediction_tiles(H, W, s)
    samples = []
    for top, left, y_take, x_take in tiles:
        in_arr, dates = build_inference_sample(
            ds, top, left, target_t, L, s, mask_bits, scale_factor, fill_value,
            y_take, x_take,
        )
        samples.append((top, left, y_take, x_take, in_arr, dates))

    recon = np.zeros((C_out, H, W), dtype=np.float32)
    for i0 in range(0, len(samples), batch_crops):
        batch = samples[i0:i0 + batch_crops]
        in_batch = torch.from_numpy(np.stack([b[4] for b in batch])).to(device)
        d_batch = torch.from_numpy(np.stack([b[5] for b in batch])).to(device)
        with torch.no_grad():
            pred = model(in_batch, d_batch).cpu().numpy()        # (B, C, s, s)
        for j, (top, left, y_take, x_take, _, _) in enumerate(batch):
            recon[:, top:top + y_take, left:left + x_take] = pred[j, :, :y_take, :x_take]
    return recon, alpha


def list_target_times_in_range(
    ds: xr.Dataset, start_date: str, end_date: str, lookback: int,
    platforms_filter: str,
) -> list[int]:
    """Indices into ds['time'] for dates in [start, end] with t >= lookback - 1.

    `platforms_filter == "HLSS30"` additionally restricts to platform == 1
    (matching training eligibility); "all" includes HLSL30 too.
    """
    times = pd.to_datetime(ds["time"].values)
    lo = pd.Timestamp(start_date)
    hi = pd.Timestamp(end_date) + pd.Timedelta(days=1)        # inclusive end
    in_range = (times >= lo) & (times < hi)
    valid_lookback = np.arange(len(times)) >= (lookback - 1)
    mask = in_range & valid_lookback
    if platforms_filter == "HLSS30":
        mask &= ds["platform"].values == _HLSS30_CODE
    return np.flatnonzero(mask).tolist()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def discover_zarrs(input_path: Path, reliable_list: Path | None = None) -> list[Path]:
    if input_path.suffix == ".zarr":
        return [input_path]
    if reliable_list is not None and reliable_list.exists():
        fids = [ln.strip() for ln in reliable_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [input_path / f"{fid}.zarr" for fid in fids
                if (input_path / f"{fid}.zarr").exists()]
    return sorted(p for p in input_path.iterdir() if p.suffix == ".zarr")


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"input not found: {args.input}", flush=True)
        return 1

    model, ck_args = load_model(args.checkpoint, args.device)
    L = int(ck_args.get("lookback", ck_args.get("seq_len", 70)))
    print(f"[setup] checkpoint = {args.checkpoint}", flush=True)
    print(f"[setup] arch       embed={ck_args.get('embed_dim')} heads={ck_args.get('num_heads')} "
          f"depth={ck_args.get('temporal_depth')}/{ck_args.get('spatial_depth')} "
          f"in_ch={ck_args.get('in_channels')} out_ch={ck_args.get('out_channels')} "
          f"L={L} crop={ck_args.get('crop_size')} bypass={ck_args.get('identity_bypass', 'hard')}",
          flush=True)
    print(f"[setup] input      = {args.input}", flush=True)
    print(f"[setup] output     = {args.output_dir}", flush=True)
    print(f"[setup] date range = {args.start_date} .. {args.end_date}  (platforms={args.platforms})",
          flush=True)

    zarrs = discover_zarrs(args.input, args.reliable_list)
    if not zarrs:
        print(f"[done] no zarrs found in {args.input}", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    total_t0 = time.time()
    n_processed = 0
    n_skipped = 0
    n_failed = 0

    for z_idx, zarr_path in enumerate(zarrs):
        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except Exception as e:
            print(f"[skip] {zarr_path.name}: open failed ({type(e).__name__}: {e})", flush=True)
            n_failed += 1
            continue

        target_ts = list_target_times_in_range(
            ds, args.start_date, args.end_date, L, args.platforms,
        )
        if not target_ts:
            print(f"[skip] {zarr_path.name}: 0 target dates in range", flush=True)
            continue

        # Pull georef ONCE per zarr (it's the same for every date).
        sr_attrs = ds["spatial_ref"].attrs if "spatial_ref" in ds else {}
        crs_wkt = sr_attrs.get("crs_wkt", None)
        gt_str = sr_attrs.get("GeoTransform", None)
        if not crs_wkt or not gt_str:
            print(f"[skip] {zarr_path.name}: no GeoZarr CRS/transform attrs", flush=True)
            n_failed += 1
            continue
        c, a, b, f, d, e = (float(v) for v in gt_str.split())
        transform = Affine(a, b, c, d, e, f)
        crs = rasterio.crs.CRS.from_wkt(crs_wkt)

        # Per-target-slot platform code: 0 = HLSL30, 1 = HLSS30.
        platforms = ds["platform"].values

        zarr_out_dir = args.output_dir / zarr_path.stem
        for t in target_ts:
            target_time = pd.Timestamp(ds["time"].values[t])
            stem = target_time.strftime("%Y%m%dT%H%M%S")
            platform = "HLSS30" if int(platforms[t]) == _HLSS30_CODE else "HLSL30"
            out_path = zarr_out_dir / f"{stem}_{platform}.tif"
            if out_path.exists() and not args.overwrite:
                n_skipped += 1
                continue
            t0 = time.time()
            try:
                recon, alpha = predict_zarr_date(
                    ds, model, ck_args, t, args.device, args.mask_bits, args.batch_crops,
                )
            except Exception as exc:                                  # noqa: BLE001
                print(f"[fail] {zarr_path.name} @ {stem}: {type(exc).__name__}: {exc}",
                      flush=True)
                n_failed += 1
                continue
            # Stack reflectance + alpha into (C+1, H, W) and write as one GeoTIFF.
            stacked = np.concatenate([recon, alpha[None, :, :]], axis=0)
            write_geotiff_atomic(
                out_path, stacked, transform, crs,
                alpha_band_index=stacked.shape[0],   # last band = alpha
            )
            if not args.no_vrt:
                # Combined sidecar: RGB true-colour + VARI/NDVI/NBR/NDMI + per-index QMLs.
                make_combined_vrt_for_tif(out_path, overwrite=True)
            dt = time.time() - t0
            n_processed += 1
            total_pairs += 1
            if (n_processed % 10 == 0) or (n_processed <= 3):
                print(f"[{z_idx + 1}/{len(zarrs)}] {zarr_path.stem}/{stem}_{platform}.tif "
                      f"({stacked.shape[0]}b, {stacked.shape[2]}x{stacked.shape[1]}, "
                      f"clear_frac={float(alpha.mean()):.3f}) "
                      f"in {dt:.1f}s  [total={n_processed}]", flush=True)
            if args.limit and total_pairs >= args.limit:
                print(f"[done] hit --limit {args.limit}; stopping early.", flush=True)
                total_dt = time.time() - total_t0
                print(f"[done] {n_processed} TIFs written, {n_skipped} skipped, {n_failed} failed "
                      f"in {total_dt:.1f}s.", flush=True)
                return 0
        try:
            ds.close()
        except Exception:                                              # noqa: BLE001
            pass

    total_dt = time.time() - total_t0
    print(f"\n[done] {n_processed} TIFs written, {n_skipped} skipped, {n_failed} failed "
          f"in {total_dt:.1f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
