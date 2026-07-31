# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""HLSGeoZarrDataset — masked-pixel reconstruction sample factory over an HLS GeoZarr.

What this is (tutorial intro)
-----------------------------
This dataset feeds an MAE-style temporal autoencoder. Given a sequence of L
recent frames whose **last slot is the target with a synthetic cloud mask
applied**, the model must reconstruct the masked pixels of that target slot
from temporal context. The synthetic mask is a real Fmask shape borrowed
from a past slot of the same crop's input window — free data augmentation
whose fraction can be steered across epochs by a per-epoch band schedule
(see `band_schedule`).

A *sample* is a dict with four entries:
    - `inputs` : (L, C+2, H, W) float32 where C is the number of reflectance
                 bands stored in the zarr (4 for the legacy zarrs, 6 for the
                 6-band clustered zarrs). Channels 0..C-1 are reflectance;
                 channel C is the per-frame Fmask aerosol level (bits 6-7,
                 normalised /3.0); channel C+1 is an **invalid-mask
                 indicator** (1.0 where reflectance was hidden / nodata at
                 that slot, 0.0 elsewhere) — gives the model an explicit
                 per-slot per-pixel valid/invalid signal so it can route
                 around NaN→0 substitutions instead of having to infer them
                 from zero reflectance values. The **last slot** is the
                 target frame with synthetic clouds applied:
                     * past slots (k < L-1): reflectance set to NaN where
                       `frame_fmask & mask_bits != 0`. Aerosol from that frame.
                     * target slot (k = L-1): reflectance set to NaN where
                       the synthetic mask `M` is True. Aerosol on `M=True`
                       pixels is replaced with the source frame's aerosol
                       so the model sees realistic "this pixel is hazy"
                       metadata for the masked region.
                 NaN→0 substitution on the reflectance channels happens
                 **inside the model**, not here. The invalid-mask channel
                 is computed from the NaN footprint before substitution and
                 stays finite throughout.
    - `target` : (C, H, W) float32. The original full-clean reflectance of
                 the target frame. Loss is computed against this.
    - `mask`   : (H, W) bool. The synthetic mask `M` — True where the target
                 reflectance was hidden. Loss is taken over these pixels only.
    - `dates`  : (L,) float32. Delta-days from the target frame, with
                 `dates[L-1] = 0`. Past slots are positive (older = larger).

Eligibility is now **per crop**, not per ROI. At construction time the ROI
is tiled into `crop_size × crop_size` windows with uniform spacing (small
overlaps when dims aren't multiples of `crop_size`); for each (crop, time)
pair we compute QC scalars directly from the Fmask pixels inside that crop
and accept the pair iff every threshold passes. `__len__` is the total
number of accepted (crop, time) pairs — typically several times more than
the legacy per-ROI count because crops "rescue" each other.

The synthetic-mask source is picked by a **band-gated walk** through the
crop's input window. At every epoch `e`, a target band `(lo, hi)` is read
from `band_schedule[e]`; only past slots whose stored mask fraction (per
`mask_bits`) falls in `[lo, hi]` are eligible. Visit `e mod n_eligible`
picks the n-th in-band slot (forward in time, so oldest first → minimal
positional shortcut from slot L-2). If the band is empty for a crop, the
slot whose fraction is closest to the band centre is used as fallback —
the model never trains on an empty mask.

How to use it
-------------
    >>> ds = HLSGeoZarrDataset("path/to/<fid>.zarr", lookback_window=70,
    ...                         band_schedule=[(0.05, 0.30)] * 33
    ...                                       + [(0.30, 0.60)] * 33
    ...                                       + [(0.60, 0.90)] * 34)
    >>> ds.set_epoch(7)                 # called from train loop each epoch
    >>> sample = ds[0]
    >>> sample["inputs"].shape          # (70, 8, 64, 64)
    >>> sample["target"].shape          # (6, 64, 64)
    >>> sample["mask"].shape            # (64, 64)
    >>> sample["dates"].shape           # (70,)

Eligibility rule (per crop)
---------------------------
A (crop, time) pair qualifies iff ALL of:
    - `platform[t] == HLSS30`
    - within the crop's pixels: pct_cloud[t] <= cloud_max,
      pct_shadow[t] <= shadow_max, the four aerosol-level fractions
      respect their respective max, and pct_valid[t] >= valid_min
    - `t >= lookback_window - 1`     (must have L-1 frames *before* t)

The target IS the last slot of `inputs`, so we need L-1 past frames + the
target itself = L frames total.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import ConcatDataset, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from eligibility import filter_crop_targets  # noqa: E402

# Default mask_bits: cirrus | cloud | adj_cloud | shadow.
# Fmask bit layout (HLS v2.0): bit 0 cirrus, bit 1 cloud, bit 2 adj_cloud,
# bit 3 shadow, bit 4 snow_ice, bit 5 water, bits 6-7 aerosol level (0..3).
_DEFAULT_MASK_BITS = 0b00001111
# Open band used when no schedule is supplied — disables the curriculum.
_DEFAULT_BAND_SCHEDULE: tuple[tuple[float, float], ...] = ((0.0, 1.0),)


class HLSGeoZarrDataset(Dataset):
    """One GeoZarr store → masked-pixel reconstruction samples.

    Parameters
    ----------
    zarr_path
        Filesystem path to a `.zarr` directory written by the cubes step
        (03_cubes).
    lookback_window
        Total sequence length L (target slot included). The model sees L-1
        past frames plus the masked target.
    crop_size
        Side of the spatial window (in pixels). Same window across all L slots
        and the target. ROIs smaller than this are top-left padded with zeros.
    mask_bits
        Fmask bitmask used for two purposes:
          (a) pixels in past slots where `frame_fmask & mask_bits != 0` get
              NaN reflectance on input;
          (b) the synthetic target mask `M` is built from a past slot's
              Fmask using the same `mask_bits`; the per-crop `frac_cache`
              used by the band gate also counts pixels triggered by
              `mask_bits`.
        Default `0b00001111` (cirrus | cloud | adj_cloud | shadow).
    cloud_max, shadow_max, cirrus_max, adj_cloud_max, water_max,
    aero_*_max, valid_min
        Per-crop quality thresholds applied to (crop, time) target candidates.
    band_schedule
        Sequence of `(lo, hi)` cloud-fraction bands, one per epoch. At epoch
        `e` only past slots whose fraction falls in `band_schedule[e]` are
        eligible as synthetic-mask sources. If `e` exceeds the schedule
        length, the last entry is reused (curriculum clamp). Default is a
        single open band `(0.0, 1.0)` — no curriculum effect, equivalent to
        "pick any past slot".
    """

    def __init__(
        self,
        zarr_path: str | Path,
        lookback_window: int = 70,
        crop_size: int = 64,
        mask_bits: int = _DEFAULT_MASK_BITS,
        cloud_max: float = 0.0,
        shadow_max: float = 0.0,
        cirrus_max: float = 5.0,
        adj_cloud_max: float = 0.0,
        water_max: float = 10.0,
        aero_climatology_max: float = 0.0,
        aero_low_max: float = 100.0,
        aero_medium_max: float = 30.0,
        aero_high_max: float = 0.0,
        valid_min: float = 90.0,
        band_schedule: Sequence[tuple[float, float]] | None = None,
        include_aero_channel: bool = True,
        input_mask_bits: int | None = None,
        qa_channels: bool = False,
    ) -> None:
        self.zarr_path = Path(zarr_path)
        self.lookback_window = int(lookback_window)
        self.crop_size = int(crop_size)
        self.mask_bits = int(mask_bits)
        # Per-slot input filter (NaN-out past reflectance). Defaults to the
        # same mask_bits used for the synthetic target mask M; set to 0 to
        # disable the per-slot filter (model sees raw reflectance, only ROI
        # nodata stays NaN). Decouples "what counts as masked in input" from
        # "what counts as the target reconstruction task".
        self.input_mask_bits = (
            int(input_mask_bits) if input_mask_bits is not None else self.mask_bits
        )
        self.include_aero_channel = bool(include_aero_channel)
        # Append 8 QA channels per slot (6 multi-hot Fmask bits + 2 binary
        # aero medium/high) between reflectance and invalid_mask.
        self.qa_channels = bool(qa_channels)
        self._epoch: int = 0

        # Lazy open: metadata + QA scalars only.
        self.ds = xr.open_zarr(self.zarr_path, consolidated=True)

        refl_attrs = self.ds["reflectance"].attrs
        self.scale_factor = float(refl_attrs.get("scale_factor", 1.0))
        self.fill_value = refl_attrs.get("_FillValue", None)

        # Per-crop eligibility + per-crop slot-mask-fraction cache.
        self._target_crops, self._frac_cache = filter_crop_targets(
            self.ds,
            crop_size=self.crop_size,
            lookback_window=self.lookback_window,
            mask_bits=self.mask_bits,
            cloud_max=cloud_max,
            shadow_max=shadow_max,
            cirrus_max=cirrus_max,
            adj_cloud_max=adj_cloud_max,
            water_max=water_max,
            aero_climatology_max=aero_climatology_max,
            aero_low_max=aero_low_max,
            aero_medium_max=aero_medium_max,
            aero_high_max=aero_high_max,
            valid_min=valid_min,
        )

        self._n_time = int(self.ds.sizes["time"])

        # Per-epoch band schedule for the synthetic-mask walk.
        bs = list(band_schedule) if band_schedule is not None else list(_DEFAULT_BAND_SCHEDULE)
        if not bs:
            raise ValueError("band_schedule must be non-empty")
        self._band_schedule: list[tuple[float, float]] = [
            (float(lo), float(hi)) for (lo, hi) in bs
        ]

        # Geo metadata for georeferenced TIFF dumps. GeoZarr convention: the
        # `spatial_ref` coord carries `crs_wkt` and a GDAL-style `GeoTransform`
        # string of six floats: "c a b f d e". Affine convention: x = a*col +
        # b*row + c; y = d*col + e*row + f. For per-sample crops we shift the
        # origin (c, f) by the (left, top) pixel offset.
        sr = self.ds["spatial_ref"] if "spatial_ref" in self.ds else None
        self.crs_wkt: str = "" if sr is None else str(sr.attrs.get("crs_wkt", ""))
        gt_str = "" if sr is None else str(sr.attrs.get("GeoTransform", ""))
        if gt_str:
            c, a, b, f, d, e = (float(v) for v in gt_str.split())
            self._affine6 = np.array([a, b, c, d, e, f], dtype=np.float64)
        else:
            self._affine6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._target_crops)

    def set_epoch(self, epoch: int) -> None:
        """Set the current training epoch.

        Drives the per-epoch band gate on the synthetic-mask walk. Called
        from the training loop before every epoch via
        `set_epoch_recursive(concat_ds, epoch)`.
        """
        self._epoch = int(epoch)

    def _materialise(self, da: xr.DataArray) -> np.ndarray:
        """Materialise an (already-isel'd) reflectance DataArray as float32.

        Pulls bytes from zarr, casts Int16 → float32, replaces `_FillValue`
        with NaN, and applies `scale_factor` so finite pixels are physical
        reflectance in [0, 1].
        """
        raw = da.values
        arr = raw.astype(np.float32)
        if self.fill_value is not None:
            arr[raw == self.fill_value] = np.nan
        arr *= self.scale_factor
        return arr

    @staticmethod
    def _aero_level(fmask_uint8: np.ndarray) -> np.ndarray:
        """Extract aerosol level (Fmask bits 6-7) as float32 in [0, 1]."""
        return ((fmask_uint8 >> 6) & 0b11).astype(np.float32) / 3.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one masked-pixel reconstruction sample.

        Keys: `inputs` (L, C+2, s, s), `target` (C, s, s), `mask` (s, s),
        `dates` (L,). C is the number of reflectance bands in the zarr.
        """
        top, left, target_t = self._target_crops[idx]
        top, left, target_t = int(top), int(left), int(target_t)
        L = self.lookback_window
        # The window covers [target_t - L + 1, target_t]. Eligibility ensures
        # target_t >= L - 1, so the start is non-negative.
        start = target_t - L + 1
        input_slice = slice(start, target_t + 1)

        refl = self.ds["reflectance"]
        fmask = self.ds["fmask"]

        # Spatial slices come straight from the (top, left) the tiler chose
        # for this sample. ROIs smaller than crop_size return (0, 0); the
        # crop is then less than s in that axis and gets bottom-right padded
        # at the end of this function.
        H = int(refl.sizes["y"])
        W = int(refl.sizes["x"])
        s = self.crop_size
        y_take = min(s, H - top)
        x_take = min(s, W - left)
        y_slice = slice(top, top + y_take)
        x_slice = slice(left, left + x_take)

        # Lazy slices.
        in_da = refl.isel(time=input_slice, y=y_slice, x=x_slice)
        fm_da = fmask.isel(time=input_slice, y=y_slice, x=x_slice)
        # Target reflectance comes from the same frame as inputs[L-1] but is
        # the *original* (un-masked) reflectance.
        tg_da = refl.isel(time=target_t, y=y_slice, x=x_slice)

        # Materialise (one disk read each).
        in_refl = self._materialise(in_da)            # (L, C, y_take, x_take), NaN at nodata
        tg_arr = self._materialise(tg_da)             # (C, y_take, x_take)
        fm_float = fm_da.values                       # (L, y_take, x_take), float (NaN nodata)

        # Some cubes-step zarrs have target frames where pct_valid=100 but the
        # reflectance array still contains NaN (reproject_match leaves
        # non-overlapping pixels as NaN; pct_valid is computed on the source
        # tile, not the destination). Detect those positions, replace target
        # NaN with 0 so it's safe to broadcast through the loss, and exclude
        # them from the synth mask we build below so loss never scores them.
        target_invalid = np.isnan(tg_arr).any(axis=0)            # (y_take, x_take)
        if target_invalid.any():
            tg_arr = np.nan_to_num(tg_arr, nan=0.0).astype(np.float32)

        # Cast Fmask: NaN → 255 (treated as "all bits set"; mask_bits will trip).
        fm_uint = np.where(np.isfinite(fm_float), fm_float, 255).astype(np.uint8)
        aero = self._aero_level(fm_uint)              # (L, y_take, x_take), [0, 1]

        # Past-slot reflectance NaN-out where input_mask_bits triggers.
        # input_mask_bits defaults to mask_bits but can be set to 0 to disable
        # the per-slot filter entirely (model sees raw reflectance even at
        # cloudy positions; ROI nodata is still NaN via the in_refl init).
        # Target slot handled below via the synthetic mask. Loop over slots
        # so the (C, y, x)←(y, x) broadcast under bool indexing is unambiguous.
        if self.input_mask_bits != 0:
            for k in range(L - 1):
                bad = (fm_uint[k] & self.input_mask_bits) != 0       # (y, x)
                if bad.any():
                    in_refl[k][:, bad] = np.nan

        # ---- Band-gated synthetic-mask source selection ---------------------
        # Eligible source slots are this crop's past-window slots whose stored
        # mask fraction (from filter_crop_targets) falls in this epoch's band.
        # Visit n = n-th eligible slot, forward in time, wrapping mod size.
        # Empty-band fallback: closest-to-band-centre slot from the past window.
        visit = self._epoch
        bs = self._band_schedule
        lo, hi = bs[min(visit, len(bs) - 1)]
        all_fracs = self._frac_cache[(top, left)]            # (T,) float32
        past_idx = np.arange(start, target_t)                # absolute time idx
        past_fracs = all_fracs[start:target_t]               # (L-1,)
        in_band = (past_fracs >= lo) & (past_fracs <= hi)
        eligible_idx = past_idx[in_band]
        if eligible_idx.size == 0:
            centre = 0.5 * (lo + hi)
            diffs = np.abs(past_fracs - centre)
            # np.nanargmin: skip past slots whose frac is NaN (fully-nodata).
            # If all past frames are NaN — a degenerate ROI with no observable
            # past — fall back to slot 0; the resulting mask will be empty.
            if np.all(np.isnan(diffs)):
                src_t = int(past_idx[0])
            else:
                src_t = int(past_idx[int(np.nanargmin(diffs))])
        else:
            n = eligible_idx.size
            src_idx_in_pool = visit % n
            src_t = int(eligible_idx[src_idx_in_pool])
            # "Try next slot": if the chosen frac is zero, walk forward in the
            # in-band pool until non-empty (or all-zero, in which case keep it).
            if float(all_fracs[src_t]) == 0.0:
                for i in range(1, n):
                    cand = int(eligible_idx[(src_idx_in_pool + i) % n])
                    if float(all_fracs[cand]) > 0.0:
                        src_t = cand
                        break

        src_fm_da = fmask.isel(time=src_t, y=y_slice, x=x_slice)
        src_fm_float = src_fm_da.values                      # (y_take, x_take)
        src_fm_uint = np.where(
            np.isfinite(src_fm_float), src_fm_float, 0
        ).astype(np.uint8)
        # NaN-fmask source pixels treated as M=False (no mask there).
        src_aero = self._aero_level(src_fm_uint)             # (y, x), [0, 1]
        M = (src_fm_uint & self.mask_bits) != 0              # (y, x) bool
        # Don't ask the model to reconstruct positions where the target itself
        # is invalid — they'd produce NaN gradients and poison training.
        M &= ~target_invalid

        # Apply M to the target slot of inputs.
        target_slot_refl = in_refl[L - 1]                    # (4, y, x)
        target_slot_aero = aero[L - 1].copy()                # (y, x)
        if M.any():
            target_slot_refl[:, M] = np.nan
            target_slot_aero[M] = src_aero[M]
        # Stitch the modified slot back into the (L, 5, ...) cube.
        # (in_refl[L-1] already updated in place above.)
        aero[L - 1] = target_slot_aero

        # Invalid-mask indicator per slot per pixel: 1.0 where any reflectance
        # band is NaN at this (slot, pixel) before the model's NaN→0 substitution.
        # Past slots: True under cloud/shadow/etc. (mask_bits) and at ROI nodata.
        # Target slot: True under the synthetic mask M (and at nodata).
        # Always finite — gives the model an explicit valid/invalid signal so it
        # doesn't have to disentangle "zero reflectance" from "hidden pixel".
        is_invalid = np.isnan(in_refl[:, 0]).astype(np.float32)        # (L, y, x)

        # Stack reflectance + (aerosol?) + (QA bits?) + invalid-mask along
        # channel axis. invalid-mask is always the last channel so the model's
        # hard gate can locate it as `x[:, L-1, in_channels - 1]` regardless
        # of which optional channels are present.
        cat = [in_refl]
        if self.include_aero_channel:
            cat.append(aero[:, None, :, :])
        if self.qa_channels:
            # 8 binary QA channels per slot:
            #   0: cirrus (Fmask bit 0)
            #   1: cloud (bit 1)
            #   2: adj_cloud (bit 2)
            #   3: shadow (bit 3)
            #   4: snow_ice (bit 4)
            #   5: water (bit 5)
            #   6: aero_medium (aero_level == 2)
            #   7: aero_high (aero_level == 3)
            # Multi-hot (Fmask bits can co-occur at cloud edges, ~1.4% of
            # valid pixels). At Fmask=255 (nodata), all QA channels are 0;
            # the invalid_mask channel covers nodata signalling.
            qa = np.zeros((L, 8, fm_uint.shape[1], fm_uint.shape[2]), dtype=np.float32)
            for bit in range(6):
                qa[:, bit] = ((fm_uint & np.uint8(1 << bit)) != 0).astype(np.float32)
            aero_level_local = (fm_uint >> 6) & 0b11
            qa[:, 6] = (aero_level_local == 2).astype(np.float32)
            qa[:, 7] = (aero_level_local == 3).astype(np.float32)
            # Zero-out QA at true-nodata pixels (Fmask=255 trips every bit).
            nodata_mask = fm_uint == 255
            qa = qa * (~nodata_mask[:, None, :, :]).astype(np.float32)
            cat.append(qa)
        cat.append(is_invalid[:, None, :, :])
        in_arr = np.concatenate(cat, axis=1).astype(np.float32)

        # ---- Date encoding (delta-days from target) -------------------------
        lb_times = self.ds["time"].isel(time=input_slice).values   # (L,) datetime64
        target_time = self.ds["time"].isel(time=target_t).values
        dates = ((target_time - lb_times) / np.timedelta64(1, "D")).astype(np.float32)
        # By construction dates[L-1] == 0.

        # ---- Pad to (s, s) if ROI smaller -----------------------------------
        if y_take < s or x_take < s:
            pad_y = s - y_take
            pad_x = s - x_take
            in_arr = np.pad(
                in_arr,
                ((0, 0), (0, 0), (0, pad_y), (0, pad_x)),
                mode="constant", constant_values=0.0,
            )
            tg_arr = np.pad(
                tg_arr,
                ((0, 0), (0, pad_y), (0, pad_x)),
                mode="constant", constant_values=0.0,
            )
            M = np.pad(
                M,
                ((0, pad_y), (0, pad_x)),
                mode="constant", constant_values=False,
            )

        # Crop-adjusted Affine for georeferenced TIFFs. Padding is
        # bottom-right only, so the origin (c, f) shifts only by the crop's
        # (left, top) — pad pixels carry valid (extrapolated) coords.
        a, b, c, d, e, f = self._affine6
        c_new = c + a * left + b * top
        f_new = f + d * left + e * top
        affine6 = np.array([a, b, c_new, d, e, f_new], dtype=np.float64)

        return {
            "inputs": torch.from_numpy(in_arr),                   # (L, C+2, s, s)
            "target": torch.from_numpy(tg_arr),                   # (C, s, s)
            "mask":   torch.from_numpy(M),                        # (s, s) bool
            "dates":  torch.from_numpy(dates),                    # (L,)
            "transform": torch.from_numpy(affine6),               # (6,) float64
            "crs_wkt":  self.crs_wkt,                             # str
        }


def set_epoch_recursive(ds: Dataset, epoch: int) -> int:
    """Propagate the current epoch through Concat/Subset wrappers to every
    underlying `HLSGeoZarrDataset` leaf.

    `torch.utils.data.ConcatDataset` and `Subset` don't forward method calls
    to their constituents, so the training loop calls this helper at the
    top of every epoch to keep the band-gated mask schedule in sync.

    Returns the number of leaf datasets whose `set_epoch` was called — useful
    for asserting the chain is what you think it is.
    """
    if isinstance(ds, ConcatDataset):
        return sum(set_epoch_recursive(sub, epoch) for sub in ds.datasets)
    if isinstance(ds, Subset):
        return set_epoch_recursive(ds.dataset, epoch)
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)
        return 1
    return 0


if __name__ == "__main__":
    zarr_path = Path(sys.argv[1] if len(sys.argv) > 1 else "./data/example.zarr")
    if not zarr_path.exists():
        sys.exit(f"Zarr store not found at {zarr_path}.")

    ds = HLSGeoZarrDataset(zarr_path, lookback_window=8, crop_size=64)
    print(f"Dataset loaded from: {zarr_path}")
    print(f"Store dims: time={ds.ds.sizes['time']}  y={ds.ds.sizes['y']}  x={ds.ds.sizes['x']}")
    print(f"Eligible (crop, time) samples: {len(ds)}")
    print(f"Crops tiled (unique top, left): {len(ds._frac_cache)}")
    if len(ds) == 0:
        sys.exit("No (crop, time) pair passed the quality filter.")

    sample = ds[0]
    x, y, M, d = sample["inputs"], sample["target"], sample["mask"], sample["dates"]
    n_bands = y.shape[0]
    print(f"inputs  : {tuple(x.shape)}  dtype={x.dtype}  (refl bands C={n_bands})")
    print(f"target  : {tuple(y.shape)}  dtype={y.dtype}")
    print(f"mask    : {tuple(M.shape)}  dtype={M.dtype}  frac={M.float().mean():.3f}")
    print(f"dates   : {tuple(d.shape)}  range=[{d.min():.0f}, {d.max():.0f}] days")

    a = x.numpy()
    print(
        f"input refl stats : nan_frac={np.isnan(a[:, :n_bands]).mean():.3f}  "
        f"min={np.nanmin(a[:, :n_bands]):.4f}  max={np.nanmax(a[:, :n_bands]):.4f}"
    )
    print(
        f"input aero stats : min={a[:, n_bands].min():.3f}  max={a[:, n_bands].max():.3f}  "
        f"mean={a[:, n_bands].mean():.3f}"
    )
    print(
        f"input invalid-mask stats : frac={a[:, n_bands + 1].mean():.3f}  "
        f"target-slot frac={a[-1, n_bands + 1].mean():.3f}"
    )
