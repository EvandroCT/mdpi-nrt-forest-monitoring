# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Eligibility logic shared between training (HLSGeoZarrDataset) and the
Phase B scanner (the download step's scan_eligibility.py).

The two callers must agree on the eligibility rule, so both import these
helpers instead of duplicating the boolean expression.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

HLSS30_CODE = 1


def enumerate_crops(H: int, W: int, crop_size: int) -> list[tuple[int, int]]:
    """Tile an HxW ROI into crop_size x crop_size windows aligned to a
    crop_size-stride grid (multiples of crop_size).

    Starts are multiples of `crop_size` so each crop reads exactly one chunk
    when the zarr chunks are crop_size-aligned. When `dim` is not a multiple
    of `crop_size`, the last crop's tail extends past `dim` and the dataset's
    existing padding fills the trailing pixels. Axes smaller than `crop_size`
    return [0] (and the dataset pads up to crop_size at `__getitem__` time).
    """
    def axis_starts(dim: int, s: int) -> list[int]:
        if dim <= s:
            return [0]
        n = math.ceil(dim / s)
        return [i * s for i in range(n)]  # multiples of s; last may extend past dim (padded)
    tops = axis_starts(H, crop_size)
    lefts = axis_starts(W, crop_size)
    return [(t, l) for t in tops for l in lefts]


def filter_targets(
    ds: xr.Dataset,
    lookback_window: int,
    cloud_max: float,
    shadow_max: float,
    aero_climatology_max: float,
    aero_low_max: float,
    aero_medium_max: float,
    aero_high_max: float,
    valid_min: float,
) -> np.ndarray:
    """Return zarr-time indices that qualify as targets.

    A timestep `t` qualifies iff platform==HLSS30, every QA percentage is on
    the safe side of its threshold, and `t >= lookback_window - 1` so there
    are at least L-1 frames before t (the target itself sits in the last slot
    of the input window).
    """
    platform = ds["platform"].values
    cloud = ds["pct_cloud"].values
    shadow = ds["pct_shadow"].values
    aero_climatology = ds["pct_aero_climatology"].values
    aero_low = ds["pct_aero_low"].values
    aero_medium = ds["pct_aero_medium"].values
    aero_high = ds["pct_aero_high"].values
    valid = ds["pct_valid"].values
    n = platform.shape[0]

    eligible = (
        (platform == HLSS30_CODE)
        & (np.nan_to_num(cloud, nan=np.inf) <= cloud_max)
        & (np.nan_to_num(shadow, nan=np.inf) <= shadow_max)
        & (np.nan_to_num(aero_climatology, nan=np.inf) <= aero_climatology_max)
        & (np.nan_to_num(aero_low, nan=np.inf) <= aero_low_max)
        & (np.nan_to_num(aero_medium, nan=np.inf) <= aero_medium_max)
        & (np.nan_to_num(aero_high, nan=np.inf) <= aero_high_max)
        & (np.nan_to_num(valid, nan=-np.inf) >= valid_min)
        & (np.arange(n) >= lookback_window - 1)
    )
    return np.flatnonzero(eligible)


def filter_crop_targets(
    ds: xr.Dataset,
    crop_size: int,
    lookback_window: int,
    mask_bits: int = 0b00001111,
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
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], np.ndarray]]:
    """Per-crop eligibility filter + per-crop slot mask-fraction cache.

    For every (top, left) tile produced by `enumerate_crops` and every time
    index `t`, computes per-crop QC scalars directly from the crop's Fmask
    pixels and applies the same eligibility rules as `filter_targets`.

    Returns
    -------
    target_crops : list of (top, left, t)
        Crops at times that pass every threshold AND have
        `t >= lookback_window - 1` (so the input window fits).
    frac_cache : dict[(top, left)] → np.ndarray of shape (T,) float32
        Per-crop mask fraction at every time index — the fraction of pixels
        whose Fmask triggers `mask_bits`. Used by the dataset's band-gated
        mask schedule (Phase 4). Built for ALL enumerated crops, not just
        eligible ones, so every dataset sample can look up its source pool.

    Notes
    -----
    Fmask bit layout (HLS v2.0): bit 0 cirrus, bit 1 cloud, bit 2 adj_cloud,
    bit 3 shadow, bit 4 snow_ice, bit 5 water, bits 6-7 aerosol level (0..3).
    NaN-Fmask pixels are cast to 255 ("all bits set"): they trip every
    threshold, so they correctly drop targets and saturate mask fractions.
    """
    H = int(ds.sizes["y"])
    W = int(ds.sizes["x"])
    T = int(ds.sizes["time"])

    fm_float = ds["fmask"].values
    fm_all = np.where(np.isfinite(fm_float), fm_float, 255).astype(np.uint8)
    platform = ds["platform"].values

    cirrus_bit    = (fm_all & 0b00000001) != 0
    cloud_bit     = (fm_all & 0b00000010) != 0
    adj_cloud_bit = (fm_all & 0b00000100) != 0
    shadow_bit    = (fm_all & 0b00001000) != 0
    water_bit     = (fm_all & 0b00100000) != 0
    aero_level    = (fm_all >> 6) & 0b11                # uint8 in 0..3
    nodata        = fm_all == 255
    mask_any      = (fm_all & np.uint8(mask_bits)) != 0  # used by the band gate

    target_crops: list[tuple[int, int, int]] = []
    frac_cache: dict[tuple[int, int], np.ndarray] = {}

    time_ok = np.arange(T) >= lookback_window - 1
    plat_ok = platform == HLSS30_CODE

    for top, left in enumerate_crops(H, W, crop_size):
        y0, y1 = top, top + crop_size
        x0, x1 = left, left + crop_size
        c_cirrus    = cirrus_bit[:,    y0:y1, x0:x1]
        c_cloud     = cloud_bit[:,     y0:y1, x0:x1]
        c_adj_cloud = adj_cloud_bit[:, y0:y1, x0:x1]
        c_shadow    = shadow_bit[:,    y0:y1, x0:x1]
        c_water     = water_bit[:,     y0:y1, x0:x1]
        c_aero      = aero_level[:,    y0:y1, x0:x1]
        c_nodata    = nodata[:,        y0:y1, x0:x1]
        c_mask      = mask_any[:,      y0:y1, x0:x1]
        npx = c_cloud.shape[1] * c_cloud.shape[2]

        # Compare counts vs count thresholds (avoids float-precision artefacts
        # like 99.999...% when count == npx).
        n_cirrus    = c_cirrus.sum(axis=(1, 2))
        n_cloud     = c_cloud.sum(axis=(1, 2))
        n_adj_cloud = c_adj_cloud.sum(axis=(1, 2))
        n_shadow    = c_shadow.sum(axis=(1, 2))
        n_water     = c_water.sum(axis=(1, 2))
        n_aero_0    = (c_aero == 0).sum(axis=(1, 2))
        n_aero_1    = (c_aero == 1).sum(axis=(1, 2))
        n_aero_2    = (c_aero == 2).sum(axis=(1, 2))
        n_aero_3    = (c_aero == 3).sum(axis=(1, 2))
        n_valid     = (~c_nodata).sum(axis=(1, 2))
        cirmax  = cirrus_max             * npx / 100.0
        cmax    = cloud_max              * npx / 100.0
        adjcmax = adj_cloud_max          * npx / 100.0
        smax    = shadow_max             * npx / 100.0
        watmax  = water_max              * npx / 100.0
        a0max   = aero_climatology_max   * npx / 100.0
        a1max   = aero_low_max           * npx / 100.0
        a2max   = aero_medium_max        * npx / 100.0
        a3max   = aero_high_max          * npx / 100.0
        vmin    = valid_min              * npx / 100.0

        eligible = (
            plat_ok
            & (n_cirrus    <= cirmax)
            & (n_cloud     <= cmax)
            & (n_adj_cloud <= adjcmax)
            & (n_shadow    <= smax)
            & (n_water     <= watmax)
            & (n_aero_0    <= a0max)
            & (n_aero_1    <= a1max)
            & (n_aero_2    <= a2max)
            & (n_aero_3    <= a3max)
            & (n_valid     >= vmin)
            & time_ok
        )
        for t in np.flatnonzero(eligible).tolist():
            target_crops.append((int(top), int(left), int(t)))

        # Mask coverage fraction, relative to the NON-NODATA area of the crop.
        # Numerator excludes nodata pixels (fm=255 sets all bits, including
        # the mask bits, and would otherwise inflate the count). Denominator
        # is the valid pixel count, not the whole crop. Fully-nodata slots
        # become NaN so the band-gated selector treats them as unusable
        # (NaN comparisons are False -> not in any band; np.nanargmin handles
        # the closest-to-centre fallback).
        n_mask_in_valid = (c_mask & ~c_nodata).sum(axis=(1, 2)).astype(np.float32)
        n_valid_f = n_valid.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_cache[(int(top), int(left))] = np.where(
                n_valid_f > 0, n_mask_in_valid / n_valid_f, np.nan
            ).astype(np.float32)

    return target_crops, frac_cache


def find_mask_sources(ds: xr.Dataset, mask_bits: int) -> np.ndarray:
    """Time indices of frames likely carrying mask_bits-positive pixels.

    Uses the cubes step's per-bit QA scalars. Bit semantics:
      bit 0 cirrus, bit 1 cloud, bit 2 adj_cloud, bit 3 shadow,
      bit 4 snow_ice, bit 5 water.
    Falls back to "any frame with non-zero pct_valid" if no QA bit is set
    in `mask_bits` (yields the 0% mask augmentation case).
    """
    n = int(ds.sizes["time"])
    any_bad = np.zeros(n, dtype=bool)

    bit_to_var = {
        0: "pct_cirrus",
        1: "pct_cloud",
        2: "pct_adj_cloud",
        3: "pct_shadow",
        4: "pct_snow",
        5: "pct_water",
    }
    for bit, var in bit_to_var.items():
        if not (mask_bits >> bit) & 1:
            continue
        if var not in ds:
            continue
        arr = ds[var].values
        any_bad |= np.nan_to_num(arr, nan=0.0) > 0.0

    idx = np.flatnonzero(any_bad)
    if idx.size == 0:
        valid = ds["pct_valid"].values
        idx = np.flatnonzero(np.nan_to_num(valid, nan=-np.inf) > 0.0)
    return idx
