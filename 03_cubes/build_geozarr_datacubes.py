#!/usr/bin/env python
# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""build_geozarr_datacubes.py — Assemble per-FID GeoZarr datacubes from HLS band TIFs.

For each FID subdirectory under INPUT_DIR, groups the HLS band TIFs by (date, platform),
loads them as an xarray.Dataset with dims (time, band, y, x), computes per-timestep
quality metrics from Fmask, and writes a GeoZarr-compliant Zarr store to OUTPUT_DIR.

Bands kept in order B02, B03, B04, NIR (= B05 on HLSL30, B8A on HLSS30 — renamed to NIR
so both platforms share the same band axis).

Quality metrics (per time):
    - pct_cirrus       Fmask bit 0 (cirrus) set            [see note 1]
    - pct_cloud        Fmask bit 1 (cloud) set
    - pct_adj_cloud    Fmask bit 2 (adjacent-to-cloud) set
    - pct_shadow       Fmask bit 3 (cloud shadow) set
    - pct_snow_ice     Fmask bit 4 (snow/ice) set
    - pct_water        Fmask bit 5 (water) set
    - aerosol_mode     mode of Fmask bits 6-7 (aerosol level)
    - pct_valid        non-nodata pixel share
    - platform         HLSS30 | HLSL30

Global attrs on the dataset:
    - fid              FID directory name
    - GeoZarr spatial:/proj: conventions (via geozarr_toolkit.from_rioxarray)

Usage:
    python build_geozarr_datacubes.py INPUT_DIR OUTPUT_DIR [--jobs N] [--sample]

Note 1: HLS v2.0 Fmask bit layout is (LSB→MSB): 0 cirrus, 1 cloud, 2 adjacent-to-cloud,
3 cloud shadow, 4 snow/ice, 5 water, 6-7 aerosol. "Cloud shadow" is bit 3.
"""

import argparse
import logging
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr
from geozarr_toolkit import from_rioxarray
from pyproj import CRS
from rasterio.enums import Resampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_geozarr")

FMASK_NODATA = np.uint8(255)
BAND_ORDER = ("B02", "B03", "B04", "NIR", "SWIR1", "SWIR2")
NIR_SOURCE = {"HLSL30": "B05", "HLSS30": "B8A"}
SWIR1_SOURCE = {"HLSL30": "B06", "HLSS30": "B11"}
SWIR2_SOURCE = {"HLSL30": "B07", "HLSS30": "B12"}
FNAME_RE = re.compile(
    r"^(?P<date>\d{8}T\d{6})_(?P<platform>HLS[LS]30)_(?P<band>[A-Za-z0-9]+)\.tif$"
)


def discover_groups(fid_dir: Path, fmask_only: bool = False) -> list[dict]:
    """Return complete groups for a FID directory.

    Default mode requires {B02, B03, B04, NIR, Fmask}; `fmask_only=True`
    relaxes the requirement to Fmask alone (Phase A of the eligibility-driven
    download workflow — reflectance arrives in a later append pass).
    """
    raw: dict[tuple[str, str], dict[str, Path]] = {}
    for tif in fid_dir.glob("*.tif"):
        m = FNAME_RE.match(tif.name)
        if not m:
            continue
        raw.setdefault((m["date"], m["platform"]), {})[m["band"]] = tif

    groups = []
    for (date, platform), paths in raw.items():
        if fmask_only:
            if "Fmask" not in paths:
                continue
            groups.append(
                {
                    "date": date,
                    "platform": platform,
                    "paths": {"Fmask": paths["Fmask"]},
                }
            )
        else:
            nir_src = NIR_SOURCE[platform]
            swir1_src = SWIR1_SOURCE[platform]
            swir2_src = SWIR2_SOURCE[platform]
            required = {"B02", "B03", "B04", nir_src, swir1_src, swir2_src, "Fmask"}
            if not required.issubset(paths):
                continue
            groups.append(
                {
                    "date": date,
                    "platform": platform,
                    "paths": {
                        "B02": paths["B02"],
                        "B03": paths["B03"],
                        "B04": paths["B04"],
                        "NIR": paths[nir_src],
                        "SWIR1": paths[swir1_src],
                        "SWIR2": paths[swir2_src],
                        "Fmask": paths["Fmask"],
                    },
                }
            )
    groups.sort(key=lambda g: (g["date"], g["platform"]))
    return groups


def load_group(
    group: dict, fmask_only: bool = False
) -> tuple[xr.DataArray | None, xr.DataArray]:
    """Load one group as (reflectance[band,y,x] | None, fmask[y,x])."""
    import rioxarray as rxr

    fmask = rxr.open_rasterio(
        group["paths"]["Fmask"], chunks={"x": -1, "y": -1}, masked=False
    ).squeeze("band", drop=True)
    if fmask_only:
        return None, fmask

    bands = []
    for b in BAND_ORDER:
        da = rxr.open_rasterio(
            group["paths"][b], chunks={"x": -1, "y": -1}, masked=False
        ).squeeze("band", drop=True)
        bands.append(da)
    refl = xr.concat(
        bands,
        dim=pd.Index(np.arange(len(BAND_ORDER), dtype=np.int8), name="band"),
    )
    return refl, fmask


def _crs_equivalent(a: CRS, b: CRS) -> bool:
    """HLSL30 and HLSS30 ship WKTs that differ textually (datum= vs ellps=) but
    encode the same UTM zone. Accept EPSG match (relaxed confidence) or the
    combo of same UTM zone + same projection method as equivalent."""
    ae = a.to_epsg(min_confidence=25)
    be = b.to_epsg(min_confidence=25)
    if ae is not None and ae == be:
        return True
    if a.is_projected and b.is_projected and a.utm_zone and a.utm_zone == b.utm_zone:
        return a.coordinate_operation.method_name == b.coordinate_operation.method_name
    return a.equals(b)


def compute_qc(fmask_arr: np.ndarray) -> dict:
    """Per-timestep QC metrics from a 2D Fmask array."""
    valid = fmask_arr != FMASK_NODATA
    n_valid = int(valid.sum())
    total = fmask_arr.size
    pct_valid = np.float32(100.0 * n_valid / total) if total else np.float32(0.0)

    if n_valid == 0:
        return {
            "pct_cirrus": np.float32(np.nan),
            "pct_cloud": np.float32(np.nan),
            "pct_adj_cloud": np.float32(np.nan),
            "pct_shadow": np.float32(np.nan),
            "pct_snow_ice": np.float32(np.nan),
            "pct_water": np.float32(np.nan),
            "pct_aero_climatology": np.float32(np.nan),
            "pct_aero_low": np.float32(np.nan),
            "pct_aero_medium": np.float32(np.nan),
            "pct_aero_high": np.float32(np.nan),
            "pct_valid": pct_valid,
        }

    fm = fmask_arr[valid]
    aerosol = ((fm >> 6) & 0b11).astype(np.uint8)

    def _aerosol_pct(code: int) -> np.float32:
        return np.float32(100.0 * (aerosol == code).mean())

    def _bit_pct(bit: int) -> np.float32:
        return np.float32(100.0 * ((fm >> bit) & 1).astype(bool).mean())

    return {
        "pct_cirrus": _bit_pct(0),
        "pct_cloud": _bit_pct(1),
        "pct_adj_cloud": _bit_pct(2),
        "pct_shadow": _bit_pct(3),
        "pct_snow_ice": _bit_pct(4),
        "pct_water": _bit_pct(5),
        "pct_aero_climatology": _aerosol_pct(0),
        "pct_aero_low": _aerosol_pct(1),
        "pct_aero_medium": _aerosol_pct(2),
        "pct_aero_high": _aerosol_pct(3),
        "pct_valid": pct_valid,
    }


def build_fid_dataset(fid_dir: Path, fmask_only: bool = False) -> xr.Dataset | None:
    groups = discover_groups(fid_dir, fmask_only=fmask_only)
    if not groups:
        required = "Fmask" if fmask_only else "B02/B03/B04/NIR/Fmask"
        log.warning("%s: no complete (%s) groups", fid_dir.name, required)
        return None

    log.info("%s: %d complete groups", fid_dir.name, len(groups))

    # Reference grid: reflectance when available, else Fmask (fmask-only mode).
    ref_refl, ref_fmask = load_group(groups[0], fmask_only=fmask_only)
    ref_grid = ref_refl if ref_refl is not None else ref_fmask
    ref_shape = (ref_grid.sizes["y"], ref_grid.sizes["x"])
    ref_crs = ref_grid.rio.crs
    ref_crs_pyproj = CRS.from_wkt(ref_crs.to_wkt())

    reflectances, fmasks, times, platforms, qcs = [], [], [], [], []
    for g in groups:
        try:
            refl, fm = load_group(g, fmask_only=fmask_only)
        except Exception as e:
            log.warning(
                "%s/%s/%s: load failed: %s", fid_dir.name, g["platform"], g["date"], e
            )
            continue

        # ROIs that cross HLS tile / UTM-zone boundaries can return frames whose
        # native grid differs from the reference (different shape, or a neighbouring
        # UTM zone). Instead of dropping these, snap them onto the reference grid:
        # nearest-neighbour resampling preserves Fmask DN values exactly, and pixels
        # outside the source footprint are written as nodata (255 for Fmask, -9999
        # for reflectance) — i.e. partial coverage is faithfully encoded.
        this_shape = (fm.sizes["y"], fm.sizes["x"])
        this_crs_pyproj = CRS.from_wkt(fm.rio.crs.to_wkt())
        same_crs = _crs_equivalent(this_crs_pyproj, ref_crs_pyproj)
        if this_shape != ref_shape or not same_crs:
            try:
                fm = fm.rio.reproject_match(
                    ref_grid,
                    resampling=Resampling.nearest,
                    nodata=int(FMASK_NODATA),
                )
                if refl is not None:
                    refl = refl.rio.reproject_match(
                        ref_grid, resampling=Resampling.nearest
                    )
            except Exception as e:
                log.warning(
                    "%s/%s/%s: reproject_match failed (%s) — skipping",
                    fid_dir.name,
                    g["platform"],
                    g["date"],
                    e,
                )
                continue
            log.info(
                "%s/%s/%s: snapped to reference grid (was shape=%s, crs=%s)",
                fid_dir.name,
                g["platform"],
                g["date"],
                this_shape,
                this_crs_pyproj.to_authority() or this_crs_pyproj.name,
            )

        # Force byte-identical x/y on every frame (native and reprojected alike).
        # Without this, concat sees sub-ULP drift between scenes as "duplicate"
        # index values along x or y under the default outer-join, blowing up the
        # whole ROI.
        fm = fm.assign_coords(x=ref_grid.x, y=ref_grid.y)
        if refl is not None:
            refl = refl.assign_coords(x=ref_grid.x, y=ref_grid.y)

        # Materialise inside a try/except so a single corrupt source TIF
        # (e.g. a truncated LZW stream) drops only that frame rather than
        # killing the whole ROI.
        try:
            fm_values = fm.values  # forces Fmask read; QC needs DN bits
            if refl is not None:
                refl = refl.compute()
        except Exception as e:
            log.warning(
                "%s/%s/%s: read failed (%s) — skipping",
                fid_dir.name,
                g["platform"],
                g["date"],
                e,
            )
            continue
        qcs.append(compute_qc(fm_values))
        if refl is not None:
            reflectances.append(refl)
        fmasks.append(fm)
        times.append(pd.to_datetime(g["date"], format="%Y%m%dT%H%M%S"))
        platforms.append(g["platform"])

    if not times:
        return None

    time_idx = pd.DatetimeIndex(times, name="time")
    fmask_cube = xr.concat(fmasks, dim=time_idx)
    refl_cube = xr.concat(reflectances, dim=time_idx) if reflectances else None

    data_vars: dict = {}
    if refl_cube is not None:
        data_vars["reflectance"] = refl_cube
    data_vars["fmask"] = fmask_cube
    data_vars.update(
        {
            "platform": (
                "time",
                np.array(
                    [0 if p == "HLSL30" else 1 for p in platforms],
                    dtype=np.int8,
                ),
            ),
            "pct_cirrus": (
                "time",
                np.array([q["pct_cirrus"] for q in qcs], dtype=np.float32),
            ),
            "pct_cloud": (
                "time",
                np.array([q["pct_cloud"] for q in qcs], dtype=np.float32),
            ),
            "pct_adj_cloud": (
                "time",
                np.array([q["pct_adj_cloud"] for q in qcs], dtype=np.float32),
            ),
            "pct_shadow": (
                "time",
                np.array([q["pct_shadow"] for q in qcs], dtype=np.float32),
            ),
            "pct_snow_ice": (
                "time",
                np.array([q["pct_snow_ice"] for q in qcs], dtype=np.float32),
            ),
            "pct_water": (
                "time",
                np.array([q["pct_water"] for q in qcs], dtype=np.float32),
            ),
            "pct_aero_climatology": (
                "time",
                np.array([q["pct_aero_climatology"] for q in qcs], dtype=np.float32),
            ),
            "pct_aero_low": (
                "time",
                np.array([q["pct_aero_low"] for q in qcs], dtype=np.float32),
            ),
            "pct_aero_medium": (
                "time",
                np.array([q["pct_aero_medium"] for q in qcs], dtype=np.float32),
            ),
            "pct_aero_high": (
                "time",
                np.array([q["pct_aero_high"] for q in qcs], dtype=np.float32),
            ),
            "pct_valid": (
                "time",
                np.array([q["pct_valid"] for q in qcs], dtype=np.float32),
            ),
        }
    )
    ds = xr.Dataset(data_vars=data_vars)

    # rioxarray's .rio is carried by the "spatial_ref" scalar coord; reassert CRS so
    # geozarr_toolkit.from_rioxarray finds it reliably on the concatenated array.
    if "reflectance" in ds:
        ds["reflectance"] = ds["reflectance"].rio.write_crs(ref_crs)
    ds["fmask"] = ds["fmask"].rio.write_crs(ref_crs)

    # Human-readable variable metadata
    if "reflectance" in ds:
        ds["reflectance"].attrs.update(
            {
                "long_name": "HLS surface reflectance (scaled Int16)",
                "_FillValue": -9999,
                "scale_factor": 0.0001,
                "band_names": " ".join(BAND_ORDER),
            }
        )
    ds["fmask"].attrs.update(
        {
            "long_name": "HLS Fmask quality layer",
            "flag_bits": "0:cirrus, 1:cloud, 2:adjacent, 3:cloud_shadow, 4:snow_ice, 5:water, 6-7:aerosol",
            "_FillValue": int(FMASK_NODATA),
        }
    )
    ds["pct_cirrus"].attrs.update(
        {"long_name": "Cirrus pixels (Fmask bit 0)", "units": "%"}
    )
    ds["pct_cloud"].attrs.update(
        {"long_name": "Cloud pixels (Fmask bit 1)", "units": "%"}
    )
    ds["pct_adj_cloud"].attrs.update(
        {"long_name": "Adjacent-to-cloud pixels (Fmask bit 2)", "units": "%"}
    )
    ds["pct_shadow"].attrs.update(
        {"long_name": "Cloud-shadow pixels (Fmask bit 3)", "units": "%"}
    )
    ds["pct_snow_ice"].attrs.update(
        {"long_name": "Snow/ice pixels (Fmask bit 4)", "units": "%"}
    )
    ds["pct_water"].attrs.update(
        {"long_name": "Water pixels (Fmask bit 5)", "units": "%"}
    )
    ds["pct_aero_climatology"].attrs.update(
        {"long_name": "Climatology aerosol pixels (Fmask bits 6-7)", "units": "%"}
    )
    ds["pct_aero_low"].attrs.update(
        {"long_name": "Low aerosol pixels (Fmask bits 6-7)", "units": "%"}
    )
    ds["pct_aero_medium"].attrs.update(
        {"long_name": "Medium aerosol pixels (Fmask bits 6-7)", "units": "%"}
    )
    ds["pct_aero_high"].attrs.update(
        {"long_name": "High aerosol pixels (Fmask bits 6-7)", "units": "%"}
    )
    ds["pct_valid"].attrs.update(
        {"long_name": "Valid (non-nodata) pixels", "units": "%"}
    )
    ds["platform"].attrs.update(
        {
            "long_name": "HLS collection",
            "flag_values": np.array([0, 1], dtype=np.int8),
            "flag_meanings": "HLSL30 HLSS30",
        }
    )

    # GeoZarr spatial:/proj: conventions from rioxarray metadata. Use whichever
    # of reflectance/fmask is present as the geometry carrier (both share the
    # same grid by construction).
    geozarr_attrs = from_rioxarray(
        ds["reflectance"] if "reflectance" in ds else ds["fmask"]
    )
    ds.attrs.update(geozarr_attrs)
    ds.attrs["fid"] = fid_dir.name

    # Chunk shape tuned for the training step's access pattern: a sample reads
    # ~70 contiguous timesteps × 64×64 spatial × all 4 reflectance bands, plus
    # one random-time fmask read of the same 64×64 window.
    #   time=32 → 70-frame window touches ~3 chunks (low overhead) and the
    #             random-time read pulls only ~32 timesteps' worth.
    #   y/x=64 → matches the model's crop_size; worst-case 2×2=4 chunks per
    #            crop when the offset straddles boundaries.
    # We force uniform spatial chunks (capped by the ROI's actual extent so
    # small ROIs don't get oversized chunks) — this also avoids the
    # non-uniform-chunk error that zarr v3 raises after `xr.concat` of
    # reprojected frames.
    chunk_y = min(64, ds.sizes["y"])
    chunk_x = min(64, ds.sizes["x"])
    chunk_t = min(32, ds.sizes["time"])
    ds = ds.chunk({"time": chunk_t, "y": chunk_y, "x": chunk_x})
    return ds


def _discover_swir_groups(fid_dir: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """Return {(date, platform): {"SWIR1": path, "SWIR2": path}} for cluster
    tifs that have BOTH SWIR1 and SWIR2 source bands on disk.
    """
    raw: dict[tuple[str, str], dict[str, Path]] = {}
    for tif in fid_dir.glob("*.tif"):
        m = FNAME_RE.match(tif.name)
        if not m:
            continue
        raw.setdefault((m["date"], m["platform"]), {})[m["band"]] = tif

    groups: dict[tuple[str, str], dict[str, Path]] = {}
    for (date, platform), paths in raw.items():
        s1 = SWIR1_SOURCE[platform]
        s2 = SWIR2_SOURCE[platform]
        if s1 in paths and s2 in paths:
            groups[(date, platform)] = {"SWIR1": paths[s1], "SWIR2": paths[s2]}
    return groups


def append_swir_to_zarr(fid_dir: Path, zarr_path: Path) -> str:
    """Extend an existing 4-band reflectance zarr with SWIR1 + SWIR2.

    Adds 2 slots to the `band` dimension (4 → 6) and writes the SWIR bands
    in slots 4 and 5. Time slots without on-disk SWIR tifs get NaN-filled.
    The existing 4 reflectance bands are untouched.
    """
    import rioxarray as rxr

    existing = xr.open_zarr(zarr_path, consolidated=True)
    if "reflectance" not in existing.data_vars:
        raise RuntimeError(
            f"{zarr_path} has no 'reflectance' — run --append-reflectance first"
        )
    cur_n_band = int(existing.reflectance.sizes["band"])
    if cur_n_band == len(BAND_ORDER):
        raise RuntimeError(
            f"{zarr_path}: reflectance already has {cur_n_band} bands "
            "(matching BAND_ORDER) — nothing to append"
        )
    if cur_n_band != 4:
        raise RuntimeError(
            f"{zarr_path}: reflectance has {cur_n_band} bands, expected 4"
        )

    n_time = int(existing.sizes["time"])
    ny = int(existing.sizes["y"])
    nx = int(existing.sizes["x"])
    times = existing["time"].values
    platforms = existing["platform"].values

    if "spatial_ref" not in existing or "crs_wkt" not in existing.spatial_ref.attrs:
        raise RuntimeError(f"{zarr_path}: missing GeoZarr spatial_ref/crs_wkt")
    ref_crs_pyproj = CRS.from_wkt(existing.spatial_ref.attrs["crs_wkt"])
    ref_grid = existing["fmask"].rio.write_crs(ref_crs_pyproj.to_wkt())

    groups = _discover_swir_groups(fid_dir)

    # Output: (T, 2, Y, X) float64 physical reflectance, NaN for missing.
    swir_cube = np.full((n_time, 2, ny, nx), np.nan, dtype=np.float64)
    n_filled = 0
    for i in range(n_time):
        date_str = pd.Timestamp(times[i]).strftime("%Y%m%dT%H%M%S")
        platform = "HLSL30" if int(platforms[i]) == 0 else "HLSS30"
        key = (date_str, platform)
        if key not in groups:
            n_filled += 1
            continue
        try:
            bands = []
            for b_key in ("SWIR1", "SWIR2"):
                da = rxr.open_rasterio(
                    groups[key][b_key], chunks={"x": -1, "y": -1}, masked=False
                ).squeeze("band", drop=True)
                this_crs_pyproj = CRS.from_wkt(da.rio.crs.to_wkt())
                if (da.sizes["y"], da.sizes["x"]) != (ny, nx) or not _crs_equivalent(
                    this_crs_pyproj, ref_crs_pyproj
                ):
                    da = da.rio.reproject_match(ref_grid, resampling=Resampling.nearest)
                da = da.assign_coords(x=ref_grid.x, y=ref_grid.y).compute()
                # Apply HLS Int16 scale & fill manually since we'll re-encode.
                raw = da.values.astype(np.int16)
                arr = raw.astype(np.float64)
                arr[raw == -9999] = np.nan
                arr *= 0.0001
                bands.append(arr)
            swir_cube[i, 0] = bands[0]
            swir_cube[i, 1] = bands[1]
        except Exception as e:
            log.warning(
                "%s/%s/%s: SWIR load failed (%s) — filling NaN",
                fid_dir.name,
                platform,
                date_str,
                e,
            )
            n_filled += 1

    swir = xr.DataArray(
        swir_cube,
        dims=("time", "band", "y", "x"),
        coords={
            "time": existing["time"],
            "band": np.array([cur_n_band, cur_n_band + 1], dtype=np.int8),
            "y": existing["y"],
            "x": existing["x"],
        },
    )
    # Encoding: match the existing reflectance's on-disk format. Chunk-band
    # must be 2 (full extent of the new slab) — zarr v3 stores the appended
    # data as one extra chunk along the band axis on top of the existing
    # chunk of size 4.
    swir.encoding = {
        "dtype": "int16",
        "_FillValue": -9999,
        "scale_factor": 0.0001,
        "chunks": (min(32, n_time), 2, min(64, ny), min(64, nx)),
    }
    new_ds = xr.Dataset({"reflectance": swir})
    new_ds.to_zarr(zarr_path, append_dim="band", consolidated=True, zarr_format=3)
    return f"OK ({n_time} timesteps, {n_filled} filled, band {cur_n_band}→{cur_n_band + 2}, {zarr_path})"


def append_reflectance_to_zarr(
    fid_dir: Path, zarr_path: Path, *, force: bool = False
) -> str:
    """Phase D: add the reflectance variable to an existing fmask-only zarr.

    Reads the existing store's `time` and `platform` coords as authority,
    locates the matching reflectance group per timestep in `fid_dir`, and
    writes a new `reflectance` variable aligned to the existing grid.
    Time slots without a reflectance group on disk get filled with the
    `_FillValue` (-9999) so dataset semantics stay consistent.

    With ``force=True``, an existing `reflectance` variable is removed from
    disk first so the function can rebuild it from scratch (e.g. after a
    re-scan exposes more eligible target dates than the previous Phase D
    run pulled).
    """
    existing = xr.open_zarr(zarr_path, consolidated=True)
    if "reflectance" in existing.data_vars:
        if not force:
            raise RuntimeError(
                f"{zarr_path} already has 'reflectance' — pass force=True "
                f"(CLI: --overwrite-reflectance) to rebuild it"
            )
        # Close the lazy handle before deleting on-disk data.
        existing.close()
        refl_dir = zarr_path / "reflectance"
        if refl_dir.exists():
            shutil.rmtree(refl_dir)
        # Re-open to pick up the missing-variable state for the rest of this
        # function. Consolidated metadata will be refreshed by the to_zarr
        # call at the end.
        existing = xr.open_zarr(zarr_path, consolidated=False)
    if "fmask" not in existing.data_vars:
        raise RuntimeError(
            f"{zarr_path} lacks 'fmask' — append-reflectance requires an "
            "fmask-only zarr built by --fmask-only first"
        )

    n_time = int(existing.sizes["time"])
    ny = int(existing.sizes["y"])
    nx = int(existing.sizes["x"])
    times = existing["time"].values
    platforms = existing["platform"].values

    # CRS doesn't survive xr.open_zarr through .rio.crs by default; pull it
    # straight from the GeoZarr `spatial_ref` coord and reattach to fmask so
    # reproject_match has a reference grid with a real CRS.
    if "spatial_ref" not in existing or "crs_wkt" not in existing.spatial_ref.attrs:
        raise RuntimeError(
            f"{zarr_path}: missing GeoZarr spatial_ref/crs_wkt — cannot align"
        )
    ref_crs_pyproj = CRS.from_wkt(existing.spatial_ref.attrs["crs_wkt"])
    ref_grid = existing["fmask"].rio.write_crs(ref_crs_pyproj.to_wkt())
    ref_crs = ref_grid.rio.crs

    # Lookup: (date_str, platform_str) → group with reflectance bands.
    full_groups = discover_groups(fid_dir, fmask_only=False)
    group_by_key = {(g["date"], g["platform"]): g for g in full_groups}

    fill = np.int16(-9999)
    fill_da = xr.DataArray(
        np.full((len(BAND_ORDER), ny, nx), fill, dtype=np.int16),
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(len(BAND_ORDER), dtype=np.int8),
            "y": ref_grid.y,
            "x": ref_grid.x,
        },
    )

    refl_per_time: list[xr.DataArray] = []
    n_filled = 0
    for i in range(n_time):
        date_str = pd.Timestamp(times[i]).strftime("%Y%m%dT%H%M%S")
        platform = "HLSL30" if int(platforms[i]) == 0 else "HLSS30"
        key = (date_str, platform)
        if key not in group_by_key:
            refl_per_time.append(fill_da)
            n_filled += 1
            continue

        refl, _ = load_group(group_by_key[key], fmask_only=False)
        this_crs_pyproj = CRS.from_wkt(refl.rio.crs.to_wkt())
        same_crs = _crs_equivalent(this_crs_pyproj, ref_crs_pyproj)
        if (refl.sizes["y"], refl.sizes["x"]) != (ny, nx) or not same_crs:
            try:
                refl = refl.rio.reproject_match(
                    ref_grid, resampling=Resampling.nearest
                )
            except Exception as e:
                log.warning(
                    "%s/%s/%s: reproject_match failed (%s) — filling NaN",
                    fid_dir.name,
                    platform,
                    date_str,
                    e,
                )
                refl_per_time.append(fill_da)
                n_filled += 1
                continue
        refl = refl.assign_coords(x=ref_grid.x, y=ref_grid.y)
        try:
            refl = refl.compute()
        except Exception as e:
            log.warning(
                "%s/%s/%s: read failed (%s) — filling NaN",
                fid_dir.name,
                platform,
                date_str,
                e,
            )
            refl_per_time.append(fill_da)
            n_filled += 1
            continue
        refl_per_time.append(refl)

    # Drop spatial_ref before concat: fill_da doesn't carry it but
    # rioxarray-loaded refl DAs do, so the default coords='different'
    # raises "spatial_ref not present in all datasets". We rewrite the
    # CRS onto the concatenated cube below.
    refl_per_time = [
        da.drop_vars("spatial_ref", errors="ignore") for da in refl_per_time
    ]
    time_idx = pd.DatetimeIndex(times, name="time")
    refl_cube = xr.concat(refl_per_time, dim=time_idx, coords="minimal")

    # Pin the cube's coords byte-for-byte to the existing store's coords so
    # xarray's mode="a" alignment doesn't trip on sub-ULP float drift.
    refl_cube = refl_cube.assign_coords(
        time=existing["time"], y=existing["y"], x=existing["x"]
    )
    refl_cube = refl_cube.rio.write_crs(ref_crs)
    refl_cube.attrs.update(
        {
            "long_name": "HLS surface reflectance (scaled Int16)",
            "_FillValue": -9999,
            "scale_factor": 0.0001,
            "band_names": " ".join(BAND_ORDER),
        }
    )

    # Match chunking to the existing fmask variable; zarr v3 rejects
    # non-uniform chunks across variables that share a dim.
    fmask_chunks = existing["fmask"].encoding.get("chunks") or (
        min(32, n_time), min(64, ny), min(64, nx)
    )
    chunk_t, chunk_y, chunk_x = fmask_chunks
    refl_cube = refl_cube.chunk(
        {"time": chunk_t, "y": chunk_y, "x": chunk_x, "band": len(BAND_ORDER)}
    )

    new_ds = xr.Dataset({"reflectance": refl_cube})
    new_ds.to_zarr(zarr_path, mode="a", consolidated=True, zarr_format=3)
    return f"OK ({n_time} timesteps, {n_filled} filled, {zarr_path})"


def process_fid(
    fid_dir: Path,
    output_dir: Path,
    fmask_only: bool = False,
    append_reflectance: bool = False,
    append_swir: bool = False,
    overwrite_reflectance: bool = False,
) -> tuple[str, str]:
    try:
        out_path = output_dir / f"{fid_dir.name}.zarr"

        if append_reflectance:
            if not out_path.exists():
                return fid_dir.name, f"FAIL: zarr missing ({out_path})"
            return fid_dir.name, append_reflectance_to_zarr(
                fid_dir, out_path, force=overwrite_reflectance
            )

        if append_swir:
            if not out_path.exists():
                return fid_dir.name, f"FAIL: zarr missing ({out_path})"
            return fid_dir.name, append_swir_to_zarr(fid_dir, out_path)

        ds = build_fid_dataset(fid_dir, fmask_only=fmask_only)
        if ds is None:
            return fid_dir.name, "EMPTY"

        if out_path.exists():
            shutil.rmtree(out_path)

        ds.to_zarr(out_path, mode="w", zarr_format=3, consolidated=True)
        return fid_dir.name, f"OK ({ds.sizes['time']} timesteps, {out_path})"
    except Exception as e:
        log.exception("%s: failed", fid_dir.name)
        return fid_dir.name, f"FAIL: {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "input_dir", type=Path, help="Root of per-FID subdirs (e.g. earthaccess/)"
    )
    ap.add_argument("output_dir", type=Path, help="Destination for *.zarr stores")
    ap.add_argument(
        "--jobs", type=int, default=1, help="Parallel workers over FIDs (default 1)"
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Process only FID 10124_curr (for testing)",
    )
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fmask-only",
        action="store_true",
        help="Skip reflectance: build zarrs with Fmask + QC scalars only. "
        "Phase A of the eligibility-driven download workflow — reflectance "
        "is added later via --append-reflectance.",
    )
    mode_group.add_argument(
        "--append-reflectance",
        action="store_true",
        help="Append the reflectance variable to existing fmask-only zarrs. "
        "Phase D: refuses to run if a zarr already has reflectance or lacks "
        "fmask. Time slots with no on-disk reflectance group are filled "
        "with the dataset's _FillValue (-9999).",
    )
    ap.add_argument(
        "--overwrite-reflectance",
        action="store_true",
        help="When combined with --append-reflectance, remove any existing "
        "reflectance variable on disk first and rebuild from scratch. Use "
        "after a re-scan exposes more eligible target dates than the previous "
        "Phase D run downloaded.",
    )
    mode_group.add_argument(
        "--append-swir",
        action="store_true",
        help="Extend an existing 4-band reflectance zarr with SWIR1+SWIR2 "
        "(band 4 and 5). Reads the SWIR1/SWIR2 tifs from the FID dir, "
        "appends along the band axis. Refuses if reflectance is absent "
        "or already at len(BAND_ORDER).",
    )
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        log.error("%s is not a directory", args.input_dir)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect: if input_dir contains .tif files directly, treat as a single
    # FID dir; otherwise treat it as a root containing per-FID subdirectories.
    if any(args.input_dir.glob("*.tif")):
        fid_dirs = [args.input_dir]
    else:
        fid_dirs = sorted(p for p in args.input_dir.iterdir() if p.is_dir())
    if args.sample:
        sample = args.input_dir / "10124_curr"
        if not sample.is_dir():
            log.error("--sample requested but %s is missing", sample)
            return 1
        fid_dirs = [sample]

    log.info("Processing %d FID(s) with %d worker(s)", len(fid_dirs), args.jobs)

    results: list[tuple[str, str]] = []
    if args.overwrite_reflectance and not args.append_reflectance:
        log.error("--overwrite-reflectance only applies with --append-reflectance")
        return 1

    if args.jobs > 1 and len(fid_dirs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {
                ex.submit(
                    process_fid,
                    d,
                    args.output_dir,
                    args.fmask_only,
                    args.append_reflectance,
                    args.append_swir,
                    args.overwrite_reflectance,
                ): d
                for d in fid_dirs
            }
            for fut in as_completed(futures):
                name, msg = fut.result()
                log.info("%s: %s", name, msg)
                results.append((name, msg))
    else:
        for d in fid_dirs:
            name, msg = process_fid(
                d, args.output_dir, args.fmask_only, args.append_reflectance,
                args.append_swir, args.overwrite_reflectance,
            )
            log.info("%s: %s", name, msg)
            results.append((name, msg))

    ok = sum(1 for _, m in results if m.startswith("OK"))
    empty = sum(1 for _, m in results if m == "EMPTY")
    failed = len(results) - ok - empty
    log.info(
        "=== Summary: %d OK, %d empty, %d failed (of %d) ===",
        ok,
        empty,
        failed,
        len(results),
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
