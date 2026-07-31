#!/usr/bin/env python
# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""scan_eligibility.py — Phase B of the eligibility-driven download pipeline.

Walks the per-ROI GeoZarr stores produced by the cubes-building step's
`--fmask-only` build (Phase A), applies the shared eligibility filter to
determine which target dates are trainable, expands each eligible target
backwards over the lookback window, and intersects the result with
`granule_list.json` to populate the optional `needed_fids` field per granule.

Phase C of the downloader (`hls_stac_download.py`) auto-applies that field:
granules outside any ROI's lookback window are skipped entirely, and granules
with non-empty `needed_fids` are clipped only for those ROIs.

Inputs
------
- `--zarrs-dir`   Root of per-ROI `<fid>.zarr` stores (Phase A output).
- `--granule-list` JSON written by the downloader's Phase C. Updated in place.

Eligibility thresholds (defaults match the pretraining screens actually used:
up to 5% cirrus, 30% moderate aerosol, 10% open water, and 10% invalid
reflectance; strict — 0% — on cloud, shadow, low-climatology aerosol, and
high aerosol, the classes that corrupt the surface signal outright):
  --lookback-window, --cloud-max, --shadow-max, --cirrus-max, --water-max,
  --aero-{climatology,low,medium,high}-max, --valid-min.

Safety policy
-------------
For every granule, the scanner only writes `needed_fids` when **all** of the
granule's `fids` have a corresponding zarr in `--zarrs-dir` (i.e. their
eligibility is actually known). Granules referencing one or more ROIs whose
zarr is missing are left untouched, so Phase C falls back to today's
behaviour (clip all `fids`) for those granules. This keeps a partial Phase A
run from silently dropping ROIs in Phase C.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Reuse the shared eligibility helper — single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from eligibility import filter_crop_targets  # noqa: E402

log = logging.getLogger("scan_eligibility")


def granule_id_to_filename_date(gid: str) -> str:
    """Convert `L30.T20LKP.2022015T135716` → `20220115T135716` to match the
    `YYYYMMDDTHHMMSS` format used by the downloader's `parse_hls_filename`
    (and therefore by the cubes step's zarr `time` coord, which is built
    from those filenames)."""
    raw_dt = gid.split(".")[2]
    date_part, time_part = raw_dt.split("T")
    dt = datetime.strptime(date_part, "%Y%j")
    return dt.strftime("%Y%m%d") + "T" + time_part


def time_value_to_filename_date(ts: np.datetime64) -> str:
    """Convert a numpy datetime64 from the zarr `time` coord to the
    `YYYYMMDDTHHMMSS` string format used in granule IDs / HLS filenames."""
    return pd.Timestamp(ts).strftime("%Y%m%dT%H%M%S")


def scan_zarr(
    zarr_path: Path,
    crop_size: int,
    lookback_window: int,
    cloud_max: float,
    shadow_max: float,
    cirrus_max: float,
    water_max: float,
    aero_climatology_max: float,
    aero_low_max: float,
    aero_medium_max: float,
    aero_high_max: float,
    valid_min: float,
) -> tuple[set[str], int, int, int]:
    """Open one zarr, apply the per-crop eligibility filter, expand lookback windows.

    Per-crop QC is computed for every (top, left) tile produced by
    `enumerate_crops`. Target times whose crops pass eligibility for at least
    one tile contribute their input window's dates to `needed`.

    Returns (needed_dates_strings, n_eligible_pairs, n_unique_target_times, n_time).
    """
    ds = xr.open_zarr(zarr_path, consolidated=True)
    n_time = int(ds.sizes["time"])
    target_crops, _ = filter_crop_targets(
        ds,
        crop_size=crop_size,
        lookback_window=lookback_window,
        cloud_max=cloud_max,
        shadow_max=shadow_max,
        cirrus_max=cirrus_max,
        water_max=water_max,
        aero_climatology_max=aero_climatology_max,
        aero_low_max=aero_low_max,
        aero_medium_max=aero_medium_max,
        aero_high_max=aero_high_max,
        valid_min=valid_min,
    )

    n_pairs = len(target_crops)
    target_times = sorted({t for (_, _, t) in target_crops})

    needed: set[str] = set()
    if not target_times:
        return needed, 0, 0, n_time

    times = ds["time"].values
    for t in target_times:
        lo = max(0, int(t) - lookback_window + 1)
        hi = int(t) + 1
        for ts in times[lo:hi]:
            needed.add(time_value_to_filename_date(ts))
    return needed, n_pairs, len(target_times), n_time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--zarrs-dir",
        type=Path,
        required=True,
        help="Root directory of per-ROI <fid>.zarr stores",
    )
    ap.add_argument(
        "--granule-list",
        type=Path,
        required=True,
        help="Granule list JSON to augment (updated in place)",
    )
    ap.add_argument(
        "--lookback-window",
        type=int,
        default=80,
        help="Lookback window L (default: 80, the retrieval length used throughout)",
    )
    ap.add_argument(
        "--crop-size",
        type=int,
        default=64,
        help="Per-crop tile side used for eligibility (matches training crop_size)",
    )
    ap.add_argument("--cloud-max", type=float, default=0.0,
                     help="Max cloud fraction admitted (default: 0.0, strict)")
    ap.add_argument("--shadow-max", type=float, default=0.0,
                     help="Max cloud-shadow fraction admitted (default: 0.0, strict)")
    ap.add_argument("--cirrus-max", type=float, default=5.0,
                     help="Max cirrus fraction admitted (default: 5.0)")
    ap.add_argument("--water-max", type=float, default=10.0,
                     help="Max open-water fraction admitted (default: 10.0)")
    ap.add_argument("--aero-climatology-max", type=float, default=0.0,
                     help="Max climatology-aerosol fraction admitted (default: 0.0, strict)")
    ap.add_argument("--aero-low-max", type=float, default=100.0,
                     help="Max low-aerosol fraction admitted (default: 100.0, unrestricted)")
    ap.add_argument("--aero-medium-max", type=float, default=30.0,
                     help="Max moderate-aerosol fraction admitted (default: 30.0)")
    ap.add_argument("--aero-high-max", type=float, default=0.0,
                     help="Max high-aerosol fraction admitted (default: 0.0, strict)")
    ap.add_argument("--valid-min", type=float, default=90.0,
                     help="Min valid-reflectance fraction required (default: 90.0, "
                          "i.e. up to 10%% invalid reflectance admitted)")
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (default: <granule-list>.scan_eligibility.log)",
    )
    args = ap.parse_args()

    log_path = args.log_file or args.granule_list.with_name(
        args.granule_list.name + ".scan_eligibility.log"
    )
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Also echo to stderr so the user sees progress without tailing.
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(sh)

    if not args.zarrs_dir.is_dir():
        log.error("zarrs-dir not found: %s", args.zarrs_dir)
        return 1
    if not args.granule_list.is_file():
        log.error("granule-list not found: %s", args.granule_list)
        return 1

    # 1. Walk zarrs → per-ROI needed_dates set.
    zarrs = sorted(p for p in args.zarrs_dir.iterdir() if p.suffix == ".zarr")
    if not zarrs:
        log.error("no *.zarr stores in %s", args.zarrs_dir)
        return 1
    log.info("Scanning %d zarrs from %s", len(zarrs), args.zarrs_dir)

    needed_dates: dict[str, set[str]] = {}
    n_no_targets = 0
    for z in zarrs:
        fid = z.stem
        try:
            dates, n_pairs, n_target_times, n_time = scan_zarr(
                z,
                crop_size=args.crop_size,
                lookback_window=args.lookback_window,
                cloud_max=args.cloud_max,
                shadow_max=args.shadow_max,
                cirrus_max=args.cirrus_max,
                water_max=args.water_max,
                aero_climatology_max=args.aero_climatology_max,
                aero_low_max=args.aero_low_max,
                aero_medium_max=args.aero_medium_max,
                aero_high_max=args.aero_high_max,
                valid_min=args.valid_min,
            )
        except Exception as e:
            log.warning("%s: scan failed (%s) — skipping", fid, e)
            continue
        needed_dates[fid] = dates
        if n_target_times == 0:
            n_no_targets += 1
        log.info(
            "%s: %d eligible (crop, time) pairs across %d unique target times "
            "/ %d frames → %d needed dates",
            fid,
            n_pairs,
            n_target_times,
            n_time,
            len(dates),
        )

    log.info(
        "Scan summary: %d ROIs scanned, %d with zero eligible targets",
        len(needed_dates),
        n_no_targets,
    )

    # 2. Augment granule_list.json.
    data = json.loads(args.granule_list.read_text(encoding="utf-8"))
    n_written = 0
    n_skipped_unknown = 0
    n_empty = 0
    for gid, entry in data.items():
        fids = entry.get("fids", [])
        unknown = [f for f in fids if f not in needed_dates]
        if unknown:
            # Some ROIs lack eligibility info → leave needed_fids untouched
            # so Phase C falls back to today's full-fids behaviour for safety.
            n_skipped_unknown += 1
            continue
        try:
            g_date_str = granule_id_to_filename_date(gid)
        except Exception as e:
            log.warning("Could not parse date from gid %s: %s — skipping", gid, e)
            continue
        nf = sorted(f for f in fids if g_date_str in needed_dates[f])
        entry["needed_fids"] = nf
        n_written += 1
        if not nf:
            n_empty += 1

    args.granule_list.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info(
        "Wrote needed_fids for %d granules (%d with empty set = will be "
        "skipped in Phase C); %d granules untouched due to ROIs without "
        "zarrs",
        n_written,
        n_empty,
        n_skipped_unknown,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
