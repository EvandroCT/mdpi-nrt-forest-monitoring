#!/usr/bin/env python3
# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""
hls_stac_download.py — Download and clip HLS v2.0 band files via NASA earthaccess.

Searches NASA's CMR per ROI polygon, building an exact granule-to-ROI mapping.
For each granule: downloads selected band COGs, clips them to every associated
ROI bounding box, and deletes the staging files before moving to the next.

Usage:
    python hls_stac_download.py --roi ROI.geojson --output /data/hls --dry-run
    python hls_stac_download.py --roi ROI.geojson --output /data/hls \
        --start-date 2024-10-01 --end-date 2026-05-13 \
        --l30-bands B02 B03 B04 B05 B06 B07 Fmask \
        --s30-bands B02 B03 B04 B05 B06 B07 B8A B11 B12 Fmask \
        --threads 16

Requirements:
    - NASA Earthdata Login account (https://urs.earthdata.nasa.gov/)
    - earthaccess, geopandas, rasterio (see pyproject.toml)
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import geopandas as gpd
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from shapely.geometry import box as shapely_box
from shapely.geometry.polygon import orient

_clip_lock = threading.Lock()

COLLECTIONS = {
    "HLSL30": {"short_name": "HLSL30"},
    "HLSS30": {"short_name": "HLSS30"},
}

DEFAULT_L30_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]
DEFAULT_S30_BANDS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "Fmask",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(log_file: str, debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_roi(path: str) -> gpd.GeoDataFrame:
    """Load ROI features, reproject to EPSG:4326 if needed."""
    gdf = gpd.read_file(path)
    logging.info(f"Loaded {len(gdf)} features from {path}")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        logging.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def parse_hls_filename(filename: str) -> dict:
    """Parse an HLS filename into its components.

    Example: HLS.L30.T20LKP.2022015T135716.v2.0.B02.tif
    """
    parts = filename.split(".")
    sensor = parts[1]  # L30 or S30
    tile = parts[2]  # T20LKP
    raw_dt = parts[3]  # 2022015T135716
    band = parts[6]  # B02, Fmask, etc.

    date_part, time_part = raw_dt.split("T")
    dt = datetime.strptime(date_part, "%Y%j")
    date_str = dt.strftime("%Y%m%d") + "T" + time_part

    collection = f"HLS{sensor}"  # HLSL30 or HLSS30
    return {
        "collection": collection,
        "tile": tile,
        "date": date_str,
        "band": band,
        "sensor": sensor,
    }


def parse_granule_date(gid: str) -> datetime:
    """Extract the acquisition date from a granule ID.

    gid format: L30.T20LKP.2022015T135716
    """
    raw_dt = gid.split(".")[2]  # 2022015T135716
    date_part = raw_dt.split("T")[0]  # 2022015
    return datetime.strptime(date_part, "%Y%j")


def get_granule_id(granule) -> str:
    """Build a short identifier for logging."""
    links = granule.data_links()
    if links:
        fname = links[0].split("/")[-1].split("?")[0]
        parts = fname.split(".")
        if len(parts) >= 4:
            return f"{parts[1]}.{parts[2]}.{parts[3]}"
    return "unknown"


# Column name we always use internally for the cluster id, regardless of the
# user's `--fid-field`. Lets the rest of the code treat the cluster table as a
# regular ROI GeoDataFrame.
_CLUSTER_FID_COL = "fid"


def _synth_id_for_index(n: int, total: int) -> str:
    width = max(4, len(str(total)))
    return f"roi_{n:0{width}d}"


def _cluster_id_from_members(member_ids: list[str]) -> str:
    """Build a cluster id from sorted member base ids. Hash if too long."""
    member_ids = sorted(member_ids)
    if len(member_ids) == 1:
        return member_ids[0]
    joined = "+".join(member_ids)
    if len(joined) <= 200:
        return joined
    digest = hashlib.sha256(joined.encode()).hexdigest()[:16]
    return f"cluster_{digest}"


def _bboxes_overlap(a: tuple, b: tuple) -> bool:
    """True iff two (minx, miny, maxx, maxy) bboxes share any area or edge."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_find_clusters(bboxes: list[tuple]) -> list[list[int]]:
    """Group bbox indices by transitive overlap (connected components)."""
    n = len(bboxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _bboxes_overlap(bboxes[i], bboxes[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def build_cluster_gdf(
    roi_gdf: gpd.GeoDataFrame,
    fid_field: str | None,
    extended_bbox_pct: float,
    fuse_overlaps: bool,
) -> tuple[gpd.GeoDataFrame, dict[str, dict]]:
    """Project ROIs into a cluster table whose rows feed the rest of the pipeline.

    Returns `(cluster_gdf, clusters_meta)`:
      - `cluster_gdf`: one row per cluster with columns `fid` (cluster id),
        `geometry` (shapely.box of the cluster's effective bbox), and
        `members` (list of base ids).
      - `clusters_meta`: `{cluster_id: {"members": [...], "bbox": [minx, miny,
        maxx, maxy]}}` for the side-car `clusters.json`.

    Semantics
    ---------
    * The per-ROI **effective bbox** is `extend_bbox(raw_bounds, pct)` when
      `extended_bbox_pct > 0`, else `raw_bounds`. This is what fusion checks
      for overlap, and what downstream clipping uses.
    * `fuse_overlaps=True` groups ROIs into connected components by overlap of
      effective bboxes; the cluster bbox is the bounds-union of its members'
      effective bboxes.
    * `fid_field is None` → synthetic base ids `roi_NNNN` ordered by centroid
      (x, then y) for stable IDs across re-runs.
    """
    n = len(roi_gdf)
    # Effective per-ROI bbox (the bbox we'd actually clip to).
    eff_bboxes: list[tuple] = []
    for _, row in roi_gdf.iterrows():
        b = row.geometry.bounds
        if extended_bbox_pct > 0:
            b = extend_bbox(b, extended_bbox_pct)
        eff_bboxes.append(b)

    # Per-ROI base id.
    if fid_field is not None:
        base_ids = [str(v) for v in roi_gdf[fid_field].tolist()]
    else:
        # Stable ordering by centroid so re-running on the same vector file
        # always produces the same synthetic ids.
        centroids = [
            (row.geometry.centroid.x, row.geometry.centroid.y)
            for _, row in roi_gdf.iterrows()
        ]
        order = sorted(range(n), key=lambda i: centroids[i])
        base_ids = [""] * n
        for rank, i in enumerate(order):
            base_ids[i] = _synth_id_for_index(rank, n)

    # Cluster: connected components if fuse-overlaps, else 1-1.
    if fuse_overlaps:
        groups = _union_find_clusters(eff_bboxes)
    else:
        groups = [[i] for i in range(n)]

    rows: list[dict] = []
    clusters_meta: dict[str, dict] = {}
    for member_idx in groups:
        member_ids = [base_ids[i] for i in member_idx]
        cluster_id = _cluster_id_from_members(member_ids)
        member_bboxes = [eff_bboxes[i] for i in member_idx]
        minx = min(b[0] for b in member_bboxes)
        miny = min(b[1] for b in member_bboxes)
        maxx = max(b[2] for b in member_bboxes)
        maxy = max(b[3] for b in member_bboxes)
        # Square the merged bbox at 0% extra growth so clusters always end up
        # as squares — single-ROI clusters are already square (the input bbox
        # passed through extend_bbox), but the bounds-union of multiple square
        # boxes is generally a rectangle.
        merged = extend_bbox((minx, miny, maxx, maxy), 0.0)
        rows.append(
            {
                _CLUSTER_FID_COL: cluster_id,
                "geometry": shapely_box(*merged),
                "members": sorted(member_ids),
            }
        )
        clusters_meta[cluster_id] = {
            "members": sorted(member_ids),
            "bbox": list(merged),
        }

    cluster_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=roi_gdf.crs)
    return cluster_gdf, clusters_meta


def extend_bbox(bounds: tuple, pct: float) -> tuple:
    """Expand a (minx, miny, maxx, maxy) bbox into a square with a margin.

    The output side equals `max(width, height) * (1 + pct/100)`, centered on
    the input bbox's centroid. `pct=0` still squares the bbox (side becomes
    the larger original dimension). Use `pct=0` cautiously — pass 0 only when
    you want raw bounds (the caller handles that by skipping this helper).
    """
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    side = max(maxx - minx, maxy - miny) * (1.0 + pct / 100.0)
    half = side / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def clip_band(src_path: Path, out_path: Path, bbox: tuple) -> None:
    """Clip a raster to a geographic bounding box.

    bbox: (minx, miny, maxx, maxy) in EPSG:4326.

    Replicates `gdal_translate -projwin` semantics (verified byte-identical
    against GDAL 3.8.5): corner-point reprojection of the bbox into the
    source CRS, floor(+1e-3) pixel offsets, half-up-rounded window size,
    and boundless reads filled with the source nodata (0 when the source
    declares none) for windows extending past the raster edge.
    """
    xmin, ymin, xmax, ymax = bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _clip_lock:
        with rasterio.open(src_path) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [xmin, xmax], [ymax, ymin])
            (ulx, lrx), (uly, lry) = xs, ys
            gt = src.transform
            col_off = math.floor((ulx - gt.c) / gt.a + 0.001)
            row_off = math.floor((uly - gt.f) / gt.e + 0.001)
            width = math.floor((lrx - ulx) / gt.a + 0.5)
            height = math.floor((lry - uly) / gt.e + 0.5)
            window = Window(col_off, row_off, width, height)
            fill = src.nodata if src.nodata is not None else 0
            data = src.read(window=window, boundless=True, fill_value=fill)
            profile = src.profile
            profile.update(
                driver="GTiff",
                width=width,
                height=height,
                transform=src.window_transform(window),
                compress="lzw",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)


# ---------------------------------------------------------------------------
# Search: per-ROI with deduplication
# ---------------------------------------------------------------------------


def _cmr_search_one(
    fid: str,
    coords: list[tuple[float, float]],
    collections: list[str],
    start_date: str,
    end_date: str,
) -> tuple[str, dict[str, object]]:
    """Worker: search CMR for one ROI across all collections.

    Returns (fid, {granule_id: DataGranule}). Per-collection retries are kept
    bounded so a single flaky ROI can't stall the whole pool.
    """
    found: dict[str, object] = {}
    for key in collections:
        coll = COLLECTIONS[key]
        results = None
        for attempt in range(5):
            try:
                results = earthaccess.search_data(
                    short_name=coll["short_name"],
                    temporal=(start_date, end_date),
                    polygon=coords,
                    count=-1,
                )
                break
            except Exception as e:
                wait = 2**attempt
                logging.warning(
                    f"Search failed for {fid}/{key} (attempt {attempt + 1}): "
                    f"{e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
        if results is None:
            logging.error(
                f"Search failed for {fid}/{key} after 5 attempts, skipping"
            )
            continue
        for g in results:
            gid = get_granule_id(g)
            if gid not in found:
                found[gid] = g
    return fid, found


def search_per_roi(
    collections: list[str],
    roi_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    fid_field: str,
    search_workers: int = 1,
) -> tuple[list, dict[str, list[str]]]:
    """Search CMR once per ROI, deduplicate granules, build granule→FID mapping.

    `search_workers > 1` runs the per-ROI CMR queries in a thread pool. The
    NASA CMR endpoint is happy with parallel reads; the search step is purely
    I/O-bound so threads (not processes) win. The cross-ROI dedup/aggregation
    is done on the main thread as workers complete to avoid lock contention.

    Returns:
        granules: deduplicated list of DataGranule objects
        granule_rois: dict mapping granule_id → list of fid values
    """
    granule_index: dict[str, object] = {}
    granule_rois: dict[str, list[str]] = {}

    work: list[tuple[str, list, list[str], str, str]] = []
    for _, row in roi_gdf.iterrows():
        fid = row[fid_field]
        poly = orient(row.geometry, sign=1.0)
        coords = [(x, y) for x, y, *_ in poly.exterior.coords]
        work.append((fid, coords, collections, start_date, end_date))

    n_total = len(work)
    log_step = max(1, n_total // 50)  # ~50 progress lines total
    completed = 0

    def _drain(result: tuple[str, dict[str, object]]) -> None:
        nonlocal completed
        fid, found = result
        for gid, g in found.items():
            if gid not in granule_index:
                granule_index[gid] = g
                granule_rois[gid] = []
            granule_rois[gid].append(fid)
        completed += 1
        if completed % log_step == 0 or completed == n_total:
            logging.info(
                f"[CMR {completed}/{n_total}] {fid}: "
                f"{len(granule_index)} unique granules so far"
            )

    if search_workers <= 1:
        for item in work:
            _drain(_cmr_search_one(*item))
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=search_workers
        ) as pool:
            futures = [pool.submit(_cmr_search_one, *item) for item in work]
            for fut in concurrent.futures.as_completed(futures):
                _drain(fut.result())

    granules = list(granule_index.values())
    logging.info(f"Search complete: {len(granules)} unique granules")
    return granules, granule_rois


def save_granule_list(
    granule_urls: dict[str, list[str]],
    granule_rois: dict[str, list[str]],
    path: Path,
    run_info: dict,
) -> None:
    """Save granule URLs and ROI mapping to a JSON file, merging with existing.

    Every save appends `run_info` to each touched granule's `runs[]` history
    so future invocations can infer coverage. Entries missing `runs[]` (from
    older script versions) get one initialised on first write.
    """
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        logging.info(f"Merging with existing {len(existing)} granules in {path}")
    else:
        existing = {}

    # Ensure every pre-existing entry has a runs[] list (upgrade old files).
    for entry in existing.values():
        entry.setdefault("runs", [])

    for gid, urls in granule_urls.items():
        new_fids = granule_rois.get(gid, [])
        if gid in existing:
            existing[gid]["fids"] = sorted(
                set(existing[gid].get("fids", [])) | set(new_fids)
            )
            # Keep URLs fresh (earthaccess may re-sign/rotate them)
            existing[gid]["urls"] = urls
            # `needed_fids` (Phase B output) is preserved verbatim; we only
            # touch it via scan_eligibility.py.
        else:
            existing[gid] = {"fids": sorted(set(new_fids)), "urls": urls, "runs": []}
        existing[gid]["runs"].append(run_info)

    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logging.info(f"Saved {len(existing)} granules to {path}")


def determine_search_gaps(
    granule_urls: dict[str, list[str]],
    dt_start: datetime,
    dt_end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Return date sub-ranges not yet covered by the cached granules.

    Coverage is inferred from granule IDs (each embeds its acquisition date).
    If the cache is empty, the whole requested range is returned. Otherwise
    the uncovered head `[dt_start, cache_min - 1d]` and tail
    `[cache_max + 1d, dt_end]` are returned (either/both may be empty).

    Blind spot: gaps *inside* `[cache_min, cache_max]` are not detected.
    Acceptable here because caches are built from contiguous searches, and
    newly added `runs[]` history will make this precise over time.
    """
    if not granule_urls:
        return [(dt_start, dt_end)]

    cached_dates = []
    for gid in granule_urls:
        try:
            cached_dates.append(parse_granule_date(gid))
        except (IndexError, ValueError):
            continue
    if not cached_dates:
        return [(dt_start, dt_end)]

    cache_min, cache_max = min(cached_dates), max(cached_dates)
    gaps: list[tuple[datetime, datetime]] = []
    if dt_start < cache_min:
        head_end = min(cache_min - timedelta(days=1), dt_end)
        if dt_start <= head_end:
            gaps.append((dt_start, head_end))
    if dt_end > cache_max:
        tail_start = max(cache_max + timedelta(days=1), dt_start)
        if tail_start <= dt_end:
            gaps.append((tail_start, dt_end))
    return gaps


def load_granule_list(
    path: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str] | None]]:
    """Load granule URLs and ROI mappings from a JSON file.

    Returns:
        granule_urls: dict mapping granule_id → list of URLs
        granule_rois: dict mapping granule_id → list of fid values
        granule_needed_fids: dict mapping granule_id → list of fid values that
            actually need this granule's date for their training lookback
            window. `None` when the entry has no `needed_fids` key (i.e. the
            scanner hasn't been run yet) — callers should fall back to
            `granule_rois[gid]` in that case.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    granule_urls = {gid: entry["urls"] for gid, entry in data.items()}
    granule_rois = {gid: entry["fids"] for gid, entry in data.items()}
    granule_needed_fids = {gid: entry.get("needed_fids") for gid, entry in data.items()}
    logging.info(f"Loaded {len(data)} granules from {path}")
    return granule_urls, granule_rois, granule_needed_fids


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def check_granule_complete(
    fids: list[str],
    bands: list[str],
    collection: str,
    date_str: str,
    output_dir: Path,
) -> bool:
    """Return True if all clips for this granule already exist."""
    for fid in fids:
        fid_dir = output_dir / fid
        for band in bands:
            path = fid_dir / f"{date_str}_{collection}_{band}.tif"
            if not path.exists():
                return False
    return True


def process_granule(
    gid: str,
    all_urls: list[str],
    fids: list[str],
    roi_gdf: gpd.GeoDataFrame,
    l30_bands: list[str],
    s30_bands: list[str],
    output_dir: Path,
    staging_dir: str,
    threads: int,
    fid_field: str,
    extended_bbox_pct: float,
    needed_fids: list[str] | None,
    granule_idx: int,
    granule_total: int,
) -> dict:
    """Download, clip, and clean up one granule. Returns stats dict."""
    stats = {"skipped": False, "clipped": 0, "error": None}
    prefix = f"[{granule_idx}/{granule_total}]"

    # --- Restrict to ROIs that actually need this granule's date ---
    if needed_fids is not None:
        effective_fids = sorted(set(fids) & set(needed_fids))
        if not effective_fids:
            logging.info(f"{prefix} {gid}: empty needed_fids — skipping")
            stats["skipped"] = True
            return stats
        fids = effective_fids

    # --- Determine collection + bands from first URL ---
    if not all_urls:
        logging.warning(f"{prefix} {gid}: no data links, skipping")
        stats["error"] = "no_links"
        return stats

    sample_fname = all_urls[0].split("/")[-1].split("?")[0]
    info = parse_hls_filename(sample_fname)
    collection = info["collection"]
    date_str = info["date"]
    bands = l30_bands if info["sensor"] == "L30" else s30_bands

    # --- Resumability ---
    if check_granule_complete(fids, bands, collection, date_str, output_dir):
        logging.info(f"{prefix} {gid}: all clips exist, skipping")
        stats["skipped"] = True
        return stats

    # --- Filter out bands already fully clipped for all ROIs ---
    needed_bands = []
    for b in bands:
        for fid in fids:
            path = output_dir / fid / f"{date_str}_{collection}_{b}.tif"
            if not path.exists():
                needed_bands.append(b)
                break

    if not needed_bands:
        logging.info(f"{prefix} {gid}: all clips exist, skipping")
        stats["skipped"] = True
        return stats

    # --- Download only needed bands to staging ---
    patterns = [f".{b}." for b in needed_bands]
    band_urls = [u for u in all_urls if any(p in u for p in patterns)]
    if not band_urls:
        logging.warning(f"{prefix} {gid}: no matching band URLs")
        stats["error"] = "no_band_urls"
        return stats

    logging.info(
        f"{prefix} {gid}: downloading {len(band_urls)} bands, "
        f"clipping to {len(fids)} ROIs"
    )

    downloaded = None
    for attempt in range(5):
        try:
            downloaded = earthaccess.download(
                band_urls, local_path=staging_dir, threads=threads
            )
            break
        except Exception as e:
            wait = 2**attempt
            logging.warning(
                f"{prefix} {gid}: download failed (attempt {attempt + 1}): "
                f"{e}. Retrying in {wait}s..."
            )
            time.sleep(wait)
    if downloaded is None:
        logging.error(f"{prefix} {gid}: download failed after 5 attempts, skipping")
        stats["error"] = "download_failed"
        return stats

    # --- Clip to each ROI ---
    roi_rows = roi_gdf[roi_gdf[fid_field].isin(fids)]

    for dl_path in downloaded:
        dl_path = Path(dl_path)
        try:
            finfo = parse_hls_filename(dl_path.name)
        except (IndexError, ValueError):
            logging.warning(f"{prefix} Could not parse filename: {dl_path.name}")
            continue

        for _, row in roi_rows.iterrows():
            fid = row[fid_field]
            bbox = row.geometry.bounds  # (minx, miny, maxx, maxy)
            if extended_bbox_pct > 0.0:
                bbox = extend_bbox(bbox, extended_bbox_pct)
            out_path = (
                output_dir
                / fid
                / f"{finfo['date']}_{finfo['collection']}_{finfo['band']}.tif"
            )
            if out_path.exists():
                continue
            try:
                clip_band(dl_path, out_path, bbox)
                stats["clipped"] += 1
            except Exception as e:
                logging.error(f"{prefix} Clip failed {out_path.name}: {e}")

    # --- Cleanup staging ---
    for dl_path in downloaded:
        Path(dl_path).unlink(missing_ok=True)

    logging.info(f"{prefix} {gid}: {stats['clipped']} clips produced")
    return stats


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary_from_urls(
    granule_urls: dict[str, list[str]],
    granule_rois: dict[str, list[str]],
    l30_bands: list[str],
    s30_bands: list[str],
) -> None:
    """Print search summary from URL index."""
    tiles = set()
    n_l30 = 0
    n_s30 = 0
    for gid in granule_urls:
        # gid format: L30.T20LKP.2022015T135716
        parts = gid.split(".")
        if len(parts) >= 2:
            tiles.add(parts[1])
        if parts[0] == "L30":
            n_l30 += 1
        elif parts[0] == "S30":
            n_s30 += 1

    total_bands = n_l30 * len(l30_bands) + n_s30 * len(s30_bands)
    total_roi_pairs = sum(len(fids) for fids in granule_rois.values())

    print(f"\n{'=' * 60}")
    print(f"  L30 granules:      {n_l30}  ({len(l30_bands)} bands each)")
    print(f"  S30 granules:      {n_s30}  ({len(s30_bands)} bands each)")
    print(f"  MGRS tiles:        {len(tiles)}")
    print(f"  Band files total:  {total_bands}")
    print(f"  Granule-ROI pairs: {total_roi_pairs}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download and clip HLS v2.0 band files via NASA earthaccess.",
    )
    parser.add_argument(
        "--roi",
        required=True,
        help="Path to ROI vector file (GeoJSON, GeoPackage, etc.)",
    )
    parser.add_argument(
        "--output", required=True, help="Root output directory for clipped band files"
    )
    parser.add_argument(
        "--start-date",
        default="2024-10-01",
        help="Start date YYYY-MM-DD (default: 2024-10-01, the retrieval start used throughout)",
    )
    parser.add_argument(
        "--end-date",
        default="2026-05-13",
        help="End date YYYY-MM-DD (default: 2026-05-13, the retrieval end used throughout)",
    )
    parser.add_argument(
        "--l30-bands",
        nargs="+",
        default=DEFAULT_L30_BANDS,
        help=f"L30 bands (default: {' '.join(DEFAULT_L30_BANDS)})",
    )
    parser.add_argument(
        "--s30-bands",
        nargs="+",
        default=DEFAULT_S30_BANDS,
        help=f"S30 bands (default: {' '.join(DEFAULT_S30_BANDS)})",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["HLSL30", "HLSS30"],
        choices=["HLSL30", "HLSS30"],
        help="HLS collections (default: both)",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Parallel download threads (default: 8)"
    )
    parser.add_argument(
        "--staging",
        help="Directory for temporary download files (default: system tmpdir)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of granules to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--granule-list",
        help="Path to a granule list JSON from a previous search (skips CMR search)",
    )
    parser.add_argument(
        "--fid-field",
        default=None,
        help="ROI feature ID column name. If omitted, synthetic ids "
        "`roi_NNNN` are generated (sorted by centroid for stability).",
    )
    parser.add_argument(
        "--extended-bbox",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Square-extend each ROI's bbox by this percent of its larger "
        "dimension before clipping (default: 0 = exact ROI bounds)",
    )
    parser.add_argument(
        "--fuse-overlaps",
        action="store_true",
        help="Merge ROIs whose (extended) bboxes overlap into a single "
        "cluster with id `<idA>+<idB>+...` and bounds-union bbox. "
        "Connected-component grouping (A-B-C chains all merge).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search and report stats without downloading",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir / "hls_download.log"), args.debug)

    logging.info("=== HLS STAC Download ===")
    logging.info(f"ROI: {args.roi}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Period: {args.start_date} to {args.end_date}")
    logging.info(f"L30 bands: {args.l30_bands}")
    logging.info(f"S30 bands: {args.s30_bands}")
    logging.info(f"Collections: {args.collections}")
    logging.info(f"Threads: {args.threads}")
    logging.info(f"Parallel granules: {args.parallel}")
    logging.info(
        f"FID field: {args.fid_field or '<synthetic roi_NNNN ids>'}"
    )
    if args.extended_bbox > 0:
        logging.info(f"Extended bbox: +{args.extended_bbox}% (squared)")
    if args.fuse_overlaps:
        logging.info("Fuse-overlaps: clusters of ROIs with overlapping bboxes")

    # 1. Load ROI and project to a cluster table.
    roi_gdf = load_roi(args.roi)
    if args.fid_field is not None and args.fid_field not in roi_gdf.columns:
        logging.error(
            f"--fid-field '{args.fid_field}' not found in ROI columns: "
            f"{list(roi_gdf.columns)}"
        )
        sys.exit(1)

    cluster_gdf, clusters_meta = build_cluster_gdf(
        roi_gdf,
        fid_field=args.fid_field,
        extended_bbox_pct=args.extended_bbox,
        fuse_overlaps=args.fuse_overlaps,
    )
    n_in = len(roi_gdf)
    n_out = len(cluster_gdf)
    n_fused = sum(1 for m in clusters_meta.values() if len(m["members"]) > 1)
    logging.info(
        f"Clusters: {n_out} (from {n_in} ROIs; {n_fused} multi-member)"
    )

    # Write clusters.json side-car for downstream traceability. Must happen
    # before any granule processing so a kill mid-run still leaves the map.
    (output_dir / "clusters.json").write_text(json.dumps(clusters_meta, indent=2), encoding="utf-8")
    logging.info(f"Saved clusters.json with {len(clusters_meta)} entries")

    # Cluster geometries already encode the (extended, merged) bbox we want
    # at clip time, so set the downstream effective field to the synthetic
    # column and the extension to 0 to avoid double-extending.
    roi_gdf = cluster_gdf
    effective_fid_field = _CLUSTER_FID_COL
    effective_extended_bbox = 0.0

    # 2. Authenticate (with timeout to avoid hanging on flaky networks)
    auth = None
    for attempt in range(3):
        result = [None]
        exc = [None]

        def _login():
            try:
                result[0] = earthaccess.login(strategy="all", persist=True)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_login, daemon=True)
        t.start()
        t.join(timeout=30)

        if t.is_alive():
            wait = 2**attempt
            logging.warning(
                f"Login timed out (attempt {attempt + 1}/3). Retrying in {wait}s..."
            )
            time.sleep(wait)
            continue
        if exc[0]:
            wait = 2**attempt
            logging.warning(
                f"Login failed (attempt {attempt + 1}/3): {exc[0]}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)
            continue
        auth = result[0]
        break

    if auth is None or not auth.authenticated:
        logging.error(
            "Authentication failed. Create an account at "
            "https://urs.earthdata.nasa.gov/ then run: "
            "earthaccess.login(strategy='interactive', persist=True)"
        )
        sys.exit(1)
    logging.info("Authenticated with NASA Earthdata")

    # 3. Load cache (if any) + CMR-search any uncovered gaps, then save.
    cache_path = (
        Path(args.granule_list) if args.granule_list
        else output_dir / "granule_list.json"
    )
    if cache_path.exists():
        granule_urls, granule_rois, granule_needed_fids = load_granule_list(cache_path)
    else:
        granule_urls, granule_rois, granule_needed_fids = {}, {}, {}

    dt_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Determine which date sub-ranges still need a CMR search by looking at
    # cached granule dates. Empty cache → search the whole requested range.
    search_ranges = determine_search_gaps(granule_urls, dt_start, dt_end)

    if not search_ranges:
        logging.info("Cached granule list already covers requested date range — skipping CMR search")
    for gap_start, gap_end in search_ranges:
        logging.info(f"CMR search for gap: {gap_start.date()} to {gap_end.date()}")
        granules, new_rois = search_per_roi(
            args.collections,
            roi_gdf,
            gap_start.strftime("%Y-%m-%d"),
            gap_end.strftime("%Y-%m-%d"),
            effective_fid_field,
            search_workers=max(1, args.parallel),
        )
        for g in granules:
            gid = get_granule_id(g)
            links = g.data_links()
            if not links:
                continue
            granule_urls[gid] = links
            granule_rois[gid] = sorted(
                set(granule_rois.get(gid, [])) | set(new_rois.get(gid, []))
            )

    if not granule_urls:
        logging.warning("No granules available. Check ROI, dates, and collections.")
        sys.exit(0)

    # Always persist the (possibly unchanged) list so runs[] accumulates history.
    run_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "collections": args.collections,
        "l30_bands": args.l30_bands,
        "s30_bands": args.s30_bands,
    }
    save_granule_list(
        granule_urls, granule_rois, output_dir / "granule_list.json", run_info
    )

    # Filter granules to current date range
    filtered_gids = []
    for gid in granule_urls:
        try:
            dt = parse_granule_date(gid)
            if dt_start <= dt <= dt_end:
                filtered_gids.append(gid)
        except (IndexError, ValueError):
            filtered_gids.append(gid)  # keep unparseable ones to be safe

    if len(filtered_gids) < len(granule_urls):
        logging.info(
            f"Date filter: {len(filtered_gids)}/{len(granule_urls)} granules "
            f"within {args.start_date} to {args.end_date}"
        )
    granule_urls = {gid: granule_urls[gid] for gid in filtered_gids}

    print_summary_from_urls(granule_urls, granule_rois, args.l30_bands, args.s30_bands)

    if args.dry_run:
        logging.info("Dry run — exiting before download")
        sys.exit(0)

    # 4. Per-granule processing (sequential or parallel)
    total = len(granule_urls)
    total_clipped = 0
    total_skipped = 0

    # Build work items
    work = []
    for i, (gid, urls) in enumerate(granule_urls.items(), 1):
        fids = granule_rois.get(gid, [])
        if fids:
            work.append((i, gid, urls, fids))

    staging_base = args.staging if args.staging else None
    if staging_base:
        Path(staging_base).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hls_staging_", dir=staging_base) as staging_root:
        if args.parallel <= 1:
            # Sequential — single staging dir
            for i, gid, urls, fids in work:
                stats = process_granule(
                    gid, urls, fids, roi_gdf,
                    args.l30_bands, args.s30_bands,
                    output_dir, staging_root, args.threads,
                    effective_fid_field, effective_extended_bbox,
                    granule_needed_fids.get(gid),
                    granule_idx=i, granule_total=total,
                )
                total_clipped += stats["clipped"]
                if stats["skipped"]:
                    total_skipped += 1
        else:
            # Parallel — each worker gets its own staging subdirectory
            def _worker(item):
                idx, gid, urls, fids = item
                worker_staging = Path(staging_root) / f"worker_{gid}"
                worker_staging.mkdir(exist_ok=True)
                return process_granule(
                    gid, urls, fids, roi_gdf,
                    args.l30_bands, args.s30_bands,
                    output_dir, str(worker_staging), args.threads,
                    effective_fid_field, effective_extended_bbox,
                    granule_needed_fids.get(gid),
                    granule_idx=idx, granule_total=total,
                )

            logging.info(f"Processing {len(work)} granules with {args.parallel} workers")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel
            ) as pool:
                for stats in pool.map(_worker, work):
                    total_clipped += stats["clipped"]
                    if stats["skipped"]:
                        total_skipped += 1

    logging.info(
        f"=== Done: {total_clipped} clips produced, "
        f"{total_skipped} granules skipped (already complete) ==="
    )


if __name__ == "__main__":
    main()
