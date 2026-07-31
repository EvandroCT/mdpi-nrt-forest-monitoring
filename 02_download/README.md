# HLS acquisition

Reference for `hls_stac_download.py`. The companion `scan_eligibility.py` is documented by
`--help` and by the repository README's **Rebuilding the corpora from source**.

## What it does

Downloads HLS (Harmonized Landsat Sentinel-2) v2.0 satellite imagery from NASA, clips each band to the bounding box of multiple areas of interest (ROIs), and organizes the output per-ROI. It is designed for batch processing thousands of granules on machines with limited disk space.

---

## 1. Per-ROI Granule Discovery

Searches NASA's CMR (Common Metadata Repository) catalog individually for each polygon in the ROI file. For each of the N features and each selected collection (HLSL30, HLSS30), a separate CMR query is issued using the feature's exact polygon as a spatial filter. Results are deduplicated across ROIs — if two ROIs overlap the same MGRS tile, the granule appears once but is mapped to both ROIs. This builds an exact **granule → [list of FID_DETERs]** mapping at search time, eliminating any need to compute or parse bounding boxes from granule metadata.

## 2. Separate Band Selection for L30 and S30

Landsat (L30) and Sentinel-2 (S30) use different band names (e.g., Sentinel has `B8A`, `B11`, `B12` where Landsat has `B05`, `B06`, `B07`). The `--l30-bands` and `--s30-bands` arguments let you specify which bands to download for each sensor independently. Defaults cover the standard VNIR/SWIR + Fmask bands for each.

## 3. Per-Granule Streaming (Disk-Safe)

Instead of downloading all granules first and then clipping, the script processes **one granule at a time**: download selected bands to a temporary staging directory → clip to every associated ROI → delete staging files → move to the next granule. This means disk usage stays roughly constant (one granule's worth of COGs at any time), making it safe to run on machines where storage is limited.

## 4. Smart Resumability

Three levels of skip logic prevent redundant work:

- **Granule-level**: `check_granule_complete()` checks if all expected output files (bands × ROIs) already exist. If so, the entire granule is skipped without downloading anything.
- **Band-level**: Before downloading, the script identifies which bands still have missing clips for at least one ROI. Only those bands are downloaded. If you previously ran with `B02 B03 B04` and now add `B05`, only `B05` gets fetched.
- **File-level**: During clipping, each individual output file is checked — if it exists, the clip is skipped even if the band was downloaded (handles partially-completed granules from interrupted runs).

## 5. Granule List as a Gap-Filling Cache

The script treats `granule_list.json` as a persistent cache rather than a frozen
snapshot. On every run:

1. **Load** — if `--granule-list PATH` is given, that file is loaded; otherwise
   the cache at `<output>/granule_list.json` is loaded if present; otherwise
   the cache starts empty.
2. **Detect coverage** — acquisition dates are inferred from cached granule IDs
   to determine `[cache_min, cache_max]`.
3. **Search gaps** — CMR is queried only for the *uncovered* portions of the
   requested range: the head `[start_date, cache_min − 1d]` and the tail
   `[cache_max + 1d, end_date]`. If neither exists, no CMR call is made.
4. **Merge** — new granules join the cache; existing entries get their FID
   union updated and their URLs refreshed (earthaccess signed URLs expire).
5. **Save** — the cache is **always** written back, even when nothing changed,
   so `runs[]` history accumulates.

**Known limitation:** coverage inference uses only `min`/`max` of cached
dates, so **interior date gaps within `[cache_min, cache_max]` are not
detected**. This is acceptable while caches are built from contiguous
searches; a future version will walk `runs[]` for precise coverage.

## 6. Run Provenance Tracking

Each granule record in `granule_list.json` carries a `runs` array. Every save
appends a new entry for every touched granule containing:

- `timestamp` — when the save happened
- `start_date`, `end_date` — temporal window requested
- `collections` — which collections were queried
- `l30_bands`, `s30_bands` — which bands were requested

Older files predating this feature (entries with only `fids` + `urls`) are
automatically upgraded on the first save — a `runs[]` list is initialised and
the current run is appended.

## 7. Date-Range Filtering at Processing Time

Before processing begins, every granule is checked against the current `--start-date` / `--end-date` window. Granules whose acquisition date falls outside the requested range are excluded — no downloads, no file checks. This applies regardless of whether granules came from a fresh CMR search or a loaded `--granule-list`. It prevents a common pitfall: if a previous run searched a wide date range but only partially downloaded, and a new run targets a narrower subset, the leftover out-of-range granules won't be touched.

## 8. Parallel Granule Processing

By default, granules are processed one at a time (download → clip → cleanup). The `--parallel N` argument allows processing N granules concurrently, overlapping network downloads across granules to maximize throughput. Each parallel worker gets its own staging subdirectory to avoid file collisions. Clipping calls are serialized via a thread lock (conservative; clipping is not a bottleneck since clipping (disk I/O) is fast compared to downloads (network I/O)). The `--staging` argument lets you choose where temporary files are stored (defaults to the system tmpdir, typically `/tmp`), which is important when processing many granules in parallel.

Note: with `--parallel 4 --threads 8`, up to 32 concurrent download connections hit NASA's servers. Reduce `--threads` when increasing `--parallel` to avoid rate-limiting (e.g., `--parallel 4 --threads 4`).

## 9. Dry-Run Mode

`--dry-run` performs the full CMR search, saves `granule_list.json`, prints a summary (L30/S30 counts, MGRS tiles, total band files, granule-ROI pairs), and exits without downloading anything. Useful for planning and verifying search parameters before committing to a long download.

## 10. Retry Logic with Exponential Backoff

Three network operations have retry protection:

- **CMR search**: 5 attempts per ROI/collection query, backoff 1s → 2s → 4s → 8s → 16s. If all fail, that ROI/collection pair is skipped and the script continues.
- **Download**: 5 attempts per granule, same backoff. If all fail, the granule is skipped.
- **Authentication**: 3 attempts with a **30-second timeout** each (via a daemon thread). Prevents the script from hanging indefinitely when NASA's auth endpoint is unreachable.

## 11. Output Layout

```
output/
├── granule_list.json          # persistent granule index
├── hls_download.log           # full log (appended across runs)
├── DR_1234/                   # one directory per FID_DETER
│   ├── 20220115T135716_HLSL30_B02.tif
│   ├── 20220115T135716_HLSL30_B03.tif
│   ├── 20220315T104019_HLSS30_B8A.tif
│   └── ...
├── DR_5678/
│   └── ...
```

Date format is `YYYYMMDDTHHMMSS` (converted from HLS day-of-year format). Each file is a LZW-compressed, tiled GeoTIFF clipped to the ROI's geographic bounding box.

## 12. CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--roi` | *(required)* | Path to ROI vector file (GeoJSON, GeoPackage, etc.) |
| `--output` | *(required)* | Root output directory |
| `--start-date` | `2024-10-01` | Temporal filter start |
| `--end-date` | `2026-05-13` | Temporal filter end |
| `--l30-bands` | `B02 B03 B04 B05 B06 B07 Fmask` | Landsat bands to download |
| `--s30-bands` | `B02 B03 B04 B05 B06 B07 B8A B11 B12 Fmask` | Sentinel-2 bands to download |
| `--collections` | `HLSL30 HLSS30` | Which collections to search |
| `--threads` | `8` | Parallel download threads per granule |
| `--parallel` | `1` | Number of granules to process concurrently |
| `--staging` | system tmpdir | Directory for temporary download files |
| `--granule-list` | *(none)* | Pre-load a saved granule JSON as cache; CMR is still queried for date gaps |
| `--dry-run` | off | Search + save list only, no download |
| `--debug` | off | Enable DEBUG-level logging |

## Dependencies

- `earthaccess` — NASA CMR search and authenticated downloads
- `geopandas` — ROI loading and CRS reprojection
- `rasterio` — windowed raster clipping (GDAL statically linked in the wheel)
- `shapely` — polygon orientation for CMR queries
