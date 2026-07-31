#!/usr/bin/env python3
# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""build_roi_candidates.py — build the selective-logging ROI candidate set.

Filters a DETER alert export down to the selective-logging classes
(CS_GEOMETRICO + CS_DESORDENADO), builds a 1920 m x 1920 m bounding box
centred on each alert's centroid (EPSG:4674, metric offsets via pyproj Geod
so each box is ~1920 m on the ground regardless of latitude), and writes out
the ROI candidate file that seeds the download step (02_download).

Usage:
    python build_roi_candidates.py --deter-export /path/to/deter_export.gpkg \
        --output-dir ./data/rois

The DETER export itself is not bundled here — obtain it from TerraBrasilis
(https://terrabrasilis.dpi.inpe.br/), filtering by view date as described in
the paper (Sec. 3.1); a single "view dates 1 January-1 May 2026" export
matches the corpus used throughout.

Outputs (under --output-dir):
    rois.gpkg          One feature per ROI candidate, 1920x1920 m bbox geometry.
    windows.csv         roi_id, window_start, window_end -- informational only,
                        not consumed by any downstream step (the eligibility
                        scan in 02_download computes coverage from the zarr
                        itself, not from this file).
    roi_ids.txt         One roi_id per line.
"""
import argparse
import csv
from pathlib import Path

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import box

HALF_M = 960.0  # half of 1920 m


def metric_halfdegs(lat: float, lon: float, half_m: float = HALF_M):
    g = Geod(ellps="WGS84")
    _, lat_n, _ = g.fwd(lon, lat, 0.0, half_m)    # north
    lon_e, _, _ = g.fwd(lon, lat, 90.0, half_m)   # east
    return abs(lat_n - lat), abs(lon_e - lon)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--deter-export", type=Path, required=True,
                     help="Path to a DETER alert export (GeoPackage), from TerraBrasilis")
    ap.add_argument("--output-dir", type=Path, default=Path("./data/rois"),
                     help="Output directory (default: ./data/rois)")
    ap.add_argument("--window-start", default="2024-10-01",
                     help="Informational window start (default: 2024-10-01, "
                          "the retrieval start used throughout)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_gpkg = args.output_dir / "rois.gpkg"
    out_windows = args.output_dir / "windows.csv"
    out_ids = args.output_dir / "roi_ids.txt"

    g = gpd.read_file(args.deter_export)
    cs = g[g["CLASSNAME"].astype(str).str.startswith("CS_")].copy()
    print(f"source CS_ rows (selective logging): {len(cs)}")

    # 'FID' is a reserved column name in GeoPackage -- rename early. The
    # source FID is already in `<num>_curr`/`<num>_hist` form, so it doubles
    # as roi_id (kept as the DETER FID throughout, per project convention).
    cs = cs.rename(columns={"FID": "FID_DETER"})
    cs["roi_id"] = cs["FID_DETER"].astype(str)
    cs = cs.reset_index(drop=True)

    # Build 1920 m x 1920 m bboxes centred on each centroid.
    centroids = cs.geometry.centroid
    boxes = []
    for cy, cx in zip(centroids.y.values, centroids.x.values):
        dlat, dlon = metric_halfdegs(cy, cx)
        boxes.append(box(cx - dlon, cy - dlat, cx + dlon, cy + dlat))
    cs_out = cs.set_geometry(gpd.GeoSeries(boxes, index=cs.index, crs=cs.crs))

    keep = ["FID_DETER", "CLASSNAME", "QUADRANT", "PATH_ROW", "VIEW_DATE",
            "SENSOR", "SATELLITE", "AREAUCKM", "UC", "AREAMUNKM",
            "MUNICIPALI", "GEOCODIBGE", "UF", "roi_id", "geometry"]
    cs_out = cs_out[[c for c in keep if c in cs_out.columns]]

    out_gpkg.unlink(missing_ok=True)
    cs_out.to_file(out_gpkg, driver="GPKG", layer="rois")
    print(f"wrote {out_gpkg}  rows={len(cs_out)}")

    with out_windows.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["roi_id", "window_start", "window_end"])
        for _, row in cs_out.iterrows():
            view_date = str(row["VIEW_DATE"])[:10]
            w.writerow([row["roi_id"], args.window_start, view_date])
    print(f"wrote {out_windows}")

    out_ids.write_text("\n".join(cs_out["roi_id"].tolist()) + "\n", encoding="utf-8")
    print(f"wrote {out_ids}  ({len(cs_out)} roi_ids)")


if __name__ == "__main__":
    main()
