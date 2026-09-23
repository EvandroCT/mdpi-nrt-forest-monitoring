# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""DETER alerts intersecting every site of the labelled dataset (input of make_deter_tables.py).

Each zarr store of the labelled dataset -- the 20 monitoring units and the 74 logging-free
sites -- covers a 1920 m box (its spatial:bbox and proj:code attributes). Every DETER alert, of
any class and date, whose polygon intersects a site's box is listed with its view date, class,
area and the share of the box it covers.

Input: the public DETER Amazonia shapefile (INPE/TerraBrasilis, CC BY-SA 4.0), unzipped from
  https://terrabrasilis.dpi.inpe.br/file-delivery/download/deter-amz/shape
The file is updated as new alerts are published: the deter_survey.csv shipped in the record was
built from the download of 15 September 2026 (457,803 alerts, view dates 2 August 2016 to
4 September 2026), and a later download can add alerts.

Writes footprints.json (per site: box, CRS, first/last observation) and deter_survey.csv (one row
per intersecting alert: site, view_date, classname, area_km2, frac_box).
Usage: uv run python 07_detector/deter_alerts/deter_survey.py /path/to/deter-amz-deter-public.shp
Env: DATA_DIR (default ./data), ZARRS; OUT_DIR (default DATA_DIR/runs/deter_alerts)."""
import csv, json, os, sys, warnings
from pathlib import Path
import geopandas as gpd
import pandas as pd
import xarray as xr
from pyproj import Transformer
from shapely.geometry import box

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ZARRS = Path(os.environ["ZARRS"]) if os.environ.get("ZARRS") else DATA_DIR / "labelled-dataset"
OUT_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else DATA_DIR / "runs" / "deter_alerts"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    fps = {}
    for p in sorted(ZARRS.glob("*.zarr")):
        z = xr.open_zarr(p, consolidated=False); t = z["time"].values
        fps[p.name[:-len(".zarr")]] = dict(bbox=[float(v) for v in z.attrs["spatial:bbox"]], epsg=z.attrs["proj:code"],
                                          first_obs=str(t[0])[:10], last_obs=str(t[-1])[:10], n_obs=int(z.sizes["time"]))
    D = gpd.read_file(sys.argv[1])
    D.columns = [c.lower() for c in D.columns]
    D["view_date"] = pd.to_datetime(D["view_date"]).dt.date
    print(f"DETER alerts: {len(D):,} (view dates {min(D.view_date)} to {max(D.view_date)}) | sites: {len(fps)}", flush=True)
    rows = []
    for site, s in fps.items():
        x0, y0, x1, y1 = s["bbox"]; tr = Transformer.from_crs(s["epsg"], D.crs, always_xy=True)
        lon0, lat0 = tr.transform(x0, y0); lon1, lat1 = tr.transform(x1, y1)
        fp = box(min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))
        hit = D.iloc[D.sindex.query(fp, predicate="intersects")].copy()
        # share of the box, in geographic coordinates (the distortion is negligible over ~2 km)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            hit["frac_box"] = hit.geometry.intersection(fp).area / fp.area
        for _, a in hit.sort_values(["view_date", "classname", "areamunkm"], kind="stable").iterrows():
            rows.append(dict(site=site, view_date=a.view_date.isoformat(), classname=a.classname,
                             area_km2=float(a.areamunkm), frac_box=round(float(a.frac_box), 3)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "footprints.json").write_text(json.dumps(fps, indent=1), encoding="utf-8")
    with open(OUT_DIR / "deter_survey.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["site", "view_date", "classname", "area_km2", "frac_box"], lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"alerts intersecting a site box: {len(rows)} over {len({r['site'] for r in rows})} of {len(fps)} sites "
          f"-> {OUT_DIR / 'deter_survey.csv'}", flush=True)
