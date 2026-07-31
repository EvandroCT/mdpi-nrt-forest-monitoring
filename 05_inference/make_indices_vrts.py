# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""make_indices_vrts.py — emit one combined VRT per prediction TIF that
serves as both the true-colour RGB view AND a live-computing multi-index
viewer (VARI on transformed RGB, plus NDVI/NBR/NDMI on raw reflectance).

For every 7-band float32 prediction TIF `<stem>.tif` produced by
`predict_geotiffs.py`, this writes a single sidecar:

  <stem>.vrt — 15-band combined visualisation VRT.

      Band 1: Red    Byte uint8, source TIF band 3 (Red)   stretched [0, 0.3] → [0, 255]
      Band 2: Green  Byte uint8, source TIF band 2 (Green) stretched [0, 0.3] → [0, 255]
      Band 3: Blue   Byte uint8, source TIF band 1 (Blue)  stretched [0, 0.3] → [0, 255]
        ↳ tagged with ColorInterp.{Red,Green,Blue} → QGIS opens as RGB
          by default (no QML needed).

      Band 4: VARI   = (G'-R')/(G'+R'-B')          [transformed RGB,
                                                    references bands 1-3]
      Band 5: NDVI   = (NIR-Red) / (NIR+Red)       [raw reflectance from src]
      Band 6: NBR    = (NIR-SWIR2)/(NIR+SWIR2)     [raw reflectance from src]
      Band 7: NDMI   = (NIR-SWIR1)/(NIR+SWIR1)     [raw reflectance from src]

      Bands 8-15: numerator/denominator intermediates for each index.
        ↳ visible in QGIS's band list but only useful diagnostically.

Self-referencing chain (band 4 reads bands 1-3 of the same .vrt; bands 8-9
also read bands 1-3) is resolved by GDAL via dataset caching — no external
intermediate files needed.

The original `<stem>.tif` (raw float32 reflectance + cloud mask) is the
single source of truth and is unchanged. No QML is shipped — ColorInterp
tags give default RGB; user manually selects pseudocolour on band 4/5/6/7
for index visualisation in QGIS layer properties.

Why this works in this GDAL (no muparser, no Python pixel functions):
self-reference within one VRT plus built-in `div`/`diff`/`sum` pixel
functions and `ComplexSource` ScaleRatio handle every operation natively.

Usage:
    python make_indices_vrts.py
        [--root ./data/predictions]
        [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

_DEFAULT_ROOT = Path("./data/predictions")

# True-colour stretch parameters.
_SCALE_LO, _SCALE_HI = 0.0, 0.3
_TARGET_HI = 255
_SCALE_RATIO = _TARGET_HI / (_SCALE_HI - _SCALE_LO)       # 850.0
_SCALE_OFFSET = -_SCALE_LO * _SCALE_RATIO                  # 0.0

_CMAP = matplotlib.colormaps["RdYlGn"]


# ---------------------------------------------------------------------------
# VRT XML builders
# ---------------------------------------------------------------------------


def _rgb_band(out_band: int, src_band: int, ci: str,
              src_name: str, W: int, H: int) -> str:
    """A regular VRT band that scales a source band to Byte (uint8 0..255)."""
    return f"""  <VRTRasterBand dataType="Byte" band="{out_band}">
    <ColorInterp>{ci}</ColorInterp>
    <ComplexSource>
      <SourceFilename relativeToVRT="1">{src_name}</SourceFilename>
      <SourceBand>{src_band}</SourceBand>
      <SourceProperties RasterXSize="{W}" RasterYSize="{H}" DataType="Float32" />
      <SrcRect xOff="0" yOff="0" xSize="{W}" ySize="{H}" />
      <DstRect xOff="0" yOff="0" xSize="{W}" ySize="{H}" />
      <ScaleOffset>{_SCALE_OFFSET:.6f}</ScaleOffset>
      <ScaleRatio>{_SCALE_RATIO:.6f}</ScaleRatio>
    </ComplexSource>
  </VRTRasterBand>
"""


def _div_band(out_band: int, src_name: str, num_band: int, den_band: int,
              desc: str) -> str:
    """A derived band that divides two bands of `src_name`."""
    return f"""  <VRTRasterBand dataType="Float32" band="{out_band}" subClass="VRTDerivedRasterBand">
    <Description>{desc}</Description>
    <PixelFunctionType>div</PixelFunctionType>
    <SourceTransferType>Float32</SourceTransferType>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{src_name}</SourceFilename>
      <SourceBand>{num_band}</SourceBand>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{src_name}</SourceFilename>
      <SourceBand>{den_band}</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
"""


def _diff_band(out_band: int, src_name: str, a: int, b: int, desc: str) -> str:
    return f"""  <VRTRasterBand dataType="Float32" band="{out_band}" subClass="VRTDerivedRasterBand">
    <Description>{desc}</Description>
    <PixelFunctionType>diff</PixelFunctionType>
    <SourceTransferType>Float32</SourceTransferType>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{src_name}</SourceFilename>
      <SourceBand>{a}</SourceBand>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{src_name}</SourceFilename>
      <SourceBand>{b}</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
"""


def _sum_band(out_band: int, src_name: str, a: int, b: int, desc: str) -> str:
    return f"""  <VRTRasterBand dataType="Float32" band="{out_band}" subClass="VRTDerivedRasterBand">
    <Description>{desc}</Description>
    <PixelFunctionType>sum</PixelFunctionType>
    <SourceTransferType>Float32</SourceTransferType>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{src_name}</SourceFilename>
      <SourceBand>{a}</SourceBand>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{src_name}</SourceFilename>
      <SourceBand>{b}</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
"""


def _vari_den_band(out_band: int, self_name: str) -> str:
    """G' + R' - B' — sum of bands 2, 1, and negated 3 of `self_name`."""
    return f"""  <VRTRasterBand dataType="Float32" band="{out_band}" subClass="VRTDerivedRasterBand">
    <Description>VARI den G'+R'-B' [transformed RGB]</Description>
    <PixelFunctionType>sum</PixelFunctionType>
    <SourceTransferType>Float32</SourceTransferType>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{self_name}</SourceFilename>
      <SourceBand>2</SourceBand>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{self_name}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
    <ComplexSource>
      <SourceFilename relativeToVRT="1">{self_name}</SourceFilename>
      <SourceBand>3</SourceBand>
      <ScaleRatio>-1</ScaleRatio>
      <ScaleOffset>0</ScaleOffset>
    </ComplexSource>
  </VRTRasterBand>
"""


def _vrt_xml(source_tif_name: str, self_vrt_name: str,
             W: int, H: int, srs_wkt: str, geotransform: str) -> str:
    """Build the 15-band combined VRT XML."""
    parts: list[str] = []
    # 1-3: stretched RGB from the source TIF.
    parts.append(_rgb_band(1, 3, "Red",   source_tif_name, W, H))
    parts.append(_rgb_band(2, 2, "Green", source_tif_name, W, H))
    parts.append(_rgb_band(3, 1, "Blue",  source_tif_name, W, H))
    # 4-7: indices (self-reference their own intermediates below).
    parts.append(_div_band(4, self_vrt_name, 8, 9,
                            "VARI = (G'-R')/(G'+R'-B') [transformed RGB]"))
    parts.append(_div_band(5, self_vrt_name, 10, 11,
                            "NDVI = (NIR-Red)/(NIR+Red) [raw reflectance]"))
    parts.append(_div_band(6, self_vrt_name, 12, 13,
                            "NBR  = (NIR-SWIR2)/(NIR+SWIR2) [raw reflectance]"))
    parts.append(_div_band(7, self_vrt_name, 14, 15,
                            "NDMI = (NIR-SWIR1)/(NIR+SWIR1) [raw reflectance]"))
    # 8-9: VARI numerator/denominator on bands 1-3 of self (= transformed RGB).
    parts.append(_diff_band(8, self_vrt_name, 2, 1,
                             "VARI num G'-R' [transformed RGB]"))
    parts.append(_vari_den_band(9, self_vrt_name))
    # 10-11: NDVI num/den on raw NIR (src band 4), Red (src band 3).
    parts.append(_diff_band(10, source_tif_name, 4, 3,
                             "NDVI num NIR-Red [raw reflectance]"))
    parts.append(_sum_band (11, source_tif_name, 4, 3,
                             "NDVI den NIR+Red [raw reflectance]"))
    # 12-13: NBR num/den (NIR=4, SWIR2=6).
    parts.append(_diff_band(12, source_tif_name, 4, 6,
                             "NBR num NIR-SWIR2 [raw reflectance]"))
    parts.append(_sum_band (13, source_tif_name, 4, 6,
                             "NBR den NIR+SWIR2 [raw reflectance]"))
    # 14-15: NDMI num/den (NIR=4, SWIR1=5).
    parts.append(_diff_band(14, source_tif_name, 4, 5,
                             "NDMI num NIR-SWIR1 [raw reflectance]"))
    parts.append(_sum_band (15, source_tif_name, 4, 5,
                             "NDMI den NIR+SWIR1 [raw reflectance]"))
    body = "".join(parts)
    return f"""<VRTDataset rasterXSize="{W}" rasterYSize="{H}">
  <SRS>{srs_wkt}</SRS>
  <GeoTransform>{geotransform}</GeoTransform>
{body}</VRTDataset>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# QML styles (one auto-loaded default + alternates for "Load Style")
# ---------------------------------------------------------------------------


def _pseudocolor_qml(band: int, vmin: float, vmax: float, n_stops: int = 17) -> str:
    """QGIS singleband-pseudocolour XML for a given band + RdYlGn ramp."""
    stops = []
    for i in range(n_stops):
        f = i / (n_stops - 1)
        v = vmin + f * (vmax - vmin)
        r, g, b, _ = _CMAP(f)
        stops.append(
            f'          <item alpha="255" value="{v:.6f}" '
            f'label="{v:+.2f}" color="#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"/>'
        )
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer band="{band}" classificationMin="{vmin}" classificationMax="{vmax}" type="singlebandpseudocolor" opacity="1" alphaBand="-1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits><extent>WholeRaster</extent><statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower><cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0" minimumValue="{vmin}" maximumValue="{vmax}">
{chr(10).join(stops)}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def _rgb_qml() -> str:
    """QGIS multibandcolor RGB renderer using bands 1/2/3 as R/G/B with explicit
    [0, 255] min/max — matches the byte stretch baked into bands 1-3."""
    return """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer redBand="1" greenBand="2" blueBand="3" alphaBand="-1" type="multibandcolor" opacity="1">
      <rasterTransparency/>
      <redContrastEnhancement>
        <minValue>0</minValue><maxValue>255</maxValue><algorithm>NoEnhancement</algorithm>
      </redContrastEnhancement>
      <greenContrastEnhancement>
        <minValue>0</minValue><maxValue>255</maxValue><algorithm>NoEnhancement</algorithm>
      </greenContrastEnhancement>
      <blueContrastEnhancement>
        <minValue>0</minValue><maxValue>255</maxValue><algorithm>NoEnhancement</algorithm>
      </blueContrastEnhancement>
    </rasterrenderer>
  </pipe>
</qgis>
"""


# Per-index style presets: (band, vmin, vmax)
_INDEX_STYLES: dict[str, tuple[int, float, float]] = {
    "vari": (4, -0.5, 0.5),       # matches train.py `_vari_uint8` clip
    "ndvi": (5, -1.0, 1.0),
    "nbr":  (6, -1.0, 1.0),
    "ndmi": (7, -1.0, 1.0),
}


def make_combined_vrt_for_tif(tif: Path, overwrite: bool = False) -> bool:
    """Write the combined VRT + auto-loaded QML + alternate QMLs next to `tif`.

    Auto-loaded QML (`<tif.stem>.qml`) renders VARI on band 4 with the
    TensorBoard RdYlGn `[-0.5, 0.5]` colormap. Alternates are shipped
    with descriptive stems so users can "Load Style" in QGIS to switch:
      <stem>_rgb.qml   — true-colour RGB
      <stem>_ndvi.qml  — NDVI on band 5
      <stem>_nbr.qml   — NBR  on band 6
      <stem>_ndmi.qml  — NDMI on band 7
    """
    out_vrt = tif.with_suffix(".vrt")
    out_qml = tif.with_suffix(".qml")          # auto-loaded
    rgb_qml = tif.with_name(tif.stem + "_rgb.qml")
    ndvi_qml = tif.with_name(tif.stem + "_ndvi.qml")
    nbr_qml = tif.with_name(tif.stem + "_nbr.qml")
    ndmi_qml = tif.with_name(tif.stem + "_ndmi.qml")
    all_paths = (out_vrt, out_qml, rgb_qml, ndvi_qml, nbr_qml, ndmi_qml)
    if all(p.exists() for p in all_paths) and not overwrite:
        return False

    import rasterio
    with rasterio.open(tif) as src:
        W, H = src.width, src.height
        srs_wkt = src.crs.wkt if src.crs else ""
        t = src.transform
        gt = f"{t.c}, {t.a}, {t.b}, {t.f}, {t.d}, {t.e}"

    # VRT
    xml = _vrt_xml(tif.name, out_vrt.name, W, H, srs_wkt, gt)
    tmp = out_vrt.with_suffix(".vrt.tmp")
    tmp.write_text(xml, encoding="utf-8")
    os.replace(tmp, out_vrt)

    # Auto-loaded QML: TB-style VARI
    vari_band, vari_lo, vari_hi = _INDEX_STYLES["vari"]
    tmp = out_qml.with_suffix(".qml.tmp")
    tmp.write_text(_pseudocolor_qml(vari_band, vari_lo, vari_hi), encoding="utf-8")
    os.replace(tmp, out_qml)

    # Alternate QMLs (manual "Load Style")
    for suffix, target in (("ndvi", ndvi_qml), ("nbr", nbr_qml), ("ndmi", ndmi_qml)):
        b, lo, hi = _INDEX_STYLES[suffix]
        tmp = target.with_suffix(".qml.tmp")
        tmp.write_text(_pseudocolor_qml(b, lo, hi), encoding="utf-8")
        os.replace(tmp, target)
    # RGB style
    tmp = rgb_qml.with_suffix(".qml.tmp")
    tmp.write_text(_rgb_qml(), encoding="utf-8")
    os.replace(tmp, rgb_qml)
    return True


# Backwards-compat aliases for predict_geotiffs.py imports.
make_vari_for_tif = make_combined_vrt_for_tif
make_indices_for_tif = make_combined_vrt_for_tif


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--root", type=Path, default=_DEFAULT_ROOT,
                   help="Walk this directory recursively for prediction *.tif files.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing <stem>.vrt sidecars.")
    args = p.parse_args()

    if not args.root.exists():
        print(f"root not found: {args.root}", flush=True)
        return 1

    # Skip files that are themselves index sidecars or other artefacts.
    skip_suffixes = ("_vari", "_indices")
    tifs = sorted(
        t for t in args.root.rglob("*.tif")
        if not any(t.stem.endswith(suf) for suf in skip_suffixes)
    )
    print(f"found {len(tifs)} source .tif files under {args.root}", flush=True)

    n_made = n_skip = n_fail = 0
    for i, t in enumerate(tifs):
        try:
            made = make_combined_vrt_for_tif(t, args.overwrite)
        except Exception as e:                                    # noqa: BLE001
            print(f"[fail] {t.name}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            continue
        if made:
            n_made += 1
        else:
            n_skip += 1
        if (i + 1) % 10000 == 0:
            print(f"  [{i + 1}/{len(tifs)}] made={n_made} skipped={n_skip} failed={n_fail}",
                  flush=True)

    print(f"[done] {n_made} combined VRTs written, {n_skip} skipped, {n_fail} failed.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
