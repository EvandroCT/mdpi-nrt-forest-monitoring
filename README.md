# Selective-logging near-real-time detection — reproducibility pipeline

Code to reproduce the trainings and inferences reported in:

> Taquary, E., Mataveli, G., Nogueira, S., Braga, D., Queiroz, G., Aragão, L. *Enhancing
> Near-Real-Time Amazon Forest Monitoring Using GeoAI: A Case Study on Selective Logging.*
> Submitted to *Remote Sensing* (MDPI).

## Scope

Starting from the companion Zenodo record, which carries the datacubes the models read, this
repository reproduces, end to end:

1. Pretraining of the spatiotemporal masked-autoencoder (ST-MAE) foundation model.
2. Training and leave-one-out cross-validation of the downstream model (a soft-label regressor).
3. The temporal detection-filter grid search (the change detector) and its inference output.
4. The evaluation of the detector: its nested leave-one-MU-out evaluation, the negative test on
   certified logging-free observations, and the survey of DETER alerts inside each analysed
   period (the paper's Tables 3 and 5 and Supplementary Tables S2 and S3).

The code that built those datacubes in the first place — the DETER-derived ROI set, the
eligibility-gated HLS acquisition, the GeoZarr assembly — is published too, under **Rebuilding
the corpora from source**, but it is not part of the path above.

## Data

The corpora and trained weights (~22 GiB) are not stored in this repository — they are published
as a companion dataset on Zenodo: **[10.5281/zenodo.21712167](https://doi.org/10.5281/zenodo.21712167)**
(this exact version; [10.5281/zenodo.21712166](https://doi.org/10.5281/zenodo.21712166) always
resolves to the latest one). See `data/README.md` for exactly what's in that record and how it
maps to the directories below.

To fetch it, into the layout every stage below expects:

```bash
uv run python 00_data/download_record.py --record <DOI or record id>            # everything
uv run python 00_data/download_record.py --record <id> --list                   # what's available
uv run python 00_data/download_record.py --record <id> \
    --categories labelled-dataset,reference-data                                 # just one stage's inputs
```

No credentials needed for this download — the record is public and the script never sends an
Authorization header. Each archive is checked against both the checksum Zenodo reports and
`data/checksums.sha256` in this repository before it is unpacked, so a truncated or altered
download fails loudly instead of quietly poisoning a run. Re-running skips whatever is already
present and verified, so an interrupted download resumes rather than starting over.

The record is the entry point: fetch it and every stage below has its inputs. Fetching everything
lets you start at pretraining; fetching `labelled-dataset,reference-data` plus the published
checkpoints lets you skip straight to the downstream model and the detection filters.

The two upstream data sources are public but not re-hosted here in raw form:

- **HLS v2.0** (NASA/USGS, via `earthaccess`): no restriction on reuse, redistribution, or sale;
  Sentinel-2-derived (S30) content must carry the notice *"Contains modified Copernicus Sentinel
  data \[year\]"* wherever it's redistributed (already included in the Zenodo record's own
  description).
- **DETER alerts** (INPE, via [TerraBrasilis](https://terrabrasilis.dpi.inpe.br/)): CC BY-SA 4.0.
  Attribution required; anything derived from the alert polygons (including the ROI candidate set
  and the labels behind the downstream model) is itself licensed CC BY-SA 4.0 for the same
  reason — see `LICENSE-DATA`.

## Environment

One environment for the whole pipeline, managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

**GPU support.** `uv sync` installs the best available PyTorch build for your platform
automatically — no flags, no edits. On **Linux x86_64** and **Windows**, this is a CUDA 13.0
build (NVIDIA driver ≥ 580 required; ~3 GB download); on **Apple Silicon macOS**, the standard
wheel with Metal (MPS) acceleration. macOS has no CUDA path (no NVIDIA GPUs since 2016), and
Intel Macs are not supported by PyTorch ≥ 2.2. Machines without an NVIDIA GPU still work on
Linux/Windows (the CUDA build falls back to CPU at runtime), at the cost of a larger download.
Verify with `python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"`.

## Pipeline

### 1. Foundation-model pretraining

`04_pretraining/train.py` takes its settings as CLI flags, not a config file. `04_pretraining/recipes/train_extend.py` encodes the full curriculum — the **seven** stages that produced the paper's 460-epoch checkpoint — and runs it automatically, from scratch or resuming from wherever `<DATA_DIR>/runs/pretraining/checkpoints/last.pt` currently stands:

```bash
uv run python 04_pretraining/recipes/train_extend.py             # every remaining stage (base included)
uv run python 04_pretraining/recipes/train_extend.py --one       # run just the next stage, then stop
uv run python 04_pretraining/recipes/train_extend.py --until 260 # stop after the ep260 stage
uv run python 04_pretraining/recipes/train_extend.py --dry-run   # print every stage's exact train.py argv
```

The curriculum is seven stages: ep0–99 (base run) → ep140 → ep180 → ep220 → ep260 → ep360 → ep460. Before launching each one, the scheduler reads the checkpoint's stored `next_epoch` and refuses to run unless it matches that stage's expected starting epoch.

Each stage runs in the foreground and tees its output to both the console and
`<DATA_DIR>/runs/pretraining/train_extend_epNNN.log`; background the scheduler yourself (tmux/screen on
Linux/macOS, `start /b` on Windows). Pass `--data-dir` to override `DATA_DIR`.

**Starting from nothing.** With no checkpoint present, the scheduler runs the base run (ep0–99)
first and continues through the extensions, so the whole 460-epoch curriculum is one command.
The base stage's settings come from `04_pretraining/config/config.yaml`, the config `train.py`
writes for a run: `max_epochs: 100`, `lr_anneal_epoch: 60`,
`lr_schedule: warm-restart-then-anneal`. Parameters the scheduler does not pass explicitly take
`train.py` defaults equal to that file's values (`in_channels` and `input_mask_bits` are derived
at parse time from `--no-aero-channel` and `--mask-bits`). A run writes its own `config.yaml`
into the run directory; diff it against the reference to check your setup.

The 460 epochs are three regimes, run as seven launches in total:

| Epochs | Launches | Learning rate |
|---|---|---|
| 1–100 | 1 (the base run) | warm restarts every 10 epochs through 59, then one last restart at 60 annealing to ~0 by 99 |
| 101–260 | 4 | one 40-epoch restart-and-anneal cycle each |
| 261–460 | 2 | one 100-epoch restart-and-anneal cycle each, giving the late trajectory room to settle |

Epoch 60 is a double boundary: the learning rate stops restarting *and* the band-gated masking
curriculum leaves low → mid → high occlusion for the full-range mixed band.

`04_pretraining/config/config.yaml` records the architecture and optimiser settings (embedding
width, attention heads, batch size). Read it as the **base run's** launch record, not as the
published model's configuration: its `max_epochs: 100` is that first launch's budget and its
`lr_anneal_epoch: 60` is the boundary inside it — the six stages after it live in
`train_extend.py`'s table, which `--dry-run` prints as exact commands.
`config/val_history.csv` is the full per-epoch validation curve for all 460 epochs, so the whole
schedule is visible there. Expect the masked-reconstruction validation loss to fall roughly
32-fold over training and reach its minimum near the end of the schedule.

### 2. Downstream model

```bash
uv run python 06_regressor/build_caches.py   # crop-level caches for the labelled sites
uv run python 06_regressor/loocv_chain.py raw  # the reported model: 20 leave-one-MU-out folds
uv run python 06_regressor/agg_oof.py        # pools the out-of-fold metrics
```

Pass `raw`. Invoked bare, `loocv_chain.py` runs five variants -- `raw all recon masked composite`,
so 100 folds and roughly five times the GPU time. Only `raw` is the configuration this paper
reports; the other four are input-variant ablations, discussed but not reported, and their
checkpoints are deliberately absent from the Zenodo record. `agg_oof.py` prints one line per
variant, so nothing is mixed either way -- the cost is time, not correctness.

`loocv_chain.py` throttles concurrent folds on the live `nvidia-smi` compute-process count
(`--n`, default 3); on a machine with no NVIDIA GPU (e.g. macOS/Apple Silicon), pass `--serial`
to run one fold at a time instead. `--help` lists every flag (`--prefix`, `--exclude`,
`--data-dir`, `--patience`, `--num-workers`), which double as the old `PREFIX=`/`EXCLUDE=`/
`DATA_DIR=`/`PAT=`/`NW=`/`N=` environment variables (still honoured as defaults).

Reproduces the leave-one-MU-out cross-validation: pooled out-of-fold MSE **0.076** (R²-like score
**0.55**), per-fold macro MSE **0.065 ± 0.038**. The corpus couples 20 cross-validated,
selectively-logged monitoring units with 57 negative-only sites. The negative candidates are
hand-picked: each is checked to carry no logging activity in its target observation or in that
target's lookback window. Which of them contribute is then decided by a deterministic draw at a
fixed seed (0), round-robin over time bins crossed with cloud-fraction strata, so a different
seed selects a different subset of the same hand-picked candidates while the total sample counts
stay seed-invariant.

### 3. Detection filters

```bash
uv run python 07_detector/sweep_kernels.py          # reproduces the grid search
uv run python 07_detector/build_dense_pred_variant.py  # reference-detector inference
```

The reference operating point is the setting at which false alarms first reach zero while all 20
monitoring units are still confirmed — not a hand-tuned choice. Reproducing it end to end should
confirm all 20 of 20 held-out units on time, 0 false positives, median confirmation lag **+15
days**.

### 4. Detector evaluation

```bash
uv run python 07_detector/nested_sweep.py                    # nested leave-one-MU-out evaluation of the grid search
uv run python 07_detector/make_nested_tables.py              # Table 5 and Table S3
uv run python 07_detector/negative_test/select_targets.py    # the certified logging-free targets no fold trained on
uv run python 07_detector/negative_test/score_targets.py     # their scores under the 20 fold models (GPU)
uv run python 07_detector/negative_test/evaluate_targets.py  # Table 3
uv run python 07_detector/deter_alerts/deter_survey.py /path/to/deter-amz-deter-public.shp  # DETER alerts at every site
uv run python 07_detector/deter_alerts/make_deter_tables.py  # Table S2 and the alert counts quoted in the text
```

The seven detector parameters are selected on the same 20 units the detector is evaluated on.
`nested_sweep.py` repeats the grid search on 19 units at a time and applies each winner, unchanged,
to the held-out unit: 18 of 20 are confirmed on time, with one early false alarm and one miss,
median confirmation lag unchanged at +15 days; 18 of the 20 folds re-select the reference
configuration. It runs on CPU, in about 8 minutes with 21 worker processes.

The negative test takes the logging-free candidates of `annotations.json` that the seed-0 negative
draw never reached: 51 targets on 41 sites, none of them a training target of any fold. Under the
reference rule no target is alerted by a majority of the 20 fold models, and 6 of the 1,020
target–model readings alert. `score_targets.py` needs the `regressor-checkpoints` category and a
CUDA GPU (about an hour on one H100); the other steps run on CPU in seconds to minutes.

`deter_survey.py` reads the public DETER Amazônia shapefile, unzipped from
[TerraBrasilis](https://terrabrasilis.dpi.inpe.br/file-delivery/download/deter-amz/shape). That file
grows as alerts are published; the survey in the record was built from the download of 15 September
2026. To reproduce Table S2 from the record's survey and targets, without the shapefile:

```bash
SURVEY=data/detector-evaluation/deter_alerts/deter_survey.csv \
TARGETS=data/detector-evaluation/negative_test/targets.csv \
    uv run python 07_detector/deter_alerts/make_deter_tables.py
```

Every step writes under `data/runs/` (`nested_sweep/`, `negative_test/`, `deter_alerts/`); the
`detector-evaluation` category holds the published outputs under the same names, in
`data/detector-evaluation/`.

## Rebuilding the corpora from source

**Not part of the reproduction path above.** The Zenodo
record already contains the GeoZarr datacubes every stage above reads, so nothing here is
needed to reproduce the paper. The scripts are published because they are how those cubes were
built. They talk to live NASA and INPE catalogues, which move, so treat them as provenance
rather than as a supported path.

**NASA Earthdata Login.** Needed only here, to fetch imagery yourself. HLS is free but
not anonymous: NASA requires a (free) [Earthdata Login](https://urs.earthdata.nasa.gov/users/new)
account, and `02_download/hls_stac_download.py` authenticates through `earthaccess` before it
fetches anything. Set it up once, either way:

```bash
# Either: store the credentials, then every later run is non-interactive.
uv run python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"

# Or: export them for the session (useful in CI or a container).
export EARTHDATA_USERNAME=... EARTHDATA_PASSWORD=...
```

`persist=True` writes `~/.netrc`, which the script reads on subsequent runs. Nothing in the
reproduction path needs an account: **downloading the Zenodo record requires no credentials**,
and neither does anything that runs off it.

### 1. ROI candidates

```bash
uv run python 01_rois/build_roi_candidates.py \
    --deter-export /path/to/your/deter_export.gpkg \
    --output-dir ./data/rois
```

Filters the DETER export down to the selective-logging classes (`CS_GEOMETRICO` +
`CS_DESORDENADO`) and writes a 1920 m × 1920 m bounding box per alert, centred on its centroid.
Obtain the DETER export from TerraBrasilis, filtering by view date; a single "view dates 1
January–1 May 2026" export reproduces the 3,288-alert catalogue the paper draws from (see
`data/README.md` for the exact filter used).

### 2. Economical acquisition (5 phases)

Downloading full-band HLS imagery for every candidate observation is expensive. The pipeline
instead screens on the QA/cloud mask alone first, and only fetches full bands for the
observations that survive eligibility — the QA band is a small fraction of the data volume of the
full 6-band product.

```bash
# Phase QA-only: fetch just the quality-assessment band for every ROI candidate.
uv run python 02_download/hls_stac_download.py \
    --roi ./data/rois/rois.gpkg --output ./data/hls_qa \
    --l30-bands Fmask --s30-bands Fmask

# Phase A: ingest the QA band alone into GeoZarr stores.
uv run python 03_cubes/build_geozarr_datacubes.py \
    ./data/hls_qa ./data/zarrs --fmask-only

# Phase B: scan the QA-only zarrs, decide which target dates are eligible, and write the
# result back into the granule cache as `needed_fids`.
uv run python 02_download/scan_eligibility.py \
    --zarrs-dir ./data/zarrs --granule-list ./data/hls_qa/granule_list.json

# Phase C: download the full bands, but only for what Phase B marked as needed.
uv run python 02_download/hls_stac_download.py \
    --roi ./data/rois/rois.gpkg --output ./data/hls_full \
    --granule-list ./data/hls_qa/granule_list.json

# Phase D: fill in the real reflectance for the granules fetched in Phase C.
uv run python 03_cubes/build_geozarr_datacubes.py \
    ./data/hls_full ./data/zarrs --append-reflectance
```

Retrieval window throughout: **1 October 2024 to 13 May 2026** (the scripts' own defaults).
Eligibility screens admit up to 5% cirrus, 30% moderate aerosol, 10% open water, and 10% invalid
reflectance, strict (0%) on cloud, cloud-shadow, low-climatology aerosol, and high aerosol — the
classes that corrupt the surface signal outright — over an `L=80`-observation lookback window.
These are `scan_eligibility.py`'s own defaults; run it with `--help` to see (and override) every
threshold explicitly.

## Repository layout

```
common/          Shared eligibility-filter logic (imported by 02_download and 04_pretraining).
00_data/         Fetch + verify the companion Zenodo record into ./data.
01_rois/         Selective-logging ROI candidate set from a DETER export.
02_download/     HLS acquisition (economical QA-first pipeline).
03_cubes/        GeoZarr datacube assembly.
04_pretraining/  ST-MAE foundation-model pretraining.
05_inference/    Inference with the pretrained/trained models.
06_regressor/    Downstream soft-label regressor + leave-one-MU-out cross-validation.
07_detector/     Temporal detection-filter grid search, reference-detector inference and the
                 detector evaluation (nested LOMO, negative test, DETER alert survey).
data/            Pointer to the companion Zenodo record — not tracked in git.
```

## License

Code: [AGPL-3.0-or-later](LICENSE). Data (via the companion Zenodo record):
[CC BY-SA 4.0](LICENSE-DATA).

## Citation

If you use this code, please cite the paper above. If you use the companion data, please also
cite the underlying HLS v2.0 (NASA/USGS) and DETER (INPE) products per their own citation
requirements, linked in `data/README.md`.
