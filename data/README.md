# Data

The corpora and trained weights referenced throughout this pipeline are not tracked in this git
repository — they are published as a companion record on Zenodo:

**DOI: [10.5281/zenodo.21712167](https://doi.org/10.5281/zenodo.21712167)** — the exact version
this repository is built against. [10.5281/zenodo.21712166](https://doi.org/10.5281/zenodo.21712166)
is the concept DOI, which always resolves to the most recent version.

Fetch it with `00_data/download_record.py` (see the repository README's **Data** section); the
archives unpack straight into this directory, in the layout the pipeline stages expect.

## What's in the record

| Item | Approx. size | Maps to |
|---|---|---|
| Pretraining corpus (`pretraining-corpus/`) | ~21.3 GiB | Unlabelled input to `04_pretraining/train.py` |
| Labelled dataset (`labelled-dataset/`) | ~1.49 GiB | Input to `06_regressor/` and `07_detector/` |
| Small reference data (`reference-data/`) | ~100 KB | Annotations, view dates, cached scores |
| Foundation-model checkpoint (`fm-checkpoint/best.pt`) | ~11 MB | The pretrained encoder used throughout |
| Downstream model, 20 leave-one-out folds (`regressor-checkpoints/`) | ~110 MB | The single configuration reported in the paper |

Three terms are used consistently here and in the code:

- **corpus** — only the unlabelled pretraining set.
- **labelled dataset** — the labelled data the downstream model trains and cross-validates on.
- **detector** — only the temporal change detector of `07_detector/`, the edge and persistence
  filters applied to the downstream model's scores. Never the downstream model itself.

The downstream model is a soft-label regressor, which is why its stage directory and checkpoints
are named `06_regressor/` and `regressor-checkpoints/`.

### The annotations

`reference-data/annotations.json` holds the interpreter annotations the soft labels are derived
from, for the 20 monitoring units the paper reports. Its four blocks:

- **`logging_activity`** — every observation where a new selective-logging event was
  unambiguously discernible in the canopy. The earliest of these is the *onset*, against which
  all detection timing is measured; the later ones are *increments*. Each entry is
  `[date, k]`, where `k` is 1-based and picks among several observations sharing one date (two
  platforms, or two tiles) — it does **not** encode the kind of activity.
- **`ambiguous_events`** — observations whose logging state could not be confidently resolved.
  Never counted as logging activity: before the onset they bound the undisturbed region, and
  after it they read as weak evidence of ongoing disturbance.
- **`first_observation`** — the first acquisition inside each unit's monitoring period; see the
  next section.
- **`oversample_pool`** — the **logging-free candidates**: candidate target observations from
  the logging-free sites, from which the negative samples are drawn. The key name is historical;
  this is the candidate pool, and it is larger than the subset actually drawn.

Everything left unmarked is taken to be undisturbed. The soft label itself is not stored: it is
derived by walking each series forward from these annotations, as described in the paper and
implemented in `common/dataset_classifier.py`.

### Monitoring period and observations interval

Two nested spans, distinguished consistently here, in the code and in the manuscript:

```
monitoring period      [ start_period ................................. end_period ]
observations interval        [ first_observation ..... last_observation ]

                       start_period <= first_observation
                                    <= last_observation <= end_period
```

- **`end_period`** — the DETER alert's view date, which closes the monitoring period. This is the
  one primary datum of the four, and it is what `view_dates.json` holds.
- **`start_period`** — `end_period` minus one calendar year.
- **`first_observation`** — the first acquisition at or after `start_period`. Because acquisitions
  are irregular, it lands a few days *inside* the period rather than on its edge: 3 to 6 days
  across the 20 monitoring units.
- **`last_observation`** — the last acquisition at or before `end_period`, 0 to 6 days early
  (it coincides with the view date in only 7 of the 20 units).

The observations interval therefore sits strictly inside the monitoring period at both ends. Only
`end_period` and `first_observation` are stored; the other two are derivable, and duplicating
derived values into a published record only invites them to drift.

Only the reported configuration is included — the paper's Discussion section compares a handful
of additional input variants, but those are not the reported model and are not part of this
record.

Zarr stores are named by their originating DETER alert identifier (e.g.
`10931_curr+11771_curr+....zarr` for a fused cluster of overlapping alerts) — this is the project's
established technical identifier and is unrelated to the manuscript's MU$_1$–MU$_{20}$ numbering
of the 20 cross-validated units, which exists purely for readability in the paper's own text and
tables.

## License and attribution

The record is licensed **CC BY-SA 4.0** (see `../LICENSE-DATA`), inherited from the DETER alert
polygons the labels and corpus selection derive from (INPE/TerraBrasilis, CC BY-SA 4.0 — the
ShareAlike clause propagates to derivatives).

The record's own description carries the required attribution notices:

- **HLS v2.0** (NASA/USGS): freely reusable, but Sentinel-2-derived (S30) content is marked
  *"Contains modified Copernicus Sentinel data \[2024–2026\]"* per the Copernicus data licence.
- **DETER** (INPE, via TerraBrasilis): CC BY-SA 4.0, attributed as the source of the alert
  polygons underlying the ROI candidate set and the downstream model's labels.

## Reproducing this data from scratch

You don't need the Zenodo record at all if you'd rather rebuild the corpora yourself — that's what
`01_rois/` through `03_cubes/` are for. The record exists so you can skip straight to pretraining
and cross-validation without a multi-day HLS download.
