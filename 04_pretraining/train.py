# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Multi-ROI training of the time-first masked-pixel autoencoder.

Design
------
- Per-crop eligibility: each ROI is tiled into `crop_size`-sided windows
  with uniform spacing; eligibility is computed per (crop, time).
- Band-gated synthetic-mask walk: each epoch defines a target cloud-fraction
  band `(lo, hi)`; only past-window slots in that band are mask sources.
  3-phase curriculum (low → mid → high) by default.
- 6-band reflectance + per-band-per-pixel α gate instead of a hard
  `torch.where` bypass.
- L1 loss with `1/band_std²` per-band weighting and `1/sqrt(mask_frac)`
  per-sample weighting.
- `StratifiedROISampler`: each epoch yields `min(N, n_crops_in_ROI)`
  samples per ROI, no ROI dominates regardless of its crop count.
- LR schedules: linear warmup → cosine warm-restart, cosine-to-zero, or
  warm-restart-then-anneal (restarts through the curriculum, then a single
  cosine anneal over the mixed phase).
- Cycle-aware early stopping: val is evaluated at the end of every cycle
  and training stops after `patience_cycles` cycle-ends without improvement.
- Per-trial run dir at `runs/<branch>/` containing `checkpoints/`,
  `train.log`, `val_history.csv`, `config.yaml`, and a copy of
  `inspect_template.ipynb` with the trial's `best.pt` path pre-substituted.
- Atomic-write checkpoint protocol; full state for seamless resume
  (model, optimizer, scheduler, RNG, cycle-end history, patience counter).

Example
-------
    python train.py \
        --branch pretraining \
        --zarrs-dir ./data/earthaccess_clustered_zarrs \
        --reliable-list ./data/earthaccess_clustered_zarrs/reliable_rois.txt \
        --max-epochs 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import HLSGeoZarrDataset, set_epoch_recursive
from model import TimeFirstAutoencoder

# NOTE on --num-workers: PyTorch's default sharing strategy passes DataLoader-worker
# tensors through /dev/shm, so multi-worker loading fails with "No space left on device"
# wherever shared memory is small -- 64 MB is a common container default. If you hit that,
# either raise the container's shm size (e.g. `docker run --shm-size=8g`) or fall back to
# --num-workers 0, which loads in-process and is slower but has no shm requirement.
# The runs behind the paper used 4 workers (24 for the first extension stage); see the
# schedule in recipes/train_extend.py.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VARI_CMAP = matplotlib.colormaps["RdYlGn"]

# Critical args that must match between a saved checkpoint and the live CLI
# at resume time. Mismatches fail loudly unless --force-mismatch is passed.
_CRITICAL_ARGS: tuple[str, ...] = (
    "embed_dim", "num_heads", "temporal_depth", "spatial_depth",
    "in_channels", "out_channels", "crop_size", "lookback",
    "dropout", "mlp_ratio", "window", "identity_bypass", "spatial_attn",
    "mask_bits", "no_aero_channel",
    "input_mask_bits", "qa_channels",
)

# 4-phase curriculum (100 ep), set 2026-05-22:
#   ep 0-19   → low     (0.05, 0.30)
#   ep 20-39  → mid     (0.30, 0.60)
#   ep 40-59  → high    (0.60, 0.90)
#   ep 60-99  → mixed   (0.05, 0.90)   full-range band
# This is a genuinely advancing curriculum, so it REQUIRES the persistent-worker
# fix below (persistent_workers=False) — otherwise a frozen worker would read a
# single schedule entry forever and the phases would never transition.
_DEFAULT_BAND_SCHEDULE: list[tuple[float, float]] = (
    [(0.05, 0.30)] * 20
    + [(0.30, 0.60)] * 20
    + [(0.60, 0.90)] * 20
    + [(0.05, 0.90)] * 40
)


# Per-pixel temporal attention runs nn.MultiheadAttention over a (B·H·W)
# "batch" dimension. Under bf16 autocast that selects the flash SDPA kernel,
# whose CUDA grid exceeds the 65535 limit at this batch size →
# cudaErrorInvalidValue. Force the math SDPA backend (already used in fp32) —
# it handles an arbitrary batch.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Inductor's coalescing analyzer (`solve_for_tiling`) asserts the simplified
# expr is free of ModularIndexing/FloorDiv; in our compute graph (loss +
# weighting + augmentation under autocast bf16) this assertion sometimes
# fires during BACKWARD compile and aborts the whole training process —
# even though `failed to eagerly compile backwards` says it tried to be
# tolerant. `suppress_errors = True` makes torch.compile fall back to eager
# for the failing subgraph instead of crashing. Throughput impact is bounded
# to that subgraph.
torch._dynamo.config.suppress_errors = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    # Data
    p.add_argument("--zarrs-dir", type=Path, required=True,
                   help="Directory containing per-ROI .zarr stores.")
    p.add_argument("--reliable-list", type=Path, required=True,
                   help="Text file with one FID per line; <FID>.zarr inside zarrs-dir.")
    # Run identity
    p.add_argument("--branch", type=str, required=True,
                   help="Branch name; also the run dir name under runs/.")
    p.add_argument("--runs-root", type=Path, default=Path(__file__).parent / "runs",
                   help="Root directory under which per-branch run folders are created.")
    # Architecture
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--temporal-depth", type=int, default=3)
    p.add_argument("--spatial-depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--in-channels", type=int, default=8)
    p.add_argument("--out-channels", type=int, default=6)
    p.add_argument("--bf16", action="store_true",
                   help="Train under torch.autocast bfloat16 (mixed precision) — "
                        "~halves activation memory, faster on tensor-core GPUs. "
                        "bf16 needs no GradScaler.")
    p.add_argument("--mask-bits", type=lambda x: int(x, 0), default=0b00001111,
                   help="Bitmask applied to Fmask for per-slot input filter, "
                        "synthetic target mask M, and frac_cache. Default "
                        "0b00001111 = cirrus|cloud|adj_cloud|shadow. Pass e.g. "
                        "0b10001111 (= 0x8F) to also mask aero level >= 2.")
    p.add_argument("--no-aero-channel", action="store_true",
                   help="Drop the aerosol input channel. When set, --in-channels "
                        "auto-adjusts to 7 (6 reflectance + invalid-mask). Useful "
                        "when mask_bits already absorbs aerosol-medium/high.")
    p.add_argument("--input-mask-bits", type=lambda x: int(x, 0), default=None,
                   help="Per-slot input filter bits (NaN-out past reflectance "
                        "where these Fmask bits trip). Default = --mask-bits. "
                        "Pass 0 to disable: model sees raw cloudy reflectance "
                        "(C15/C16 noisy-input ablation).")
    p.add_argument("--qa-channels", action="store_true",
                   help="Append 8 binary QA channels per slot (6 multi-hot "
                        "Fmask bits + 2 binary aero medium/high). Inserted "
                        "between reflectance and invalid_mask. When set, "
                        "in_channels grows by 8.")
    p.add_argument("--lookback", type=int, default=80)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--spatial-attn", choices=["global", "window"], default="global",
                   help="Spatial decoder attention. 'global' = every pixel attends to "
                        "every pixel in the crop (no 8x8 window seams, quadratic mem). "
                        "'window' = Swin-style windowed MHA with alternating shifts.")
    p.add_argument("--identity-bypass", choices=["hard", "per_band_per_pixel_alpha"],
                   default="hard",
                   help="hard: torch.where(fill_mask, head, identity) — strict, content-aware. "
                        "per_band_per_pixel_alpha: static learnable blend (ablation C7).")
    # Loss
    p.add_argument("--loss", choices=["l1", "l2"], default="l1")
    p.add_argument("--band-weighting", choices=["inv-var", "inv-std", "uniform"],
                   default="inv-var",
                   help="Per-band loss reweighting: 1/std² (default), 1/std, or uniform.")
    p.add_argument("--sample-weight-exp", type=float, default=0.5,
                   help="Per-sample weight = 1/mask_frac^x. 0=uniform, 0.5=1/sqrt (default), 1=1/m.")
    # Sampler
    p.add_argument("--sampler", choices=["stratified", "uniform"],
                   default="stratified")
    p.add_argument("--n-per-roi", type=int, default=64,
                   help="StratifiedROISampler: max crops per ROI per epoch.")
    # Schedule
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr-schedule",
                   choices=["cosine-warm-restart", "cosine-to-zero",
                            "warm-restart-then-anneal"],
                   default="cosine-warm-restart")
    p.add_argument("--T0", type=int, default=10,
                   help="Cosine warm-restart period (epochs). Ignored for cosine-to-zero.")
    p.add_argument("--T-mult", type=int, default=1)
    p.add_argument("--lr-anneal-epoch", type=int, default=60,
                   help="warm-restart-then-anneal only: epoch at which warm restarts "
                        "stop and a single cosine decay to ~0 begins (the mixed "
                        "curriculum phase). Ignored by the other schedules.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip-norm", type=float, default=1.0)
    # Early stopping
    p.add_argument("--patience-cycles", type=int, default=3,
                   help="Stop if val doesn't improve across N consecutive cycle ends.")
    p.add_argument("--early-stop-start-epoch", type=int, default=0,
                   help="Cycle-aware early stopping only counts cycle-ends at or "
                        "after this epoch. Earlier cycle-ends are still logged to "
                        "val_history.csv but don't accrue patience and can't "
                        "trigger a stop. Lets a curriculum's warmup phase (e.g. "
                        "the low-band phase) run without early-stop interference.")
    p.add_argument("--reset-early-stop", action="store_true",
                   help="On resume: clear cycle-end history + patience counter.")
    p.add_argument("--extend-patience", type=int, default=0,
                   help="On resume: subtract this many cycles from the saved patience counter.")
    # Resume + safety
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint to resume from (default: <run-dir>/checkpoints/last.pt).")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing last.pt and start fresh.")
    p.add_argument("--force-mismatch", action="store_true",
                   help="Skip critical-arg mismatch check on resume.")
    # Per-epoch snapshot archive
    p.add_argument("--archive-every", type=int, default=0,
                   help="If > 0, also save checkpoints/ep_<NNN>.pt every N epochs in addition "
                        "to last/best/final. Preserves a mid-trajectory history for later "
                        "diagnostics. 0 disables (default).")
    # Misc
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--img-log-every", type=int, default=200)
    p.add_argument("--val-img-samples", type=int, default=16,
                   help="Number of val samples logged to TB each epoch. The samples are "
                        "pre-picked once at setup — one per randomly-chosen ROI — so "
                        "they're diverse and consistent across epochs.")
    p.add_argument("--val-loss-samples", type=int, default=4000,
                   help="Number of val samples used to compute val_loss each epoch. "
                        "A fixed random subset of size N is drawn once at setup (seeded "
                        "by args.seed) so the val curve is comparable across epochs. "
                        "0 = use the full val set. Does NOT change the train/val split.")
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--tif-dir", type=Path, default=None)
    p.add_argument("--no-tifs", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", choices=["on", "off"], default="on",
                   help="Whether to torch.compile the model.")
    args = p.parse_args()
    # Auto-derive in_channels so it stays consistent with the optional
    # channel flags. 6 reflectance + (1 aero) + (8 QA) + 1 invalid_mask.
    derived = 6 + (0 if args.no_aero_channel else 1) + (8 if args.qa_channels else 0) + 1
    args.in_channels = derived
    # input_mask_bits defaults to mask_bits if not provided.
    if args.input_mask_bits is None:
        args.input_mask_bits = args.mask_bits
    return args


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_datasets(
    zarrs_dir: Path,
    reliable_list: Path,
    lookback: int,
    crop_size: int,
    val_frac: float,
    band_schedule: list[tuple[float, float]],
    mask_bits: int,
    include_aero_channel: bool,
    input_mask_bits: int,
    qa_channels: bool,
) -> tuple[ConcatDataset, ConcatDataset,
           dict[str, list[int]],
           list[tuple[str, int, int, int]]]:
    """Open every reliable ROI, per-ROI temporal split, return concat datasets.

    Returns
    -------
    train_concat, val_concat : ConcatDataset
    train_roi_to_indices : dict[fid → list of indices INTO train_concat]
        Used by StratifiedROISampler to balance crops across ROIs.
    summary : list of (fid, n_total, n_train, n_val) per ROI included.
    """
    fids = [ln.strip() for ln in reliable_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    train_subs: list[Subset] = []
    val_subs: list[Subset] = []
    train_roi_to_indices: dict[str, list[int]] = {}
    summary: list[tuple[str, int, int, int]] = []
    skipped: list[tuple[str, str]] = []
    train_running_offset = 0

    for fid in fids:
        zarr_path = zarrs_dir / f"{fid}.zarr"
        if not zarr_path.exists():
            skipped.append((fid, "zarr not found"))
            continue
        try:
            # Two instances per ROI: train's epoch follows the training loop
            # (drives the band-gated mask walk), val's stays pinned at 0 so
            # the val loss curve tracks model improvement rather than mask
            # drift. Doubles startup IO; standard pattern.
            ds_train = HLSGeoZarrDataset(
                zarr_path,
                lookback_window=lookback,
                crop_size=crop_size,
                mask_bits=mask_bits,
                input_mask_bits=input_mask_bits,
                qa_channels=qa_channels,
                include_aero_channel=include_aero_channel,
                band_schedule=band_schedule,
            )
            ds_val = HLSGeoZarrDataset(
                zarr_path,
                lookback_window=lookback,
                crop_size=crop_size,
                mask_bits=mask_bits,
                input_mask_bits=input_mask_bits,
                qa_channels=qa_channels,
                include_aero_channel=include_aero_channel,
                band_schedule=band_schedule,
            )
        except Exception as e:                                    # noqa: BLE001
            skipped.append((fid, f"open failed: {type(e).__name__}: {e}"))
            continue
        n = len(ds_train)
        if n == 0:
            skipped.append((fid, "0 eligible (crop, time) pairs"))
            continue
        n_val = max(1, int(round(n * val_frac)))
        n_train = n - n_val
        if n_train == 0:
            skipped.append((fid, f"only {n} sample(s) — no room for split"))
            continue
        train_subs.append(Subset(ds_train, list(range(n_train))))
        val_subs.append(Subset(ds_val,   list(range(n_train, n))))
        # Record this ROI's indices in the *concat* index space.
        train_roi_to_indices[fid] = list(
            range(train_running_offset, train_running_offset + n_train)
        )
        train_running_offset += n_train
        summary.append((fid, n, n_train, n_val))

    if skipped:
        print(f"[data] skipped {len(skipped)} ROI(s):", flush=True)
        for fid, reason in skipped[:30]:
            print(f"  - {fid}: {reason}", flush=True)
        if len(skipped) > 30:
            print(f"  ... and {len(skipped) - 30} more", flush=True)

    return (
        ConcatDataset(train_subs),
        ConcatDataset(val_subs),
        train_roi_to_indices,
        summary,
    )


def compute_band_std(
    train_concat: ConcatDataset, n_samples: int = 256
) -> torch.Tensor:
    """Estimate per-band reflectance standard deviation on a small sample.

    Used to set per-band loss weights to `1/std²` (inverse-variance
    equalisation). Sampling once is cheap and gives a stable estimate;
    fresh estimates per run avoid bundling stats into a static config.
    """
    n = min(n_samples, len(train_concat))
    idxs = np.random.default_rng(0).choice(len(train_concat), size=n, replace=False)
    bands_acc: list[torch.Tensor] = []
    for i in idxs:
        s = train_concat[int(i)]
        bands_acc.append(s["target"].flatten(1))                  # (C, H*W)
    cat = torch.cat(bands_acc, dim=1)                              # (C, N*H*W)
    std = cat.std(dim=1).clamp(min=1e-6)                          # (C,)
    return std


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class StratifiedROISampler(Sampler[int]):
    """Each epoch yields up to `n_per_roi` indices per ROI, drawn at random
    (without replacement) from that ROI's index pool.

    The sample count per epoch is therefore
    `sum(min(n_per_roi, n_crops_in_roi) for roi)` — independent of how
    unbalanced ROI sizes are. ROIs are processed in alphabetical fid order,
    then the full epoch is shuffled to avoid ROI-correlated mini-batches.
    """

    def __init__(self, roi_to_indices: dict[str, list[int]], n_per_roi: int,
                 seed: int = 0):
        self.roi_to_indices = {k: list(v) for k, v in roi_to_indices.items()}
        self.n_per_roi = int(n_per_roi)
        self._epoch = 0
        self._seed = int(seed)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        out: list[int] = []
        for fid in sorted(self.roi_to_indices):
            pool = self.roi_to_indices[fid]
            k = min(self.n_per_roi, len(pool))
            chosen = rng.choice(pool, size=k, replace=False)
            out.extend(int(x) for x in chosen)
        rng.shuffle(out)
        return iter(out)

    def __len__(self) -> int:
        return sum(min(self.n_per_roi, len(v)) for v in self.roi_to_indices.values())


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def weighted_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    band_w: torch.Tensor,
    sample_w_exp: float,
    kind: str = "l1",
) -> torch.Tensor:
    """L1 or L2 over masked pixels, with per-band and per-sample reweighting.

    pred, target : (B, C, H, W)
    mask         : (B, H, W) bool
    band_w       : (C,) per-band weights (e.g., 1/std²), normalised to mean 1
    sample_w_exp : per-sample weight = 1/mask_frac^exp; 0 disables.
    """
    diff = pred - target
    if kind == "l1":
        per_pix = diff.abs()
    elif kind == "l2":
        per_pix = diff.pow(2)
    else:
        raise ValueError(f"unknown loss kind: {kind!r}")

    mb = mask.unsqueeze(1).to(per_pix.dtype)                       # (B, 1, H, W)
    per_pix = per_pix * mb
    per_pix = per_pix * band_w.view(1, -1, 1, 1)

    mask_pixels = mask.flatten(1).sum(dim=1).clamp(min=1).to(per_pix.dtype)
    per_sample = per_pix.sum(dim=(1, 2, 3)) / (mask_pixels * pred.shape[1])

    if sample_w_exp > 0.0:
        mask_frac = (mask_pixels / mask[0].numel()).clamp(min=1e-3)
        sw = mask_frac.pow(-sample_w_exp)
        sw = sw / sw.mean().clamp(min=1e-6)
        per_sample = per_sample * sw

    return per_sample.mean()


def masked_band_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-band masked RMSE → (C,) tensor (unweighted; for logging only)."""
    diff2 = (pred - target).pow(2)
    m = mask.unsqueeze(1).to(diff2.dtype)
    num = (diff2 * m).sum(dim=(0, 2, 3))
    denom = m.sum().clamp(min=1.0)
    return (num / denom).sqrt()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def lr_at(
    step: int,
    lr_max: float,
    warmup: int,
    steps_per_epoch: int,
    schedule: str,
    T0: int,
    T_mult: int,
    total_steps: int,
    anneal_epoch: int = 0,
) -> float:
    """Linear warmup → cosine warm-restart, cosine-to-zero, or
    warm-restart-then-anneal: warm restarts through the curriculum phases,
    then one final restart to lr_max at epoch `anneal_epoch` followed by a
    single cosine decay to ~0 over the mixed phase (consolidation)."""
    if step < warmup:
        return lr_max * (step + 1) / max(1, warmup)
    post = step - warmup
    if schedule == "cosine-to-zero":
        progress = post / max(1, total_steps - warmup)
        progress = min(progress, 1.0)
        return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "warm-restart-then-anneal":
        anneal_post = anneal_epoch * steps_per_epoch
        if post >= anneal_post:
            # Mixed phase: one restart to lr_max at anneal_post, then a single
            # cosine decay to ~0 over the remaining epochs.
            span = max(1, total_steps - warmup - anneal_post)
            progress = min((post - anneal_post) / span, 1.0)
            return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))
        # Curriculum phases: fall through to the warm-restart path below.
    # cosine-warm-restart
    cycle_steps = T0 * steps_per_epoch
    Ti = cycle_steps
    cycle_pos = post
    while cycle_pos >= Ti:
        cycle_pos -= Ti
        Ti *= T_mult
    progress = cycle_pos / max(1, Ti)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _rgb_uint8(refl_chw: torch.Tensor) -> torch.Tensor:
    """N-band reflectance → uint8 (3, H, W) RGB tensor for TB add_image."""
    a = refl_chw.detach().cpu().numpy()
    rgb = a[[2, 1, 0]] if a.shape[0] >= 3 else np.stack([a[0]] * 3)
    rgb = np.clip(rgb / 0.3, 0, 1)
    return torch.from_numpy((rgb * 255).astype(np.uint8))


def _vari(refl_chw: np.ndarray) -> np.ndarray:
    """VARI = (G - R) / (G + R - B). Band order 0=B02, 1=B03, 2=B04, …"""
    blue, green, red = refl_chw[0], refl_chw[1], refl_chw[2]
    return (green - red) / (green + red - blue + 1e-6)


def _vari_uint8(refl_chw: torch.Tensor) -> torch.Tensor:
    a = refl_chw.detach().cpu().numpy().astype(np.float32)
    v = np.clip(_vari(a), -0.5, 0.5)
    rgba = _VARI_CMAP(v + 0.5)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _save_reflectance_tif(
    path: Path,
    refl_chw: torch.Tensor,
    *,
    transform: torch.Tensor | np.ndarray | None = None,
    crs_wkt: str | None = None,
) -> None:
    a = refl_chw.detach().cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    bands, h, w = a.shape
    aff = None
    crs = None
    if transform is not None:
        t = transform.cpu().numpy() if isinstance(transform, torch.Tensor) else np.asarray(transform)
        aff = Affine(*[float(x) for x in t[:6]])
    if crs_wkt:
        crs = rasterio.crs.CRS.from_wkt(crs_wkt)
    profile = dict(
        driver="GTiff", height=h, width=w, count=bands, dtype="float32",
        compress="deflate", predictor=3,
    )
    if aff is not None:
        profile["transform"] = aff
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a)


def log_triplet(
    writer: SummaryWriter,
    step: int,
    inputs_b: torch.Tensor,
    target_b: torch.Tensor,
    pred_b: torch.Tensor,
    mask_b: torch.Tensor,
    *,
    out_channels: int,
    phase: str,
    tif_dir: Path | None = None,
    epoch: int | None = None,
    sample_idx: int = 0,
    tag_suffix: str = "",
    transform: torch.Tensor | None = None,
    crs_wkt: str | None = None,
) -> None:
    target_slot = inputs_b[sample_idx, -1, :out_channels]
    pred = pred_b[sample_idx]
    target = target_b[sample_idx]
    M = mask_b[sample_idx].cpu().numpy().astype(bool)
    sx = tag_suffix

    masked_input_rgb = _rgb_uint8(target_slot).numpy().copy()
    masked_input_rgb[0][M] = 255
    masked_input_rgb[1][M] = 0
    masked_input_rgb[2][M] = 0
    writer.add_image(f"{phase}/rgb/input_masked{sx}", torch.from_numpy(masked_input_rgb), step)
    writer.add_image(f"{phase}/rgb/prediction{sx}",   _rgb_uint8(pred),               step)
    writer.add_image(f"{phase}/rgb/target{sx}",       _rgb_uint8(target),             step)
    writer.add_image(f"{phase}/vari/input_masked{sx}", _vari_uint8(target_slot), step)
    writer.add_image(f"{phase}/vari/prediction{sx}",   _vari_uint8(pred),         step)
    writer.add_image(f"{phase}/vari/target{sx}",       _vari_uint8(target),       step)

    if tif_dir is not None:
        ep_tag = f"ep{epoch:03d}_" if epoch is not None else ""
        stem = f"{ep_tag}step{step:06d}{sx}"
        sub = tif_dir / phase
        _save_reflectance_tif(sub / f"{stem}_input.tif",  target_slot,
                              transform=transform, crs_wkt=crs_wkt)
        _save_reflectance_tif(sub / f"{stem}_pred.tif",   pred,
                              transform=transform, crs_wkt=crs_wkt)
        _save_reflectance_tif(sub / f"{stem}_target.tif", target,
                              transform=transform, crs_wkt=crs_wkt)


# ---------------------------------------------------------------------------
# Resume + run dir scaffolding
# ---------------------------------------------------------------------------


def save_ckpt_atomic(path: Path, state: dict) -> None:
    """Atomic save: write to .tmp, fsync, rename to final.

    Crash mid-write leaves either the old `.pt` intact or `.pt.tmp` (which
    we ignore on resume). No corrupted-checkpoint failure mode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def scaffold_run_dir(run_dir: Path, args: argparse.Namespace) -> None:
    """Create the per-trial run directory layout. Idempotent."""
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        # Minimal YAML emit so we don't pull in PyYAML for one file.
        lines = []
        for k, v in sorted(vars(args).items()):
            if isinstance(v, Path):
                v = str(v)
            if isinstance(v, str):
                lines.append(f"{k}: {v!r}")
            elif v is None:
                lines.append(f"{k}: null")
            elif isinstance(v, bool):
                lines.append(f"{k}: {'true' if v else 'false'}")
            else:
                lines.append(f"{k}: {v}")
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    val_csv = run_dir / "val_history.csv"
    if not val_csv.exists():
        val_csv.write_text(
            "epoch,train_mean,val_loss,val_rmse_mean,lr_at_epoch_end,cycle_end\n",
            encoding="utf-8",
        )


def scaffold_inspect(run_dir: Path, branch: str) -> None:
    """Copy inspect_template.ipynb to <run_dir>/inspect.ipynb with the
    trial's best.pt path pre-substituted. Idempotent: skips if already
    present so user edits survive resume."""
    target = run_dir / "inspect.ipynb"
    if target.exists():
        return
    template = Path(__file__).parent / "inspect_template.ipynb"
    if not template.exists():
        # Soft failure: no template available yet. Training still proceeds.
        print(f"[warn] inspect_template.ipynb not found at {template}; "
              f"skipping per-trial notebook copy.", flush=True)
        return
    nb = json.loads(template.read_text(encoding="utf-8"))
    best_pt = run_dir.absolute() / "checkpoints" / "best.pt"
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_src = []
        for line in cell["source"]:
            new_src.append(
                line.replace("REPLACE_ME_BEFORE_FIRST_RUN", str(best_pt))
                    .replace("REPLACE_ME", branch)
            )
        cell["source"] = new_src
    target.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def resume_from(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    train_concat: ConcatDataset,
    args: argparse.Namespace,
) -> dict:
    """Load checkpoint, restore state, apply resume overrides. Returns a dict
    of resume-relevant state for the training loop to consume."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # 1. Critical-arg guard
    saved = ck.get("args", {})
    mismatches = []
    for k in _CRITICAL_ARGS:
        live = getattr(args, k, None)
        if k in saved and saved[k] != live:
            mismatches.append((k, saved[k], live))
    if mismatches and not args.force_mismatch:
        msg = "Resume-config mismatch on critical args:\n"
        for k, s, l in mismatches:
            msg += f"  {k}: ckpt={s!r} vs cli={l!r}\n"
        msg += "Add --force-mismatch to override (use with care)."
        raise RuntimeError(msg)
    if mismatches and args.force_mismatch:
        for k, s, l in mismatches:
            print(f"[resume][force] mismatch on {k}: ckpt={s!r} vs cli={l!r}", flush=True)

    # 2. RNG state restoration BEFORE anything random fires.
    if "rng_torch" in ck:
        torch.set_rng_state(ck["rng_torch"].cpu())
    if "rng_cuda" in ck and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
    if "rng_numpy" in ck:
        np.random.set_state(ck["rng_numpy"])
    if "rng_python" in ck:
        random.setstate(ck["rng_python"])

    # 3. Model / optim / scheduler.
    # If torch.compile wrapped the model, params are prefixed with _orig_mod.
    # The checkpoint stores unprefixed keys (see ck_state writer below), so
    # we load into the unwrapped module to match.
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(ck["model"], strict=True)
    optimizer.load_state_dict(ck["optimizer"])
    if "scheduler" in ck and ck["scheduler"] is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ck["scheduler"])

    # 4. Dataset epoch state for the band-gated mask.
    n_set = set_epoch_recursive(train_concat, ck["next_epoch"])
    print(f"[resume] set_epoch propagated to {n_set} leaf dataset(s)", flush=True)

    # 5. Early-stop state with optional override.
    cycle_end_history = list(ck.get("cycle_end_history", []))
    patience_counter  = int(ck.get("patience_counter", 0))
    if args.reset_early_stop:
        cycle_end_history, patience_counter = [], 0
        print("[resume] --reset-early-stop: cleared cycle-end history + patience", flush=True)
    elif args.extend_patience > 0:
        patience_counter = max(0, patience_counter - args.extend_patience)
        print(f"[resume] --extend-patience {args.extend_patience}: "
              f"patience_counter now {patience_counter}", flush=True)

    return dict(
        start_epoch=int(ck["next_epoch"]),
        step=int(ck.get("step", 0)),
        best_val=float(ck.get("best_val", float("inf"))),
        best_epoch=int(ck.get("best_epoch", -1)),
        train_losses=list(ck.get("train_losses", [])),
        val_losses=list(ck.get("val_losses", [])),
        cycle_end_history=cycle_end_history,
        patience_counter=patience_counter,
        log_dir=Path(ck["log_dir"]) if "log_dir" in ck else None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    run_dir = args.runs_root / args.branch
    scaffold_run_dir(run_dir, args)
    scaffold_inspect(run_dir, args.branch)

    log_dir = run_dir / "tb" / f"recon_{datetime.now():%Y%m%d-%H%M%S}"
    ckpt_dir = run_dir / "checkpoints"
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    final_path = ckpt_dir / "final.pt"
    tif_dir: Path | None = (
        None if args.no_tifs
        else (args.tif_dir if args.tif_dir is not None
              else run_dir / "recon_tifs")
    )

    print(f"[setup] branch       = {args.branch}")
    print(f"[setup] run_dir      = {run_dir}")
    print(f"[setup] device       = {args.device}")
    print(f"[setup] zarrs_dir    = {args.zarrs_dir}")
    print(f"[setup] reliable     = {args.reliable_list}")
    print(f"[setup] arch         embed={args.embed_dim} heads={args.num_heads} "
          f"t_depth={args.temporal_depth} s_depth={args.spatial_depth} "
          f"dropout={args.dropout} in_ch={args.in_channels} out_ch={args.out_channels}",
          flush=True)
    print(f"[setup] data         lookback={args.lookback} crop={args.crop_size} "
          f"batch={args.batch_size} max_ep={args.max_epochs}", flush=True)
    print(f"[setup] schedule     lr={args.lr} warmup={args.warmup_steps} "
          f"sched={args.lr_schedule} T0={args.T0} T_mult={args.T_mult} "
          f"wd={args.weight_decay} clip={args.clip_norm}", flush=True)
    print(f"[setup] loss         kind={args.loss} band_w={args.band_weighting} "
          f"sample_w_exp={args.sample_weight_exp}", flush=True)
    print(f"[setup] early stop   patience={args.patience_cycles} cycles", flush=True)

    print(f"\n[data] loading ROIs from {args.reliable_list}", flush=True)
    t0 = time.time()
    train_set, val_set, roi_idx, summary = build_datasets(
        args.zarrs_dir, args.reliable_list,
        args.lookback, args.crop_size, args.val_frac,
        _DEFAULT_BAND_SCHEDULE,
        args.mask_bits,
        not args.no_aero_channel,
        args.input_mask_bits,
        args.qa_channels,
    )
    print(f"[data] {len(summary)} ROI(s) loaded in {time.time()-t0:.1f}s", flush=True)
    if len(summary) == 0:
        print("[data] no ROIs available; aborting.", flush=True)
        sys.exit(2)
    n_train, n_val = len(train_set), len(val_set)
    print(f"[data] total samples: train={n_train}  val={n_val}", flush=True)
    smallest = min(summary, key=lambda x: x[1])
    largest = max(summary, key=lambda x: x[1])
    print(f"[data] ROI samples per FID: min={smallest[1]} ({smallest[0]})  "
          f"max={largest[1]} ({largest[0]})  "
          f"median={int(np.median([s[1] for s in summary]))}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Sampler
    if args.sampler == "stratified":
        train_sampler = StratifiedROISampler(roi_idx, args.n_per_roi, seed=args.seed)
        epoch_size = len(train_sampler)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, drop_last=True,
            persistent_workers=False,  # workers re-fork each epoch so __getitem__
                                       # sees the advancing _epoch (curriculum fix)
            # drop_last=True: last-batch shape variance triggers torch.compile
            # to recompile the backward graph at end-of-epoch, which hits a
            # PyTorch Inductor assertion in `analyze_memory_coalescing` /
            # `_solve_simple_expr` (ModularIndexing/FloorDiv in tiling expr).
            # Dropping <batch_size samples per epoch is a 0.03% loss; fixes
            # the recompile crash at root.
        )
    else:
        train_sampler = None
        g = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, generator=g,
            num_workers=args.num_workers, drop_last=True,  # see comment in stratified branch
            persistent_workers=False,  # workers re-fork each epoch so __getitem__
                                       # sees the advancing _epoch (curriculum fix)
        )
        epoch_size = n_train
    # Val-loss subset — a fixed random sample of val_loss_samples indices,
    # to speed up the per-epoch val pass without changing the train/val split.
    # The complement is the held-out TEST set: never iterated during training,
    # written to disk for post-hoc evaluation. Set 0 (or value >= len(val_set))
    # to use the full val set (no held-out test).
    if 0 < args.val_loss_samples < len(val_set):
        loss_rng = np.random.default_rng(args.seed)
        val_loss_indices = loss_rng.choice(
            len(val_set), size=args.val_loss_samples, replace=False).tolist()
        val_loss_set = set(val_loss_indices)
        test_indices = [i for i in range(len(val_set)) if i not in val_loss_set]
        val_loss_dataset = Subset(val_set, val_loss_indices)
        (run_dir / "val_loss_indices.txt").write_text(
            "\n".join(map(str, val_loss_indices)) + "\n", encoding="utf-8")
        (run_dir / "test_indices.txt").write_text(
            "\n".join(map(str, test_indices)) + "\n", encoding="utf-8")
        print(f"[data] val-loss subset: {len(val_loss_indices)} of "
              f"{len(val_set)} samples (seeded); test partition: "
              f"{len(test_indices)} samples (held-out, written to "
              f"run_dir/{{val_loss,test}}_indices.txt)", flush=True)
    else:
        val_loss_dataset = val_set
        val_loss_set = None
        print(f"[data] val-loss: full val set ({len(val_set)} samples; "
              f"no held-out test partition)", flush=True)
    val_loader = DataLoader(
        val_loss_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=True,  # see train_loader comment
        persistent_workers=False,  # consistent with train loader (curriculum fix)
    )

    # Viz loader — pre-pick val_img_samples indices, one per randomly-chosen
    # ROI, RESTRICTED to (a) the val_loss subset (so the held-out test
    # partition is never iterated even for visualization) AND (b) ROIs
    # whose native zarr is at least crop_size x crop_size (smaller ROIs
    # get heavily zero-padded thumbnails that are useless for visual
    # inspection). Picks are stable across epochs (seeded).
    viz_rng = np.random.default_rng(args.seed)
    roi_order = list(range(len(val_set.datasets)))
    viz_rng.shuffle(roi_order)
    viz_indices: list[int] = []
    skipped_small = 0
    for roi_i in roi_order:
        if len(viz_indices) >= args.val_img_samples:
            break
        sub = val_set.datasets[roi_i]
        if len(sub) == 0:
            continue
        # Skip small ROIs — TB thumbnails of a 37x37 ROI padded to 64x64
        # are dominated by the black padding border.
        inner = sub.dataset  # the underlying HLSGeoZarrDataset
        if (inner.ds.sizes["y"] < args.crop_size or
                inner.ds.sizes["x"] < args.crop_size):
            skipped_small += 1
            continue
        start = val_set.cumulative_sizes[roi_i - 1] if roi_i > 0 else 0
        end = val_set.cumulative_sizes[roi_i]
        if val_loss_set is not None:
            pool = [j for j in range(start, end) if j in val_loss_set]
            if not pool:
                continue
            viz_indices.append(pool[int(viz_rng.integers(0, len(pool)))])
        else:
            local_idx = int(viz_rng.integers(0, len(sub)))
            viz_indices.append(start + local_idx)
    viz_subset = Subset(val_set, viz_indices)
    if skipped_small:
        print(f"[data] viz: skipped {skipped_small} ROI(s) smaller than "
              f"{args.crop_size}x{args.crop_size}", flush=True)
    viz_loader = DataLoader(
        viz_subset, batch_size=args.val_img_samples,
        shuffle=False, num_workers=0,
    )
    print(f"[data] viz loader: {len(viz_indices)} sample(s) "
          f"from {len(viz_indices)} different ROI(s)", flush=True)

    steps_per_epoch = math.ceil(epoch_size / args.batch_size)
    total_steps = args.max_epochs * steps_per_epoch
    print(f"[data] sampler={args.sampler}  epoch_size={epoch_size}  "
          f"steps/epoch={steps_per_epoch}  total_steps={total_steps}", flush=True)

    # Model
    model = TimeFirstAutoencoder(
        in_channels=args.in_channels, out_channels=args.out_channels,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        temporal_depth=args.temporal_depth, spatial_depth=args.spatial_depth,
        seq_len=args.lookback, crop_size=args.crop_size, window=args.window,
        mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        identity_bypass=args.identity_bypass,
        spatial_attn=args.spatial_attn,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params = {n_params:,}", flush=True)
    if args.compile == "on":
        model = torch.compile(model)
        print("[model] torch.compile enabled", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Band-std estimate for inverse-variance loss weighting
    if args.band_weighting == "uniform":
        band_w = torch.ones(args.out_channels, device=args.device)
    else:
        print("[data] estimating per-band std on a 256-sample subset...", flush=True)
        std = compute_band_std(train_set, n_samples=256).to(args.device)
        if args.band_weighting == "inv-var":
            band_w = 1.0 / std.pow(2)
        else:
            band_w = 1.0 / std
        band_w = band_w / band_w.mean().clamp(min=1e-6)
    print(f"[loss] band_w = {band_w.cpu().tolist()}", flush=True)

    # Resume
    train_losses: list[float] = []
    val_losses: list[float] = []
    cycle_end_history: list[float] = []
    patience_counter = 0
    step = 0
    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    resume_path: Path | None = None
    if not args.no_resume:
        resume_path = args.resume if args.resume is not None else (
            last_path if last_path.exists() else None
        )
    if resume_path is not None and resume_path.exists():
        rs = resume_from(resume_path, model, optimizer, None, train_set, args)
        start_epoch = rs["start_epoch"]
        step = rs["step"]
        best_val = rs["best_val"]
        best_epoch = rs["best_epoch"]
        train_losses = rs["train_losses"]
        val_losses = rs["val_losses"]
        cycle_end_history = rs["cycle_end_history"]
        patience_counter = rs["patience_counter"]
        if rs["log_dir"] is not None:
            log_dir = rs["log_dir"]
        print(f"[resume] from {resume_path}: epoch {start_epoch}/{args.max_epochs}, "
              f"step {step}, best_val={best_val:.5f} @ ep{best_epoch}, "
              f"patience_counter={patience_counter}", flush=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[tb] log_dir = {log_dir}", flush=True)

    if start_epoch >= args.max_epochs:
        print("[done] already at max_epochs.", flush=True)
        writer.close()
        return

    # --------------------- Train loop ---------------------
    val_csv = run_dir / "val_history.csv"
    print("\n[train] starting", flush=True)
    train_t0 = time.time()
    # Pin val datasets at epoch 0 once for the entire run — train and val use
    # separate HLSGeoZarrDataset instances (see build_datasets) so this stays
    # frozen while train's epoch advances.
    set_epoch_recursive(val_set, 0)
    for epoch in range(start_epoch, args.max_epochs):
        # Push current epoch into every per-ROI training dataset (drives the
        # band-gated mask schedule).
        set_epoch_recursive(train_set, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_t0 = time.time()
        epoch_losses: list[float] = []

        for batch_idx, batch in enumerate(train_loader):
            b = {k: v.to(args.device, non_blocking=True)
                 for k, v in batch.items() if isinstance(v, torch.Tensor)}
            cur_lr = lr_at(step, args.lr, args.warmup_steps, steps_per_epoch,
                           args.lr_schedule, args.T0, args.T_mult, total_steps,
                           args.lr_anneal_epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                pred = model(b["inputs"], b["dates"])
                loss = weighted_recon_loss(
                    pred, b["target"], b["mask"],
                    band_w=band_w, sample_w_exp=args.sample_weight_exp,
                    kind=args.loss,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("lr", cur_lr, step)
            with torch.no_grad():
                rmse_bands = masked_band_rmse(pred, b["target"], b["mask"])
                for i, r in enumerate(rmse_bands):
                    writer.add_scalar(f"rmse_train/band{i}", r.item(), step)
                writer.add_scalar("rmse_train/mean", rmse_bands.mean().item(), step)
                writer.add_scalar("mask/frac", b["mask"].float().mean().item(), step)

            if step % args.img_log_every == 0:
                with torch.no_grad():
                    tr0 = b["transform"][0] if "transform" in b else None
                    crs0 = batch["crs_wkt"][0] if "crs_wkt" in batch else None
                    log_triplet(
                        writer, step, b["inputs"], b["target"], pred, b["mask"],
                        out_channels=args.out_channels,
                        phase="train", tif_dir=tif_dir, epoch=epoch,
                        transform=tr0, crs_wkt=crs0,
                    )

            if step % args.print_every == 0:
                print(f"[step {step:>6d}  ep {epoch:>3d}  "
                      f"batch {batch_idx + 1:>4d}/{steps_per_epoch}]  "
                      f"loss={loss.item():.5f}  lr={cur_lr:.2e}  "
                      f"mask_frac={b['mask'].float().mean().item():.3f}",
                      flush=True)
            step += 1

        train_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_dt = time.time() - epoch_t0

        # ---- Validation ----
        model.eval()
        val_t0 = time.time()
        v_losses: list[float] = []
        v_rmse_acc: list[torch.Tensor] = []
        v_cache: dict[str, list] = {"inputs": [], "target": [], "pred": [], "mask": [],
                                    "tform": [], "crs": []}
        with torch.no_grad():
            for batch in val_loader:
                b = {k: v.to(args.device, non_blocking=True)
                     for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    pred = model(b["inputs"], b["dates"])
                    v_loss_b = weighted_recon_loss(
                        pred, b["target"], b["mask"],
                        band_w=band_w, sample_w_exp=args.sample_weight_exp,
                        kind=args.loss,
                    )
                v_losses.append(v_loss_b.item())
                v_rmse_acc.append(masked_band_rmse(pred, b["target"], b["mask"]).cpu())

            # Populate v_cache from the pre-picked diverse viz indices.
            for batch in viz_loader:
                b = {k: v.to(args.device, non_blocking=True)
                     for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    pred = model(b["inputs"], b["dates"])
                v_cache["inputs"].append(b["inputs"].detach())
                v_cache["target"].append(b["target"].detach())
                v_cache["pred"].append(pred.detach())
                v_cache["mask"].append(b["mask"].detach())
                if "transform" in batch:
                    v_cache["tform"].append(batch["transform"])
                if "crs_wkt" in batch:
                    v_cache["crs"].extend(batch["crs_wkt"])
                break  # batch_size == val_img_samples, so one batch is enough

        v_loss = float(np.mean(v_losses)) if v_losses else float("nan")
        v_rmse_per_band = (torch.stack(v_rmse_acc).mean(dim=0).tolist()
                          if v_rmse_acc else [float("nan")] * args.out_channels)
        v_rmse_mean = float(np.mean(v_rmse_per_band))
        val_losses.append(v_loss)
        writer.add_scalar("loss/val", v_loss, epoch)
        writer.add_scalar("rmse_val/mean", v_rmse_mean, epoch)
        for i, r in enumerate(v_rmse_per_band):
            writer.add_scalar(f"rmse_val/band{i}", r, epoch)

        if v_cache["inputs"]:
            v_inputs = torch.cat(v_cache["inputs"], dim=0)
            v_target = torch.cat(v_cache["target"], dim=0)
            v_pred   = torch.cat(v_cache["pred"],   dim=0)
            v_mask   = torch.cat(v_cache["mask"],   dim=0)
            v_tform  = torch.cat(v_cache["tform"], dim=0) if v_cache["tform"] else None
            n_log = v_inputs.shape[0]
            with torch.no_grad():
                for i in range(n_log):
                    log_triplet(
                        writer, step, v_inputs, v_target, v_pred, v_mask,
                        out_channels=args.out_channels,
                        phase="val", tif_dir=tif_dir, epoch=epoch,
                        sample_idx=i, tag_suffix=f"_{i}",
                        transform=(v_tform[i] if v_tform is not None else None),
                        crs_wkt=(v_cache["crs"][i] if i < len(v_cache["crs"]) else None),
                    )
        val_dt = time.time() - val_t0

        # ---- Cycle-aware early stopping ----
        # Only counts cycle-ends at/after --early-stop-start-epoch, so a
        # curriculum warmup phase (e.g. low band) runs without early-stop.
        next_epoch_count = epoch + 1
        cycle_end = (next_epoch_count % args.T0 == 0)
        early_stop_active = epoch >= args.early_stop_start_epoch
        cycle_label = ""
        if cycle_end and early_stop_active:
            cycle_end_history.append(v_loss)
            best_so_far = min(cycle_end_history)
            if v_loss <= best_so_far + 1e-12:
                patience_counter = 0
                cycle_label = f"  cycle-end best ({len(cycle_end_history)} cycle ends)"
            else:
                patience_counter += 1
                cycle_label = f"  cycle-end no-improve ({patience_counter}/{args.patience_cycles})"
        elif cycle_end:
            cycle_label = "  cycle-end (early-stop inactive: pre-start-epoch)"

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            best_epoch = epoch
        print(f"[epoch {epoch:>3d}/{args.max_epochs - 1}]  "
              f"train_mean={train_mean:.5f}  val={v_loss:.5f}  "
              f"val_rmse_mean={v_rmse_mean:.5f}  "
              f"per_band={[round(r, 4) for r in v_rmse_per_band]}  "
              f"train_t={epoch_dt:.1f}s  val_t={val_dt:.1f}s  "
              f"step={step}{'  *NEW BEST*' if improved else ''}{cycle_label}",
              flush=True)

        with val_csv.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_mean:.6f},{v_loss:.6f},{v_rmse_mean:.6f},"
                    f"{cur_lr:.6e},{int(cycle_end)}\n")

        # ---- Checkpoint ----
        ck_state = {
            "next_epoch": epoch + 1,
            "step": step,
            "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "cycle_end_history": cycle_end_history,
            "patience_counter": patience_counter,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "log_dir": str(log_dir),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "args": vars(args),
        }
        save_ckpt_atomic(last_path, ck_state)
        if improved:
            save_ckpt_atomic(best_path, ck_state)
        # Always preserve a per-cycle snapshot at every cycle end (T0 epochs)
        # so the cycle-to-cycle progression can be inspected later.
        if cycle_end:
            save_ckpt_atomic(ckpt_dir / f"cycle_ep{epoch:03d}.pt", ck_state)
        # Preserve every new "best val" inside the warm-restart-then-anneal
        # mixed phase (ep >= lr_anneal_epoch), so each consolidation
        # improvement is kept (not just the global best.pt).
        if (improved and args.lr_schedule == "warm-restart-then-anneal"
                and epoch >= args.lr_anneal_epoch):
            save_ckpt_atomic(ckpt_dir / f"best_ep{epoch:03d}.pt", ck_state)
        if args.archive_every > 0 and (epoch + 1) % args.archive_every == 0:
            save_ckpt_atomic(ckpt_dir / f"ep_{epoch:03d}.pt", ck_state)

        # ---- Stop condition ----
        if cycle_end and early_stop_active and patience_counter >= args.patience_cycles:
            print(f"[early-stop] patience exhausted after {patience_counter} "
                  f"cycle-ends without improvement (cycles: {len(cycle_end_history)})",
                  flush=True)
            break

    # Save final.pt regardless of why we exited.
    save_ckpt_atomic(final_path, ck_state)
    writer.close()
    total_dt = time.time() - train_t0
    n_ep = epoch + 1 - start_epoch
    print(f"\n[done] {n_ep} epoch(s) in {total_dt:.1f}s "
          f"({total_dt / max(1, n_ep):.1f}s/epoch)", flush=True)
    if val_losses:
        print(f"[done] final val: {val_losses[-1]:.5f}  "
              f"best val: {best_val:.5f} @ ep{best_epoch}", flush=True)


if __name__ == "__main__":
    main()
