# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""train_extend.py -- scheduler for the 460-epoch pretraining curriculum.

Reads `next_epoch` from `<DATA_DIR>/runs/pretraining/checkpoints/last.pt`, works out which
stages remain, and runs them in order. With no checkpoint present it starts at the base run.

    ep0-99 (base) -> ep140 -> ep180 -> ep220 -> ep260 -> ep360 -> ep460

The learning rate follows `warm-restart-then-anneal` throughout. In the base run it warm-restarts
every 10 epochs (T0=10) to epoch 59, then restarts once more at 60 and anneals to ~0 by 99; epoch
60 is also where the masking curriculum moves to the full-range mixed band. Each later stage is
one restart-and-anneal cycle: 40 epochs through ep260, 100 epochs for ep360 and ep460.

Passing `--lr-anneal-epoch N` retriggers that lambda's restart-to-lr_max + cosine-anneal branch;
the anneal span is `(max_epochs - anneal_epoch) * spe - warmup`. `lr_schedule`, `lr_anneal_epoch`
and `max_epochs` are outside train.py's `_CRITICAL_ARGS`, so overriding them on resume needs no
`--force-mismatch`.

Runtime on one H100: the 100-epoch stages take about 17.5 h each (~628 s/epoch), the 40-epoch
stages proportionally less. All stages are GPU-bound (100% utilisation, 90/96 GB), so
`num_workers` above 4 buys no throughput; it affects input-pipeline speed only, not the
optimisation, and any value reproduces the same model.

Cross-platform notes:
  - Launches train.py via `sys.executable` rather than `uv run`: this script already runs inside
    the target interpreter, and `uv run` does not resolve identically from a subprocess on every
    OS.
  - Each stage runs in the foreground and tees stdout to the console and to the stage log.
    Background this script yourself with tmux/screen (Linux/macOS) or `start /b` (Windows).
  - On Windows `--compile on` is downgraded to `--compile off` (Triton/Inductor is unavailable
    there); the paper's numbers come from a Linux run with compile on.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

RECIPES = Path(__file__).resolve().parent
PRETRAINING_DIR = RECIPES.parent
TRAIN_PY = PRETRAINING_DIR / "train.py"
BRANCH = "pretraining"


@dataclass(frozen=True)
class Stage:
    max_epochs: int
    lr_anneal_epoch: int   # == the epoch this stage resumes from, for every EXTENSION stage
    num_workers: int
    note: str            # what the stage does, shown by --dry-run
    resume: bool = True    # False only for BASE, which starts from scratch


# The base run, ep0-99. Its launch is not lost: 04_pretraining/config/config.yaml is the
# config train.py wrote for it, and it records `resume: null`, `max_epochs: 100` and
# `lr_anneal_epoch: 60` -- warm restarts through ep59, then a single anneal to ep99. Every
# parameter that _stage_argv does not pass explicitly has a train.py default equal to that
# file's value (in_channels and input_mask_bits are derived at parse time, from
# --no-aero-channel and --mask-bits respectively), so this Stage reproduces it exactly.
BASE = Stage(100, 60, 2, "base run: warm restarts ep0-59, then one anneal ep60-99 "
                         "(config/config.yaml records this launch)", resume=False)
BASE_EPOCHS = BASE.max_epochs

SCHEDULE: tuple[Stage, ...] = (
    Stage(140, 100, 24, "40-epoch restart-and-anneal cycle, ep100-139"),
    Stage(180, 140, 4,  "40-epoch restart-and-anneal cycle, ep140-179"),
    Stage(220, 180, 4,  "40-epoch restart-and-anneal cycle, ep180-219"),
    Stage(260, 220, 4,  "40-epoch restart-and-anneal cycle, ep220-259"),
    Stage(360, 260, 4,  "100-epoch restart-and-anneal cycle, ep260-359 (~17.5 h)"),
    Stage(460, 360, 4,  "100-epoch restart-and-anneal cycle, ep360-459 (~17.5 h)"),
)

# The table must not be internally inconsistent: every stage's lr_anneal_epoch
# has to equal the epoch the checkpoint will be at when that stage starts,
# i.e. the previous stage's max_epochs (or BASE_EPOCHS for the first stage).
for _i, _stage in enumerate(SCHEDULE):
    _expected = BASE_EPOCHS if _i == 0 else SCHEDULE[_i - 1].max_epochs
    assert _stage.lr_anneal_epoch == _expected, (
        f"SCHEDULE inconsistency at index {_i} (ep{_stage.max_epochs}): "
        f"lr_anneal_epoch={_stage.lr_anneal_epoch} but expected {_expected}"
    )
del _i, _stage, _expected

# BASE first, then the extensions. BASE is exempt from the assertion above: its
# lr_anneal_epoch (60) is a point INSIDE the stage, where warm restarts give way to the
# anneal, not a junction with a preceding stage.
ALL_STAGES: tuple[Stage, ...] = (BASE,) + SCHEDULE


def _read_next_epoch(ckpt: Path) -> int:
    """Read `next_epoch` out of a pretraining checkpoint.

    weights_only=False is required: the checkpoint's state dict also carries
    numpy RNG state and vars(args) (which includes Path objects) -- see
    train.py:1199-1218 (writer) -- so this mirrors train.py's own loader at
    train.py:714 exactly.
    """
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}, and it was expected to exist. Run this scheduler "
            f"with no checkpoint present to train the base run (ep0-99) from scratch."
        ) from exc
    return int(state["next_epoch"])


def _stage_argv(stage: Stage, data_dir: Path) -> list[str]:
    """Build the exact train.py argv for one stage. Nothing here varies except
    max_epochs / lr_anneal_epoch / num_workers, which come from the Stage."""
    ckpt = data_dir / "runs" / "pretraining" / "checkpoints" / "last.pt"
    zarrs = data_dir / "earthaccess_clustered_zarrs"
    reliable = zarrs / "reliable_rois.txt"
    runs_root = data_dir / "runs"

    compile_mode = "off" if platform.system() == "Windows" else "on"

    return [
        sys.executable, str(TRAIN_PY),
        *(["--resume", str(ckpt)] if stage.resume else []),
        "--zarrs-dir", str(zarrs),
        "--reliable-list", str(reliable),
        "--branch", BRANCH,
        "--runs-root", str(runs_root),
        "--embed-dim", "64",
        "--num-heads", "2",
        "--bf16",
        "--mask-bits", "0b10001111",
        "--no-aero-channel",
        "--max-epochs", str(stage.max_epochs),
        "--lr-anneal-epoch", str(stage.lr_anneal_epoch),
        "--lr-schedule", "warm-restart-then-anneal",
        "--batch-size", "24",
        "--num-workers", str(stage.num_workers),
        "--compile", compile_mode,
        "--spatial-attn", "global",
        "--lr", "2e-4",
        "--early-stop-start-epoch", "999",
        "--no-tifs",
        "--val-img-samples", "4",
    ]


def _run_stage(stage: Stage, data_dir: Path) -> int:
    """Launch one stage in the foreground, teeing its output to console + log.
    Returns the child process's exit code."""
    argv = _stage_argv(stage, data_dir)

    if platform.system() == "Windows":
        print("[schedule] Windows detected: --compile on -> off (Triton/Inductor is not "
              "available there; the paper's numbers come from a Linux run with compile on)",
              flush=True)

    log = data_dir / "runs" / BRANCH / f"train_extend_ep{stage.max_epochs}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUTF8"] = "1"   # avoid cp1252 UnicodeEncodeError on Windows when stdout is redirected

    with log.open("a", encoding="utf-8") as logf:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        rc = proc.wait()
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic scheduler for the pretraining-extension curriculum "
                     "(ep0-99 base -> ep140 -> ep180 -> ep220 -> ep260 -> ep360 -> ep460)."
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                         help="Override DATA_DIR (env var DATA_DIR takes precedence over the "
                              "./data default; this flag takes precedence over both).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print every remaining stage's exact train.py argv, annotated with "
                              "its rationale, and exit without touching a checkpoint or running "
                              "anything.")
    parser.add_argument("--one", action="store_true",
                         help="Run just the next stage, then stop (the historical workflow: "
                              "launch, inspect curves, continue).")
    parser.add_argument("--until", type=int, default=None,
                         choices=[s.max_epochs for s in ALL_STAGES],
                         help="Stop after this stage completes.")
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else Path(os.environ.get("DATA_DIR", "./data"))

    if args.dry_run:
        # Must work in a completely empty clone: no checkpoint, no data required.
        for stage in ALL_STAGES:
            print(f"# ep{stage.max_epochs}: {stage.note}")
            print(" ".join(_stage_argv(stage, data_dir)))
        return 0

    ckpt = data_dir / "runs" / "pretraining" / "checkpoints" / "last.pt"
    # No checkpoint at all means nothing has been trained yet: start at epoch 0 and let the
    # loop pick BASE. Only a checkpoint that exists but cannot be read is an error.
    epoch = _read_next_epoch(ckpt) if ckpt.exists() else 0

    for stage in ALL_STAGES:
        if epoch >= stage.max_epochs:
            print(f"[schedule] ep{stage.max_epochs} already complete "
                  f"(checkpoint next_epoch={epoch})", flush=True)
            continue

        if not (stage.lr_anneal_epoch <= epoch < stage.max_epochs):
            print(f"[schedule] refusing to launch ep{stage.max_epochs}: "
                  f"checkpoint next_epoch={epoch}, stage ep{stage.max_epochs} expects start "
                  f"{stage.lr_anneal_epoch}", file=sys.stderr)
            return 1

        if epoch > stage.lr_anneal_epoch:
            print(f"[schedule] resuming ep{stage.max_epochs} MID-STAGE from epoch {epoch} "
                  f"(interrupted stage restart; train.py handles this natively)", flush=True)

        rc = _run_stage(stage, data_dir)
        if rc != 0:
            print(f"[schedule] stage ep{stage.max_epochs} exited rc={rc}; aborting the chain",
                  file=sys.stderr)
            return rc

        epoch = _read_next_epoch(ckpt)
        if epoch != stage.max_epochs:
            print(f"[schedule] post-stage check failed for ep{stage.max_epochs}: expected "
                  f"checkpoint next_epoch={stage.max_epochs}, got {epoch} "
                  f"(did train.py exit without training? see its '[done] already at max_epochs' "
                  f"path)", file=sys.stderr)
            return 1

        if args.one:
            print(f"[schedule] --one: stopping after ep{stage.max_epochs}", flush=True)
            return 0
        if args.until is not None and stage.max_epochs == args.until:
            print(f"[schedule] --until {args.until}: stopping", flush=True)
            return 0

    print(f"[schedule] all stages complete; next_epoch={ALL_STAGES[-1].max_epochs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
