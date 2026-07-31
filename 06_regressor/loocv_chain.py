# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""loocv_chain.py -- definitive LOOCV launcher. Runs N folds CONCURRENTLY on one GPU,
throttling on the REAL GPU compute-process count (nvidia-smi) so it coexists with
externally-launched folds (e.g. manual measurement folds) without exceeding N or
duplicating ROIs.

Order: VISIBLE (raw) first; the recon/masked CACHE is built ONLY AFTER raw (GPU free),
before the variants that need it (all/recon/masked).

Usage:
    python loocv_chain.py [variants...]
    uv run python loocv_chain.py raw all recon masked composite
    python loocv_chain.py --n 3 --exclude 11874_curr,11984_curr,11310_curr
    python loocv_chain.py --serial          # no GPU polling: 1 fold at a time (no NVIDIA GPU)

Each fold trains into <data-dir>/runs/<prefix>-<variant>-loo<roi>/ and writes its console
output to a .log of the same name alongside it. Every flag also reads a matching
environment variable (PREFIX, EXCLUDE, DATA_DIR, PAT, NW, N) as its default, since
`NAME=value cmd` is not valid syntax in Windows' cmd.exe.

Notes:
  - DATA_DIR is resolved to an absolute path up front and passed to every child by
    environment variable, so the launcher works from any working directory.
  - The liveness check that skips folds already running elsewhere uses psutil:
    os.kill(pid, 0) is not a liveness probe on Windows (it can only send
    CTRL_C/CTRL_BREAK there -- any other value actually KILLS the process), and /proc
    is Linux-only.
  - --serial runs one fold at a time and never calls nvidia-smi: use it on machines
    with no NVIDIA GPU (e.g. macOS/Apple Silicon).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

RECIPE_DIR = Path(__file__).resolve().parent
TRAIN_CLASSIFIER_PY = RECIPE_DIR / "train_classifier.py"
BUILD_CACHES_PY = RECIPE_DIR / "build_caches.py"
DEFAULT_VARIANTS = ["raw", "all", "recon", "masked", "composite"]

_CHILDREN: list[subprocess.Popen] = []   # every Popen this process launches; terminated on Ctrl-C


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("variants", nargs="*", default=list(DEFAULT_VARIANTS),
                    help=f"Variants to run, in order (default: {' '.join(DEFAULT_VARIANTS)}).")
    p.add_argument("--n", type=int, default=int(os.environ.get("N", 3)),
                    help="Max concurrent GPU jobs (env N). Default 3.")
    p.add_argument("--prefix", default=os.environ.get("PREFIX", "regressor"),
                    help="Run namespace: branch + log-file prefix (env PREFIX). The default names "
                         "the run directories agg_oof.py and the detector stage look for.")
    p.add_argument("--exclude", default=os.environ.get("EXCLUDE", "11874_curr,11984_curr,11310_curr"),
                    help="Comma-separated positive ROIs excluded ENTIRELY (env EXCLUDE). Default "
                         "excludes the three cloud-affected ROIs, matching the paper's reported folds.")
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "./data")),
                    help="Data directory (env DATA_DIR). Resolved to an absolute path.")
    p.add_argument("--patience", type=int, default=int(os.environ.get("PAT", 35)),
                    help="--patience-epochs passed to train_classifier.py (env PAT).")
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NW", 4)),
                    help="--num-workers passed to train_classifier.py (env NW).")
    p.add_argument("--serial", action="store_true",
                    help="No GPU-count polling: launch one fold at a time via p.wait(). "
                         "Use on machines without nvidia-smi (e.g. macOS/Apple Silicon).")
    return p


def load_rois(data_dir: Path, exclude: set[str]) -> list[str]:
    ps = json.loads((data_dir / "annotations.json").read_text(encoding="utf-8"))["logging_activity"]
    vd = json.loads((data_dir / "view_dates.json").read_text(encoding="utf-8"))
    return sorted(a for a in ps if a in vd and a not in exclude)


def gpu_count() -> int:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                        capture_output=True, text=True)
    return sum(1 for ln in r.stdout.splitlines() if ln.strip())


def is_running(branch: str) -> bool:
    """Portable substitute for `pgrep -f "train_classifier.py .*--branch $BR( |$)"`."""
    pat = re.compile(rf"train_classifier\.py .*--branch {re.escape(branch)}( |$)")
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if pat.search(" ".join(cmdline)):
            return True
    return False


def ensure_cache(data_dir: Path, exclude: str, runs_log_dir: Path) -> None:
    """Build the recon+masked cache once (after raw, GPU free). Raises SystemExit(1) on failure,
    matching the .sh's `... || exit 1`."""
    done = data_dir / "pred_cache_recon" / ".done"
    if done.exists():
        return
    print("[chain] building recon+masked cache (GPU free post-raw)...", flush=True)
    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    env["EXCLUDE"] = exclude
    env["PYTHONUTF8"] = "1"
    log_path = runs_log_dir / "build_caches.log"
    with log_path.open("w", encoding="utf-8") as logf:
        rc = subprocess.run([sys.executable, str(BUILD_CACHES_PY), "--which", "both"],
                             stdout=logf, stderr=subprocess.STDOUT, env=env).returncode
    if rc == 0:
        done.touch()
        print("[chain] cache done", flush=True)
    else:
        print("[chain] CACHE FAILED (see build_caches.log)", flush=True)
        raise SystemExit(1)


def launch_fold(variant: str, roi: str, args: argparse.Namespace, runs_dir: Path,
                 runs_log_dir: Path, data_dir: Path) -> tuple[str, subprocess.Popen, object] | None:
    """One fold; skip if already trained OR already running (externally). Returns
    (branch, Popen, open log file handle) or None if skipped."""
    branch = f"{args.prefix}-{variant}-loo{roi}"
    if (runs_dir / branch / "checkpoints" / "best.pt").exists():
        print(f"[chain] SKIP {branch} (done)", flush=True)
        return None
    if is_running(branch):
        print(f"[chain] SKIP {branch} (running)", flush=True)
        return None
    print(f"[chain] start {branch}", flush=True)
    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    env["PYTHONUTF8"] = "1"
    cmd = [sys.executable, str(TRAIN_CLASSIFIER_PY),
           "--variant", variant, "--loo-roi", roi, "--branch", branch,
           "--exclude-rois", args.exclude,
           "--unfreeze-spatial", "--bf16", "--lr", "1e-5", "--weight-decay", "0.1",
           "--head-hidden", "256", "--patience-epochs", str(args.patience),
           "--num-workers", str(args.num_workers)]
    log_path = runs_log_dir / f"{args.prefix}-{variant}-loo{roi}.log"
    logf = log_path.open("w", encoding="utf-8")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    _CHILDREN.append(p)
    return branch, p, logf


def _reap(launched: list[tuple[str, subprocess.Popen, object]]) -> None:
    """Non-blocking: remove finished own-Popens from `launched`, closing their log and
    printing [chain] FAILED for nonzero exit -- the Python equivalent of the bash
    `launch_fold ... || echo FAILED`, done opportunistically while we'd otherwise just sleep."""
    still = []
    for branch, p, logf in launched:
        rc = p.poll()
        if rc is None:
            still.append((branch, p, logf))
        else:
            logf.close()
            if rc != 0:
                print(f"[chain] FAILED {branch}", flush=True)
    launched[:] = still


def _wait_all(launched: list[tuple[str, subprocess.Popen, object]]) -> None:
    for branch, p, logf in launched:
        rc = p.wait()
        logf.close()
        if rc != 0:
            print(f"[chain] FAILED {branch}", flush=True)
    launched.clear()


def run_variant(variant: str, args: argparse.Namespace, runs_dir: Path, runs_log_dir: Path,
                 data_dir: Path, rois: list[str]) -> None:
    print(f"[chain] === variant {variant} (N={args.n}{', serial' if args.serial else ''}) ===",
          flush=True)
    launched: list[tuple[str, subprocess.Popen, object]] = []
    for roi in rois:
        branch = f"{args.prefix}-{variant}-loo{roi}"
        if (runs_dir / branch / "checkpoints" / "best.pt").exists():
            # already done -> skip instantly (no 100s GPU-register wait)
            print(f"[chain] SKIP {branch} (done)", flush=True)
            continue

        if args.serial:
            result = launch_fold(variant, roi, args, runs_dir, runs_log_dir, data_dir)
            if result is None:
                continue
            branch, p, logf = result
            rc = p.wait()
            logf.close()
            if rc != 0:
                print(f"[chain] FAILED {branch}", flush=True)
            continue

        while gpu_count() >= args.n:       # wait for a free GPU slot
            _reap(launched)
            time.sleep(20)
        before = gpu_count()
        result = launch_fold(variant, roi, args, runs_dir, runs_log_dir, data_dir)
        if result is None:
            continue
        branch, p, logf = result
        launched.append((branch, p, logf))
        w = 0                              # let the new job register on the GPU
        while gpu_count() <= before and w < 100:
            if p.poll() is not None:       # crashed before registering: don't wait the full 100s
                break
            time.sleep(10); w += 10
    _wait_all(launched)
    print(f"[chain] === variant {variant} complete ===", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    exclude = set(a for a in args.exclude.split(",") if a)
    runs_dir = data_dir / "runs"
    runs_log_dir = data_dir / "runs"
    runs_log_dir.mkdir(parents=True, exist_ok=True)

    if not args.serial and shutil.which("nvidia-smi") is None:
        raise SystemExit("[chain] nvidia-smi not found on PATH. On a machine without an NVIDIA "
                          "GPU (e.g. macOS/Apple Silicon), rerun with --serial.")

    rois = load_rois(data_dir, exclude)
    print(f"[chain] {len(rois)} ROIs, variants: {' '.join(args.variants)}, N={args.n} concurrent"
          f"{' (serial)' if args.serial else ''}", flush=True)

    try:
        for variant in args.variants:
            if variant in ("all", "recon", "masked"):
                ensure_cache(data_dir, args.exclude, runs_log_dir)
            run_variant(variant, args, runs_dir, runs_log_dir, data_dir, rois)
    except KeyboardInterrupt:
        for p in _CHILDREN:
            if p.poll() is None:
                p.terminate()
        raise
    print("[chain] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
