#!/usr/bin/env python3
# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""download_record.py -- fetch and extract this paper's companion Zenodo record.

Runs on any machine (Linux, Windows, macOS) and needs no credentials: the record is a
published, public, open-access Zenodo deposit, and this script never sends an
Authorization header. Every archive it downloads is verified twice before use -- against
the checksum Zenodo itself reports, and against data/checksums.sha256 in this repository,
so a corrupted or tampered download fails loudly instead of silently producing wrong
results. data/record_manifest.json maps archive filenames to categories; it is the only
source of truth for that mapping, so no naming convention is ever guessed.

Categories: pretraining-corpus, labelled-dataset, reference-data, fm-checkpoint,
regressor-checkpoints. Not every downstream step needs every category -- for example,
`labelled-dataset,reference-data` alone is enough to run 07_detector/sweep_kernels.py:
it consumes the already-shipped dense-prediction cache (dense_predictions.json, part of
reference-data) and the zarr stores it re-derives features from (labelled-dataset), and
does not need fm-checkpoint or regressor-checkpoints at all.

Usage:
    uv run python 00_data/download_record.py --record 1234567
    uv run python 00_data/download_record.py --record 10.5281/zenodo.1234567
    uv run python 00_data/download_record.py --record https://zenodo.org/records/1234567
    uv run python 00_data/download_record.py --record 1234567 --sandbox
    uv run python 00_data/download_record.py --record 1234567 --categories labelled-dataset,reference-data
    uv run python 00_data/download_record.py --record 1234567 --list
    uv run python 00_data/download_record.py --archives-dir /path/to/local/zips --categories reference-data

Exit codes: 0 success, 1 verification/extraction failure, 2 usage/argument error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS_PATH = REPO_ROOT / "data" / "checksums.sha256"
RECORD_MANIFEST_PATH = REPO_ROOT / "data" / "record_manifest.json"

CATEGORY_NAMES = [
    "pretraining-corpus", "labelled-dataset", "reference-data", "fm-checkpoint", "regressor-checkpoints",
]
# The two zarr-chunk-heavy categories are stored uncompressed (their chunks are already
# compressed), the rest are deflated -- used ONLY as a rough uncompressed-size estimate for
# the disk-space preflight below, nothing more. Archive names always come from
# record_manifest.json, never from a guessed naming convention.
STORED_CATEGORIES = {"pretraining-corpus", "labelled-dataset"}

PRODUCTION_API = "https://zenodo.org/api/"
SANDBOX_API = "https://sandbox.zenodo.org/api/"

PROGRESS_INTERVAL_S = 2.0
CHUNK_SIZE = 1024 * 1024
MAX_DOWNLOAD_ATTEMPTS = 2  # 1 try + 1 retry, per spec


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _human_size(n: Optional[float]) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.2f} {unit}"
        n /= 1024.0
    return f"{n:,.2f} TiB"


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds or seconds < 0:  # NaN/negative guard
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def hash_file(path: Path) -> tuple[str, str]:
    """Single pass over path -> (md5_hex, sha256_hex)."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


# ---------------------------------------------------------------------------
# --record parsing: bare id / DOI (prod or sandbox) / URL
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"10\.(5281|5072)/zenodo\.(\d+)")
_URL_ID_RE = re.compile(r"records?/(\d+)")


def parse_record_ref(raw: str) -> tuple[int, Optional[bool]]:
    """Parse a bare numeric id, a Zenodo DOI, or a Zenodo record URL.

    Returns (record_id, sandbox_hint): sandbox_hint is True/False if the input itself
    disambiguated it (a 10.5072 DOI or a sandbox.zenodo.org URL means sandbox; a 10.5281
    DOI or a zenodo.org URL means production), or None if it didn't (a bare numeric id).
    """
    s = raw.strip()
    if s.isdigit():
        return int(s), None

    m = _DOI_RE.search(s)
    if m:
        prefix, num = m.group(1), m.group(2)
        return int(num), (prefix == "5072")

    if "sandbox.zenodo.org" in s:
        sandbox_hint: Optional[bool] = True
    elif "zenodo.org" in s:
        sandbox_hint = False
    else:
        sandbox_hint = None

    m = _URL_ID_RE.search(s)
    if m:
        return int(m.group(1)), sandbox_hint

    raise ValueError(
        f"Could not parse a Zenodo record id out of {raw!r}. Expected a bare numeric id "
        "(e.g. 1234567), a DOI (10.5281/zenodo.1234567 for production, "
        "10.5072/zenodo.1234567 for sandbox), or a URL "
        "(https://zenodo.org/records/1234567 or https://sandbox.zenodo.org/records/1234567)."
    )


def resolve_sandbox(cli_sandbox: bool, hint: Optional[bool]) -> bool:
    """Combine --sandbox with whatever parse_record_ref's hint says. The only real
    conflict is --sandbox given alongside an unambiguously-production reference."""
    if hint is None:
        return cli_sandbox
    if cli_sandbox and not hint:
        raise ValueError(
            "--sandbox was given, but --record resolved to a production Zenodo reference "
            "(zenodo.org / a 10.5281 DOI). Drop --sandbox, or pass a sandbox reference "
            "(sandbox.zenodo.org / a 10.5072 DOI) instead."
        )
    return hint


# ---------------------------------------------------------------------------
# Zenodo API (unauthenticated, public-record reads only)
# ---------------------------------------------------------------------------

def fetch_record(base_url: str, record_id: int) -> dict:
    url = f"{base_url}records/{record_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "files" not in data:
        raise RuntimeError(
            f"Zenodo record response has no 'files' key -- top-level keys found: "
            f"{sorted(data.keys())}. URL: {url}"
        )
    return data


def file_download_url(file_entry: dict) -> str:
    links = file_entry.get("links", {})
    for key in ("content", "self"):
        if isinstance(links, dict) and key in links:
            return links[key]
    raise RuntimeError(
        f"Could not find a download link on file entry {file_entry.get('key', '?')!r}. "
        f"Entry keys: {sorted(file_entry.keys())}. "
        f"links: {sorted(links.keys()) if isinstance(links, dict) else links!r}"
    )


def file_checksum_md5(file_entry: dict) -> Optional[str]:
    chk = file_entry.get("checksum")
    if isinstance(chk, str) and chk.startswith("md5:"):
        return chk[len("md5:"):]
    return None


# ---------------------------------------------------------------------------
# data/checksums.sha256 + data/record_manifest.json (repo-shipped, read-only here)
# ---------------------------------------------------------------------------

def load_checksums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. It ships with this repository and holds the reference "
            "checksums for every archive in the record; without it nothing can be "
            "verified, so this is a hard precondition rather than a skippable step."
        )
    checksums: dict[str, str] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or not digest or not name:
            raise ValueError(f"Malformed line {i} in {path}: {line!r} "
                              f"(expected GNU coreutils format: '<sha256>  <filename>')")
        checksums[name] = digest
    return checksums


def load_record_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. It ships with this repository and maps each archive "
            "filename to its category; without it archives cannot be grouped."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_archive_plan(manifest: dict, categories: list[str]) -> list[tuple[str, str]]:
    """[(archive_name, category), ...] for the requested categories, per record_manifest.json."""
    plan = []
    cats = manifest.get("categories", {})
    for cat in categories:
        if cat not in cats:
            print(f"WARNING: category {cat!r} has no archives in {RECORD_MANIFEST_PATH} -- skipping",
                  file=sys.stderr)
            continue
        for name in cats[cat]["archives"]:
            plan.append((name, cat))
    return plan


# ---------------------------------------------------------------------------
# Download with resume
# ---------------------------------------------------------------------------

def _print_progress(label: str, done: int, total: Optional[int], done_this_run: int, elapsed: float) -> None:
    pct = f"{100 * done / total:5.1f}%" if total else "   ?  "
    rate = done_this_run / elapsed if elapsed > 0 else 0.0
    rate_s = f"{_human_size(rate)}/s" if rate > 0 else "-- /s"
    eta = _format_eta((total - done) / rate) if (total and rate > 0) else "--:--"
    print(f"  {label}: {_human_size(done)} / {_human_size(total)} ({pct})  {rate_s}  ETA {eta}", flush=True)


def download_with_resume(url: str, dest_part: Path, expected_size: Optional[int], label: str) -> None:
    dest_part.parent.mkdir(parents=True, exist_ok=True)
    resume_from = dest_part.stat().st_size if dest_part.exists() else 0
    headers = {}
    mode = "wb"
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resume_from > 0 and resp.status_code == 200:
            # Server ignored the Range request -- the body is the FULL file from byte 0,
            # not a continuation. Discard whatever partial bytes we had and restart clean.
            print(f"  {label}: server did not honour Range resume (got 200, not 206) -- "
                  "restarting this file from scratch", flush=True)
            resume_from = 0
            mode = "wb"
        elif resp.status_code not in (200, 206):
            resp.raise_for_status()

        total: Optional[int] = expected_size
        if "Content-Length" in resp.headers:
            content_len = int(resp.headers["Content-Length"])
            total = resume_from + content_len if resp.status_code == 206 else content_len

        downloaded_this_run = 0
        start = time.monotonic()
        last_print = start
        with dest_part.open(mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded_this_run += len(chunk)
                now = time.monotonic()
                if now - last_print >= PROGRESS_INTERVAL_S:
                    _print_progress(label, resume_from + downloaded_this_run, total,
                                     downloaded_this_run, now - start)
                    last_print = now
        _print_progress(label, resume_from + downloaded_this_run, total,
                         downloaded_this_run, max(time.monotonic() - start, 1e-9))


# ---------------------------------------------------------------------------
# Path-traversal-safe extraction
# ---------------------------------------------------------------------------

def _is_safe_member_name(name: str) -> bool:
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:", name):  # Windows drive letter, e.g. "C:\evil"
        return False
    if name.startswith("\\\\"):  # UNC path
        return False
    # Zip member names conventionally use '/', but a crafted zip isn't obliged to --
    # and this script targets Windows too, where '\' is the real separator. Check both,
    # not just the posix one, so a "..\x" member can't slip past as a single opaque
    # component (it would be harmless-by-accident on Linux, but not on Windows).
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return ".." not in parts


def validate_zip_members(zf: zipfile.ZipFile, data_dir: Path) -> list[str]:
    problems = []
    data_dir_resolved = data_dir.resolve()
    for info in zf.infolist():
        name = info.filename
        if not _is_safe_member_name(name):
            problems.append(f"unsafe member path: {name!r}")
            continue
        target = (data_dir / name).resolve()
        if not target.is_relative_to(data_dir_resolved):
            problems.append(f"member resolves outside data-dir: {name!r} -> {target}")
    return problems


def extract_zip(zip_path: Path, data_dir: Path) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        problems = validate_zip_members(zf, data_dir)
        if problems:
            shown = "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else "")
            raise RuntimeError(f"{len(problems)} unsafe member(s), refusing to extract: {shown}")
        members = zf.infolist()
        data_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(data_dir)
        return len(members)


# ---------------------------------------------------------------------------
# Completion stamps
# ---------------------------------------------------------------------------

def stamp_path_for(data_dir: Path, archive_name: str) -> Path:
    return data_dir / ".zenodo_fetch" / f"{archive_name}.json"


def read_valid_stamp(stamp_path: Path, expected_sha256: str) -> bool:
    if not stamp_path.is_file():
        return False
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stamp.get("sha256") == expected_sha256


def write_stamp(stamp_path: Path, sha256: str, size: int, member_count: int) -> None:
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = {
        "sha256": sha256,
        "size": size,
        "extracted_utc": datetime.now(timezone.utc).isoformat(),
        "member_count": member_count,
    }
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-archive task + pipeline
# ---------------------------------------------------------------------------

@dataclass
class ArchiveTask:
    name: str
    category: str
    expected_sha256: str
    download_url: Optional[str] = None
    zenodo_size: Optional[int] = None
    zenodo_md5: Optional[str] = None
    local_source: Optional[Path] = None


def process_archive(task: ArchiveTask, data_dir: Path, keep_archives: bool, force: bool) -> bool:
    stamp_path = stamp_path_for(data_dir, task.name)
    if not force and read_valid_stamp(stamp_path, task.expected_sha256):
        print(f"{task.name}: already fetched and verified -- skipping (--force to redo)", flush=True)
        return True

    if task.local_source is not None:
        return _process_local_archive(task, data_dir, stamp_path)
    return _process_remote_archive(task, data_dir, stamp_path, keep_archives)


def _process_local_archive(task: ArchiveTask, data_dir: Path, stamp_path: Path) -> bool:
    zip_path = task.local_source
    assert zip_path is not None
    if not zip_path.is_file():
        print(f"{task.name}: not found in --archives-dir ({zip_path})", file=sys.stderr)
        return False
    _, sha256_actual = hash_file(zip_path)
    if sha256_actual != task.expected_sha256:
        print(f"{task.name}: sha256 mismatch in --archives-dir mode "
              f"(expected {task.expected_sha256}, got {sha256_actual}). FAILED.", file=sys.stderr)
        return False
    try:
        member_count = extract_zip(zip_path, data_dir)
    except Exception as e:
        print(f"{task.name}: extraction FAILED: {e}", file=sys.stderr)
        return False
    write_stamp(stamp_path, task.expected_sha256, zip_path.stat().st_size, member_count)
    print(f"{task.name}: extracted {member_count} member(s) from --archives-dir, stamp written", flush=True)
    return True  # never delete the user's own --archives-dir source


def _process_remote_archive(task: ArchiveTask, data_dir: Path, stamp_path: Path, keep_archives: bool) -> bool:
    downloads_dir = data_dir / "downloads"
    zip_path = downloads_dir / task.name
    part_path = downloads_dir / (task.name + ".part")
    assert task.download_url is not None

    ok = False
    attempt = 0
    while attempt < MAX_DOWNLOAD_ATTEMPTS:
        attempt += 1
        try:
            if not zip_path.is_file():
                print(f"{task.name}: downloading ({_human_size(task.zenodo_size)}) ...", flush=True)
                download_with_resume(task.download_url, part_path, task.zenodo_size, task.name)
                part_path.rename(zip_path)
            md5_actual, sha256_actual = hash_file(zip_path)
            if task.zenodo_md5 is None:
                print(f"{task.name}: Zenodo reported no md5 checksum for this file -- "
                      "skipping the fast md5 pre-check, verifying sha256 only", flush=True)
            elif md5_actual != task.zenodo_md5:
                raise ValueError(f"md5 mismatch (Zenodo reports {task.zenodo_md5}, got {md5_actual})")
            if sha256_actual != task.expected_sha256:
                raise ValueError(f"sha256 mismatch (data/checksums.sha256 expects "
                                  f"{task.expected_sha256}, got {sha256_actual})")
            ok = True
            break
        except Exception as e:
            print(f"{task.name}: attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS} failed: {e}", file=sys.stderr)
            zip_path.unlink(missing_ok=True)
            part_path.unlink(missing_ok=True)

    if not ok:
        print(f"{task.name}: FAILED after {attempt} attempt(s)", file=sys.stderr)
        return False

    try:
        member_count = extract_zip(zip_path, data_dir)
    except Exception as e:
        print(f"{task.name}: extraction FAILED: {e}", file=sys.stderr)
        return False

    write_stamp(stamp_path, task.expected_sha256, zip_path.stat().st_size, member_count)
    print(f"{task.name}: extracted {member_count} member(s), stamp written", flush=True)

    if not keep_archives:
        zip_path.unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# Disk-space preflight
# ---------------------------------------------------------------------------

def preflight_disk_space(data_dir: Path, tasks: list[ArchiveTask]) -> Optional[str]:
    """Returns an error message if there's not enough free space, else None."""
    total_needed = 0
    for t in tasks:
        zip_size = t.local_source.stat().st_size if t.local_source else (t.zenodo_size or 0)
        # ZIP_STORED categories: extraction needs ~the same footprint as the zip itself.
        # ZIP_DEFLATED (small reference-data/checkpoint categories): conservative 2x headroom
        # since we can't know the true uncompressed size before downloading the zip's
        # central directory.
        extract_multiplier = 1.0 if t.category in STORED_CATEGORIES else 2.0
        total_needed += int(zip_size * extract_multiplier)
        # The zip only occupies data_dir when we download it. Under --archives-dir it is read
        # in place from wherever the caller keeps it, so charging for it here doubled the
        # requirement: staging the 20.5 GiB pretraining corpus from local archives asked for
        # 40.9 GiB free and refused a run that would have fit.
        if t.local_source is None:
            total_needed += zip_size

    data_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_dir)
    if usage.free < total_needed:
        return (f"Insufficient disk space at {data_dir}: need ~{_human_size(total_needed)}, "
                f"only {_human_size(usage.free)} free. Aborting before starting any download.")
    return None


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def do_list(record_id: Optional[int], sandbox: bool, archives_dir: Optional[Path]) -> int:
    if archives_dir is not None:
        try:
            manifest = load_record_manifest(RECORD_MANIFEST_PATH)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        cats = manifest.get("categories", {})
        files_meta = manifest.get("files", {})
        print(f"Archives expected per {RECORD_MANIFEST_PATH} (offline mode, source: {archives_dir}):")
        for cat in CATEGORY_NAMES:
            if cat not in cats:
                continue
            print(f"\n{cat}:")
            for name in cats[cat]["archives"]:
                local = archives_dir / name
                status = "present" if local.is_file() else "MISSING"
                size = files_meta.get(name, {}).get("size_bytes")
                print(f"  {name}  ({_human_size(size)})  [{status}]")
        return 0

    if record_id is None:
        print("ERROR: --list without --archives-dir needs --record", file=sys.stderr)
        return 2

    base = SANDBOX_API if sandbox else PRODUCTION_API
    try:
        record = fetch_record(base, record_id)
    except Exception as e:
        print(f"ERROR: could not fetch record {record_id} from {base}: {e}", file=sys.stderr)
        return 1
    files_by_key = {f["key"]: f for f in record["files"]}

    manifest = None
    if RECORD_MANIFEST_PATH.is_file():
        manifest = load_record_manifest(RECORD_MANIFEST_PATH)

    if manifest:
        cats = manifest.get("categories", {})
        for cat in CATEGORY_NAMES:
            if cat not in cats:
                continue
            print(f"\n{cat}:")
            for name in cats[cat]["archives"]:
                f = files_by_key.get(name)
                size = _human_size(f["size"]) if f else "NOT ON RECORD"
                print(f"  {name}  ({size})")
    else:
        print(f"{RECORD_MANIFEST_PATH} not found -- listing raw record file entries "
              "(no category grouping available):")
        for key in sorted(files_by_key):
            print(f"  {key}  ({_human_size(files_by_key[key].get('size'))})")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download and extract this paper's companion Zenodo record.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--record", default=None,
                    help="Zenodo record id, DOI (10.5281/zenodo.NNNNNN or "
                         "10.5072/zenodo.NNNNNN), or URL. Required unless --archives-dir is "
                         "given (archives-dir mode never talks to Zenodo).")
    p.add_argument("--sandbox", action="store_true",
                    help="Use sandbox.zenodo.org instead of zenodo.org")
    p.add_argument("--categories", default="all",
                    help=f"Comma-separated category names, or 'all'. One of: {CATEGORY_NAMES}")
    p.add_argument("--data-dir", type=Path, default=Path("./data"),
                    help="Destination directory for extracted data")
    p.add_argument("--archives-dir", type=Path, default=None,
                    help="Read already-downloaded zips from this local directory instead of "
                         "the network (offline test mode)")
    p.add_argument("--keep-archives", action="store_true",
                    help="Don't delete downloaded zips after successful extraction")
    p.add_argument("--force", action="store_true",
                    help="Ignore existing completion stamps; re-verify and re-extract anyway")
    p.add_argument("--list", action="store_true",
                    help="Print available categories/archives and exit, no download")
    return p


def _resolve_categories(raw: str) -> Optional[list[str]]:
    if raw == "all":
        return list(CATEGORY_NAMES)
    names = [c.strip() for c in raw.split(",") if c.strip()]
    bad = [c for c in names if c not in CATEGORY_NAMES]
    if bad:
        print(f"ERROR: unknown categor{'y' if len(bad) == 1 else 'ies'}: {bad}. "
              f"Valid: {CATEGORY_NAMES}", file=sys.stderr)
        return None
    return names


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    categories = _resolve_categories(args.categories)
    if categories is None:
        return 2

    # --archives-dir never talks to Zenodo, so it never needs --record -- not just when
    # combined with --list, but for a real (offline) extraction run too.
    if args.record is None and args.archives_dir is None:
        print("ERROR: --record is required unless --archives-dir is given "
              "(--archives-dir mode never talks to Zenodo, live or --list)", file=sys.stderr)
        return 2

    record_id: Optional[int] = None
    sandbox = args.sandbox
    if args.record is not None:
        try:
            record_id, sandbox_hint = parse_record_ref(args.record)
            sandbox = resolve_sandbox(args.sandbox, sandbox_hint)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    if args.list:
        return do_list(record_id, sandbox, args.archives_dir)

    try:
        checksums = load_checksums(CHECKSUMS_PATH)
        manifest = load_record_manifest(RECORD_MANIFEST_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    plan_entries = resolve_archive_plan(manifest, categories)
    if not plan_entries:
        print("Nothing to do: no archives resolved for the requested categories.", file=sys.stderr)
        return 1

    live_files_by_key: dict[str, dict] = {}
    if args.archives_dir is None:
        base = SANDBOX_API if sandbox else PRODUCTION_API
        try:
            record = fetch_record(base, record_id)
        except Exception as e:
            print(f"ERROR: could not fetch record {record_id} from {base}: {e}", file=sys.stderr)
            return 1
        live_files_by_key = {f["key"]: f for f in record["files"]}

    tasks: list[ArchiveTask] = []
    for name, cat in plan_entries:
        if name not in checksums:
            print(f"ERROR: {name} has no entry in {CHECKSUMS_PATH} -- cannot verify, skipping",
                  file=sys.stderr)
            continue
        expected_sha256 = checksums[name]
        if args.archives_dir is not None:
            tasks.append(ArchiveTask(name=name, category=cat, expected_sha256=expected_sha256,
                                      local_source=args.archives_dir / name))
        else:
            fentry = live_files_by_key.get(name)
            if fentry is None:
                print(f"ERROR: {name} is not present on the live Zenodo record ({record_id})",
                      file=sys.stderr)
                continue
            try:
                url = file_download_url(fentry)
            except RuntimeError as e:
                print(f"ERROR: {name}: {e}", file=sys.stderr)
                continue
            tasks.append(ArchiveTask(name=name, category=cat, expected_sha256=expected_sha256,
                                      download_url=url, zenodo_size=fentry.get("size"),
                                      zenodo_md5=file_checksum_md5(fentry)))

    if not tasks:
        print("No archives could be resolved to a download source. Aborting.", file=sys.stderr)
        return 1

    pending = [t for t in tasks
               if args.force or not read_valid_stamp(stamp_path_for(args.data_dir, t.name), t.expected_sha256)]
    err = preflight_disk_space(args.data_dir, pending)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    failures = []
    for t in tasks:
        if not process_archive(t, args.data_dir, args.keep_archives, args.force):
            failures.append(t.name)

    if failures:
        print(f"\n{len(failures)}/{len(tasks)} archive(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(tasks)} archive(s) OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
