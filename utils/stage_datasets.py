"""
Stage external dataset files into the framework-local `dataset/` folder.

This script is intentionally conservative:
- Default mode is to *symlink* to avoid copying large artifacts.
- Use `--copy` to physically copy files/directories.

Usage:
  python utils/stage_datasets.py --link
  python utils/stage_datasets.py --copy --include-ultrachat
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "dataset"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _rm_if_exists(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        return


def _link_or_copy(src: Path, dst: Path, *, copy: bool, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        _rm_if_exists(dst)

    _ensure_dir(dst.parent)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return

    # Prefer symlinks to avoid huge copies.
    os.symlink(src, dst, target_is_directory=src.is_dir())


def _maybe(src: Optional[Path]) -> Optional[Path]:
    if src is None:
        return None
    return src if src.exists() else None


def _glob_existing(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        try:
            if p.exists():
                out.append(p)
        except Exception:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage datasets into DeepScan/dataset/")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--link", action="store_true", help="Symlink files/directories (default)")
    mode.add_argument("--copy", action="store_true", help="Copy files/directories")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination dataset directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing staged files")
    parser.add_argument(
        "--include-ultrachat",
        action="store_true",
        help="Stage UltraChat local cache dir for X-Boundary (can be very large)",
    )

    # Allow explicit overrides for non-standard source locations
    parser.add_argument("--tellme-dir", default="/root/code/TELLME", help="Path to TELLME repo (contains test.csv)")
    parser.add_argument("--spin-dir", default="/root/code/SPIN/data", help="Path to SPIN CSV directory")
    parser.add_argument("--xboundary-dir", default="/root/code/X-Boundary/data", help="Path to X-Boundary data directory")
    parser.add_argument("--mipeaks-csv", default="/root/code/MI-Peaks/src/data/math_train_12k.csv", help="Path to MI-Peaks CSV")

    args = parser.parse_args()
    copy = bool(args.copy)
    dest = Path(args.dest).expanduser().resolve()
    overwrite = bool(args.overwrite)

    _ensure_dir(dest)

    # --- TELLME ---
    tellme_dir = Path(args.tellme_dir)
    tellme_dst = dest / "tellme"
    _ensure_dir(tellme_dst)
    for name in ("test.csv", "train.csv"):
        src = _maybe(tellme_dir / name)
        if src:
            _link_or_copy(src, tellme_dst / name, copy=copy, overwrite=overwrite)

    # --- SPIN ---
    spin_dir = Path(args.spin_dir)
    spin_dst = dest / "spin"
    _ensure_dir(spin_dst)
    # Stage all CSVs we can find (small + large samples); keeps filenames stable.
    if spin_dir.exists():
        for csv_path in sorted(spin_dir.glob("*.csv")):
            _link_or_copy(csv_path, spin_dst / csv_path.name, copy=copy, overwrite=overwrite)

    # --- MI-PEAKS ---
    mi_csv = Path(args.mipeaks_csv)
    mi_dst = dest / "mi_peaks"
    _ensure_dir(mi_dst)
    if mi_csv.exists():
        _link_or_copy(mi_csv, mi_dst / "math_train_12k.csv", copy=copy, overwrite=overwrite)

    # --- X-BOUNDARY ---
    xb_src = Path(args.xboundary_dir)
    xb_dst = dest / "xboundary"
    _ensure_dir(xb_dst)
    # Required jsons
    for fname in ("circuit_breakers_train_2400.json", "ORbench_retain_set.json"):
        src = _maybe(xb_src / fname)
        if src:
            _link_or_copy(src, xb_dst / fname, copy=copy, overwrite=overwrite)

    # Optional local UltraChat cache dir
    if args.include_ultrachat:
        ultrachat_src = _maybe(xb_src / "ultrachat_200k_local")
        if ultrachat_src:
            _link_or_copy(ultrachat_src, xb_dst / "ultrachat_200k_local", copy=copy, overwrite=overwrite)

    print(f"Staging complete → {dest}")
    print("Tip: set your YAML dataset paths to point inside this folder.")


if __name__ == "__main__":
    main()

