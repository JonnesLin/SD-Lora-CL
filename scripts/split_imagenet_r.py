#!/usr/bin/env python3
"""
Split an ImageNet-R style folder (synset subdirs) into train/test.

- Default split ratio: 80% train / 20% test (to match TFDS usage in the code).
- Deterministic per-class shuffle with a global seed.
-
Expected folder structure before:
  <root>/n01443537/*.jpg
  <root>/n01484850/*.jpg
  ...

After:
  <root>/train/n01443537/*.jpg
  <root>/train/n01484850/*.jpg
  <root>/test/n01443537/*.jpg
  <root>/test/n01484850/*.jpg

Notes:
  - By default we MOVE files (storage-friendly). Use --copy to keep originals.
  - If <root>/train or <root>/test already exist and are non-empty, use
    --overwrite to remove and recreate them.
  - Writes a split_manifest.json with counts, seed, and ratio.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIFF", ".GIF",
}


def is_synset_dir(name: str) -> bool:
    """Return True if `name` looks like an ImageNet synset directory.

    Typical pattern is 'n' followed by 8 digits, e.g., n01443537
    """
    return bool(re.fullmatch(r"n\d{8}", name))


def list_class_dirs(root: Path) -> List[Path]:
    """List synset-style class directories under `root`.

    Skips pre-existing 'train'/'test' folders and any non-synset dirs.
    """
    result = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name in {"train", "test", "val", "validation"}:
            continue
        if is_synset_dir(p.name):
            result.append(p)
    result.sort()
    return result


def list_images(d: Path) -> List[Path]:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
    files.sort()
    return files


@dataclass
class SplitArgs:
    root: Path
    train_ratio: float = 0.8
    seed: int = 0
    copy: bool = False
    overwrite: bool = False
    dry_run: bool = False
    ensure_min_test: bool = True  # ensure at least 1 test when class has >=2


def prepare_out_dirs(root: Path, overwrite: bool) -> Tuple[Path, Path]:
    train_dir = root / "train"
    test_dir = root / "test"
    for d in (train_dir, test_dir):
        if d.exists():
            # check if non-empty
            non_empty = any(True for _ in d.rglob("*"))
            if non_empty and not overwrite:
                raise SystemExit(
                    f"Output dir '{d}' already exists and is non-empty. Use --overwrite to replace.")
            if overwrite:
                shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    return train_dir, test_dir


def split_class(
    cls_dir: Path,
    train_dir: Path,
    test_dir: Path,
    args: SplitArgs,
) -> Tuple[int, int]:
    images = list_images(cls_dir)
    if not images:
        return 0, 0

    rnd = random.Random(f"{args.seed}-{cls_dir.name}")
    rnd.shuffle(images)

    n = len(images)
    n_train = int(n * args.train_ratio)
    if args.ensure_min_test and n >= 2:
        # keep at least 1 for test and at least 1 for train
        n_train = max(1, min(n - 1, n_train))

    train_subset = images[:n_train]
    test_subset = images[n_train:]

    out_train_cls = train_dir / cls_dir.name
    out_test_cls = test_dir / cls_dir.name
    out_train_cls.mkdir(parents=True, exist_ok=True)
    out_test_cls.mkdir(parents=True, exist_ok=True)

    mover = shutil.copy2 if args.copy else shutil.move
    for src in train_subset:
        dst = out_train_cls / src.name
        if not args.dry_run:
            mover(src, dst)
    for src in test_subset:
        dst = out_test_cls / src.name
        if not args.dry_run:
            mover(src, dst)

    return len(train_subset), len(test_subset)


def maybe_remove_empty_dirs(root: Path) -> None:
    """Remove any now-empty synset dirs under root (best-effort)."""
    for p in root.iterdir():
        if p.is_dir() and is_synset_dir(p.name):
            try:
                next(p.iterdir())
            except StopIteration:
                # empty
                try:
                    p.rmdir()
                except Exception:
                    pass


def run(args: SplitArgs) -> None:
    if not args.root.exists() or not args.root.is_dir():
        raise SystemExit(f"Root not found or not a directory: {args.root}")

    classes = list_class_dirs(args.root)
    if not classes:
        raise SystemExit(
            f"No synset-style class dirs found under {args.root}. Expected e.g. n01443537, n02113799, ...")

    train_dir, test_dir = prepare_out_dirs(args.root, args.overwrite)

    total_train = 0
    total_test = 0
    per_class_counts: Dict[str, Dict[str, int]] = {}

    for i, cls_dir in enumerate(classes, 1):
        t_cnt, v_cnt = split_class(cls_dir, train_dir, test_dir, args)
        total_train += t_cnt
        total_test += v_cnt
        per_class_counts[cls_dir.name] = {"train": t_cnt, "test": v_cnt}
        print(f"[{i:03d}/{len(classes)}] {cls_dir.name}: train={t_cnt} test={v_cnt}")

    manifest = {
        "root": str(args.root),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "copy": args.copy,
        "total": {"train": total_train, "test": total_test},
        "per_class": per_class_counts,
    }

    if not args.dry_run:
        with open(args.root / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # If we moved files, remove now-empty original class dirs for tidiness.
        if not args.copy:
            maybe_remove_empty_dirs(args.root)

    print("\nDone.")
    print(f"Total train images: {total_train}")
    print(f"Total test  images: {total_test}")
    print(f"Manifest written to: {(args.root / 'split_manifest.json')}\n")


def parse_args(argv: List[str]) -> SplitArgs:
    p = argparse.ArgumentParser(
        description="Split ImageNet-R style folder into train/test.")
    p.add_argument("--root", required=True, type=Path,
                   help="Dataset root containing synset subfolders.")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Train ratio (default 0.8 => 80/20 split).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for deterministic per-class shuffle.")
    p.add_argument("--copy", action="store_true",
                   help="Copy instead of move (keeps originals; uses more space).")
    p.add_argument("--overwrite", action="store_true",
                   help="If train/test already exist, remove them before splitting.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute split and print summary without moving/copying files.")
    p.add_argument("--no-ensure-min-test", action="store_true",
                   help="Do not force at least 1 test image when class has >=2.")
    args_ns = p.parse_args(argv)
    return SplitArgs(
        root=args_ns.root,
        train_ratio=args_ns.train_ratio,
        seed=args_ns.seed,
        copy=args_ns.copy,
        overwrite=args_ns.overwrite,
        dry_run=args_ns.dry_run,
        ensure_min_test=not args_ns.no_ensure_min_test,
    )


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])

