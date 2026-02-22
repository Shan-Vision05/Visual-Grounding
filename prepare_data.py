#!/usr/bin/env python3
"""Download and prepare RefCOCOg data for Visual Grounding training.

This script handles everything end-to-end:
  1. Downloads COCO 2014 train images  (~13 GB zip, ~83k images)
  2. Downloads RefCOCOg annotations    (from refer / refcocog-umd split)
  3. Converts into the flat format expected by VG_Dataset:
       data/{train,test}/images/<image_id>.jpg
       data/{train,test}/annotations_1.json

Each annotation entry: {"image_id": <int>, "text": <str>, "bbox": [x, y, w, h]}

Usage
-----
  # Run from the repo root (local or on a compute node)
  python prepare_data.py                       # defaults to ./data
  python prepare_data.py --data_dir /scratch/alpine/$USER/vg_data

On the cluster, run this BEFORE submitting the training job, e.g. from an
`acompile` or `sinteractive` session so you have internet access.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* (skip if file exists)."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["wget", "--quiet", "--show-progress", "-O", str(dest), url],
    )


def _unzip(src: Path, dest: Path) -> None:
    """Extract *src* zip into *dest* (skip if dest already populated)."""
    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {dest} already extracted")
        return
    print(f"  Extracting {src.name} ...")
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dest)


# ---------------------------------------------------------------------------
# RefCOCOg loading (handles both .p pickle and .json)
# ---------------------------------------------------------------------------
REFCOCOG_URLS = {
    # UMD split (Google-refexp partition, most common in papers)
    "refs": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    "instances": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"


def _load_refs(refs_path: Path) -> list[dict]:
    """Load RefCOCOg refs from pickle or JSON."""
    suffix = refs_path.suffix.lower()
    if suffix == ".p" or suffix == ".pkl":
        with open(refs_path, "rb") as f:
            return pickle.load(f)
    elif suffix == ".json":
        with open(refs_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown refs format: {refs_path}")


def _load_instances(instances_path: Path) -> dict[int, dict]:
    """Load COCO instances and return {ann_id: annotation} mapping."""
    with open(instances_path, "r") as f:
        data = json.load(f)
    return {ann["id"]: ann for ann in data["annotations"]}


# ---------------------------------------------------------------------------
# Build annotations
# ---------------------------------------------------------------------------
def _build_split_annotations(
    refs: list[dict],
    coco_anns: dict[int, dict],
    split: str,
) -> list[dict]:
    """Convert RefCOCOg refs for a given split into our flat format."""
    entries: list[dict] = []
    for ref in refs:
        if ref.get("split") != split:
            continue
        ann_id = ref["ann_id"]
        if ann_id not in coco_anns:
            continue
        coco_ann = coco_anns[ann_id]
        bbox = coco_ann["bbox"]  # [x, y, w, h] — already in COCO format
        image_id = ref["image_id"]
        for sentence in ref.get("sentences", []):
            text = sentence.get("sent") or sentence.get("raw", "")
            if not text:
                continue
            entries.append({
                "image_id": image_id,
                "text": text,
                "bbox": bbox,
            })
    return entries


# ---------------------------------------------------------------------------
# Symlink / copy images
# ---------------------------------------------------------------------------
def _link_images(
    annotations: list[dict],
    coco_img_dir: Path,
    dest_dir: Path,
) -> None:
    """Create symlinks (or copies) for only the images we need."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    needed_ids = {ann["image_id"] for ann in annotations}
    linked = 0
    for img_id in needed_ids:
        # COCO 2014 filename convention
        src = coco_img_dir / f"COCO_train2014_{img_id:012d}.jpg"
        dst = dest_dir / f"{img_id}.jpg"
        if dst.exists():
            continue
        if not src.exists():
            continue
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            # Filesystem doesn't support symlinks (e.g. some scratch mounts)
            shutil.copy2(src, dst)
        linked += 1
    print(f"  Linked/copied {linked} new images → {dest_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RefCOCOg data")
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Root output directory (will contain train/ and test/)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./.cache",
        help="Where to cache downloaded zips",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # ── 1. Download COCO 2014 train images ──────────────────────────
    print("\n[1/4] COCO 2014 train images")
    coco_zip = cache / "train2014.zip"
    _download(COCO_IMAGES_URL, coco_zip)
    coco_extract = cache / "coco"
    _unzip(coco_zip, coco_extract)
    coco_img_dir = coco_extract / "train2014"
    if not coco_img_dir.exists():
        # Some zips nest differently
        alt = list(coco_extract.rglob("train2014"))
        if alt:
            coco_img_dir = alt[0]
        else:
            print("ERROR: Could not find train2014/ after extraction.", file=sys.stderr)
            sys.exit(1)
    print(f"  COCO images at: {coco_img_dir}")

    # ── 2. Download RefCOCOg annotations ────────────────────────────
    print("\n[2/4] RefCOCOg annotations")
    refcocog_zip = cache / "refcocog.zip"
    _download(REFCOCOG_URLS["refs"], refcocog_zip)
    refcocog_extract = cache / "refcocog"
    _unzip(refcocog_zip, refcocog_extract)

    # Find refs file (could be refs(umd).p, refs(google).p, etc.)
    refs_candidates = list(refcocog_extract.rglob("refs*.p")) + list(refcocog_extract.rglob("refs*.json"))
    # Prefer UMD split
    refs_path = None
    for c in refs_candidates:
        if "umd" in c.name.lower():
            refs_path = c
            break
    if refs_path is None and refs_candidates:
        refs_path = refs_candidates[0]
    if refs_path is None:
        print("ERROR: No refs file found in RefCOCOg download.", file=sys.stderr)
        sys.exit(1)
    print(f"  Using refs: {refs_path}")

    # COCO instances for bbox lookup
    instances_zip = cache / "annotations_trainval2014.zip"
    _download(REFCOCOG_URLS["instances"], instances_zip)
    instances_extract = cache / "coco_ann"
    _unzip(instances_zip, instances_extract)
    instances_path = instances_extract / "annotations" / "instances_train2014.json"
    if not instances_path.exists():
        instances_path = list(instances_extract.rglob("instances_train2014.json"))[0]

    # ── 3. Build annotation JSONs ───────────────────────────────────
    print("\n[3/4] Building annotations")
    refs = _load_refs(refs_path)
    coco_anns = _load_instances(instances_path)

    splits = {
        "train": "train",
        "test": "val",  # RefCOCOg 'val' split → our test split
    }

    for our_split, refcocog_split in splits.items():
        split_dir = data_dir / our_split
        split_dir.mkdir(parents=True, exist_ok=True)

        anns = _build_split_annotations(refs, coco_anns, refcocog_split)
        out_path = split_dir / "annotations_1.json"
        with open(out_path, "w") as f:
            json.dump(anns, f)
        print(f"  {our_split}: {len(anns)} referring expressions → {out_path}")

    # ── 4. Link images ──────────────────────────────────────────────
    print("\n[4/4] Linking images to split directories")
    for our_split in splits:
        split_dir = data_dir / our_split
        with open(split_dir / "annotations_1.json") as f:
            anns = json.load(f)
        _link_images(anns, coco_img_dir, split_dir / "images")

    print(f"\n✓  Data ready at {data_dir.resolve()}")
    print("  You can now run training:")
    print(f"    python main.py --data_dir {data_dir}")


if __name__ == "__main__":
    main()
