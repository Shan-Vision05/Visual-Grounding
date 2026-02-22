#!/usr/bin/env python3
"""Download and prepare RefCOCOg data for Visual Grounding training.

This script handles everything end-to-end:
  1. Downloads COCO 2014 train images  (~13 GB zip, ~83k images)
  2. Downloads RefCOCOg annotations    (from HuggingFace or Wayback Machine)
  3. Converts into the flat format expected by VG_Dataset:
       data/{train,test}/images/<image_id>.jpg
       data/{train,test}/annotations_1.json

Each annotation entry: {"image_id": <int>, "text": <str>, "bbox": [x, y, w, h]}

Usage
-----
  # Run from the repo root (on a login / compile node with internet)
  python prepare_data.py                       # defaults to ./data
  python prepare_data.py --data_dir /scratch/alpine/$USER/vg_data

On CU Alpine, compute nodes have NO internet access.
Run this BEFORE submitting the training job, from a compile node:
  acompile
  conda activate vg_env
  python prepare_data.py --data_dir ./data --cache_dir ./.cache
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
def _is_valid_zip(path: Path) -> bool:
    """Return True if *path* is a valid zip archive."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.testzip()
        return True
    except (zipfile.BadZipFile, Exception):
        return False


def _download(url: str | list[str], dest: Path, retries: int = 3) -> None:
    """Download *url* to *dest* with retries and fallback to curl.

    *url* can be a single URL string or a list of mirror URLs to try
    in order.
    """
    if dest.exists():
        if dest.suffix.lower() == ".zip" and not _is_valid_zip(dest):
            print(f"  {dest.name} exists but is corrupt — re-downloading")
            dest.unlink()
        else:
            print(f"  [skip] {dest.name} already exists")
            return

    urls = [url] if isinstance(url, str) else list(url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for mirror_idx, mirror_url in enumerate(urls):
        label = f"[mirror {mirror_idx + 1}/{len(urls)}] " if len(urls) > 1 else ""
        print(f"  {label}Downloading {mirror_url} ...")

        for attempt in range(1, retries + 1):
            try:
                subprocess.check_call(
                    ["wget", "--no-check-certificate", "--quiet",
                     "--show-progress", "-O", str(dest), mirror_url],
                )
                if dest.suffix.lower() == ".zip" and not _is_valid_zip(dest):
                    print(f"  Downloaded file is not a valid zip, trying next ...")
                    dest.unlink()
                    break
                return
            except subprocess.CalledProcessError:
                if dest.exists():
                    dest.unlink()
                if attempt < retries:
                    print(f"  wget attempt {attempt}/{retries} failed, retrying ...")

        if not dest.exists():
            print(f"  wget failed; trying curl ...")
            try:
                subprocess.check_call(
                    ["curl", "-L", "-k", "--progress-bar", "-o",
                     str(dest), mirror_url],
                )
                if dest.suffix.lower() == ".zip" and not _is_valid_zip(dest):
                    print(f"  Downloaded file is not a valid zip, trying next ...")
                    dest.unlink()
                    continue
                return
            except subprocess.CalledProcessError:
                if dest.exists():
                    dest.unlink()

    print(f"  ERROR: Could not download from any mirror.", file=sys.stderr)
    print(f"  You can manually download the file and place it at: {dest}",
          file=sys.stderr)
    sys.exit(1)


def _unzip(src: Path, dest: Path) -> None:
    """Extract *src* zip into *dest* (skip if dest already populated)."""
    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {dest} already extracted")
        return
    print(f"  Extracting {src.name} ...")
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dest)


# ---------------------------------------------------------------------------
# COCO images URL
# ---------------------------------------------------------------------------
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

# Wayback Machine mirrors for RefCOCOg (fallback only)
REFCOCOG_ZIP_URLS = [
    "https://web.archive.org/web/20230307030235/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
]


# ---------------------------------------------------------------------------
# HuggingFace datasets approach (PRIMARY — fast and reliable)
# ---------------------------------------------------------------------------
def _try_hf_refcocog(data_dir: Path, coco_ann_cache: Path) -> bool:
    """Download RefCOCOg annotations via HuggingFace datasets library.

    Returns True on success, False if the library is unavailable or fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' library not installed — will try zip mirrors")
        return False

    try:
        print("  Loading RefCOCOg from HuggingFace (jxu124/refcocog) ...")
        ds = load_dataset("jxu124/refcocog", trust_remote_code=True)
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        return False

    # Discover available splits
    available = list(ds.keys())
    print(f"  HF splits available: {available}")

    # Peek at column names
    first_split = ds[available[0]]
    cols = first_split.column_names
    print(f"  Columns: {cols}")

    # Determine which column holds the referring text
    text_col = None
    for candidate in ("sent", "sentence", "sentences", "text", "caption", "expression"):
        if candidate in cols:
            text_col = candidate
            break

    # Determine bbox column
    bbox_col = None
    for candidate in ("bbox", "bounding_box", "box"):
        if candidate in cols:
            bbox_col = candidate
            break

    if text_col is None or bbox_col is None:
        # If HF dataset doesn't have bbox directly, we need COCO instances
        # to look up bbox from ann_id.  Fall through to handle that.
        if "ann_id" in cols:
            return _try_hf_with_coco_bbox(ds, data_dir, coco_ann_cache, text_col)
        print(f"  Could not find text column ({text_col}) or bbox column ({bbox_col}) — falling back")
        return False

    # Map HF split names → our split names
    split_map = {}
    for s in available:
        if s == "train":
            split_map[s] = "train"
        elif s in ("val", "validation", "test"):
            split_map[s] = "test"

    if not split_map:
        print("  No usable splits found — falling back")
        return False

    for hf_split, our_split in split_map.items():
        split_data = ds[hf_split]
        entries = []
        for row in split_data:
            # Handle text — could be a string or list of sentences
            text = row[text_col]
            if isinstance(text, list):
                # Multiple sentences per ref — expand
                for t in text:
                    if isinstance(t, dict):
                        t = t.get("sent") or t.get("raw") or t.get("text", "")
                    if t:
                        entries.append({
                            "image_id": row["image_id"],
                            "text": str(t),
                            "bbox": list(row[bbox_col]),
                        })
            elif text:
                entries.append({
                    "image_id": row["image_id"],
                    "text": str(text),
                    "bbox": list(row[bbox_col]),
                })

        split_dir = data_dir / our_split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "annotations_1.json"
        with open(out_path, "w") as f:
            json.dump(entries, f)
        print(f"  {our_split}: {len(entries)} referring expressions → {out_path}")

    return True


def _try_hf_with_coco_bbox(ds, data_dir: Path, coco_ann_cache: Path,
                            text_col: str | None) -> bool:
    """HF dataset has ann_id but no bbox — look up bbox from COCO instances."""
    # Download COCO annotations for bbox lookup
    instances_zip = coco_ann_cache / "annotations_trainval2014.zip"
    _download(COCO_ANN_URL, instances_zip)
    instances_extract = coco_ann_cache / "coco_ann"
    _unzip(instances_zip, instances_extract)
    instances_path = instances_extract / "annotations" / "instances_train2014.json"
    if not instances_path.exists():
        candidates = list(instances_extract.rglob("instances_train2014.json"))
        if not candidates:
            return False
        instances_path = candidates[0]

    with open(instances_path, "r") as f:
        coco_data = json.load(f)
    coco_anns = {ann["id"]: ann for ann in coco_data["annotations"]}

    available = list(ds.keys())
    split_map = {}
    for s in available:
        if s == "train":
            split_map[s] = "train"
        elif s in ("val", "validation", "test"):
            split_map[s] = "test"

    for hf_split, our_split in split_map.items():
        split_data = ds[hf_split]
        entries = []
        for row in split_data:
            ann_id = row.get("ann_id")
            if ann_id is None or ann_id not in coco_anns:
                continue
            bbox = coco_anns[ann_id]["bbox"]

            # Get text
            if text_col and text_col in row:
                text = row[text_col]
            else:
                text = row.get("sent") or row.get("sentences") or ""

            if isinstance(text, list):
                for t in text:
                    if isinstance(t, dict):
                        t = t.get("sent") or t.get("raw") or ""
                    if t:
                        entries.append({"image_id": row["image_id"], "text": str(t), "bbox": bbox})
            elif text:
                entries.append({"image_id": row["image_id"], "text": str(text), "bbox": bbox})

        split_dir = data_dir / our_split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "annotations_1.json"
        with open(out_path, "w") as f:
            json.dump(entries, f)
        print(f"  {our_split}: {len(entries)} referring expressions → {out_path}")

    return True


# ---------------------------------------------------------------------------
# Legacy zip-based approach (FALLBACK — Wayback Machine mirrors)
# ---------------------------------------------------------------------------
def _load_refs(refs_path: Path) -> list[dict]:
    """Load RefCOCOg refs from pickle or JSON."""
    suffix = refs_path.suffix.lower()
    if suffix in (".p", ".pkl"):
        with open(refs_path, "rb") as f:
            return pickle.load(f)
    elif suffix == ".json":
        with open(refs_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown refs format: {refs_path}")


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
        bbox = coco_ann["bbox"]
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


def _fallback_zip_refcocog(data_dir: Path, cache: Path) -> None:
    """Download RefCOCOg from Wayback Machine zip mirrors (slow fallback)."""
    print("  Trying Wayback Machine mirrors (may be slow) ...")
    refcocog_zip = cache / "refcocog.zip"
    _download(REFCOCOG_ZIP_URLS, refcocog_zip)
    refcocog_extract = cache / "refcocog"
    _unzip(refcocog_zip, refcocog_extract)

    # Find refs file
    refs_candidates = (
        list(refcocog_extract.rglob("refs*.p"))
        + list(refcocog_extract.rglob("refs*.json"))
    )
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
    _download(COCO_ANN_URL, instances_zip)
    instances_extract = cache / "coco_ann"
    _unzip(instances_zip, instances_extract)
    instances_path = instances_extract / "annotations" / "instances_train2014.json"
    if not instances_path.exists():
        instances_path = list(instances_extract.rglob("instances_train2014.json"))[0]

    refs = _load_refs(refs_path)
    with open(instances_path, "r") as f:
        coco_data = json.load(f)
    coco_anns = {ann["id"]: ann for ann in coco_data["annotations"]}

    splits = {"train": "train", "test": "val"}
    for our_split, refcocog_split in splits.items():
        split_dir = data_dir / our_split
        split_dir.mkdir(parents=True, exist_ok=True)
        anns = _build_split_annotations(refs, coco_anns, refcocog_split)
        out_path = split_dir / "annotations_1.json"
        with open(out_path, "w") as f:
            json.dump(anns, f)
        print(f"  {our_split}: {len(anns)} referring expressions → {out_path}")


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
        src = coco_img_dir / f"COCO_train2014_{img_id:012d}.jpg"
        dst = dest_dir / f"{img_id}.jpg"
        if dst.exists():
            continue
        if not src.exists():
            continue
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
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
        alt = list(coco_extract.rglob("train2014"))
        if alt:
            coco_img_dir = alt[0]
        else:
            print("ERROR: Could not find train2014/ after extraction.", file=sys.stderr)
            sys.exit(1)
    print(f"  COCO images at: {coco_img_dir}")

    # ── 2. Download RefCOCOg annotations ────────────────────────────
    #    Primary:  HuggingFace datasets (fast, reliable)
    #    Fallback: Wayback Machine zip mirrors (slow, often broken)
    print("\n[2/4] RefCOCOg annotations")

    # Check if annotations already exist
    train_ann = data_dir / "train" / "annotations_1.json"
    test_ann = data_dir / "test" / "annotations_1.json"
    if train_ann.exists() and test_ann.exists():
        print("  [skip] Annotations already built")
    else:
        # Try HuggingFace first
        success = _try_hf_refcocog(data_dir, cache)
        if not success:
            # Fall back to zip mirrors
            _fallback_zip_refcocog(data_dir, cache)

    # ── 3. Verify annotations ──────────────────────────────────────
    print("\n[3/4] Verifying annotations")
    for split in ("train", "test"):
        ann_path = data_dir / split / "annotations_1.json"
        if ann_path.exists():
            with open(ann_path) as f:
                count = len(json.load(f))
            print(f"  {split}: {count} entries")
        else:
            print(f"  ERROR: {ann_path} not found!", file=sys.stderr)
            sys.exit(1)

    # ── 4. Link images ──────────────────────────────────────────────
    print("\n[4/4] Linking images to split directories")
    for split in ("train", "test"):
        split_dir = data_dir / split
        with open(split_dir / "annotations_1.json") as f:
            anns = json.load(f)
        _link_images(anns, coco_img_dir, split_dir / "images")

    print(f"\n✓  Data ready at {data_dir.resolve()}")
    print("  You can now run training:")
    print(f"    python main.py --data_dir {data_dir}")


if __name__ == "__main__":
    main()
