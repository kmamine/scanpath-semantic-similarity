"""
Data loading utilities for CocoFreeView dataset.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Fixation:
    """Represents a single fixation point."""

    x: float
    y: float
    duration: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.duration], dtype=np.float32)

    def to_xy_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class Scanpath:
    """Represents a complete scanpath with fixations."""

    image_id: str
    image_path: Path
    subject_id: int
    condition: str
    split: str
    fixations: List[Fixation] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        if not self.fixations:
            return np.array([], dtype=np.float32).reshape(0, 3)
        return np.array([f.to_array() for f in self.fixations], dtype=np.float32)

    def to_xy_array(self) -> np.ndarray:
        if not self.fixations:
            return np.array([], dtype=np.float32).reshape(0, 2)
        return np.array([f.to_xy_array() for f in self.fixations], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "name": self.image_id,
            "subject": self.subject_id,
            "condition": self.condition,
            "X": [f.x for f in self.fixations],
            "Y": [f.y for f in self.fixations],
            "T": [f.duration for f in self.fixations],
            "length": len(self.fixations),
            "split": self.split,
        }


def build_image_index(imgs_dir: Path) -> Dict[str, Path]:
    """Scan image directory and build a mapping from filename to full path."""
    index: Dict[str, Path] = {}

    if not imgs_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {imgs_dir}")

    for item in imgs_dir.iterdir():
        if item.is_dir():
            for img_file in item.glob("*.jpg"):
                index[img_file.name] = img_file
        elif item.suffix.lower() == ".jpg":
            index[item.name] = item

    log.info(f"Built image index with {len(index)} images from {imgs_dir}")
    return index


def load_cocofreeview(
    json_path: Path,
    imgs_dir: Path,
    n_images: Optional[int] = None,
    n_scanpaths_per_image: Optional[int] = None,
    split: Optional[str] = None,
    seed: int = 42,
    use_reduced: bool = False,
) -> Tuple[Dict[str, Path], Dict[str, List[Scanpath]]]:
    """Load CocoFreeView dataset."""
    rng = np.random.default_rng(seed)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path) as f:
        records = json.load(f)

    log.info(f"Loaded {len(records)} records from {json_path}")

    if use_reduced:
        image_index = {}
        for img_file in imgs_dir.glob("*.jpg"):
            image_index[img_file.name] = img_file
        log.info(f"Built flat image index with {len(image_index)} images")
    else:
        image_index = build_image_index(imgs_dir)

    valid_records = []
    for record in records:
        if split is not None and record.get("split") != split:
            continue

        image_name = record["name"]
        if image_name not in image_index:
            continue

        valid_records.append(record)

    log.info(f"Found {len(valid_records)} valid records with existing images")

    images_to_scanpaths: Dict[str, List[dict]] = {}
    for record in valid_records:
        img_name = record["name"]
        if img_name not in images_to_scanpaths:
            images_to_scanpaths[img_name] = []
        images_to_scanpaths[img_name].append(record)

    all_images = list(images_to_scanpaths.keys())

    if n_images is not None and n_images < len(all_images):
        rng.shuffle(all_images)
        all_images = all_images[:n_images]

    image_paths: Dict[str, Path] = {}
    scanpaths: Dict[str, List[Scanpath]] = {}

    for img_name in all_images:
        img_records = images_to_scanpaths[img_name]

        if n_scanpaths_per_image is not None and n_scanpaths_per_image < len(
            img_records
        ):
            indices = rng.choice(
                len(img_records), size=n_scanpaths_per_image, replace=False
            )
            img_records = [img_records[i] for i in indices]

        sp_list: List[Scanpath] = []
        for record in img_records:
            fixations = []
            xs = record.get("X", [])
            ys = record.get("Y", [])
            ts = record.get("T", [])

            for i in range(len(xs)):
                duration = ts[i] if i < len(ts) else 0.0
                fixations.append(Fixation(x=xs[i], y=ys[i], duration=duration))

            sp = Scanpath(
                image_id=img_name,
                image_path=image_index[img_name],
                subject_id=record.get("subject", -1),
                condition=record.get("condition", "unknown"),
                split=record.get("split", "unknown"),
                fixations=fixations,
            )
            sp_list.append(sp)

        image_paths[img_name] = image_index[img_name]
        scanpaths[img_name] = sp_list

    total_scanpaths = sum(len(sps) for sps in scanpaths.values())
    log.info(f"Loaded {len(image_paths)} images with {total_scanpaths} scanpaths")

    return image_paths, scanpaths


def create_reduced_dataset(
    json_path: Path,
    imgs_dir: Path,
    output_json: Path,
    output_imgs_dir: Path,
    n_images: int = 100,
    n_scanpaths_per_image: int = 5,
    split: str = "valid",
    seed: int = 42,
) -> Tuple[int, int]:
    """Create a reduced dataset by sampling images and scanpaths."""
    rng = np.random.default_rng(seed)

    with open(json_path) as f:
        records = json.load(f)

    image_index = build_image_index(imgs_dir)

    valid_records = [
        r for r in records if r.get("split") == split and r["name"] in image_index
    ]

    images_to_records: Dict[str, List[dict]] = {}
    for record in valid_records:
        img_name = record["name"]
        if img_name not in images_to_records:
            images_to_records[img_name] = []
        images_to_records[img_name].append(record)

    all_images = list(images_to_records.keys())
    log.info(f"Found {len(all_images)} images in {split} split with existing files")

    if len(all_images) < n_images:
        log.warning(f"Requested {n_images} images but only {len(all_images)} available")
        n_images = len(all_images)

    rng.shuffle(all_images)
    selected_images = all_images[:n_images]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_imgs_dir.mkdir(parents=True, exist_ok=True)

    reduced_records: List[dict] = []

    for img_name in selected_images:
        img_records = images_to_records[img_name]

        if len(img_records) > n_scanpaths_per_image:
            indices = rng.choice(
                len(img_records), size=n_scanpaths_per_image, replace=False
            )
            img_records = [img_records[i] for i in indices]

        reduced_records.extend(img_records)

        src_path = image_index[img_name]
        dst_path = output_imgs_dir / img_name
        shutil.copy2(src_path, dst_path)

    with open(output_json, "w") as f:
        json.dump(reduced_records, f, indent=2)

    log.info(
        f"Created reduced dataset: {n_images} images, {len(reduced_records)} scanpaths"
    )
    log.info(f"Output JSON: {output_json}")
    log.info(f"Output images: {output_imgs_dir}")

    return n_images, len(reduced_records)


def load_jsonl(path: Path) -> List[dict]:
    """Load records from JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """Append a record to JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_jsonl(path: Path, records: List[dict]) -> None:
    """Save all records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
