"""Deterministic, trajectory-level stratified dataset splits and portable manifests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trajectory import TABLE_SUFFIXES

CLASS_NAMES = ("shuttle", "skip_row", "circuit")
CLASS_DIRECTORIES = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_DIRECTORIES.update({"梭行法": 0, "套行法": 1, "绕行法": 2})


@dataclass(frozen=True)
class FileRecord:
    path: Path
    label: int


def _class_files(root: Path):
    records = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(
            path.resolve()
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not path.name.startswith("~$")
        )
        if files and folder.name not in CLASS_DIRECTORIES:
            raise ValueError(f"Unknown class directory: {folder.name}")
        records.extend(FileRecord(path, CLASS_DIRECTORIES[folder.name]) for path in files)
    return records


def _allocate(size, fractions):
    if size < 3:
        raise ValueError("At least three trajectories per present class are needed for train/val/test")
    desired = np.asarray(fractions) * size
    counts = np.maximum(np.floor(desired).astype(int), 1)
    while counts.sum() < size:
        counts[np.argmax(desired - counts)] += 1
    while counts.sum() > size:
        candidates = np.where(counts > 1, counts - desired, -np.inf)
        counts[np.argmax(candidates)] -= 1
    return counts.tolist()


def discover_splits(
    root: Path,
    seed=42,
    validation_fraction=0.2,
    test_fraction=0.1,
    allow_missing_classes=False,
    limit_per_class=None,
    policy="stratified",
    classes=(0, 1, 2),
):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Dataset root is not a directory")
    fractions = (1 - validation_fraction - test_fraction, validation_fraction, test_fraction)
    if not all(np.isfinite(fraction) and 0 < fraction < 1 for fraction in fractions):
        raise ValueError("Train/validation/test fractions must be positive and sum to one")
    directories = {path.name.lower(): path for path in root.iterdir() if path.is_dir()}
    provided = {}
    if "train" in directories:
        for name, aliases in (("train", ("train",)), ("val", ("val", "validation")), ("test", ("test",))):
            directory = next((directories[key] for key in aliases if key in directories), None)
            provided[name] = _class_files(directory) if directory is not None else []
    else:
        provided["train"] = _class_files(root)
    provided = {
        split: [record for record in records if record.label in classes]
        for split, records in provided.items()
    }
    all_records = [record for records in provided.values() for record in records]
    if not all_records:
        raise ValueError("No trajectory tables found in supported class directories")
    # Repeated names commonly indicate a copied source trajectory across splits.
    names = [(record.label, record.path.name.casefold()) for record in all_records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate source filenames within a class; remove copies before splitting")
    if policy == "provided":
        if set(provided) != {"train", "val", "test"} or any(not values for values in provided.values()):
            raise ValueError("provided split policy requires nonempty Train, Val and Test")
        splits = provided
    elif policy == "stratified":
        splits = {split: [] for split in ("train", "val", "test")}
        for label in sorted({record.label for record in all_records}):
            group = sorted(
                (record for record in all_records if record.label == label),
                key=lambda record: record.path.relative_to(root).as_posix(),
            )
            random.Random(seed + label).shuffle(group)
            train_count, val_count, _ = _allocate(len(group), fractions)
            for split, records in zip(
                splits,
                (
                    group[:train_count],
                    group[train_count : train_count + val_count],
                    group[train_count + val_count :],
                ),
            ):
                splits[split].extend(records)
    else:
        raise ValueError("split policy must be stratified or provided")
    for split, records in splits.items():
        missing = set(classes) - {record.label for record in records}
        if missing and not allow_missing_classes:
            raise ValueError(
                f"{split} is missing classes {[CLASS_NAMES[index] for index in sorted(missing)]}"
            )
        if not records:
            raise ValueError(f"Empty {split} split")
        if limit_per_class is not None:
            if limit_per_class < 1:
                raise ValueError("limit_per_class must be positive")
            selected = []
            for label in classes:
                group = [record for record in records if record.label == label]
                random.Random(seed + label).shuffle(group)
                selected.extend(group[:limit_per_class])
            splits[split] = selected
    return splits
