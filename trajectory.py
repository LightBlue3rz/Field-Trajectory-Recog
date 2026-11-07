"""Shared cleaning, four-second sampling, features and train-only normalization."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from openpyxl import load_workbook

FEATURE_NAMES = (
    "longitude",
    "latitude",
    "speed",
    "heading",
    "projected_longitude",
    "projected_latitude",
    "acceleration",
    "angular_speed",
)
FEATURE_DIM = len(FEATURE_NAMES)
TABLE_SUFFIXES = {".csv", ".xlsx"}
IGNORE_LABEL = -100
ALIASES = {
    "longitude": ("longitude", "lng", "经度"),
    "latitude": ("latitude", "lat", "纬度"),
    "speed": ("speed", "速度(km/h)", "速度"),
    "heading": ("heading", "航向"),
    "timestamp": ("timestamp", "time", "GPS时间", "GNSS时间", "时间"),
    "label": ("work_state", "工作状态"),
}


@dataclass(frozen=True)
class PreprocessConfig:
    interval_seconds: float = 4.0
    stop_speed_kmh: float = 0.1
    max_speed_kmh: float = 60.0
    max_gap_seconds: float = 12.0

    def __post_init__(self):
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("Preprocessing parameters must be finite")
        if self.interval_seconds <= 0 or self.max_gap_seconds < self.interval_seconds:
            raise ValueError("Sampling interval must be positive and no larger than max_gap_seconds")
        if not 0 <= self.stop_speed_kmh < self.max_speed_kmh:
            raise ValueError("Speed thresholds must satisfy 0 <= stop_speed_kmh < max_speed_kmh")


@dataclass
class Trajectory:
    features: np.ndarray
    coordinates: np.ndarray
    timestamps: np.ndarray
    segment_ids: np.ndarray
    point_labels: np.ndarray
    source_rows: np.ndarray
    path: Path
    label: int
    diagnostics: dict


def read_table(path: Path) -> tuple[list[str], list[tuple]]:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = iter(workbook.worksheets[0].iter_rows(values_only=True))
            header, values = next(rows, ()), list(rows)
        finally:
            workbook.close()
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream)
            header, values = next(rows, ()), list(rows)
    else:
        raise ValueError(f"Unsupported trajectory format: {path.name}")
    columns = [str(value).strip() if value is not None else "" for value in header]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f"Empty or duplicate column names: {path.name}")
    return columns, values


def _seconds(value) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return (value - datetime(1970, 1, 1)).total_seconds()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return _seconds(datetime.fromisoformat(text.replace("/", "-")))
    except ValueError:
        for pattern in (
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S.%f",
        ):
            try:
                return _seconds(datetime.strptime(text, pattern))
            except ValueError:
                continue
        return float(text)


def distances_metres(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Vectorized great-circle distance for longitude/latitude pairs."""
    lon1, lat1 = np.radians(np.asarray(first)).T
    lon2, lat2 = np.radians(np.asarray(second)).T
    hav = np.sin((lat2 - lat1) / 2) ** 2
    hav += np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))


def _index(columns, name, required=True):
    match = next((alias for alias in ALIASES[name] if alias in columns), None)
    if match is None and required:
        raise ValueError(f"Missing {name} column; accepted names: {ALIASES[name]}")
    return columns.index(match) if match is not None else None


def prepare_rows(
    columns: list[str],
    rows: Sequence[tuple],
    config: PreprocessConfig,
    path: Path = Path("trajectory.csv"),
    label: int = -1,
    require_labels: bool = False,
    label_column: str | None = None,
) -> Trajectory:
    """Clean once, interpolate short gaps, sample each moving segment uniformly.

    Stops and gaps exceeding max_gap_seconds create segment boundaries. They
    are never bridged by interpolation, graph edges, local sequences or mileage.
    Categorical annotations use nearest-neighbor sampling (earlier on ties).
    """
    indices = [_index(columns, name) for name in FEATURE_NAMES[:4]]
    time_index = _index(columns, "timestamp")
    if label_column is not None:
        label_index = columns.index(label_column) if label_column in columns else None
    else:
        label_index = _index(columns, "label", required=False)
    if require_labels and label_index is None:
        raise ValueError("Point training requires work_state or an explicit --label-column")
    stats = {
        name: 0
        for name in (
            "invalid_rows",
            "duplicate_timestamps",
            "stop_rows",
            "drift_rows",
            "gap_count",
            "interpolated_points",
        )
    }
    stats["input_rows"] = len(rows)
    valid = []
    for row_number, row in enumerate(rows):
        try:
            values = [float(row[index]) for index in indices]
            timestamp = _seconds(row[time_index])
            if not np.isfinite([*values, timestamp]).all():
                raise ValueError("Non-finite raw value")
            if not (-180 <= values[0] <= 180 and -90 <= values[1] <= 90 and values[2] >= 0):
                raise ValueError("Raw value outside its valid range")
            values[3] %= 360
        except (TypeError, ValueError, IndexError, OverflowError):
            stats["invalid_rows"] += 1
            continue
        point_label = IGNORE_LABEL
        if require_labels:
            try:
                candidate = float(row[label_index])
                if candidate in (0, 1):
                    point_label = int(candidate)
            except (TypeError, ValueError, IndexError):
                pass
        valid.append([*values, timestamp, point_label, row_number])
    points = np.asarray(valid, dtype=np.float64).reshape(-1, 7)
    points = points[np.argsort(points[:, 4], kind="stable")]
    unique = np.ones(len(points), dtype=bool)
    if len(points) > 1:
        unique[1:] = np.diff(points[:, 4]) > 0
    stats["duplicate_timestamps"] = int((~unique).sum())
    points = points[unique]
    groups, moving = [], []
    for point in points:
        if point[2] <= config.stop_speed_kmh:
            stats["stop_rows"] += 1
            if moving:
                groups.append(np.asarray(moving))
                moving = []
            continue
        if point[2] > config.max_speed_kmh:
            stats["drift_rows"] += 1
            continue
        if moving:
            elapsed = point[4] - moving[-1][4]
            if elapsed > config.max_gap_seconds:
                stats["gap_count"] += 1
                groups.append(np.asarray(moving))
                moving = []
            elif (
                distances_metres(np.asarray(moving[-1][:2]), point[:2]) / elapsed * 3.6 > config.max_speed_kmh
            ):
                stats["drift_rows"] += 1
                continue
        moving.append(point)
    if moving:
        groups.append(np.asarray(moving))
    if not groups:
        raise ValueError(f"No moving trajectory points remain: {path.name}")
    feature_parts, coordinate_parts, time_parts, segment_parts, targets, source_rows = [], [], [], [], [], []
    for segment, group in enumerate(groups):
        time = group[:, 4]
        count = int(np.floor((time[-1] - time[0]) / config.interval_seconds + 1e-9)) + 1
        grid = time[0] + np.arange(count) * config.interval_seconds
        raw = np.zeros((count, 4), dtype=np.float64)
        for index in (0, 3):
            unwrapped = np.unwrap(np.radians(group[:, index]))
            raw[:, index] = np.degrees(np.interp(grid, time, unwrapped))
        raw[:, 0] = (raw[:, 0] + 180) % 360 - 180
        raw[:, 3] %= 360
        raw[:, 1] = np.interp(grid, time, group[:, 1])
        raw[:, 2] = np.interp(grid, time, group[:, 2])
        right = np.searchsorted(time, grid).clip(0, len(time) - 1)
        left = (right - 1).clip(0)
        nearest = np.where(np.abs(grid - time[left]) <= np.abs(time[right] - grid), left, right)
        stats["interpolated_points"] += int((np.abs(time[nearest] - grid) > 1e-6).sum())
        derived = np.zeros((count, 4), dtype=np.float64)
        if count > 1:
            coordinate_delta = np.diff(raw[:, :2], axis=0)
            coordinate_delta[:, 0] = (coordinate_delta[:, 0] + 180) % 360 - 180
            derived[1:, :2] = coordinate_delta * np.cos(np.radians(raw[1:, 3]))[:, None]
            derived[1:, 2] = np.diff(raw[:, 2]) / config.interval_seconds
            derived[1:, 3] = ((np.diff(raw[:, 3]) + 180) % 360 - 180) / config.interval_seconds
        feature_parts.append(np.concatenate((raw, derived), axis=1))
        coordinate_parts.append(raw[:, :2])
        time_parts.append(grid)
        segment_parts.append(np.full(count, segment, dtype=np.int64))
        targets.append(group[nearest, 5].astype(np.int64))
        source_rows.append(group[nearest, 6].astype(np.int64))
    # Keep coordinate precision until training normalization converts to float32.
    features = np.concatenate(feature_parts)
    labels = np.concatenate(targets)
    if require_labels and not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Every retained point needs a binary 0/1 annotation: {path.name}")
    stats.update(
        valid_points=len(features), segment_count=len(groups), interval_seconds=config.interval_seconds
    )
    return Trajectory(
        features,
        np.concatenate(coordinate_parts),
        np.concatenate(time_parts),
        np.concatenate(segment_parts),
        labels,
        np.concatenate(source_rows),
        Path(path),
        label,
        stats,
    )


def load_trajectory(record, config: PreprocessConfig, require_labels=False, label_column=None):
    path = Path(record.path).resolve()
    columns, rows = read_table(path)
    return prepare_rows(columns, rows, config, path, record.label, require_labels, label_column)


def relative_path(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, trajectories: Iterable[Trajectory]):
        count = 0
        mean = np.zeros(FEATURE_DIM, dtype=np.float64)
        squared = np.zeros(FEATURE_DIM, dtype=np.float64)
        for trajectory in trajectories:
            values = trajectory.features.astype(np.float64)
            if not len(values):
                continue
            size, batch_mean = len(values), values.mean(axis=0)
            delta = batch_mean - mean
            total = count + size
            squared += ((values - batch_mean) ** 2).sum(axis=0) + delta**2 * count * size / total
            mean += delta * size / total
            count = total
        if count == 0:
            raise ValueError("Cannot fit normalization without training points")
        scale = np.sqrt(squared / count)
        scale[scale < 1e-8] = 1
        return cls(mean, scale)

    def transform(self, values):
        return ((values - self.mean) / self.scale).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
        if mean.shape != (FEATURE_DIM,) or scale.shape != mean.shape or not np.isfinite([mean, scale]).all():
            raise ValueError("Incompatible normalization schema; train a new eight-feature checkpoint")
        if (scale <= 0).any():
            raise ValueError("Normalization scales must be positive")
        return cls(mean, scale)
