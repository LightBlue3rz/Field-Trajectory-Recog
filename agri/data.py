"""Graph samples and segment-aware batching over the shared trajectory schema."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from splits import CLASS_NAMES, FileRecord, discover_splits
from trajectory import (
    FEATURE_DIM,
    FEATURE_NAMES,
    PreprocessConfig,
    Standardizer,
    Trajectory,
    load_trajectory,
    relative_path,
)

__all__ = [
    "CLASS_NAMES",
    "FEATURE_NAMES",
    "FileRecord",
    "PreprocessConfig",
    "Standardizer",
    "Trajectory",
    "TrajectoryDataset",
    "TrajectoryCollator",
    "discover_splits",
    "load_trajectory",
    "relative_path",
]


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, normalizer, path_root=None):
        self.trajectories = trajectories
        self.features = [torch.from_numpy(normalizer.transform(item.features)) for item in trajectories]
        self.path_root = path_root

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        item = self.trajectories[index]
        path = relative_path(item.path, self.path_root) if self.path_root is not None else item.path.name
        return self.features[index], item.label, path, torch.from_numpy(item.segment_ids)


@dataclass
class TrajectoryCollator:
    graph_size: int = 160

    def __call__(self, samples):
        if self.graph_size < 1:
            raise ValueError("graph_size must be positive")
        file_graphs = []
        for sample in samples:
            values, _, _ = sample[:3]
            segments = sample[3] if len(sample) > 3 else torch.zeros(len(values), dtype=torch.long)
            if len(values) == 0:
                raise ValueError("Cannot batch an empty trajectory")
            boundaries = torch.where(segments[1:] != segments[:-1])[0].add(1).tolist()
            starts, ends = [0, *boundaries], [*boundaries, len(values)]
            graphs = []
            for start, end in zip(starts, ends):
                for offset in range(start, end, self.graph_size):
                    graphs.append(
                        (values[offset : min(offset + self.graph_size, end)], int(segments[offset]))
                    )
            file_graphs.append(graphs)
        graph_count = max(len(graphs) for graphs in file_graphs)
        shape = (len(samples), graph_count, self.graph_size)
        features = torch.zeros(*shape, FEATURE_DIM)
        mask = torch.zeros(shape, dtype=torch.bool)
        segment_ids = torch.full(shape, -1, dtype=torch.long)
        for file_index, graphs in enumerate(file_graphs):
            for graph_index, (values, segment) in enumerate(graphs):
                features[file_index, graph_index, : len(values)] = values
                mask[file_index, graph_index, : len(values)] = True
                segment_ids[file_index, graph_index, : len(values)] = segment
        return {
            "features": features,
            "mask": mask,
            "segment_ids": segment_ids,
            "labels": torch.tensor([sample[1] for sample in samples], dtype=torch.long),
            "paths": [sample[2] for sample in samples],
        }
