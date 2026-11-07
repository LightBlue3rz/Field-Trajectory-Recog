"""Point sequences use the same cleaned points and eight features as Agri."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from splits import CLASS_NAMES
from trajectory import relative_path


class TimeSeriesDataset(Dataset):
    """One complete continuous segment per sample; files are split beforehand."""

    def __init__(self, trajectories, normalizer, path_root=None, include_metadata=False):
        self.samples = []
        self.windows = []
        self.paths = []
        self.metadata = []
        self.include_metadata = include_metadata
        for file_index, item in enumerate(trajectories):
            values = torch.from_numpy(normalizer.transform(item.features))
            labels = torch.from_numpy(item.point_labels)
            boundaries = torch.where(torch.from_numpy(item.segment_ids[1:] != item.segment_ids[:-1]))[0]
            starts = [0, *boundaries.add(1).tolist()]
            ends = [*boundaries.add(1).tolist(), len(values)]
            for start, end in zip(starts, ends):
                self.samples.append((values[start:end], labels[start:end]))
                self.windows.append((file_index, start, end))
                path = relative_path(item.path, path_root) if path_root else item.path.name
                self.paths.append(path)
                operation_id = int(item.label) if item.label in (0, 1) else None
                self.metadata.append(
                    {
                        "path": path,
                        "operation_id": operation_id,
                        "operation_name": CLASS_NAMES[operation_id] if operation_id is not None else None,
                        "segment_id": int(item.segment_ids[start]),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.include_metadata:
            return (*self.samples[index], dict(self.metadata[index]))
        return self.samples[index]


def collate_fn(samples):
    if len({len(sample) for sample in samples}) != 1:
        raise ValueError("Point samples must consistently include or omit metadata")
    features, labels = zip(*(sample[:2] for sample in samples))
    lengths = torch.tensor([len(values) for values in features], dtype=torch.long)
    padded = pad_sequence(features, batch_first=True)
    targets = pad_sequence(labels, batch_first=True, padding_value=-100)
    mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    if len(samples[0]) == 3:
        return {
            "features": padded,
            "labels": targets,
            "mask": mask,
            "metadata": [sample[2] for sample in samples],
        }
    return padded, targets, mask
