"""Travel distance and straight-working travel ratio on preprocessed points."""

from __future__ import annotations

import numpy as np

from trajectory import distances_metres


def travel_statistics(trajectory, labels=None, assignment="start"):
    """Attribute each edge to its start-point label by default.

    Only consecutive points within the same moving segment contribute. An
    optional end-point convention is available for explicitly chosen analyses.
    """
    if assignment not in {"start", "end"}:
        raise ValueError("Distance assignment must be start or end")
    size = len(trajectory.coordinates)
    valid = trajectory.segment_ids[1:] == trajectory.segment_ids[:-1]
    distance = (
        distances_metres(trajectory.coordinates[:-1], trajectory.coordinates[1:])
        if size > 1
        else np.array([])
    )
    total = float(distance[valid].sum())
    result = {
        "total_distance_m": total,
        "valid_edge_count": int(valid.sum()),
        "straight_distance_m": None,
        "turn_distance_m": None,
        "working_travel_ratio": None,
        "distance_assignment": assignment,
    }
    if labels is None:
        return result
    labels = np.asarray(labels)
    if labels.shape != (size,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Mileage requires one binary label per cleaned point")
    edge_labels = labels[:-1] if assignment == "start" else labels[1:]
    work = float(distance[valid & (edge_labels == 1)].sum())
    turn = float(distance[valid & (edge_labels == 0)].sum())
    result.update(
        straight_distance_m=work,
        turn_distance_m=turn,
        working_travel_ratio=work / total if total > 0 else None,
    )
    return result
