"""Hierarchical operation recognition, point labeling and working travel ratio."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from efficiency import travel_statistics
from evaluation import POINT_REPORT_CLASS_LABELS, POINT_REPORT_CLASS_NAMES
from splits import FileRecord
from training import choose_device, input_files, load_checkpoint, new_output, predict_trajectory, write_json
from trajectory import load_trajectory, relative_path


def analyze_trajectory(trajectory, pattern_bundle, point_bundle, device, assignment="start"):
    if asdict(pattern_bundle.preprocessing) != asdict(point_bundle.preprocessing):
        raise ValueError("Both stages must use identical preprocessing settings")
    pattern = predict_trajectory(pattern_bundle, trajectory, device)
    result = {
        "operation": pattern,
        "preprocessing": trajectory.diagnostics,
        "timestamps": trajectory.timestamps.tolist(),
        "segment_ids": trajectory.segment_ids.tolist(),
        "source_rows": trajectory.source_rows.tolist(),
    }
    if pattern["class_id"] == 2:
        result.update(
            status="local_behavior_not_applicable",
            point_labels=None,
            point_probabilities=None,
            travel=travel_statistics(trajectory, assignment=assignment),
        )
    else:
        points = predict_trajectory(point_bundle, trajectory, device)
        result.update(
            status="complete",
            point_labels=points["labels"],
            point_probabilities=points["probabilities"],
            travel=travel_statistics(trajectory, points["labels"], assignment),
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hierarchical recognition and working travel ratio")
    parser.add_argument("--pattern-checkpoint", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--distance-assignment", choices=("start", "end"), default="start")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        device = choose_device(args.device)
        pattern = load_checkpoint(args.pattern_checkpoint, device, "pattern")
        point = load_checkpoint(args.point_checkpoint, device, "point")
        if asdict(pattern.preprocessing) != asdict(point.preprocessing):
            raise ValueError("The two checkpoints use different preprocessing settings")
        files, root = input_files(args.input)
        results = []
        for path in files:
            trajectory = load_trajectory(FileRecord(path, -1), pattern.preprocessing)
            results.append(
                {
                    "path": relative_path(path, root),
                    **analyze_trajectory(trajectory, pattern, point, device, args.distance_assignment),
                }
            )
        output = new_output(args.output)
        write_json(
            output / "analysis.json",
            {
                "point_classes": {"0": "turn", "1": "straight"},
                "point_probability_class_names": list(POINT_REPORT_CLASS_NAMES),
                "point_probability_class_labels": list(POINT_REPORT_CLASS_LABELS),
                "results": results,
            },
        )
        print(f"Analyzed {len(results)} trajectory files", flush=True)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
