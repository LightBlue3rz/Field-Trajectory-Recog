"""Distance accounting and hierarchical routing use the same cleaned points."""

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from efficiency import travel_statistics
from pipeline import analyze_trajectory
from trajectory import PreprocessConfig, distances_metres, prepare_rows


class TravelTests(unittest.TestCase):
    def setUp(self):
        self.item = SimpleNamespace(
            coordinates=np.array([[0, 0], [0.001, 0], [0.002, 0]]),
            segment_ids=np.array([0, 0, 0]),
        )

    def test_great_circle_distance(self):
        self.assertAlmostEqual(
            float(distances_metres(np.array([0, 0]), np.array([1, 0]))), np.pi * 6_371_000 / 180, places=6
        )

    def test_start_labels_determine_working_travel_ratio(self):
        result = travel_statistics(self.item, [1, 0, 1])
        self.assertAlmostEqual(result["working_travel_ratio"], 0.5)
        self.assertAlmostEqual(
            result["total_distance_m"], result["straight_distance_m"] + result["turn_distance_m"]
        )

    def test_explicit_end_label_convention(self):
        self.assertAlmostEqual(travel_statistics(self.item, [1, 1, 0], "start")["working_travel_ratio"], 1)
        self.assertAlmostEqual(travel_statistics(self.item, [1, 1, 0], "end")["working_travel_ratio"], 0.5)

    def test_turn_only_and_straight_only(self):
        self.assertEqual(travel_statistics(self.item, [0, 0, 0])["working_travel_ratio"], 0)
        self.assertEqual(travel_statistics(self.item, [1, 1, 1])["working_travel_ratio"], 1)

    def test_gap_edges_are_excluded(self):
        self.item.segment_ids = np.array([0, 0, 1])
        result = travel_statistics(self.item, [1, 0, 0])
        self.assertEqual(result["valid_edge_count"], 1)
        self.assertEqual(result["working_travel_ratio"], 1)

    def test_zero_distance_and_singletons_have_no_ratio(self):
        self.item.coordinates[:] = 0
        self.assertIsNone(travel_statistics(self.item, [1, 1, 1])["working_travel_ratio"])
        self.item.coordinates = np.array([[0, 0]])
        self.item.segment_ids = np.array([0])
        self.assertIsNone(travel_statistics(self.item, [1])["working_travel_ratio"])

    def test_invalid_labels_and_conventions_are_rejected(self):
        for labels, convention in (([1, 0], "start"), ([1, 2, 0], "start"), ([1, 0, 1], "other")):
            with self.assertRaises(ValueError):
                travel_statistics(self.item, labels, convention)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        config = PreprocessConfig()
        self.pattern = SimpleNamespace(preprocessing=config)
        self.point = SimpleNamespace(preprocessing=config)
        self.item = prepare_rows(
            ["longitude", "latitude", "speed", "heading", "timestamp"],
            [(0, 0, 4, 0, 0), (1e-5, 0, 4, 0, 4), (2e-5, 0, 4, 0, 8)],
            config,
            path=Path("synthetic.csv"),
        )

    def test_shuttle_and_skip_row_call_point_recognition(self):
        for class_id in (0, 1):
            with patch(
                "pipeline.predict_trajectory",
                side_effect=[
                    {"class_id": class_id, "predicted": "operation"},
                    {"labels": [1, 0, 0], "probabilities": [[0.1, 0.9], [0.8, 0.2], [0.8, 0.2]]},
                ],
            ) as prediction:
                result = analyze_trajectory(self.item, self.pattern, self.point, torch.device("cpu"))
                self.assertEqual(prediction.call_count, 2)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["point_labels"], [1, 0, 0])
                self.assertAlmostEqual(result["travel"]["working_travel_ratio"], 0.5)

    def test_circuit_skips_the_point_model_and_ratio(self):
        with patch(
            "pipeline.predict_trajectory", return_value={"class_id": 2, "predicted": "circuit"}
        ) as prediction:
            result = analyze_trajectory(self.item, self.pattern, self.point, torch.device("cpu"))
        self.assertEqual(prediction.call_count, 1)
        self.assertEqual(result["status"], "local_behavior_not_applicable")
        self.assertIsNone(result["travel"]["working_travel_ratio"])
        self.assertIsNone(result["point_labels"])

    def test_preprocessing_mismatch_cannot_silently_mix_stages(self):
        self.point.preprocessing = replace(PreprocessConfig(), stop_speed_kmh=0.2)
        with self.assertRaisesRegex(ValueError, "identical preprocessing"):
            analyze_trajectory(self.item, self.pattern, self.point, torch.device("cpu"))
