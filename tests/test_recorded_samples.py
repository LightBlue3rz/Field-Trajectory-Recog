"""Functional checks on de-identified recorded trajectories, without checkpoints."""

import unittest
from pathlib import Path

import numpy as np
import torch

from agri.model import ModelConfig, build_model
from dataset import TimeSeriesDataset
from efficiency import travel_statistics
from model import PointModelConfig, TransformerNodeClassifier
from splits import FileRecord
from training import make_loader, run_epoch
from trajectory import FEATURE_NAMES, PreprocessConfig, Standardizer, load_trajectory, read_table

FIXTURES = Path(__file__).parent / "fixtures/recorded"
# Path, operation ID, raw rows, sampled points, segments, turn points, distance (m).
SAMPLES = (
    ("shuttle/recorded_01.csv", 0, 1002, 504, 58, 135, 1220.4616942071596),
    ("shuttle/recorded_02.csv", 0, 1034, 506, 5, 124, 3909.8231822194693),
    ("skip_row/recorded_01.csv", 1, 621, 496, 3, 52, 5790.822049183116),
    ("skip_row/recorded_02.csv", 1, 380, 339, 11, 67, 3961.242661063743),
)


class RecordedSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.trajectories = [
            load_trajectory(FileRecord(FIXTURES / path, label), PreprocessConfig(), require_labels=True)
            for path, label, *_ in SAMPLES
        ]
        # This normalization is only for functional tests, not an accuracy evaluation.
        cls.normalizer = Standardizer.fit(cls.trajectories)

    def setUp(self):
        torch.manual_seed(42)

    def test_recorded_files_use_anonymous_coordinates_and_relative_time(self):
        self.assertEqual(len(list(FIXTURES.rglob("*.csv"))), 4)
        for path, _, raw_rows, *_ in SAMPLES:
            with self.subTest(path=path):
                columns, rows = read_table(FIXTURES / path)
                self.assertEqual(columns, [*FEATURE_NAMES[:4], "timestamp", "work_state"])
                self.assertEqual(len(rows), raw_rows)
                coordinates = np.asarray([[float(row[0]), float(row[1])] for row in rows])
                timestamps = np.asarray([float(row[4]) for row in rows])
                self.assertLess(float(np.abs(coordinates).max()), 0.01)
                self.assertEqual(float(timestamps.min()), 0)
                self.assertLess(float(timestamps.max()), 86400)

    def test_default_preprocessing_preserves_reference_counts_and_labels(self):
        for item, sample in zip(self.trajectories, SAMPLES):
            path, _, _, points, segments, turn_points, _ = sample
            with self.subTest(path=path):
                self.assertEqual(item.features.shape, (points, len(FEATURE_NAMES)))
                self.assertEqual(item.diagnostics["segment_count"], segments)
                self.assertEqual(int((item.point_labels == 0).sum()), turn_points)
                self.assertEqual(int((item.point_labels == 1).sum()), points - turn_points)
                self.assertTrue(np.isfinite(item.features).all())
                connected = item.segment_ids[1:] == item.segment_ids[:-1]
                np.testing.assert_array_equal(np.diff(item.timestamps)[connected], 4)

    def test_default_pattern_model_runs_without_pretrained_weights(self):
        model = build_model(ModelConfig())
        loader = make_loader(self.trajectories, self.normalizer, "pattern", 2, root=FIXTURES)
        result = run_epoch(model, loader, torch.device("cpu"), "pattern")
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(np.asarray(result["confusion_matrix"]).sum(axis=1).tolist(), [2, 2, 0])

    def test_default_point_model_retains_operation_labels_without_pretrained_weights(self):
        model = TransformerNodeClassifier(PointModelConfig())
        loader = make_loader(self.trajectories, self.normalizer, "point", 4, root=FIXTURES)
        result = run_epoch(model, loader, torch.device("cpu"), "point", collect_details=True)
        self.assertTrue(np.isfinite(result["metrics"]["loss"]))
        self.assertEqual(result["operation_summary"]["trajectory_counts"], {"shuttle": 2, "skip_row": 2})
        self.assertEqual(result["metrics"]["sample_count"], sum(sample[3] for sample in SAMPLES))
        self.assertEqual(result["metrics"]["class_names"], ["turn", "straight"])
        self.assertEqual(result["metrics"]["positive_label"], 0)
        self.assertEqual(result["metrics"]["per_class"]["turn"]["support"], sum(s[5] for s in SAMPLES))
        self.assertEqual({item["path"] for item in result["per_trajectory"]}, {s[0] for s in SAMPLES})

    def test_training_step_updates_fresh_point_model_without_saving_weights(self):
        dataset = TimeSeriesDataset(self.trajectories, self.normalizer)
        features, labels = max(dataset, key=lambda item: len(item[0]))
        model = TransformerNodeClassifier(PointModelConfig(d_model=16, num_layers=1, dropout=0))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        before = model.classifier.weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(features[None])[0], labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(
            all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        )
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_annotated_travel_distances_match_recorded_references(self):
        for item, sample in zip(self.trajectories, SAMPLES):
            with self.subTest(path=sample[0]):
                result = travel_statistics(item, item.point_labels)
                self.assertAlmostEqual(result["total_distance_m"], sample[6], places=3)
                self.assertAlmostEqual(
                    result["straight_distance_m"] + result["turn_distance_m"], result["total_distance_m"]
                )
                self.assertGreater(result["working_travel_ratio"], 0)
                self.assertLess(result["working_travel_ratio"], 1)
