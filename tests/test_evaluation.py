"""Operation labels survive batching and point reports retain raw counts."""

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from evaluation import POINT_CLASS_NAMES, PointEvaluationAccumulator, metrics_from_confusion
from splits import FileRecord
from training import make_loader, predict_trajectory, run_epoch
from trajectory import PreprocessConfig, Standardizer, load_trajectory


class GroupedReportTests(unittest.TestCase):
    def updates(self):
        return [
            (
                {"path": "shuttle/a.csv", "operation_id": 0, "operation_name": "shuttle", "segment_id": 0},
                [0, 1],
                [1, 1],
                0.4,
            ),
            (
                {"path": "shuttle/a.csv", "operation_id": 0, "operation_name": "shuttle", "segment_id": 1},
                [0, 1],
                [0, 1],
                0.6,
            ),
            (
                {"path": "skip_row/b.csv", "operation_id": 1, "operation_name": "skip_row", "segment_id": 0},
                [0, 1, 1],
                [0, 0, 1],
                0.9,
            ),
        ]

    def report(self, updates=None):
        accumulator = PointEvaluationAccumulator()
        for values in self.updates() if updates is None else updates:
            accumulator.update(*values)
        metrics = metrics_from_confusion(np.array([[2, 1], [1, 3]]), POINT_CLASS_NAMES)
        metrics["loss"] = 1.9 / 7
        return accumulator.report(metrics)

    def test_operations_and_raw_confusion_matrices_are_retained(self):
        report = self.report()
        self.assertEqual(report["by_operation"]["shuttle"]["metrics"]["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(report["by_operation"]["skip_row"]["metrics"]["confusion_matrix"], [[1, 0], [1, 1]])
        self.assertEqual(report["operation_summary"]["point_counts"], {"shuttle": 4, "skip_row": 3})
        self.assertEqual(report["by_operation"]["shuttle"]["operation_id"], 0)
        self.assertEqual(report["by_operation"]["skip_row"]["operation_id"], 1)

    def test_multiple_segments_count_as_one_source_trajectory(self):
        report = self.report()
        self.assertEqual(report["operation_summary"]["trajectory_count"], 2)
        self.assertEqual(report["operation_summary"]["trajectory_counts"], {"shuttle": 1, "skip_row": 1})
        shuttle = next(item for item in report["per_trajectory"] if item["operation_id"] == 0)
        self.assertEqual(shuttle["segment_count"], 2)
        self.assertEqual(shuttle["point_count"], 4)
        self.assertEqual(shuttle["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(report["point_class_names"], ["turn", "straight"])
        self.assertEqual(report["point_class_labels"], [0, 1])

    def test_overall_metrics_remain_pooled_point_metrics(self):
        result = self.report()["metrics"]
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 3]])
        self.assertAlmostEqual(result["accuracy"], 5 / 7)
        self.assertEqual(result["sample_count"], 7)

    def test_batch_order_does_not_change_the_report(self):
        self.assertEqual(self.report(), self.report(list(reversed(self.updates()))))

    def test_missing_operation_is_explicit(self):
        accumulator = PointEvaluationAccumulator()
        accumulator.update(*self.updates()[0])
        overall = metrics_from_confusion(np.array([[0, 1], [0, 1]]), POINT_CLASS_NAMES)
        result = accumulator.report(overall)
        self.assertEqual(result["by_operation"]["skip_row"]["trajectory_count"], 0)
        self.assertIsNone(result["by_operation"]["skip_row"]["metrics"])

    def test_unknown_or_inconsistent_operation_labels_are_rejected(self):
        for metadata in (
            {"path": "a.csv", "operation_id": 2, "segment_id": 0},
            {"path": "a.csv", "operation_id": 0, "operation_name": "skip_row", "segment_id": 0},
        ):
            with self.assertRaises(ValueError):
                PointEvaluationAccumulator().update(metadata, [0], [0], 0.1)

    def test_inconsistent_overall_counts_are_rejected(self):
        accumulator = PointEvaluationAccumulator()
        accumulator.update(*self.updates()[0])
        with self.assertRaisesRegex(ValueError, "do not match"):
            accumulator.report(metrics_from_confusion(np.ones((2, 2), dtype=np.int64), POINT_CLASS_NAMES))


class AlwaysStraight(nn.Module):
    def forward(self, features, mask=None):
        logits = features.new_zeros(*features.shape[:2], 2)
        logits[..., 1] = 1
        return logits


class TurnPositiveMetricsTests(unittest.TestCase):
    def test_headline_metrics_use_turn_as_the_positive_class(self):
        result = metrics_from_confusion(np.array([[8, 2], [3, 17]]), POINT_CLASS_NAMES)
        self.assertEqual(result["positive_class"], "turn")
        self.assertEqual(result["positive_label"], 0)
        self.assertEqual(result["positive_class_index"], 0)
        self.assertEqual(result["class_names"], ["turn", "straight"])
        self.assertEqual(result["class_labels"], [0, 1])
        self.assertAlmostEqual(result["precision"], 8 / 11)
        self.assertAlmostEqual(result["recall"], 8 / 10)
        self.assertAlmostEqual(result["f1"], 16 / 21)
        self.assertEqual(result["binary_confusion"], {"tp": 8, "fp": 3, "fn": 2, "tn": 17})
        for name in ("precision", "recall", "f1"):
            self.assertEqual(result[name], result["per_class"]["turn"][name])
        self.assertNotEqual(result["precision"], result["precision_macro"])

    def test_no_predicted_turns_have_zero_positive_metrics(self):
        result = metrics_from_confusion(np.array([[0, 2], [0, 8]]), POINT_CLASS_NAMES)
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0, 0, 0))
        self.assertEqual(result["binary_confusion"], {"tp": 0, "fp": 0, "fn": 2, "tn": 8})

    def test_missing_turn_support_has_zero_recall(self):
        result = metrics_from_confusion(np.array([[0, 0], [2, 18]]), POINT_CLASS_NAMES)
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0, 0, 0))
        self.assertEqual(result["missing_classes"], ["turn"])

    def test_positive_class_follows_the_declared_class_order(self):
        result = metrics_from_confusion(np.array([[17, 3], [2, 8]]), ("straight", "turn"))
        self.assertEqual(result["positive_label"], 0)
        self.assertEqual(result["positive_class_index"], 0)
        self.assertAlmostEqual(result["precision"], 8 / 11)
        self.assertEqual(result["binary_confusion"], {"tp": 8, "fp": 3, "fn": 2, "tn": 17})

    def test_default_matrix_order_is_turn_then_straight(self):
        matrix = np.array([[8, 2], [3, 17]])
        result = metrics_from_confusion(matrix)
        self.assertEqual(result["class_names"], ["turn", "straight"])
        self.assertEqual(result["confusion_matrix"], matrix.tolist())
        self.assertAlmostEqual(result["precision"], 8 / 11)
        self.assertAlmostEqual(result["recall"], 8 / 10)
        self.assertAlmostEqual(result["f1"], 16 / 21)

    def test_explicit_straight_first_input_matches_the_default_report(self):
        default_order = metrics_from_confusion(np.array([[8, 2], [3, 17]]))
        straight_first = metrics_from_confusion(np.array([[17, 3], [2, 8]]), ("straight", "turn"))
        self.assertEqual(default_order, straight_first)

    def test_matrix_shape_must_match_the_input_class_order(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            metrics_from_confusion(np.eye(3, dtype=np.int64))

    def test_operation_pattern_metrics_keep_the_multiclass_definition(self):
        result = metrics_from_confusion(np.eye(3, dtype=np.int64), ("shuttle", "skip_row", "circuit"))
        self.assertNotIn("positive_class", result)
        self.assertNotIn("precision", result)
        self.assertEqual(result["precision_macro"], 1)

    def test_grouped_operations_use_the_same_turn_positive_definition(self):
        report = GroupedReportTests().report()
        for group in report["by_operation"].values():
            result = group["metrics"]
            self.assertEqual(result["positive_class"], "turn")
            self.assertEqual(result["precision"], result["per_class"]["turn"]["precision"])
            self.assertEqual(result["recall"], result["per_class"]["turn"]["recall"])
            self.assertEqual(result["f1"], result["per_class"]["turn"]["f1"])


class PointEvaluationIntegrationTests(unittest.TestCase):
    def trajectory(self):
        root = Path(__file__).parent / "fixtures/data"
        return load_trajectory(
            FileRecord(root / "Train/shuttle/shuttle_train_01.csv", 0),
            PreprocessConfig(),
            require_labels=True,
        )

    def test_point_prediction_retains_a_known_operation_label(self):
        item = self.trajectory()
        bundle = SimpleNamespace(task="point", model=AlwaysStraight(), normalizer=Standardizer.fit([item]))
        result = predict_trajectory(bundle, item, torch.device("cpu"))
        self.assertEqual(result["operation_id"], 0)
        self.assertEqual(result["operation_name"], "shuttle")
        self.assertEqual(result["labels"], [1] * len(item.features))
        self.assertEqual(result["class_names"], ["turn", "straight"])
        self.assertEqual(result["class_labels"], [0, 1])
        decoded = np.asarray(result["class_labels"])[np.asarray(result["probabilities"]).argmax(axis=1)]
        self.assertEqual(decoded.tolist(), result["labels"])

    def test_point_prediction_leaves_unknown_operation_labels_null(self):
        item = replace(self.trajectory(), label=-1)
        bundle = SimpleNamespace(task="point", model=AlwaysStraight(), normalizer=Standardizer.fit([item]))
        result = predict_trajectory(bundle, item, torch.device("cpu"))
        self.assertIsNone(result["operation_id"])
        self.assertIsNone(result["operation_name"])

    def test_loader_and_epoch_keep_ground_truth_operation_labels(self):
        root = Path(__file__).parent / "fixtures/data"
        records = [
            FileRecord(root / "Train/shuttle/shuttle_train_01.csv", 0),
            FileRecord(root / "Train/skip_row/skip_row_train_01.csv", 1),
        ]
        items = [load_trajectory(record, PreprocessConfig(), require_labels=True) for record in records]
        normalizer = Standardizer.fit(items)
        loader = make_loader(items, normalizer, "point", batch_size=2, root=root)
        batch = next(iter(loader))
        self.assertEqual([item["operation_name"] for item in batch["metadata"]], ["shuttle", "skip_row"])
        report = run_epoch(AlwaysStraight(), loader, torch.device("cpu"), "point", collect_details=True)
        self.assertEqual(report["operation_summary"]["trajectory_counts"], {"shuttle": 1, "skip_row": 1})
        self.assertEqual(report["metrics"]["confusion_matrix"], [[0, 4], [0, 6]])
        for operation in ("shuttle", "skip_row"):
            self.assertEqual(
                report["by_operation"][operation]["metrics"]["confusion_matrix"], [[0, 2], [0, 3]]
            )
        plain = run_epoch(AlwaysStraight(), loader, torch.device("cpu"), "point")
        self.assertEqual(plain, report["metrics"])
