"""Graph equations, segment-aware batching, model gradients and split isolation."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agri.data import TrajectoryCollator, TrajectoryDataset
from agri.model import VARIANTS, ChainGraphConv, GlobalEncoder, ModelConfig, build_model
from cli import make_parser
from splits import FileRecord, _allocate, discover_splits
from training import ValidationMonitor, classification_metrics
from trajectory import PreprocessConfig, Standardizer, load_trajectory

FIXTURES = Path(__file__).parent / "fixtures/data"


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.batch = TrajectoryCollator()(
            [
                (torch.randn(161, 8), 0, "long.csv"),
                (torch.randn(7, 8), 1, "short.csv"),
            ]
        )

    def test_normalized_chain_matches_dense_gcn(self):
        layer = ChainGraphConv(8, 6)
        values, mask = self.batch["features"].flatten(1, 2), self.batch["mask"].flatten(1, 2)
        result = layer(values, mask)
        for index, size in enumerate((161, 7)):
            adjacent = torch.eye(size) + torch.diag(torch.ones(size - 1), 1)
            adjacent += torch.diag(torch.ones(size - 1), -1)
            degree = adjacent.sum(dim=1).rsqrt()
            expected = (degree[:, None] * adjacent * degree[None, :]) @ layer.projection(values[index, :size])
            torch.testing.assert_close(result[index, :size], expected)
            self.assertEqual(int(torch.count_nonzero(result[index, size:])), 0)

    def test_graph_edges_do_not_bridge_segments(self):
        layer = ChainGraphConv(8, 8)
        with torch.no_grad():
            layer.projection.weight.copy_(torch.eye(8))
        values = torch.arange(32, dtype=torch.float).reshape(1, 4, 8)
        output = layer(values, torch.ones(1, 4, dtype=torch.bool), torch.tensor([[0, 0, 1, 1]]))
        torch.testing.assert_close(output[0, 0], values[0, :2].mean(dim=0))
        torch.testing.assert_close(output[0, 2], values[0, 2:].mean(dim=0))

    def test_global_windows_are_independent_graphs(self):
        encoder = GlobalEncoder(ModelConfig(hidden_dim=8, dropout=0)).eval()
        features = torch.randn(1, 2, 160, 8)
        mask = torch.ones(1, 2, 160, dtype=torch.bool)
        changed = features.clone()
        changed[:, 1] += 100
        with torch.no_grad():
            torch.testing.assert_close(encoder(features, mask)[:, 0], encoder(changed, mask)[:, 0])

    def test_collator_keeps_tail_and_segment_boundaries(self):
        item = (torch.randn(164, 8), 0, "segments.csv", torch.tensor([0] * 161 + [1] * 3))
        batch = TrajectoryCollator()([item])
        self.assertEqual(batch["mask"].sum(dim=-1).tolist(), [[160, 1, 3]])
        self.assertEqual(batch["segment_ids"][0, :, 0].tolist(), [0, 0, 1])

    def test_all_variants_have_finite_gradients(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = build_model(ModelConfig(hidden_dim=8, variant=variant, dropout=0))
                output = model(self.batch["features"], self.batch["mask"], self.batch["segment_ids"])
                self.assertEqual(output.shape, (2, 3))
                torch.nn.functional.cross_entropy(output, self.batch["labels"]).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_padding_extra_graphs_and_batch_peers_do_not_change_outputs(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = build_model(ModelConfig(hidden_dim=8, variant=variant, dropout=0)).eval()
                features, mask = self.batch["features"], self.batch["mask"]
                extra = torch.cat(
                    (features.masked_fill(~mask[..., None], float("nan")), torch.full((2, 1, 160, 8), 999.0)),
                    dim=1,
                )
                extra_mask = torch.cat((mask, torch.zeros(2, 1, 160, dtype=torch.bool)), dim=1)
                with torch.no_grad():
                    expected = model(features, mask)
                    torch.testing.assert_close(model(extra, extra_mask), expected, atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(
                        model(features[1:, :1], mask[1:, :1])[0], expected[1], atol=2e-5, rtol=2e-5
                    )

    def test_multiple_short_segments_and_single_points(self):
        for size in (1, 160, 320):
            batch = TrajectoryCollator()([(torch.randn(size, 8), 0, "file", torch.arange(size) // 3)])
            model = build_model(ModelConfig(hidden_dim=8))
            output = model(batch["features"], batch["mask"], batch["segment_ids"])
            output.sum().backward()
            self.assertTrue(torch.isfinite(output).all())


class SplitTests(unittest.TestCase):
    def test_split_allocator_respects_requested_proportions(self):
        self.assertEqual(_allocate(100, (0.7, 0.2, 0.1)), [70, 20, 10])
        self.assertEqual(_allocate(100, (0.7, 0.15, 0.15)), [70, 15, 15])

    def test_validation_and_test_counts_are_ceiled_per_class(self):
        for size, expected in (
            (143, [99, 29, 15]),
            (157, [109, 32, 16]),
            (297, [207, 60, 30]),
        ):
            with self.subTest(size=size):
                counts = _allocate(size, (0.7, 0.2, 0.1))
                self.assertEqual(counts, expected)
                self.assertEqual(sum(counts), size)

    def test_exact_decimal_products_do_not_round_to_an_extra_file(self):
        self.assertEqual(_allocate(100, (0.83, 0.07, 0.1)), [83, 7, 10])

    def test_ceiling_that_leaves_no_training_files_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no training"):
            _allocate(3, (0.2, 0.4, 0.4))

    def test_discovery_applies_ceiling_independently_to_each_class(self):
        records = [
            FileRecord((FIXTURES / f"class{label}_{index}.csv").resolve(), label)
            for label, size in enumerate((143, 157, 297))
            for index in range(size)
        ]
        with patch("splits._class_files", side_effect=[records, []]):
            splits = discover_splits(FIXTURES)
        for label, expected in enumerate(([99, 29, 15], [109, 32, 16], [207, 60, 30])):
            self.assertEqual(
                [sum(record.label == label for record in group) for group in splits.values()], expected
            )

    def test_default_splits_allocate_seven_two_one_for_each_class(self):
        records = [
            FileRecord((FIXTURES / f"class{label}_{index}.csv").resolve(), label)
            for label in range(3)
            for index in range(100)
        ]
        with patch("splits._class_files", side_effect=[records, []]):
            splits = discover_splits(FIXTURES)
        for label in range(3):
            self.assertEqual(
                [sum(record.label == label for record in group) for group in splits.values()], [70, 20, 10]
            )

    def test_splits_are_disjoint_reproducible_and_class_stratified(self):
        first = discover_splits(FIXTURES)
        self.assertEqual(first, discover_splits(FIXTURES))
        groups = [{record.path for record in records} for records in first.values()]
        for index, group in enumerate(groups):
            for other in groups[index + 1 :]:
                self.assertFalse(group & other)
        for records in first.values():
            self.assertEqual({record.label for record in records}, {0, 1, 2})

    def test_point_task_excludes_circuit(self):
        splits = discover_splits(FIXTURES, validation_fraction=0.2, test_fraction=0.1, classes=(0, 1))
        self.assertEqual({record.label for records in splits.values() for record in records}, {0, 1})

    def test_duplicate_filenames_and_incomplete_provided_layout_fail(self):
        records = [
            FileRecord(Path(f"class{label}_{index}.csv").resolve(), label)
            for label in range(3)
            for index in range(3)
        ]
        with patch("splits._class_files", return_value=records):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                discover_splits(FIXTURES)
        with self.assertRaisesRegex(ValueError, "provided"):
            discover_splits(FIXTURES, policy="provided")

    def test_missing_class_is_explicit(self):
        records = [FileRecord((FIXTURES / f"sample_{index}.csv").resolve(), 0) for index in range(5)]
        with patch("splits._class_files", side_effect=[records[:3], records[3:]]):
            with self.assertRaisesRegex(ValueError, "missing"):
                discover_splits(FIXTURES)

    def test_source_paths_are_relative(self):
        item = load_trajectory(
            FileRecord(FIXTURES / "Train/shuttle/shuttle_train_01.csv", 0), PreprocessConfig()
        )
        sample = TrajectoryDataset([item], Standardizer.fit([item]), FIXTURES)[0]
        self.assertEqual(sample[2], "Train/shuttle/shuttle_train_01.csv")


class ReportingTests(unittest.TestCase):
    def test_class_order_and_missing_class_metrics(self):
        result = classification_metrics([0, 0, 1], [0, 1, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(result["missing_classes"], ["circuit"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)

    def test_minimum_loss_weights_are_saved_even_with_early_stop_delta(self):
        monitor = ValidationMonitor(2, 0.1)
        self.assertEqual(monitor.update(1.0), (True, False))
        self.assertEqual(monitor.update(0.96), (True, False))
        self.assertEqual(monitor.update(0.95), (True, True))

    def test_agri_defaults(self):
        args = make_parser().parse_args(["train", "--data-root", "data", "--output", "out"])
        self.assertEqual((args.epochs, args.batch_size, args.learning_rate), (80, 1, 0.001))
        self.assertEqual((args.lr_step, args.lr_gamma, args.graph_size), (5, 0.1, 160))
        self.assertEqual((args.validation_fraction, args.test_fraction), (0.2, 0.1))
        self.assertEqual(ModelConfig().input_dim, 8)

    def test_normalizer_uses_eight_training_features(self):
        values = np.arange(80).reshape(10, 8)
        scaler = Standardizer.fit([SimpleNamespace(features=values)])
        np.testing.assert_allclose(scaler.mean, values.mean(axis=0))
