"""Mathematical checks for sparse attention, temporal decay and point labeling."""

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset import TimeSeriesDataset, collate_fn
from model import POINT_VARIANTS, DistanceAttention, PointModelConfig, TransformerNodeClassifier
from pointwise import make_parser
from splits import FileRecord
from trajectory import PreprocessConfig, Standardizer, load_trajectory, prepare_rows

FIXTURE = Path(__file__).parent / "fixtures/data/Train/shuttle/shuttle_train_01.csv"


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def dense_reference(self, attention, values, valid):
        values = values.masked_fill(~valid[..., None], 0)
        batch, length, channels = values.shape
        q, k, v = (
            attention.qkv(values).reshape(batch, length, 3, attention.nhead, attention.head_dim).unbind(dim=2)
        )
        scores = torch.einsum("bihd,bjhd->bihj", q, k) / math.sqrt(attention.head_dim)
        positions = torch.arange(length)
        distance = (positions[:, None] - positions[None, :]).abs()
        scores *= torch.exp(-attention.decay * distance)[None, :, None, :]
        allowed = valid[:, None, None, :].expand(batch, length, attention.nhead, length)
        if attention.radius is not None:
            allowed = allowed & (distance <= attention.radius)[None, :, None, :]
        scores = scores.masked_fill(~allowed, -torch.inf).masked_fill(~valid[:, :, None, None], 0)
        weights = scores.softmax(dim=-1).masked_fill(~allowed, 0)
        result = attention.output(
            torch.einsum("bihj,bjhd->bihd", weights, v).reshape(batch, length, channels)
        )
        return result.masked_fill(~valid[..., None], 0)

    def test_sparse_and_dense_modes_match_the_attention_equation(self):
        values = torch.randn(2, 17, 16)
        valid = torch.arange(17)[None, :] < torch.tensor([17, 7])[:, None]
        for radius in (0, 2, 6, None):
            for decay in (0, 0.1):
                with self.subTest(radius=radius, decay=decay):
                    attention = DistanceAttention(16, 4, radius, decay, 0)
                    torch.testing.assert_close(
                        attention(values, valid), self.dense_reference(attention, values, valid)
                    )

    def test_window_blocks_distant_points_but_keeps_future_neighbors(self):
        attention = DistanceAttention(8, 2, radius=1, decay=0.1, dropout=0)
        values, mask = torch.randn(1, 8, 8), torch.ones(1, 8, dtype=torch.bool)
        expected = attention(values, mask)[:, 3]
        far = values.clone()
        far[:, 0] += 100
        torch.testing.assert_close(attention(far, mask)[:, 3], expected)
        near_future = values.clone()
        near_future[:, 4] += 100
        self.assertFalse(torch.allclose(attention(near_future, mask)[:, 3], expected))

    def test_distance_decay_changes_attention_scores(self):
        first, second = DistanceAttention(8, 2, 6, 0, 0), DistanceAttention(8, 2, 6, 0.1, 0)
        second.load_state_dict(first.state_dict())
        values, mask = torch.randn(1, 12, 8), torch.ones(1, 12, dtype=torch.bool)
        self.assertFalse(torch.allclose(first(values, mask), second(values, mask)))

    def test_sparse_mode_accepts_a_long_sequence(self):
        attention = DistanceAttention(8, 2, 6, 0.1, 0)
        output = attention(torch.randn(1, 8192, 8), torch.ones(1, 8192, dtype=torch.bool))
        self.assertEqual(output.shape, (1, 8192, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_padding_noise_cannot_contaminate_valid_attention(self):
        attention = DistanceAttention(8, 2, 6, 0.1, 0)
        values = torch.randn(1, 9, 8)
        mask = torch.arange(9)[None, :] < 3
        expected = attention(values, mask)
        torch.testing.assert_close(
            attention(values.masked_fill(~mask[..., None], float("nan")), mask), expected
        )


class PointModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_defaults_and_ablation_structure(self):
        config = PointModelConfig()
        self.assertEqual(
            (
                config.d_model,
                config.nhead,
                config.num_layers,
                config.local_window,
                config.time_decay,
                config.dropout,
            ),
            (128, 4, 3, 6, 0.1, 0.1),
        )
        for variant in POINT_VARIANTS:
            model = TransformerNodeClassifier(d_model=16, variant=variant)
            self.assertEqual(model.lstm is not None, variant == "improved")
            self.assertEqual(
                model.encoder[0].attention.radius, 6 if variant in {"improved", "window-only"} else None
            )
            self.assertEqual(
                model.encoder[0].attention.decay, 0.1 if variant in {"improved", "decay-only"} else 0
            )

    def test_all_variants_have_valid_logits_and_gradients(self):
        for variant in POINT_VARIANTS:
            with self.subTest(variant=variant):
                model = TransformerNodeClassifier(d_model=16, variant=variant)
                values = torch.randn(2, 9, 8)
                mask = torch.arange(9)[None, :] < torch.tensor([9, 3])[:, None]
                output = model(values, mask)
                torch.nn.functional.cross_entropy(output[mask], torch.randint(0, 2, (12,))).backward()
                self.assertEqual(output.shape, (2, 9, 2))
                self.assertTrue(
                    all(
                        parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    )
                )

    def test_padding_and_batch_peers_do_not_change_valid_predictions(self):
        model = TransformerNodeClassifier(d_model=16, dropout=0).eval()
        values = torch.randn(2, 11, 8)
        mask = torch.arange(11)[None, :] < torch.tensor([11, 3])[:, None]
        with torch.no_grad():
            expected = model(values, mask)
            single = model(values[1:, :3], mask[1:, :3])
            noisy = model(values.masked_fill(~mask[..., None], float("nan")), mask)
        torch.testing.assert_close(expected[1, :3], single[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(noisy, expected)

    def test_shared_features_and_label_order(self):
        trajectory = load_trajectory(FileRecord(FIXTURE, 0), PreprocessConfig(), require_labels=True)
        scaler = Standardizer.fit([trajectory])
        dataset = TimeSeriesDataset([trajectory], scaler)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0][1].tolist(), [1, 1, 0, 0, 1])
        np.testing.assert_array_equal(dataset[0][0].numpy(), scaler.transform(trajectory.features))
        features, labels, mask = collate_fn([dataset[0], (dataset[0][0][:1], dataset[0][1][:1])])
        self.assertEqual(features.shape, (2, 5, 8))
        self.assertEqual(int(mask.sum()), 6)
        self.assertTrue((labels[~mask] == -100).all())

    def test_training_defaults_match_the_local_profile(self):
        args = make_parser().parse_args(["train", "--data-root", "data", "--output", "out"])
        self.assertEqual((args.learning_rate, args.weight_decay), (0.0001, 0.001))
        self.assertEqual((args.validation_fraction, args.test_fraction), (0.2, 0.1))
        self.assertEqual((args.num_layers, args.nhead, args.d_model), (3, 4, 128))

    def test_stop_boundaries_split_local_sequences_and_keep_point_order(self):
        item = prepare_rows(
            ["longitude", "latitude", "speed", "heading", "timestamp", "work_state"],
            [
                (0, 0, 4, 0, 0, 1),
                (1e-5, 0, 4, 0, 4, 1),
                (1e-5, 0, 0, 0, 8, 0),
                (2e-5, 0, 4, 0, 12, 0),
                (3e-5, 0, 4, 0, 16, 0),
            ],
            PreprocessConfig(),
            require_labels=True,
        )
        dataset = TimeSeriesDataset([item], Standardizer.fit([item]))
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.windows, [(0, 0, 2), (0, 2, 4)])
        self.assertEqual([labels.tolist() for _, labels in dataset], [[1, 1], [0, 0]])
