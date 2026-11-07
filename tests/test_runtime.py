"""Checkpoint compatibility and portable, disjoint split manifests."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import mock_open, patch

import torch

from model import PointModelConfig, TransformerNodeClassifier
from training import CHECKPOINT_VERSION, POINT_CLASS_NAMES, load_checkpoint, read_manifest
from trajectory import FEATURE_NAMES, PreprocessConfig


class ManifestTests(unittest.TestCase):
    def manifest(self):
        return {
            "format_version": CHECKPOINT_VERSION,
            "task": "point",
            "splits": {
                "train": [{"path": "shuttle/train.csv", "label": 0}],
                "val": [{"path": "shuttle/val.csv", "label": 0}],
                "test": [{"path": "shuttle/test.csv", "label": 0}],
            },
        }

    def read(self, value, root=Path("dataset")):
        with patch("pathlib.Path.open", mock_open(read_data=json.dumps(value))):
            return read_manifest(Path("splits.json"), root, "point")

    def test_relocation_changes_only_the_dataset_root(self):
        original = self.read(self.manifest(), Path("dataset"))
        relocated = self.read(self.manifest(), Path("relocated_dataset"))
        self.assertEqual(
            original["test"][0].path.relative_to(Path("dataset").resolve()).as_posix(),
            relocated["test"][0].path.relative_to(Path("relocated_dataset").resolve()).as_posix(),
        )

    def test_cross_split_trajectory_reuse_is_rejected(self):
        manifest = self.manifest()
        manifest["splits"]["test"] = manifest["splits"]["train"]
        with self.assertRaisesRegex(ValueError, "multiple"):
            self.read(manifest)

    def test_escaping_dataset_root_is_rejected(self):
        manifest = self.manifest()
        manifest["splits"]["test"][0]["path"] = "../outside.csv"
        with self.assertRaisesRegex(ValueError, "inside"):
            self.read(manifest)

    def test_incompatible_manifest_is_rejected(self):
        manifest = self.manifest()
        manifest["format_version"] = 1
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.read(manifest)


class CheckpointTests(unittest.TestCase):
    def test_legacy_checkpoint_requires_retraining(self):
        with patch("torch.load", return_value={"format_version": 1}):
            with self.assertRaisesRegex(ValueError, "retrain"):
                load_checkpoint(Path("old.pt"), torch.device("cpu"))

    def test_task_mismatch_is_rejected_before_loading_weights(self):
        state = {
            "format_version": CHECKPOINT_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "task": "pattern",
        }
        with patch("torch.load", return_value=state):
            with self.assertRaisesRegex(ValueError, "different recognition"):
                load_checkpoint(Path("pattern.pt"), torch.device("cpu"), "point")

    def test_model_and_preprocessing_round_trip(self):
        torch.set_num_threads(2)
        config = PointModelConfig(d_model=16, dropout=0)
        model = TransformerNodeClassifier(config).eval()
        state = {
            "format_version": CHECKPOINT_VERSION,
            "task": "point",
            "feature_names": list(FEATURE_NAMES),
            "class_names": list(POINT_CLASS_NAMES),
            "model_config": asdict(config),
            "model_state": model.state_dict(),
            "preprocessing": asdict(PreprocessConfig()),
            "normalizer": {"mean": [0] * 8, "scale": [1] * 8},
        }
        with patch("torch.load", return_value=state):
            bundle = load_checkpoint(Path("point.pt"), torch.device("cpu"), "point")
        values = torch.randn(1, 9, 8)
        with torch.no_grad():
            torch.testing.assert_close(bundle.model(values), model(values))
        self.assertEqual(bundle.preprocessing, PreprocessConfig())
