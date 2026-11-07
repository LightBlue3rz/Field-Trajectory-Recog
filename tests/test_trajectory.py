"""Shared preprocessing and feature equations on synthetic raw trajectories."""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook

from trajectory import FEATURE_NAMES, PreprocessConfig, Standardizer, _seconds, prepare_rows, read_table

COLUMNS = ["longitude", "latitude", "speed", "heading", "timestamp", "work_state"]


def prepare(rows, **kwargs):
    return prepare_rows(COLUMNS, rows, PreprocessConfig(), **kwargs)


class PreprocessingTests(unittest.TestCase):
    def test_xlsx_reads_the_first_worksheet_when_another_sheet_is_active(self):
        workbook = Workbook()
        workbook.worksheets[0].append(COLUMNS)
        workbook.worksheets[0].append([0, 0, 4, 0, 0, 1])
        workbook.create_sheet("notes").append(["Unrelated worksheet"])
        workbook.active = 1
        with patch("trajectory.load_workbook", return_value=workbook):
            columns, rows = read_table(Path("trajectory.xlsx"))
        self.assertEqual(columns, COLUMNS)
        self.assertEqual(rows, [(0, 0, 4, 0, 0, 1)])

    def test_eight_features_and_derivatives_use_actual_sample_interval(self):
        item = prepare([(0, 0, 4, 359, 0, 1), (1e-5, 2e-5, 8, 1, 4, 0)])
        self.assertEqual(item.features.shape, (2, 8))
        np.testing.assert_allclose(item.features[1, :4], [1e-5, 2e-5, 8, 1])
        np.testing.assert_allclose(
            item.features[1, 4:], [1e-5 * np.cos(np.radians(1)), 2e-5 * np.cos(np.radians(1)), 1, 0.5]
        )
        np.testing.assert_array_equal(item.features[0, 4:], 0)

    def test_two_second_data_is_sampled_at_four_seconds(self):
        item = prepare([(i * 1e-5, 0, 4, 0, 2 * i, i % 2) for i in range(6)])
        np.testing.assert_array_equal(item.timestamps, [0, 4, 8])
        np.testing.assert_array_equal(item.source_rows, [0, 2, 4])

    def test_short_gap_interpolation_and_circular_heading(self):
        item = prepare([(0, 0, 4, 359, 0, 1), (2e-5, 0, 8, 1, 8, 0)], require_labels=True)
        np.testing.assert_array_equal(item.timestamps, [0, 4, 8])
        np.testing.assert_allclose(item.features[:, 3], [359, 0, 1], atol=1e-8)
        np.testing.assert_allclose(item.features[:, 2], [4, 6, 8])
        self.assertEqual(item.point_labels.tolist(), [1, 1, 0])
        self.assertEqual(item.diagnostics["interpolated_points"], 1)

    def test_stop_removal_creates_a_boundary_without_interpolation(self):
        item = prepare(
            [(0, 0, 4, 0, 0, 1), (1e-5, 0, 0, 0, 4, 1), (2e-5, 0, 4, 0, 8, 1), (3e-5, 0, 4, 0, 12, 1)]
        )
        self.assertEqual(item.timestamps.tolist(), [0, 8, 12])
        self.assertEqual(item.segment_ids.tolist(), [0, 1, 1])
        self.assertEqual(item.diagnostics["stop_rows"], 1)
        np.testing.assert_array_equal(item.features[1, 4:], 0)

    def test_long_gaps_are_not_bridged(self):
        item = prepare(
            [(0, 0, 4, 0, 0, 1), (1e-5, 0, 4, 0, 4, 1), (1, 1, 4, 0, 40, 0), (1.00001, 1, 4, 0, 44, 0)]
        )
        self.assertEqual(item.timestamps.tolist(), [0, 4, 40, 44])
        self.assertEqual(item.segment_ids.tolist(), [0, 0, 1, 1])
        self.assertEqual(item.diagnostics["gap_count"], 1)

    def test_drift_is_filtered_against_the_last_accepted_point(self):
        item = prepare([(0, 0, 4, 0, 0, 1), (1, 1, 4, 0, 4, 0), (2e-5, 0, 4, 0, 8, 1)])
        self.assertEqual(item.diagnostics["drift_rows"], 1)
        self.assertEqual(item.diagnostics["interpolated_points"], 1)
        np.testing.assert_allclose(item.coordinates[-1], [2e-5, 0], atol=1e-10)

    def test_invalid_rows_and_duplicate_timestamps(self):
        item = prepare(
            [(1e-5, 0, 4, 0, 4, 1), (0, 0, 4, 0, 0, 0), (1e-5, 0, 9, 0, 4, 0), (None, 0, 4, 0, 8, 0)]
        )
        self.assertEqual(item.timestamps.tolist(), [0, 4])
        self.assertEqual(item.diagnostics["duplicate_timestamps"], 1)
        self.assertEqual(item.diagnostics["invalid_rows"], 1)

    def test_raw_coordinates_retain_precision_before_normalization(self):
        item = prepare([(120, 30, 4, 0, 0, 1), (120.0000001, 30, 4, 0, 4, 1)])
        self.assertAlmostEqual(item.features[1, 0] - item.features[0, 0], 1e-7, places=12)

    def test_point_labels_never_change_pattern_input(self):
        rows = [(0, 0, 4, 0, 0, 1), (1e-5, 0, 4, 0, 4, 0)]
        first = prepare(rows)
        second = prepare([(*row[:5], "unused") for row in rows])
        np.testing.assert_array_equal(first.features, second.features)

    def test_binary_labels_are_required_after_resampling(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            prepare([(0, 0, 4, 0, 0, 3)], require_labels=True)
        with self.assertRaisesRegex(ValueError, "work_state"):
            prepare_rows(COLUMNS[:5], [(0, 0, 4, 0, 0)], PreprocessConfig(), require_labels=True)

    def test_all_stopped_and_missing_timestamps_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "No moving"):
            prepare([(0, 0, 0, 0, 0, 1)])
        with self.assertRaisesRegex(ValueError, "timestamp"):
            prepare_rows(COLUMNS[:4], [(0, 0, 4, 0)], PreprocessConfig())

    def test_dates_and_timezone_aware_inputs(self):
        self.assertEqual(_seconds("2016/9/3 2:04:12"), _seconds(datetime(2016, 9, 3, 2, 4, 12)))
        utc = datetime(2020, 1, 1, tzinfo=timezone.utc)
        local = utc.astimezone(timezone(timedelta(hours=8)))
        self.assertEqual(_seconds(utc), _seconds(local))

    def test_invalid_preprocessing_configurations(self):
        for kwargs in (
            {"interval_seconds": 0},
            {"max_gap_seconds": 1},
            {"max_speed_kmh": 0},
            {"stop_speed_kmh": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(PreprocessConfig(), **kwargs)

    def test_chinese_aliases_produce_the_same_features(self):
        rows = [(0, 0, 4, 0, 0, 1), (1e-5, 0, 8, 90, 4, 0)]
        chinese = ["经度", "纬度", "速度(km/h)", "航向", "GPS时间", "工作状态"]
        item = prepare_rows(chinese, rows, PreprocessConfig(), path=Path("synthetic.csv"))
        np.testing.assert_array_equal(item.features, prepare(rows).features)


class NormalizationTests(unittest.TestCase):
    def test_training_population_statistics_and_round_trip(self):
        values = np.arange(80, dtype=np.float64).reshape(10, 8)
        scaler = Standardizer.fit(
            [SimpleNamespace(features=values[:3]), SimpleNamespace(features=values[3:])]
        )
        np.testing.assert_allclose(scaler.mean, values.mean(axis=0))
        np.testing.assert_allclose(scaler.scale, values.std(axis=0))
        restored = Standardizer.from_state_dict(scaler.state_dict())
        np.testing.assert_array_equal(restored.transform(values), scaler.transform(values))

    def test_constant_features_and_legacy_schema(self):
        scaler = Standardizer.fit([SimpleNamespace(features=np.ones((3, len(FEATURE_NAMES))))])
        np.testing.assert_array_equal(scaler.transform(np.ones((3, 8))), 0)
        with self.assertRaisesRegex(ValueError, "schema"):
            Standardizer.from_state_dict({"mean": [0] * 4, "scale": [1] * 4})
