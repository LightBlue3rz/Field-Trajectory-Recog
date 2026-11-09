# Field Trajectory Recognition

PyTorch models for operation-pattern classification, turn/straight point recognition, and working travel ratio calculation.

## Install

Requires Python 3.12 or newer. Run commands from the repository root.

```bash
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -r requirements.txt
```

CUDA is used when available. Add `--device cpu` to run on CPU.

## Data

Use UTF-8 CSV or XLSX files (first worksheet), with one complete field trajectory per file:

```text
data/
  shuttle/trajectory_001.csv
  skip_row/trajectory_002.csv
  circuit/trajectory_003.csv
```

Class directories also accept `梭行法`, `套行法` and `绕行法`, respectively.

| Column | Format |
| --- | --- |
| `longitude`, `latitude` | Decimal degrees |
| `speed` | km/h |
| `heading` | Degrees |
| `timestamp` | Date-time or numeric seconds |
| `work_state` / `工作状态` | Point training/evaluation only: **turn=0, straight=1** |

Point training uses shuttle and skip-row trajectories. Operation classification uses all three classes. Preprocessing, 4-second sampling, feature extraction and normalization are automatic; inference does not require point annotations.

## Train

```bash
# Operation classification
python -m agri train --data-root data --output ../Model/pattern

# Turn/straight point recognition
python train.py --data-root data --output ../Model/point
```

Default limits are 80 epochs for operation classification and 100 for point recognition, with batch size 1. Use `--epochs`, `--batch-size`, `--learning-rate` or `--help` to adjust training.

Both tasks use file-level, class-stratified **70/20/10** train/validation/test splits with seed 42. Validation and test counts are rounded up per class; training receives the remainder. At least three trajectories per class are required. Use `--split-policy provided` to preserve existing `Train/Val/Test` directories, or add `--split-manifest ../Model/pattern/splits.json` to point training to share the operation model's partitions.

Use a new or empty output directory for each run. Training saves checkpoints including `best_val.pt`, actual file partitions in `splits.json`, parameters in `config.json`, learning history in `history.json`, and test metrics in `test_results.json`.

## Evaluate and predict

```bash
python -m agri eval --data-root data --checkpoint ../Model/pattern/best_val.pt --manifest ../Model/pattern/splits.json --output ../Model/pattern_eval
python eval.py --data-root data --checkpoint ../Model/point/best_val.pt --manifest ../Model/point/splits.json --output ../Model/point_eval

python -m agri predict --checkpoint ../Model/pattern/best_val.pt --input data/new_trajectory.xlsx --output ../Model/pattern_predict
python predict.py --checkpoint ../Model/point/best_val.pt --input data/new_trajectory.xlsx --output ../Model/point_predict
```

Evaluation uses the test partition by default (`--split val` selects validation) and writes `evaluation.json`. Prediction accepts a file or directory and writes `predictions.json`.

Metrics are fractions from 0 to 1. Point precision, recall and F1 use **turn as the positive class**; confusion matrices use `[turn, straight]`, with true classes on rows and predictions on columns. Point reports also retain operation labels and per-trajectory results.

## Hierarchical analysis

```bash
python pipeline.py --pattern-checkpoint ../Model/pattern/best_val.pt --point-checkpoint ../Model/point/best_val.pt --input data/new_trajectory.xlsx --output ../Model/analysis
```

Use checkpoints with matching preprocessing settings. `analysis.json` contains the operation class, local behavior labels, distances and `working_travel_ratio = straight_distance / total_distance`. Local behavior and the ratio are computed for shuttle and skip-row operations; circuit results have no local labels or ratio.

## Tests

Tests run without pretrained weights:

```bash
# All tests
python -B -m unittest discover -s tests -p "test_*.py" -v

# Four recorded trajectory examples only
python -B -m unittest discover -s tests -p "test_recorded_samples.py" -v
```

The [recorded samples](tests/fixtures/recorded/README.md) include two de-identified shuttle and two skip-row trajectories. Tests cover preprocessing, both models, an optimizer step and distance accounting using in-memory models. Recognition accuracy requires trained checkpoints and held-out data.

For a one-epoch training check, use `--data-root tests/fixtures/data --epochs 1` with either training command. These three-class synthetic fixtures support all three partitions; the four recorded examples alone have too few files per class for that split.
