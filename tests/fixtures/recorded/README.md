# Recorded trajectory samples

Four complete recorded trajectories are included as CSV fixtures: two shuttle operations and two skip-row operations. Their measured speed, heading, relative timing and binary annotations are retained. Point labels are `turn=0` and `straight=1`.

| File | Raw rows | Points after default preprocessing | Moving segments | Turn points | Straight points |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shuttle/recorded_01.csv` | 1002 | 504 | 58 | 135 | 369 |
| `shuttle/recorded_02.csv` | 1034 | 506 | 5 | 124 | 382 |
| `skip_row/recorded_01.csv` | 621 | 496 | 3 | 52 | 444 |
| `skip_row/recorded_02.csv` | 380 | 339 | 11 | 67 | 272 |

## De-identification

Each CSV contains only `longitude`, `latitude`, `speed`, `heading`, `timestamp` and `work_state`. Original filenames, device identifiers, calendar dates and absolute field locations are omitted.

Coordinates are relocated near `(0, 0)`: latitude is offset from the first valid observation, and longitude is offset and multiplied by the cosine of the source trajectory's mean latitude. The cosine uses radians. This preserves local trajectory shape and approximately preserves geographic distances. Numeric timestamps are elapsed seconds from the beginning of each recording. Original sampling gaps, stops and behavior annotations are retained. Default preprocessing produces the same point labels and moving-segment membership as the source recordings.

## Run without checkpoints

From the repository root, after installing `requirements.txt`:

```bash
python -B -m unittest discover -s tests -p "test_recorded_samples.py" -v
```

The tests cover raw CSV input, four-second sampling, reference point counts, both default model architectures, operation-labeled point reports, one optimizer step on an in-memory model, and distance accounting from the supplied annotations. Models are initialized in memory; no pretrained weights, downloads or saved checkpoints are needed. Recognition accuracy is evaluated separately with trained checkpoints and held-out data.

There are only two trajectories per operation and no circuit examples here. Use the larger dataset for train/validation/test experiments; the existing `tests/fixtures/data` directory provides small three-class synthetic fixtures for training quick checks.
