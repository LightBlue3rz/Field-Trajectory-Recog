"""Shared training, checkpoint schema and inference for both recognition stages."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from agri.data import TrajectoryCollator, TrajectoryDataset
from agri.model import ModelConfig, build_model
from dataset import TimeSeriesDataset, collate_fn
from evaluation import (
    POINT_CLASS_NAMES,
    POINT_REPORT_CLASS_LABELS,
    POINT_REPORT_CLASS_NAMES,
    PointEvaluationAccumulator,
    metrics_from_confusion,
)
from evaluation import classification_metrics as classification_metrics
from model import PointModelConfig, TransformerNodeClassifier
from splits import CLASS_DIRECTORIES, CLASS_NAMES, SPLIT_ROUNDING, FileRecord, discover_splits
from trajectory import (
    FEATURE_NAMES,
    TABLE_SUFFIXES,
    PreprocessConfig,
    Standardizer,
    load_trajectory,
    relative_path,
)

CHECKPOINT_VERSION = 2


@dataclass
class ValidationMonitor:
    patience: int
    min_delta: float = 0
    best_loss: float = float("inf")
    reference_loss: float = float("inf")
    stale_epochs: int = 0

    def update(self, loss):
        improved = loss < self.best_loss
        self.best_loss = min(self.best_loss, loss)
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss, self.stale_epochs = loss, 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience


def write_json(path: Path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def new_output(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device cpu")
    return device


def make_loader(
    trajectories, normalizer, task, batch_size, graph_size=160, shuffle=False, seed=42, root=None
):
    if task == "pattern":
        dataset, collator = TrajectoryDataset(trajectories, normalizer, root), TrajectoryCollator(graph_size)
    else:
        dataset = TimeSeriesDataset(trajectories, normalizer, root, include_metadata=True)
        collator = collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collator,
    )


def run_epoch(model, loader, device, task="pattern", optimizer=None, collect_details=False):
    model.train(optimizer is not None)
    names = CLASS_NAMES if task == "pattern" else POINT_CLASS_NAMES
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    total_loss, count = 0.0, 0
    grouped = PointEvaluationAccumulator() if task == "point" and collect_details else None
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            if task == "pattern":
                logits = model(
                    batch["features"].to(device), batch["mask"].to(device), batch["segment_ids"].to(device)
                )
                labels = batch["labels"].to(device)
            else:
                features = batch["features"].to(device)
                point_labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                point_logits = model(features, mask)
                logits, labels = point_logits[mask], point_labels[mask]
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite classification loss")
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
        actual = labels.detach().cpu().numpy()
        predicted = logits.detach().argmax(dim=-1).cpu().numpy()
        np.add.at(confusion, (actual, predicted), 1)
        total_loss += loss.item() * len(actual)
        count += len(actual)
        if grouped is not None:
            with torch.no_grad():
                point_losses = nn.functional.cross_entropy(
                    point_logits.detach().transpose(1, 2), point_labels, reduction="none"
                )
                point_predictions = point_logits.detach().argmax(dim=-1)
                for index, metadata in enumerate(batch["metadata"]):
                    valid = mask[index]
                    grouped.update(
                        metadata,
                        point_labels[index, valid].cpu().numpy(),
                        point_predictions[index, valid].cpu().numpy(),
                        point_losses[index, valid].sum().item(),
                    )
    if not count:
        raise ValueError("No classification samples remain")
    result = metrics_from_confusion(confusion, names)
    result["loss"] = total_loss / count
    if collect_details:
        return grouped.report(result) if grouped is not None else {"metrics": result}
    return result


@dataclass
class ModelBundle:
    model: nn.Module
    normalizer: Standardizer
    preprocessing: PreprocessConfig
    task: str
    state: dict


def load_checkpoint(path, device, expected_task=None):
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("format_version") != CHECKPOINT_VERSION or state.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Incompatible checkpoint schema; retrain an eight-feature model")
    task = state.get("task")
    if task not in {"pattern", "point"} or (expected_task is not None and task != expected_task):
        raise ValueError("Checkpoint belongs to a different recognition task")
    names = CLASS_NAMES if task == "pattern" else POINT_CLASS_NAMES
    if state.get("class_names") != list(names):
        raise ValueError("Incompatible checkpoint class order")
    model = (
        build_model(ModelConfig(**state["model_config"]))
        if task == "pattern"
        else TransformerNodeClassifier(PointModelConfig(**state["model_config"]))
    )
    model.load_state_dict(state["model_state"])
    return ModelBundle(
        model.to(device).eval(),
        Standardizer.from_state_dict(state["normalizer"]),
        PreprocessConfig(**state["preprocessing"]),
        task,
        state,
    )


def read_manifest(path, root, task=None):
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Incompatible split manifest")
    if task is not None and manifest.get("task") != task:
        raise ValueError("Manifest belongs to a different recognition task")
    splits = {}
    seen = set()
    for split in ("train", "val", "test"):
        records = []
        for item in manifest["splits"][split]:
            path = (Path(root).resolve() / item["path"]).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError("Manifest paths must remain inside data-root")
            if path in seen:
                raise ValueError("Source trajectory occurs in multiple manifest entries")
            seen.add(path)
            label = int(item["label"])
            if label not in range(len(CLASS_NAMES)):
                raise ValueError("Manifest operation label must be 0, 1 or 2")
            records.append(FileRecord(path, label))
        splits[split] = records
    return splits


def train_experiment(args, task="pattern"):
    if min(args.epochs, args.batch_size, args.patience, args.lr_step) < 1:
        raise ValueError("Epoch, batch, patience and scheduler counts must be positive")
    if not all(
        math.isfinite(value)
        for value in (args.learning_rate, args.weight_decay, args.lr_gamma, args.min_delta)
    ):
        raise ValueError("Training parameters must be finite")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 < args.lr_gamma <= 1 or args.min_delta < 0:
        raise ValueError("Invalid optimizer, scheduler or early-stopping settings")
    seed_everything(args.seed)
    device, root = choose_device(args.device), args.data_root.resolve()
    preprocessing = PreprocessConfig(
        args.interval_seconds, args.stop_speed_kmh, args.max_speed_kmh, args.max_gap_seconds
    )
    classes = (0, 1, 2) if task == "pattern" else (0, 1)
    if args.split_manifest:
        splits = read_manifest(args.split_manifest, root)
        splits = {
            split: [record for record in records if record.label in classes]
            for split, records in splits.items()
        }
        if any(not records for records in splits.values()):
            raise ValueError("Shared manifest has an empty split after selecting task classes")
        if args.limit_per_class is not None:
            if args.limit_per_class < 1:
                raise ValueError("limit_per_class must be positive")
            for split, records in splits.items():
                selected = []
                for label in classes:
                    group = [record for record in records if record.label == label]
                    random.Random(args.seed + label).shuffle(group)
                    selected.extend(group[: args.limit_per_class])
                splits[split] = selected
    else:
        splits = discover_splits(
            root,
            args.seed,
            args.validation_fraction,
            args.test_fraction,
            args.allow_missing_classes,
            args.limit_per_class,
            args.split_policy,
            classes,
        )
    for split, records in splits.items():
        missing_labels = set(classes) - {record.label for record in records}
        if missing_labels and not args.allow_missing_classes:
            raise ValueError(f"{split} is missing required operation classes")
    if task == "pattern":
        config = ModelConfig(
            graph_size=args.graph_size,
            local_nodes=args.local_nodes,
            hidden_dim=args.hidden_dim,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            variant=args.variant,
        )
        model = build_model(config).to(device)
    else:
        config = PointModelConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            local_window=args.local_window,
            time_decay=args.time_decay,
            lstm_layers=args.lstm_layers,
            variant=args.variant,
        )
        model = TransformerNodeClassifier(config).to(device)
    trajectories = {}
    for split, records in splits.items():
        print(f"Preparing {split}: {len(records)} files", flush=True)
        trajectories[split] = []
        for index, record in enumerate(records):
            trajectories[split].append(
                load_trajectory(record, preprocessing, task == "point", args.label_column)
            )
            if (index + 1) % 25 == 0:
                print(f"Prepared {index + 1}/{len(records)} {split} files", flush=True)
    normalizer = Standardizer.fit(trajectories["train"])
    output = new_output(args.output)
    manifest = {
        "format_version": CHECKPOINT_VERSION,
        "task": task,
        "seed": args.seed,
        "policy": "manifest" if args.split_manifest else args.split_policy,
        "rounding": SPLIT_ROUNDING if not args.split_manifest and args.split_policy == "stratified" else None,
        "fractions": [
            1 - args.validation_fraction - args.test_fraction,
            args.validation_fraction,
            args.test_fraction,
        ]
        if not args.split_manifest and args.split_policy == "stratified"
        else None,
        "counts": {split: len(records) for split, records in splits.items()},
        "class_counts": {
            split: {CLASS_NAMES[label]: sum(record.label == label for record in records) for label in classes}
            for split, records in splits.items()
        },
        "splits": {
            split: [{"path": relative_path(record.path, root), "label": record.label} for record in records]
            for split, records in splits.items()
        },
    }
    write_json(output / "splits.json", manifest)
    write_json(
        output / "preprocessing.json",
        {
            "config": asdict(preprocessing),
            "feature_names": FEATURE_NAMES,
            "normalizer": normalizer.state_dict(),
            "diagnostics": {
                split: [{"path": relative_path(item.path, root), **item.diagnostics} for item in items]
                for split, items in trajectories.items()
            },
        },
    )
    write_json(
        output / "config.json",
        {
            "task": task,
            "model": asdict(config),
            "preprocessing": asdict(preprocessing),
            "training": {
                key: value
                for key, value in vars(args).items()
                if key != "handler" and not isinstance(value, Path)
            },
            "torch_version": str(torch.__version__),
        },
    )
    graph_size = config.graph_size if task == "pattern" else 160
    loaders = {
        split: make_loader(
            items, normalizer, task, args.batch_size, graph_size, split == "train", args.seed, root
        )
        for split, items in trajectories.items()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    monitor, history, best_train = ValidationMonitor(args.patience, args.min_delta), [], float("inf")
    names = CLASS_NAMES if task == "pattern" else POINT_CLASS_NAMES
    missing = [
        CLASS_NAMES[index] for index in classes if index not in {record.label for record in splits["train"]}
    ]

    def save(name, epoch):
        torch.save(
            {
                "format_version": CHECKPOINT_VERSION,
                "task": task,
                "feature_names": list(FEATURE_NAMES),
                "class_names": list(names),
                "model_config": asdict(config),
                "model_state": model.state_dict(),
                "preprocessing": asdict(preprocessing),
                "normalizer": normalizer.state_dict(),
                "label_column": args.label_column,
                "missing_training_classes": missing,
                "epoch": epoch,
                "seed": args.seed,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            output / name,
        )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, task, optimizer)
        val_metrics = run_epoch(model, loaders["val"], device, task)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        scheduler.step()
        if train_metrics["loss"] < best_train:
            best_train = train_metrics["loss"]
            save("best_train.pt", epoch)
        improved, stop = monitor.update(val_metrics["loss"])
        if improved:
            save("best_val.pt", epoch)
        save("last.pt", epoch)
        write_json(output / "history.json", history)
        print(
            f"Epoch {epoch}/{args.epochs} train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_accuracy={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if stop:
            break
    selected = load_checkpoint(output / "best_val.pt", device, task)
    test_result = run_epoch(selected.model, loaders["test"], device, task, collect_details=True)
    test_result.update(selected_epoch=selected.state["epoch"], missing_training_classes=missing)
    write_json(output / "test_results.json", test_result)
    print(f"Test accuracy={test_result['metrics']['accuracy']:.4f}; results: {output.name}", flush=True)
    return output


def evaluate_checkpoint(args, task="pattern"):
    device = choose_device(args.device)
    bundle = load_checkpoint(args.checkpoint, device, task)
    records = read_manifest(args.manifest, args.data_root, task)[args.split]
    trajectories = [
        load_trajectory(record, bundle.preprocessing, task == "point", bundle.state["label_column"])
        for record in records
    ]
    graph_size = bundle.model.config.graph_size if task == "pattern" else 160
    loader = make_loader(
        trajectories, bundle.normalizer, task, args.batch_size, graph_size, root=args.data_root
    )
    result = run_epoch(bundle.model, loader, device, task, collect_details=True)
    result["split"] = args.split
    output = new_output(args.output)
    write_json(output / "evaluation.json", result)
    print(f"Evaluation accuracy={result['metrics']['accuracy']:.4f}", flush=True)
    return output


def predict_trajectory(bundle, trajectory, device):
    if bundle.task == "pattern":
        dataset = TrajectoryDataset([trajectory], bundle.normalizer)
        batch = TrajectoryCollator(bundle.model.config.graph_size)([dataset[0]])
        with torch.inference_mode():
            probabilities = (
                bundle.model(
                    batch["features"].to(device), batch["mask"].to(device), batch["segment_ids"].to(device)
                )
                .softmax(dim=-1)[0]
                .cpu()
                .tolist()
            )
        return {
            "class_id": int(np.argmax(probabilities)),
            "predicted": CLASS_NAMES[int(np.argmax(probabilities))],
            "probabilities": dict(zip(CLASS_NAMES, probabilities)),
        }
    dataset = TimeSeriesDataset([trajectory], bundle.normalizer)
    labels, probabilities = [], []
    with torch.inference_mode():
        for features, _ in dataset:
            scores = bundle.model(features[None].to(device)).softmax(dim=-1)[0].cpu()
            labels.extend(scores.argmax(dim=-1).tolist())
            probabilities.extend(scores.tolist())
    operation_id = int(trajectory.label) if trajectory.label in (0, 1) else None
    return {
        "labels": labels,
        "probabilities": probabilities,
        "class_names": list(POINT_REPORT_CLASS_NAMES),
        "class_labels": list(POINT_REPORT_CLASS_LABELS),
        "operation_id": operation_id,
        "operation_name": CLASS_NAMES[operation_id] if operation_id is not None else None,
    }


def input_files(source):
    source = Path(source).resolve()
    files = (
        sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not path.name.startswith("~$")
        )
        if source.is_dir()
        else [source]
    )
    if not files:
        raise ValueError("No trajectory CSV/XLSX files found")
    return files, source if source.is_dir() else source.parent


def predict_files(args, task="pattern"):
    device = choose_device(args.device)
    bundle = load_checkpoint(args.checkpoint, device, task)
    files, root = input_files(args.input)
    results = []
    for path in files:
        operation_id = -1
        if task == "point":
            operation_id = next(
                (
                    CLASS_DIRECTORIES[parent.name]
                    for parent in path.parents
                    if parent.name in CLASS_DIRECTORIES
                ),
                -1,
            )
        trajectory = load_trajectory(FileRecord(path, operation_id), bundle.preprocessing)
        results.append(
            {
                "path": relative_path(path, root),
                **predict_trajectory(bundle, trajectory, device),
                "timestamps": trajectory.timestamps.tolist(),
                "segment_ids": trajectory.segment_ids.tolist(),
                "source_rows": trajectory.source_rows.tolist(),
                "preprocessing": trajectory.diagnostics,
            }
        )
    output = new_output(args.output)
    write_json(output / "predictions.json", {"task": task, "predictions": results})
    print(f"Predicted {len(results)} files", flush=True)
    return output
