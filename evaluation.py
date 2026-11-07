"""Classification metrics and operation-labeled point evaluation reports."""

from __future__ import annotations

import numpy as np

from splits import CLASS_NAMES

POINT_CLASS_NAMES = ("turn", "straight")
POINT_REPORT_CLASS_NAMES = POINT_CLASS_NAMES
POINT_REPORT_CLASS_LABELS = tuple(POINT_CLASS_NAMES.index(name) for name in POINT_REPORT_CLASS_NAMES)


def point_report_confusion(confusion, class_names=POINT_CLASS_NAMES):
    """Display turn before straight, matching model and annotation label IDs."""
    order = [list(class_names).index(name) for name in POINT_REPORT_CLASS_NAMES]
    return np.asarray(confusion)[np.ix_(order, order)]


def classification_metrics(truth, predictions, class_names=CLASS_NAMES):
    if not truth or len(truth) != len(predictions):
        raise ValueError("Metrics require nonempty, equally sized labels and predictions")
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for actual, predicted in zip(truth, predictions):
        if not (0 <= actual < len(class_names) and 0 <= predicted < len(class_names)):
            raise ValueError("Class ID outside the configured class order")
        confusion[actual, predicted] += 1
    return metrics_from_confusion(confusion, class_names)


def metrics_from_confusion(confusion, class_names=POINT_REPORT_CLASS_NAMES):
    """Interpret the input order explicitly and display point classes turn-first."""
    confusion = np.asarray(confusion)
    if confusion.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix dimensions must match the supplied class order")
    is_point = len(class_names) == 2 and set(class_names) == set(POINT_CLASS_NAMES)
    if is_point:
        confusion = point_report_confusion(confusion, class_names)
        class_names = POINT_REPORT_CLASS_NAMES
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    precision = np.divide(np.diag(confusion), predicted, out=np.zeros(len(support)), where=predicted > 0)
    recall = np.divide(np.diag(confusion), support, out=np.zeros(len(support)), where=support > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(len(support)), where=precision + recall > 0
    )
    present = support > 0
    result = {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "f1_macro_present_classes": float(f1[present].mean()),
        "class_names": list(class_names),
        "missing_classes": [name for i, name in enumerate(class_names) if not present[i]],
        "confusion_matrix": confusion.tolist(),
        "sample_count": int(confusion.sum()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }
    if is_point:
        positive = list(class_names).index("turn")
        negative = 1 - positive
        result.update(
            positive_class="turn",
            class_labels=list(POINT_REPORT_CLASS_LABELS),
            positive_label=POINT_CLASS_NAMES.index("turn"),
            positive_class_index=positive,
            precision=float(precision[positive]),
            recall=float(recall[positive]),
            f1=float(f1[positive]),
            binary_confusion={
                "tp": int(confusion[positive, positive]),
                "fp": int(confusion[negative, positive]),
                "fn": int(confusion[positive, negative]),
                "tn": int(confusion[negative, negative]),
            },
        )
    return result


class PointEvaluationAccumulator:
    """Merge segments by source file while retaining the annotated operation.

    Confusion matrices contain raw point counts. Overall metrics retain the
    existing pooled-point definition. A trajectory is counted once regardless
    of how many moving segments it contains or how batches are ordered.
    """

    def __init__(self):
        self.records = {}

    def update(self, metadata, actual, predicted, loss_sum):
        operation_id = metadata["operation_id"]
        if operation_id not in (0, 1):
            raise ValueError("Point evaluation requires a shuttle or skip_row operation label")
        operation_name = CLASS_NAMES[operation_id]
        if metadata.get("operation_name", operation_name) != operation_name:
            raise ValueError("Operation name does not match operation_id")
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
            raise ValueError("Each evaluated segment requires aligned, nonempty labels")
        if not np.isin(actual, (0, 1)).all() or not np.isin(predicted, (0, 1)).all():
            raise ValueError("Point IDs must be turn=0 or straight=1")
        key = (operation_id, metadata["path"])
        if key not in self.records:
            self.records[key] = {
                "path": metadata["path"],
                "operation_id": operation_id,
                "operation_name": operation_name,
                "segments": set(),
                "confusion": np.zeros((2, 2), dtype=np.int64),
                "loss_sum": 0.0,
            }
        record = self.records[key]
        np.add.at(record["confusion"], (actual.astype(int), predicted.astype(int)), 1)
        record["segments"].add(metadata["segment_id"])
        record["loss_sum"] += float(loss_sum)

    @staticmethod
    def _metrics(confusion, loss_sum):
        result = metrics_from_confusion(confusion, POINT_CLASS_NAMES)
        result["loss"] = float(loss_sum / confusion.sum())
        return result

    def report(self, overall_metrics):
        if not self.records:
            raise ValueError("No operation-labeled trajectories were evaluated")
        per_trajectory, by_operation = [], {}
        for _, record in sorted(self.records.items()):
            per_trajectory.append(
                {
                    "path": record["path"],
                    "operation_id": record["operation_id"],
                    "operation_name": record["operation_name"],
                    "segment_count": len(record["segments"]),
                    "point_count": int(record["confusion"].sum()),
                    "confusion_matrix": point_report_confusion(record["confusion"]).tolist(),
                    "loss": float(record["loss_sum"] / record["confusion"].sum()),
                }
            )
        pooled = np.zeros((2, 2), dtype=np.int64)
        for operation_id, operation_name in enumerate(CLASS_NAMES[:2]):
            records = [record for record in self.records.values() if record["operation_id"] == operation_id]
            confusion = sum((record["confusion"] for record in records), np.zeros((2, 2), dtype=np.int64))
            pooled += confusion
            by_operation[operation_name] = {
                "operation_id": operation_id,
                "operation_name": operation_name,
                "trajectory_count": len(records),
                "point_count": int(confusion.sum()),
                "metrics": self._metrics(confusion, sum(record["loss_sum"] for record in records))
                if records
                else None,
            }
        if not np.array_equal(point_report_confusion(pooled), overall_metrics["confusion_matrix"]):
            raise ValueError("Operation matrices do not match the overall evaluated point counts")
        return {
            "metrics": overall_metrics,
            "point_class_names": list(POINT_REPORT_CLASS_NAMES),
            "point_class_labels": list(POINT_REPORT_CLASS_LABELS),
            "operation_summary": {
                "trajectory_count": len(per_trajectory),
                "point_count": int(pooled.sum()),
                "trajectory_counts": {
                    name: value["trajectory_count"] for name, value in by_operation.items()
                },
                "point_counts": {name: value["point_count"] for name, value in by_operation.items()},
            },
            "by_operation": by_operation,
            "per_trajectory": per_trajectory,
        }
