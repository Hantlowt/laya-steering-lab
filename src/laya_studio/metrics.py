from __future__ import annotations

import math
from typing import Any

import numpy as np


def classification_metrics(
    expected: list[str],
    probabilities: list[dict[str, float]],
    labels: list[str],
    ece_bins: int = 10,
) -> dict[str, Any]:
    if not expected:
        raise ValueError("metrics require at least one example")
    label_index = {label: i for i, label in enumerate(labels)}
    if any(label not in label_index for label in expected):
        raise ValueError("expected labels contain a label outside the decision schema")
    matrix = np.asarray(
        [[float(row.get(label, 0.0)) for label in labels] for row in probabilities], dtype=float
    )
    if matrix.shape != (len(expected), len(labels)) or not np.isfinite(matrix).all():
        raise ValueError("probabilities must be a finite row for every example")
    sums = matrix.sum(1)
    if (matrix < 0).any() or (sums <= 0).any():
        raise ValueError("probabilities must be non-negative with positive row sums")
    matrix /= sums[:, None]
    y = np.asarray([label_index[x] for x in expected])
    pred = matrix.argmax(1)
    per_class, recalls = {}, []
    for i, label in enumerate(labels):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int((y == i).sum()),
        }
        recalls.append(recall)
    one_hot = np.eye(len(labels))[y]
    confidence = matrix.max(1)
    correct = pred == y
    ece = 0.0
    for low in np.linspace(0, 1, ece_bins, endpoint=False):
        high = low + 1 / ece_bins
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    clipped = np.clip(matrix[np.arange(len(y)), y], 1e-12, 1.0)
    return {
        "n": len(expected),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "per_class": per_class,
        "brier_score": float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1))),
        "ece": ece,
        "log_loss": float(-np.log(clipped).mean()),
    }


def paired_summary(
    task_scores: dict[str, float],
    baseline_scores: dict[str, float],
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    common = sorted(set(task_scores) & set(baseline_scores))
    if not common:
        raise ValueError("no common tasks for paired comparison")
    values = np.asarray([task_scores[k] for k in common], dtype=float)
    differences = np.asarray([task_scores[k] - baseline_scores[k] for k in common], dtype=float)
    n = len(values)
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    diff_std = float(differences.std(ddof=1)) if n > 1 else 0.0
    radius = 1.96 * diff_std / math.sqrt(n) if n > 1 else 0.0
    result = {
        "tasks": n,
        "mean": float(values.mean()),
        "std": std,
        "paired_difference": float(differences.mean()),
        "difference_ci95": [float(differences.mean() - radius), float(differences.mean() + radius)],
        "wins": int((differences > 1e-12).sum()),
        "ties": int((np.abs(differences) <= 1e-12).sum()),
        "losses": int((differences < -1e-12).sum()),
    }
    if bootstrap_samples:
        rng = np.random.default_rng(seed)
        samples = np.asarray(
            [rng.choice(differences, size=n, replace=True).mean() for _ in range(bootstrap_samples)]
        )
        result["bootstrap_difference_ci95"] = list(map(float, np.quantile(samples, [0.025, 0.975])))
    return result


def paraphrase_consistency(predictions: list[str], pair_ids: list[str | None]) -> float | None:
    grouped: dict[str, list[str]] = {}
    for prediction, pair_id in zip(predictions, pair_ids):
        if pair_id:
            grouped.setdefault(pair_id, []).append(prediction)
    valid = [values for values in grouped.values() if len(values) > 1]
    if not valid:
        return None
    return float(np.mean([len(set(values)) == 1 for values in valid]))


def option_order_instability(predictions_by_order: list[list[str]]) -> float:
    if len(predictions_by_order) < 2:
        return 0.0
    matrix = np.asarray(predictions_by_order, dtype=object)
    if len({len(row) for row in predictions_by_order}) != 1:
        raise ValueError("option-order predictions must have equal length")
    return float(np.mean([len(set(matrix[:, i])) > 1 for i in range(matrix.shape[1])]))
