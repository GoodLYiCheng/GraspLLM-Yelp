from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_validation_threshold(labels, probabilities) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    best = None
    for threshold in candidates:
        pred = probabilities >= threshold
        row = (
            f1_score(labels, pred, zero_division=0),
            precision_score(labels, pred, zero_division=0),
            recall_score(labels, pred, zero_division=0),
            -float(threshold),
        )
        if best is None or row > best[0]:
            best = (row, float(threshold))
    return {
        "threshold": best[1],
        "selection_split": "2020 validation",
        "objective": "maximize Fraud F1; tie-break precision, recall, then lower threshold",
        "validation_fraud_f1": float(best[0][0]),
    }


def _top_metrics(labels, probabilities, fraction: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = max(1, int(np.ceil(len(labels) * fraction)))
    order = np.lexsort((np.arange(len(labels)), -probabilities))[:count]
    positives = int(labels[order].sum())
    return {
        "count": count,
        "precision": positives / count,
        "recall": positives / max(1, int(labels.sum())),
    }


def binary_metrics(labels, probabilities, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities >= threshold
    return {
        "rows": int(len(labels)),
        "fraud": int(labels.sum()),
        "fraud_prevalence": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "fraud_f1": float(f1_score(labels, prediction, zero_division=0)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "top_1_percent": _top_metrics(labels, probabilities, 0.01),
        "top_5_percent": _top_metrics(labels, probabilities, 0.05),
    }


def annual_report(years, labels, probabilities, threshold: float) -> dict:
    years = np.asarray(years, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    annual = {
        str(year): binary_metrics(labels[years == year], probabilities[years == year], threshold)
        for year in sorted(np.unique(years))
    }
    return {
        "threshold": float(threshold),
        "annual": annual,
        "macro_pr_auc": float(np.mean([value["pr_auc"] for value in annual.values()])),
        "pooled": binary_metrics(labels, probabilities, threshold),
    }


def macro_year_pr_auc(years, labels, probabilities) -> float:
    years = np.asarray(years)
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    return float(np.mean([
        average_precision_score(labels[years == year], probabilities[years == year])
        for year in np.unique(years)
    ]))
