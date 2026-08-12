from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .evaluation import binary_metrics, select_f1_threshold


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels, probabilities = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            labels.append(int(record["ground_truth"]))
            probabilities.append(float(record["fraud_probability"]))
    if not labels:
        raise ValueError(f"no predictions in {path}")
    return np.asarray(labels, dtype=np.int8), np.asarray(probabilities, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select validation threshold and score held-out test predictions")
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    val_y, val_p = _load(args.validation)
    test_y, test_p = _load(args.test)
    threshold = select_f1_threshold(val_y, val_p)
    payload = {
        "threshold_source": "validation",
        "threshold_metric": "fraud_f1",
        "validation": binary_metrics(val_y, val_p, threshold=threshold),
        "test": binary_metrics(test_y, test_p, threshold=threshold),
        "test_fraud_prevalence": float(test_y.mean()),
        "validation_rows": int(val_y.size),
        "test_rows": int(test_y.size),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

