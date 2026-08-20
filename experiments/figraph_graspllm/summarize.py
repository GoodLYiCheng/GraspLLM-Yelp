from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FiGraph metric reports as support-seed mean and std")
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    fields = [
        ("macro_pr_auc", lambda row: row["test"]["macro_pr_auc"]),
        ("pooled_pr_auc", lambda row: row["test"]["pooled"]["pr_auc"]),
        ("pooled_roc_auc", lambda row: row["test"]["pooled"]["roc_auc"]),
        ("pooled_fraud_f1", lambda row: row["test"]["pooled"]["fraud_f1"]),
        ("year_2021_pr_auc", lambda row: row["test"]["annual"]["2021"]["pr_auc"]),
        ("year_2022_pr_auc", lambda row: row["test"]["annual"]["2022"]["pr_auc"]),
    ]
    summary = {}
    for name, getter in fields:
        values = np.asarray([getter(report) for report in reports], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "runs": len(values),
        }
    payload = {"status": "COMPLETED", "summary": summary, "reports": [str(path.resolve()) for path in args.reports]}
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
