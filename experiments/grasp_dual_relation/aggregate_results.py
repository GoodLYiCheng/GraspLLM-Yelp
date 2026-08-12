from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate held-out metrics over support seeds")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    metric_names = sorted(set.intersection(*(set(item["test"]) for item in payloads)))
    summary = {}
    for metric in metric_names:
        values = np.asarray([float(item["test"][metric]) for item in payloads], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "values": values.tolist(),
        }
    output = {"runs": len(payloads), "inputs": [str(path) for path in args.inputs], "test": summary}
    write_json(args.output, output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
