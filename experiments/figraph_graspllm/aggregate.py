from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Average aligned FiGraph probabilities across support seeds")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = []
    for path in args.inputs:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        runs.append({str(row["node_key"]): row for row in rows})
    keys = set(runs[0])
    if any(set(run) != keys for run in runs[1:]):
        raise ValueError("seed outputs are not aligned")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for key in sorted(keys):
            rows = [run[key] for run in runs]
            identity = {(int(row["year"]), int(row["ground_truth"])) for row in rows}
            if len(identity) != 1:
                raise ValueError(f"seed outputs disagree for {key}")
            probabilities = np.asarray([float(row["fraud_probability"]) for row in rows])
            row = dict(rows[0])
            row["fraud_probability"] = float(probabilities.mean())
            row["fraud_probability_std"] = float(probabilities.std(ddof=1)) if len(probabilities) > 1 else 0.0
            row["aggregated_runs"] = len(probabilities)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"status": "COMPLETED", "runs": len(runs), "rows": len(keys), "inputs": [str(path.resolve()) for path in args.inputs], "output": str(args.output.resolve())}
    write_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
