from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import write_json
from .evaluation import annual_report, select_validation_threshold


def _read(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Select threshold on 2020 and score future FiGraph years")
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validation, test = _read(args.validation), _read(args.test)
    if {int(row["year"]) for row in validation} != {2020}:
        raise ValueError("threshold input must contain only 2020 validation rows")
    if set(map(int, [row["year"] for row in test])) - {2021, 2022}:
        raise ValueError("test input may contain only 2021/2022")
    threshold = select_validation_threshold(
        [row["ground_truth"] for row in validation],
        [row["fraud_probability"] for row in validation],
    )
    report = annual_report(
        [row["year"] for row in test],
        [row["ground_truth"] for row in test],
        [row["fraud_probability"] for row in test],
        threshold["threshold"],
    )
    payload = {"status": "COMPLETED", "threshold_selection": threshold, "test": report}
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
