from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine RUR/RBR YelpZip base results")
    parser.add_argument("--rur-probability", required=True, type=Path)
    parser.add_argument("--rbr-probability", required=True, type=Path)
    parser.add_argument("--rur-accuracy", required=True, type=Path)
    parser.add_argument("--rbr-accuracy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    probability = {
        "RUR": _load(args.rur_probability),
        "RBR": _load(args.rbr_probability),
    }
    accuracy = {"RUR": _load(args.rur_accuracy), "RBR": _load(args.rbr_accuracy)}
    if probability["RUR"]["mask_hash"] != probability["RBR"]["mask_hash"]:
        raise ValueError("RUR and RBR mask hashes differ")
    if probability["RUR"]["checkpoint"] != probability["RBR"]["checkpoint"]:
        raise ValueError("RUR and RBR did not use the same projector checkpoint")
    rows = {}
    for relation in ("RUR", "RBR"):
        rows[relation] = {
            "free_text_accuracy": float(accuracy[relation]["accuracy"]),
            **probability[relation]["test"],
            "graph": probability[relation]["graph"],
        }
    payload = {
        "protocol": "static_transductive_zero_shot",
        "projector_source": "arxiv",
        "mask_hash": probability["RUR"]["mask_hash"],
        "checkpoint": probability["RUR"]["checkpoint"],
        "relations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
