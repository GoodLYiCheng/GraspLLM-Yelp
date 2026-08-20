from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import file_sha256, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an explicit frozen-projector source manifest")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-datasets", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirm-no-figraph-training", action="store_true", required=True)
    args = parser.parse_args()
    if any("figraph" in value.lower() for value in args.source_datasets):
        raise ValueError("strict transfer forbids FiGraph among projector source datasets")
    projector = args.model_path / "mm_projector.bin"
    if not projector.is_file():
        raise FileNotFoundError(projector)
    payload = {
        "source_datasets": args.source_datasets,
        "training_uses_figraph": False,
        "mm_projector_sha256": file_sha256(projector),
        "model_path_at_declaration": str(args.model_path.resolve()),
        "declaration": "Operator confirmed that neither supervised nor self-supervised FiGraph data trained this projector.",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
