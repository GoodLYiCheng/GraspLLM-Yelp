from __future__ import annotations

import json
from pathlib import Path

from .artifacts import file_sha256


def _contains_figraph(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_figraph(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_figraph(item) for item in value)
    return "figraph" in str(value).lower()


def validate_gnn_checkpoint(path: str | Path) -> dict:
    import torch

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = dict(checkpoint.get("args") or {})
    datasets = list(args.get("datasets") or [])
    if not datasets:
        raise ValueError("MotifGNN checkpoint lacks args.datasets provenance")
    if _contains_figraph(datasets):
        raise ValueError("MotifGNN checkpoint provenance contains FiGraph")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "source_datasets": datasets, "args": args}


def validate_projector_provenance(model_path: str | Path, provenance_path: str | Path) -> dict:
    model_path, provenance_path = Path(model_path), Path(provenance_path)
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    sources = list(record.get("source_datasets") or [])
    if not sources:
        raise ValueError("projector provenance must list source_datasets")
    if record.get("training_uses_figraph") is not False or _contains_figraph(record):
        raise ValueError("projector provenance does not establish zero FiGraph training")
    projector = model_path / "mm_projector.bin"
    if not projector.is_file():
        raise FileNotFoundError(projector)
    expected = record.get("mm_projector_sha256")
    actual = file_sha256(projector)
    if expected != actual:
        raise ValueError(f"projector hash mismatch: provenance={expected!r}, actual={actual}")
    return {
        "provenance_path": str(provenance_path.resolve()),
        "model_path": str(model_path.resolve()),
        "mm_projector_sha256": actual,
        "source_datasets": sources,
        "training_uses_figraph": False,
    }
