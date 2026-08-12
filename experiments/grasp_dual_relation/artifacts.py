from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, cls=_Encoder, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, cls=_Encoder), encoding="utf-8")


def build_manifest(*, config: dict[str, Any], node_order_hash: str, artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "material_passport": {
            "origin": "GraspLLM dual-relation temporal v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version": "dual_relation_tag_v1_temporal",
        },
        "config": config,
        "config_hash": stable_hash(config),
        "node_order_hash": node_order_hash,
        "artifacts": artifacts,
    }

