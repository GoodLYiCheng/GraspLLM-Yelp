from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TemporalExperimentConfig:
    raw_path: Path
    output_dir: Path
    structure_ratio: float = 0.60
    alignment_ratio: float = 0.10
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    max_user_history: int = 16
    max_business_history: int = 32
    max_depth: int = 1
    user_context_size: int = 8
    business_context_size: int = 8
    beta_user: float = 0.55
    beta_business: float = 0.55
    graph_tokens_per_relation: int = 4
    seed: int = 42

    def validate(self) -> None:
        ratios = (
            self.structure_ratio,
            self.alignment_ratio,
            self.validation_ratio,
            self.test_ratio,
        )
        if any(value <= 0 for value in ratios):
            raise ValueError("all temporal split ratios must be positive")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError(f"temporal split ratios must sum to 1, got {sum(ratios)}")
        for name in (
            "max_user_history",
            "max_business_history",
            "max_depth",
            "user_context_size",
            "business_context_size",
            "graph_tokens_per_relation",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("beta_user", "beta_business"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_manifest_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["raw_path"] = str(self.raw_path.resolve())
        result["output_dir"] = str(self.output_dir.resolve())
        return result


def default_config(repo_root: Path) -> TemporalExperimentConfig:
    raw = repo_root.parents[2] / "source_data" / "Yelp-Dataset" / "yelpzip.csv"
    output = repo_root / "artifacts" / "grasp_dual_relation" / "yelpzip"
    return TemporalExperimentConfig(raw_path=raw, output_dir=output)

