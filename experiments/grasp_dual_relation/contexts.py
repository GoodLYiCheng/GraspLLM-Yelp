from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import ReviewEventTable
from .graph import CausalRelationGraph, causal_dfs
from .sampling import ocs_select, random_select, text_topk_select


SelectionMethod = Literal["ocs", "random", "text_topk"]
ContextVariant = Literal["dual", "user_only", "business_only", "merged", "text_only"]


@dataclass(frozen=True)
class DualContext:
    center: int
    user_nodes: list[int]
    business_nodes: list[int]


def _select(
    method: SelectionMethod,
    center: int,
    candidates: list[int],
    text_embeddings: np.ndarray,
    structure_embeddings: np.ndarray,
    graph: CausalRelationGraph,
    *,
    k: int,
    beta: float,
    seed: int,
) -> list[int]:
    if method == "ocs":
        return ocs_select(
            center, candidates, structure_embeddings, graph, k=k, beta=beta
        )
    if method == "random":
        return random_select(center, candidates, k=k, seed=seed)
    if method == "text_topk":
        return text_topk_select(center, candidates, text_embeddings, k=k)
    raise ValueError(f"unknown selection method: {method}")


def build_dual_context(
    center: int,
    events: ReviewEventTable,
    user_graph: CausalRelationGraph,
    business_graph: CausalRelationGraph,
    text_embeddings: np.ndarray,
    user_structure_embeddings: np.ndarray,
    business_structure_embeddings: np.ndarray,
    *,
    method: SelectionMethod = "ocs",
    max_depth: int = 1,
    user_k: int = 8,
    business_k: int = 8,
    beta_user: float = 0.55,
    beta_business: float = 0.55,
    seed: int = 42,
) -> DualContext:
    user_candidates = causal_dfs(center, user_graph, events.timestamps, max_depth=max_depth)
    business_candidates = causal_dfs(
        center, business_graph, events.timestamps, max_depth=max_depth
    )
    return DualContext(
        center=int(center),
        user_nodes=_select(
            method,
            center,
            user_candidates,
            text_embeddings,
            user_structure_embeddings,
            user_graph,
            k=user_k,
            beta=beta_user,
            seed=seed,
        ),
        business_nodes=_select(
            method,
            center,
            business_candidates,
            text_embeddings,
            business_structure_embeddings,
            business_graph,
            k=business_k,
            beta=beta_business,
            seed=seed + 1,
        ),
    )


def training_record(
    events: ReviewEventTable,
    context: DualContext,
    *,
    variant: ContextVariant = "dual",
    merged_nodes: list[int] | None = None,
) -> dict:
    label = "Fraudulent" if int(events.labels[context.center]) == 1 else "Legitimate"
    graph_specs = {
        "dual": [
            {"type": "user", "nodes": context.user_nodes},
            {"type": "business", "nodes": context.business_nodes},
        ],
        "user_only": [{"type": "user", "nodes": context.user_nodes}],
        "business_only": [{"type": "business", "nodes": context.business_nodes}],
        "merged": [{"type": "generic", "nodes": merged_nodes or []}],
        "text_only": [],
    }[variant]
    graph_prompt = {
        "dual": "User-side graph context:\n<user_graph>\n\nBusiness-side graph context:\n<business_graph>\n\n",
        "user_only": "User-side graph context:\n<user_graph>\n\n",
        "business_only": "Business-side graph context:\n<business_graph>\n\n",
        "merged": "Merged graph context:\n<graph>\n\n",
        "text_only": "",
    }[variant]
    return {
        "id": str(events.review_ids[context.center]),
        "node_id": int(context.center),
        "timestamp": int(events.timestamps[context.center]),
        "graphs": graph_specs,
        "variant": variant,
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"Target review:\n{events.texts[context.center]}\n\n"
                    f"{graph_prompt}"
                    "Determine whether the target review is Fraudulent or Legitimate."
                ),
            },
            {"from": "gpt", "value": label},
        ],
    }
