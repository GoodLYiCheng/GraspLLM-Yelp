from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from gnn.gnn import MotifGNN
from utils.constants import DEFAULT_GRAPH_PAD_ID

from .artifacts import file_sha256, write_json
from .provenance import validate_gnn_checkpoint


def _load_embedding(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return (value["emb"] if isinstance(value, dict) else value).float()


def infer_structure_embeddings(data, text_embeddings: torch.Tensor, checkpoint_path: Path, device: str) -> tuple[torch.Tensor, dict]:
    provenance = validate_gnn_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = provenance["args"]
    model = MotifGNN(
        in_dim=int(text_embeddings.shape[1]),
        shared_dim=int(cfg.get("shared_dim", 256)),
        hidden_channels=int(cfg.get("hidden_channels", 256)),
        out_channels=int(cfg.get("out_channels", 128)),
        motif_names=["edge", "triangle", "4-cycle", "4-clique"],
        tau=float(cfg.get("tau", 0.4)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    target = torch.device(device)
    model = model.to(target).eval()
    graph = Data(x=text_embeddings.to(target), edge_index=data.edge_index.to(target))
    motif_adj = {name: edge_index.to(target) for name, edge_index in data.motif_adj.items()}
    with torch.inference_mode():
        output = model(graph, motif_adj).float().cpu()
    return output, provenance


def _adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    result = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        if src != dst:
            result[int(src)].add(int(dst))
    return [sorted(values) for values in result]


def _unit(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=-1, eps=1e-12)


def select_context(
    center: int,
    candidates: list[int],
    text_embeddings: torch.Tensor,
    structure_embeddings: torch.Tensor,
    *,
    method: str,
    seed: int,
    max_nodes: int = 32,
    structure_weight: float = 0.55,
) -> list[int]:
    if max_nodes < 1:
        raise ValueError("max_nodes must include the center")
    candidates = list(dict.fromkeys(node for node in candidates if int(node) != int(center)))
    limit = max_nodes - 1
    if method == "random":
        digest = hashlib.sha256(f"{seed}:{center}".encode("ascii")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "little"))
        rng.shuffle(candidates)
        selected = candidates[:limit]
    elif method == "ocs":
        if not 0 <= structure_weight <= 1:
            raise ValueError("structure_weight must be in [0, 1]")
        if candidates:
            indices = torch.tensor(candidates, dtype=torch.long)
            text = _unit(text_embeddings)
            structure = _unit(structure_embeddings)
            semantic = text[indices] @ text[center]
            coherence = structure[indices] @ structure[center]
            scores = structure_weight * coherence + (1 - structure_weight) * semantic
            order = np.lexsort((np.asarray(candidates), -scores.cpu().numpy()))
            selected = [candidates[index] for index in order[:limit]]
        else:
            selected = []
    else:
        raise ValueError(f"unknown context method: {method}")
    values = [int(center), *map(int, selected)]
    return values + [DEFAULT_GRAPH_PAD_ID] * (max_nodes - len(values))


def build_context_records(data, text_embeddings, structure_embeddings, *, nodes, method: str, seed: int, max_nodes: int = 32):
    adjacency = _adjacency(data.edge_index, data.num_nodes)
    records = []
    for center in map(int, nodes):
        values = select_context(
            center,
            adjacency[center],
            text_embeddings,
            structure_embeddings,
            method=method,
            seed=seed,
            max_nodes=max_nodes,
        )
        years = {int(data.years[node]) for node in values if node != DEFAULT_GRAPH_PAD_ID}
        if years != {int(data.years[center])}:
            raise RuntimeError(f"cross-year context for {data.node_keys[center]}: {years}")
        records.append({
            "node_index": center,
            "node_key": data.node_keys[center],
            "year": int(data.years[center]),
            "ground_truth": int(data.y[center]),
            "method": method,
            "nodes": values,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer frozen MotifGNN embeddings and FiGraph contexts")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--text-embedding", required=True, type=Path)
    parser.add_argument("--gnn-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nodes", type=int, default=32)
    args = parser.parse_args()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    text_embeddings = _load_embedding(args.text_embedding)
    if len(text_embeddings) != data.num_nodes:
        raise ValueError("text embedding node count differs from processed graph")
    structure, provenance = infer_structure_embeddings(data, text_embeddings, args.gnn_checkpoint, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    structure_path = args.output_dir / "structure_embedding.pt"
    torch.save({"emb": structure, "metadata": {"gnn": provenance}}, structure_path)
    nodes = torch.where(data.support_mask | data.val_mask | data.test_mask)[0].tolist()
    outputs = {}
    for method in ("ocs", "random"):
        records = build_context_records(
            data, text_embeddings, structure, nodes=nodes, method=method,
            seed=args.seed, max_nodes=args.max_nodes,
        )
        path = args.output_dir / f"contexts_{method}_seed{args.seed}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[method] = str(path.resolve())
    manifest = {
        "status": "COMPLETED",
        "max_nodes": args.max_nodes,
        "seed": args.seed,
        "selection": "one-hop direct-company; 0.55 structure cosine + 0.45 semantic cosine",
        "gnn": provenance,
        "text_embedding_sha256": file_sha256(args.text_embedding),
        "structure_embedding": str(structure_path.resolve()),
        "contexts": outputs,
    }
    write_json(args.output_dir / "context_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
