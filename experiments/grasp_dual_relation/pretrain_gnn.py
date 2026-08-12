from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
from tqdm import trange

from gnn.gnn import MotifGNN, drop_feature


def _load_embedding(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    value = obj["emb"] if isinstance(obj, dict) else obj
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("embedding file must contain a [N,D] tensor or {'emb': tensor}")
    return value


def _relation_arrays(bundle, relation: str) -> tuple[np.ndarray, np.ndarray]:
    return bundle[f"{relation}_indptr"], bundle[f"{relation}_indices"]


def _sample_subgraph(
    indptr: np.ndarray,
    indices: np.ndarray,
    structure_nodes: int,
    *,
    max_nodes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, torch.Tensor]:
    degrees = np.diff(indptr[:structure_nodes + 1])
    eligible = np.flatnonzero(degrees > 0)
    if eligible.size == 0:
        raise RuntimeError("the structure prefix has no causal edges for this relation")
    seed_count = min(max(1, max_nodes // 4), eligible.size)
    centers = rng.choice(eligible, size=seed_count, replace=False)
    selected = set(map(int, centers))
    frontier = list(map(int, centers))
    while frontier and len(selected) < max_nodes:
        current = frontier.pop()
        start, end = int(indptr[current]), int(indptr[current + 1])
        for predecessor in indices[start:end][::-1]:
            node = int(predecessor)
            if node >= structure_nodes or node in selected:
                continue
            selected.add(node)
            frontier.append(node)
            if len(selected) >= max_nodes:
                break
    nodes = np.asarray(sorted(selected), dtype=np.int64)
    mapping = {int(node): offset for offset, node in enumerate(nodes)}
    src, dst = [], []
    for global_dst in nodes:
        start, end = int(indptr[global_dst]), int(indptr[global_dst + 1])
        for global_src in indices[start:end]:
            local_src = mapping.get(int(global_src))
            if local_src is not None:
                src.append(local_src)
                dst.append(mapping[int(global_dst)])
    if not src:
        raise RuntimeError("sampled subgraph has no causal edges; increase --max-nodes")
    return nodes, torch.tensor([src, dst], dtype=torch.long)


def train_relation(
    relation: str,
    bundle,
    embeddings: torch.Tensor,
    *,
    output_dir: Path,
    epochs: int,
    max_nodes: int,
    lr: float,
    seed: int,
    device: torch.device,
    show_progress: bool,
) -> Path:
    assignments = bundle["assignments"]
    structure_nodes = int(np.count_nonzero(assignments == 0))
    if not np.all(assignments[:structure_nodes] == 0):
        raise ValueError("structure split must be the chronological prefix")
    indptr, indices = _relation_arrays(bundle, relation)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    model = MotifGNN(
        in_dim=int(embeddings.shape[1]),
        shared_dim=256,
        hidden_channels=256,
        out_channels=128,
        motif_names=["edge"],
        tau=0.4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    iterator = trange(epochs, desc=f"pretrain-{relation}", disable=not show_progress)
    for _ in iterator:
        nodes, edge_index = _sample_subgraph(
            indptr, indices, structure_nodes, max_nodes=max_nodes, rng=rng
        )
        x = embeddings[nodes].float().to(device)
        edge_index = edge_index.to(device)
        edge1, _ = dropout_edge(edge_index, p=0.4)
        edge2, _ = dropout_edge(edge_index, p=0.2)
        view1 = Data(x=drop_feature(x, 0.4), edge_index=edge1)
        view2 = Data(x=drop_feature(x, 0.3), edge_index=edge2)
        optimizer.zero_grad(set_to_none=True)
        z1 = model(view1, {"edge": edge1})
        z2 = model(view2, {"edge": edge2})
        loss = model.loss(z1, z2, batch_size=512)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{relation} contrastive loss became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        iterator.set_postfix(loss=f"{losses[-1]:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{relation}_structure_learner.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "relation": relation,
            "motif_names": ["edge"],
            "in_dim": int(embeddings.shape[1]),
            "out_dim": 128,
            "epochs": epochs,
            "max_nodes": max_nodes,
            "lr": lr,
            "seed": seed,
            "losses": losses,
            "training_scope": "structure_prefix_only",
            "message_direction": "history_to_current",
        },
        checkpoint,
    )
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain independent causal YelpZip relation GNNs")
    parser.add_argument("--graph-bundle", required=True, type=Path)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--relations", nargs="+", choices=["user", "business"], default=["user", "business"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-nodes", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    bundle = np.load(args.graph_bundle)
    embeddings = _load_embedding(args.embedding)
    if embeddings.shape[0] != bundle["timestamps"].shape[0]:
        raise ValueError("embedding rows do not match temporal graph nodes")
    checkpoints = {}
    for offset, relation in enumerate(args.relations):
        checkpoint = train_relation(
            relation,
            bundle,
            embeddings,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_nodes=args.max_nodes,
            lr=args.lr,
            seed=args.seed + offset,
            device=torch.device(args.device),
            show_progress=not args.no_progress,
        )
        checkpoints[relation] = str(checkpoint)
    print(json.dumps({"status": "COMPLETED", "checkpoints": checkpoints}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
