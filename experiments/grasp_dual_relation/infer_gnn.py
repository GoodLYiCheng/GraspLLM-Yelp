from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from gnn.gnn import MotifGNN


def _load_embedding(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return (obj["emb"] if isinstance(obj, dict) else obj).float()


def infer_relation(
    relation: str,
    bundle,
    embeddings: torch.Tensor,
    checkpoint_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("training_scope") != "structure_prefix_only":
        raise ValueError("checkpoint does not declare structure-prefix-only training")
    model = MotifGNN(
        in_dim=int(checkpoint["in_dim"]),
        shared_dim=256,
        hidden_channels=256,
        out_channels=int(checkpoint["out_dim"]),
        motif_names=["edge"],
        tau=0.4,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    indptr = bundle[f"{relation}_indptr"]
    indices = bundle[f"{relation}_indices"].astype(np.int64, copy=False)
    dst = np.repeat(np.arange(embeddings.shape[0], dtype=np.int64), np.diff(indptr))
    edge_index = torch.from_numpy(np.stack([indices, dst], axis=0))
    data = Data(x=embeddings, edge_index=edge_index)
    loader = NeighborLoader(
        data,
        input_nodes=torch.arange(embeddings.shape[0]),
        num_neighbors=[-1, -1],
        batch_size=batch_size,
        shuffle=False,
        directed=True,
    )
    output = torch.empty((embeddings.shape[0], int(checkpoint["out_dim"])), dtype=torch.float32)
    iterator = tqdm(loader, desc=f"infer-{relation}", disable=not show_progress)
    cursor = 0
    with torch.inference_mode():
        for batch in iterator:
            batch = batch.to(device)
            values = model(batch, {"edge": batch.edge_index})[: batch.batch_size]
            end = cursor + batch.batch_size
            output[cursor:end] = values.float().cpu()
            cursor = end
    if cursor != embeddings.shape[0]:
        raise RuntimeError(f"inference produced {cursor} rows, expected {embeddings.shape[0]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "emb": output,
            "relation": relation,
            "message_direction": "history_to_current",
            "checkpoint": str(checkpoint_path),
        },
        output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Causal inductive inference for frozen relation GNNs")
    parser.add_argument("--graph-bundle", required=True, type=Path)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--relations", nargs="+", choices=["user", "business"], default=["user", "business"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    bundle = np.load(args.graph_bundle)
    embeddings = _load_embedding(args.embedding)
    outputs = {}
    for relation in args.relations:
        output_path = args.output_dir / f"{relation}_structure_emb.pt"
        infer_relation(
            relation,
            bundle,
            embeddings,
            args.checkpoint_dir / f"{relation}_structure_learner.pth",
            output_path,
            batch_size=args.batch_size,
            device=torch.device(args.device),
            show_progress=not args.no_progress,
        )
        outputs[relation] = str(output_path)
    print(json.dumps({"status": "COMPLETED", "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

