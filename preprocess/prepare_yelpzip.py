from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


RELATIONS = {"yelpzip_rur": "user_id", "yelpzip_rbr": "prod_id"}


def _hash_lines(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _entity_seed(entity: str, seed: int) -> int:
    raw = hashlib.sha256(f"{seed}\0{entity}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "little")


def stratified_masks(labels: np.ndarray, *, val_size: int, test_size: int, seed: int):
    labels = np.asarray(labels, dtype=np.int8)
    if val_size <= 0 or test_size <= 0 or val_size + test_size > labels.size:
        raise ValueError("val_size and test_size must be positive and fit in the dataset")
    rng = np.random.default_rng(seed)
    selected: dict[str, list[int]] = {"val": [], "test": []}
    remaining_by_class = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    if any(nodes.size == 0 for nodes in remaining_by_class.values()):
        raise ValueError("both Legitimate and Fraudulent classes are required")

    for split_name, size in (("val", val_size), ("test", test_size)):
        available = sum(nodes.size for nodes in remaining_by_class.values())
        exact = {label: size * nodes.size / available for label, nodes in remaining_by_class.items()}
        quota = {label: int(np.floor(value)) for label, value in exact.items()}
        remainder = size - sum(quota.values())
        for label in sorted(exact, key=lambda item: (-(exact[item] - quota[item]), item))[:remainder]:
            quota[label] += 1
        for label, nodes in remaining_by_class.items():
            chosen = rng.choice(nodes, size=quota[label], replace=False)
            selected[split_name].extend(map(int, chosen))
            remaining_by_class[label] = np.setdiff1d(nodes, chosen, assume_unique=False)

    masks = {}
    for split_name, nodes in selected.items():
        mask = torch.zeros(labels.size, dtype=torch.bool)
        mask[torch.as_tensor(sorted(nodes), dtype=torch.long)] = True
        masks[split_name] = mask
    return masks["val"], masks["test"]


def _component_edges(nodes: np.ndarray, entity: str, *, max_neighbors: int, seed: int):
    nodes = np.asarray(nodes, dtype=np.int64)
    size = int(nodes.size)
    if size <= 1:
        return np.empty((2, 0), dtype=np.int64), "isolated"
    if size <= max_neighbors + 1:
        src = np.repeat(nodes, size - 1)
        dst = np.concatenate([np.delete(nodes, offset) for offset in range(size)])
        return np.stack([src, dst]), "clique"

    rng = np.random.default_rng(_entity_seed(entity, seed))
    ordered = rng.permutation(nodes)
    half = max_neighbors // 2
    chunks = []
    for offset in range(1, half + 1):
        other = np.roll(ordered, -offset)
        chunks.append(np.stack([ordered, other]))
        chunks.append(np.stack([other, ordered]))
    return np.concatenate(chunks, axis=1), "ring"


def build_bounded_relation_graph(
    entity_ids: np.ndarray, *, max_neighbors: int, seed: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    if max_neighbors < 6 or max_neighbors % 2:
        raise ValueError("max_neighbors must be an even integer >= 6")
    groups: dict[str, list[int]] = defaultdict(list)
    for node, entity in enumerate(entity_ids):
        groups[str(entity)].append(node)

    motif_chunks = {name: [] for name in ("edge", "triangle", "4-cycle", "4-clique")}
    component_modes = defaultdict(int)
    for entity in sorted(groups):
        nodes = np.asarray(groups[entity], dtype=np.int64)
        edges, mode = _component_edges(nodes, entity, max_neighbors=max_neighbors, seed=seed)
        component_modes[mode] += 1
        if edges.shape[1] == 0:
            continue
        motif_chunks["edge"].append(edges)
        size = int(nodes.size)
        # For a clique, all edges participate in the listed motifs once the
        # component is large enough. For the degree-32 circular lattice used
        # for large groups, every edge participates in all three motifs.
        if (mode == "clique" and size >= 3) or mode == "ring":
            motif_chunks["triangle"].append(edges)
        if (mode == "clique" and size >= 4) or mode == "ring":
            motif_chunks["4-cycle"].append(edges)
            motif_chunks["4-clique"].append(edges)

    motif_adj = {}
    for name, chunks in motif_chunks.items():
        array = np.concatenate(chunks, axis=1) if chunks else np.empty((2, 0), dtype=np.int64)
        motif_adj[name] = torch.from_numpy(array).long()
    edge_index = motif_adj["edge"]
    degree = torch.bincount(edge_index[0], minlength=len(entity_ids))
    stats = {
        "nodes": int(len(entity_ids)),
        "entities": int(len(groups)),
        "directed_edges": int(edge_index.shape[1]),
        "isolated_nodes": int((degree == 0).sum()),
        "max_degree": int(degree.max()) if degree.numel() else 0,
        **{f"components_{name}": int(value) for name, value in component_modes.items()},
        **{f"{name}_edges": int(value.shape[1]) for name, value in motif_adj.items()},
    }
    return edge_index, motif_adj, stats


def compute_sparse_motifs(edge_index: torch.Tensor, num_nodes: int) -> dict[str, torch.Tensor]:
    """Exact small-graph motif membership used by unit tests and audits."""
    adjacency = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        if src != dst:
            adjacency[src].add(dst)
    undirected = sorted({(min(u, v), max(u, v)) for u in range(num_nodes) for v in adjacency[u] if u != v})
    hits = {name: [] for name in ("edge", "triangle", "4-cycle", "4-clique")}
    for u, v in undirected:
        hits["edge"].extend(((u, v), (v, u)))
        common = adjacency[u] & adjacency[v]
        if common:
            hits["triangle"].extend(((u, v), (v, u)))
        clique = any(b in adjacency[a] for a in common for b in common if a < b)
        if clique:
            hits["4-clique"].extend(((u, v), (v, u)))
        cycle = any(
            b in adjacency[a]
            for a in adjacency[u] - {v}
            for b in adjacency[v] - {u, a}
        )
        if cycle:
            hits["4-cycle"].extend(((u, v), (v, u)))
    result = {}
    for name, pairs in hits.items():
        result[name] = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs else torch.empty((2, 0), dtype=torch.long)
        )
    return result


def prepare(args: argparse.Namespace) -> None:
    import pandas as pd

    columns = ["Unnamed: 0", "user_id", "prod_id", "label", "text"]
    frame = pd.read_csv(args.raw_path, usecols=columns, nrows=args.max_rows, encoding="utf-8")
    if frame[columns].isna().any().any():
        raise ValueError("YelpZip contains missing required values")
    frame["_review_sort"] = pd.to_numeric(frame["Unnamed: 0"], errors="raise")
    if frame["_review_sort"].duplicated().any():
        raise ValueError("review IDs must be unique")
    frame = frame.sort_values("_review_sort", kind="stable").reset_index(drop=True)
    raw_labels = set(frame["label"].astype(int).unique())
    if not raw_labels.issubset({-1, 1}):
        raise ValueError(f"unexpected YelpZip labels: {sorted(raw_labels)}")

    labels = (frame["label"].to_numpy(dtype=np.int8) == -1).astype(np.int64)
    val_mask, test_mask = stratified_masks(
        labels, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )
    train_mask = ~(val_mask | test_mask)
    review_ids = frame["Unnamed: 0"].astype(str).tolist()
    raw_texts = frame["text"].astype(str).tolist()
    text_hash = _hash_lines(raw_texts)
    review_id_hash = _hash_lines(review_ids)
    mask_hash = _hash_lines(
        np.flatnonzero(val_mask.numpy()).tolist() + ["TEST"] + np.flatnonzero(test_mask.numpy()).tolist()
    )

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in RELATIONS:
        output_dir = args.dataset_root / dataset_name
        processed_path = output_dir / "processed_data.pt"
        if processed_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{processed_path} exists; pass --overwrite to invalidate and rebuild Yelp artifacts"
            )
    if args.overwrite:
        derived_names = (
            "processed_data.pt", "qwen3_emb_x.pt", "ocs_train.jsonl",
            "ocs_val.jsonl", "ocs_test.jsonl", "ocs_manifest.json", "run_manifest.json",
        )
        for dataset_name in RELATIONS:
            output_dir = args.dataset_root / dataset_name
            for name in derived_names:
                path = output_dir / name
                if path.is_file():
                    path.unlink()
    common = {
        "y": torch.from_numpy(labels).long(),
        "train_mask": train_mask,
        "pretrain_mask": train_mask.clone(),
        "val_mask": val_mask,
        "test_mask": test_mask,
        "raw_texts": raw_texts,
        "review_ids": review_ids,
        "label_texts": ["Legitimate", "Fraudulent"],
    }
    manifests = {}
    for dataset_name, entity_column in RELATIONS.items():
        edge_index, motif_adj, graph_stats = build_bounded_relation_graph(
            frame[entity_column].astype(str).to_numpy(),
            max_neighbors=args.max_neighbors,
            seed=args.seed,
        )
        metadata = {
            "dataset": dataset_name,
            "source_domain": "yelp",
            "raw_path": str(args.raw_path.resolve()),
            "relation": "RUR" if entity_column == "user_id" else "RBR",
            "relation_text": "user" if entity_column == "user_id" else "business",
            "entity_column": entity_column,
            "node_feature": "review_text_only",
            "static_transductive": True,
            "training_scope": "train_induced",
            "evaluation_scope": "full_static_transductive_no_labels",
            "uses_time": False,
            "uses_rating": False,
            "uses_tag": False,
            "uses_labels_for_graph": False,
            "seed": args.seed,
            "max_neighbors": args.max_neighbors,
            "review_id_hash": review_id_hash,
            "text_hash": text_hash,
            "mask_hash": mask_hash,
            "val_rows": int(val_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_rows": int(train_mask.sum()),
            "embedding_group": "yelpzip",
            **graph_stats,
        }
        data = Data(edge_index=edge_index, num_nodes=len(frame), **common)
        data.motif_adj = motif_adj
        data.metadata = metadata
        output_dir = args.dataset_root / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_dir / "processed_data.pt")
        (output_dir / "run_manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifests[dataset_name] = metadata
    print(json.dumps({"status": "COMPLETED", "datasets": manifests}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare static YelpZip RUR/RBR TAG baselines")
    parser.add_argument("--raw-path", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="smoke-only row cap")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
