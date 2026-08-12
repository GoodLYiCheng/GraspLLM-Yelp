from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from preprocess.prepare_yelpzip import (
    build_bounded_relation_graph,
    compute_sparse_motifs,
    stratified_masks,
)


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.t().tolist()))


def test_stratified_masks_are_exact_disjoint_and_deterministic():
    labels = np.asarray(([0] * 8 + [1] * 2) * 20, dtype=np.int8)
    first_val, first_test = stratified_masks(labels, val_size=30, test_size=50, seed=42)
    second_val, second_test = stratified_masks(labels, val_size=30, test_size=50, seed=42)
    assert int(first_val.sum()) == 30
    assert int(first_test.sum()) == 50
    assert not torch.any(first_val & first_test)
    assert torch.equal(first_val, second_val)
    assert torch.equal(first_test, second_test)
    assert set(labels[first_val.numpy()]) == {0, 1}


def test_relation_graph_is_symmetric_bounded_and_relation_pure():
    entities = np.asarray(["a"] * 40 + ["b"] * 4 + ["c"], dtype=object)
    edge_index, motifs, stats = build_bounded_relation_graph(
        entities, max_neighbors=32, seed=42
    )
    edges = _edge_set(edge_index)
    assert all(src != dst for src, dst in edges)
    assert all((dst, src) in edges for src, dst in edges)
    assert all(entities[src] == entities[dst] for src, dst in edges)
    assert stats["max_degree"] <= 32
    assert stats["isolated_nodes"] == 1
    assert set(motifs) == {"edge", "triangle", "4-cycle", "4-clique"}


def test_sparse_motif_membership_on_known_graphs():
    triangle = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    tri = compute_sparse_motifs(triangle, 3)
    assert _edge_set(tri["triangle"]) == _edge_set(triangle)
    assert tri["4-cycle"].numel() == 0
    assert tri["4-clique"].numel() == 0

    cycle = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    cyc = compute_sparse_motifs(cycle, 4)
    assert _edge_set(cyc["4-cycle"]) == _edge_set(cycle)
    assert cyc["triangle"].numel() == 0
    assert cyc["4-clique"].numel() == 0

    pairs = [(u, v) for u in range(4) for v in range(4) if u != v]
    clique = torch.tensor(pairs, dtype=torch.long).t()
    k4 = compute_sparse_motifs(clique, 4)
    assert _edge_set(k4["triangle"]) == set(pairs)
    assert _edge_set(k4["4-cycle"]) == set(pairs)
    assert _edge_set(k4["4-clique"]) == set(pairs)


def test_stage0_reuses_embedding_only_for_identical_contract(tmp_path, monkeypatch):
    from preprocess import build_qwen3_embeddings as encoder

    monkeypatch.setattr(encoder, "DATASET_ROOT", str(tmp_path))
    metadata = {"review_id_hash": "ids", "text_hash": "text", "mask_hash": "mask"}
    for name in ("yelpzip_rur", "yelpzip_rbr"):
        directory = tmp_path / name
        directory.mkdir()
        data = Data(num_nodes=2)
        data.metadata = metadata
        torch.save(data, directory / "processed_data.pt")
    source = tmp_path / "yelpzip_rur" / "qwen3_emb_x.pt"
    torch.save({"emb": torch.ones(2, 4)}, source)
    encoder.reuse_yelpzip_embedding()
    target = tmp_path / "yelpzip_rbr" / "qwen3_emb_x.pt"
    assert target.exists()
    assert torch.equal(torch.load(target, weights_only=False)["emb"], torch.ones(2, 4))
