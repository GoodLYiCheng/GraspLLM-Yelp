from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from gnn import get_matrix
from gnn.seq import balanced_train_indices


def test_stage1_loader_applies_pretrain_mask_and_reuses_feature_cache(tmp_path, monkeypatch):
    data_paths = {}
    embedding_paths = {}
    full_embedding = torch.arange(24, dtype=torch.float16).reshape(6, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    for name in ("left", "right"):
        directory = tmp_path / name
        directory.mkdir()
        data = Data(
            edge_index=edge_index,
            num_nodes=6,
            train_mask=torch.tensor([True, True, True, True, False, False]),
            pretrain_mask=torch.tensor([True, True, True, True, False, False]),
        )
        data.metadata = {
            "embedding_group": "shared",
            "text_hash": "text",
            "mask_hash": "mask",
            "training_scope": "train_induced",
        }
        data_paths[name] = directory / "processed_data.pt"
        embedding_paths[name] = directory / "qwen3_emb_x.pt"
        torch.save(data, data_paths[name])
        torch.save(full_embedding, embedding_paths[name])
    monkeypatch.setattr(get_matrix, "processed_data_path", lambda name: str(data_paths[name]))
    monkeypatch.setattr(get_matrix, "qwen3_emb_path", lambda name: str(embedding_paths[name]))
    cache = {}
    left_x, left_edge, left_manifest = get_matrix.load_stage1_data("left", cache)
    right_x, right_edge, _ = get_matrix.load_stage1_data("right", cache)
    assert left_x.data_ptr() == right_x.data_ptr()
    assert left_x.shape == (4, 4)
    assert int(left_edge.max()) < 4
    assert torch.equal(left_edge, right_edge)
    assert left_manifest["training_scope"] == "train_induced"


def test_balanced_train_indices_are_exact_disjoint_and_deterministic():
    labels = torch.tensor([0] * 20 + [1] * 20)
    mask = torch.ones(40, dtype=torch.bool)
    mask[[0, 20]] = False
    first = balanced_train_indices(labels, mask, total=10, seed=42)
    second = balanced_train_indices(labels, mask, total=10, seed=42)
    assert np.array_equal(first, second)
    assert len(first) == 10
    assert int((labels[first] == 0).sum()) == 5
    assert int((labels[first] == 1).sum()) == 5
    assert 0 not in first and 20 not in first
