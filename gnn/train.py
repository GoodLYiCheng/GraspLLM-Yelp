from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, k_hop_subgraph, subgraph

# allow `python gnn/train.py` to find utils/paths.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .gnn import MotifGNN, drop_feature
    from .get_matrix import (
        load_data, load_stage1_data, convert_edge_index_to_adj_list,
        compute_motifs_torch, compute_motifs_subgraph,
    )
except ImportError:
    from gnn import MotifGNN, drop_feature  # local script import
    from get_matrix import (
        load_data, load_stage1_data, convert_edge_index_to_adj_list,
        compute_motifs_torch, compute_motifs_subgraph,
    )
from utils.paths import CHECKPOINT_ROOT, STAGE1_SOURCE_DATASETS  # noqa: E402


def graph_sampling(data, num_samples, method="n-hop", n_hop=2, walk_length=10):
    device = data.x.device
    num_nodes = data.num_nodes

    if method == "probability":
        probs = torch.rand(num_nodes, device=device)
        sampled_nodes = torch.topk(probs, num_samples).indices

    elif method == "n-hop":
        degree = torch.bincount(data.edge_index[0], minlength=num_nodes)
        eligible = torch.where(degree > 0)[0]
        if eligible.numel() == 0:
            raise RuntimeError("cannot n-hop sample a graph without edges")
        seed_count = min(max(1, num_samples // max(2, n_hop * 4)), eligible.numel())
        starts = eligible[torch.randperm(eligible.numel(), device=device)[:seed_count]]
        sampled_nodes, _, _, _ = k_hop_subgraph(
            starts, n_hop, data.edge_index, relabel_nodes=False, num_nodes=num_nodes
        )
        if sampled_nodes.numel() < min(num_samples, num_nodes):
            selected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            selected[sampled_nodes] = True
            remaining = torch.where(~selected)[0]
            need = min(num_samples - sampled_nodes.numel(), remaining.numel())
            fill = remaining[torch.randperm(remaining.numel(), device=device)[:need]]
            sampled_nodes = torch.cat([sampled_nodes, fill])
        if sampled_nodes.numel() > num_samples:
            order = torch.randperm(sampled_nodes.numel(), device=device)[:num_samples]
            sampled_nodes = sampled_nodes[order]

    elif method == "random-walk":
        sampled_nodes = set()
        for _ in range(num_samples):
            current = torch.randint(0, num_nodes, (1,), device=device).item()
            for _ in range(walk_length):
                neighbors = data.edge_index[1][data.edge_index[0] == current]
                if len(neighbors) == 0:
                    break
                current = neighbors[torch.randint(0, len(neighbors), (1,)).item()].item()
                sampled_nodes.add(current)
        sampled_nodes = torch.tensor(list(sampled_nodes), device=device)
    else:
        raise ValueError(f"Invalid sampling method: {method}")

    sampled_nodes = sampled_nodes[:num_samples].unique(sorted=True)
    new_ei, _ = subgraph(
        sampled_nodes, data.edge_index, relabel_nodes=True, num_nodes=num_nodes
    )
    return Data(x=data.x[sampled_nodes], edge_index=new_ei), sampled_nodes, None


def generate_motif_views(motif_adj, drop_rate_1: float, drop_rate_2: float):
    motif_adj_1, motif_adj_2 = {}, {}
    for motif, edge_index in motif_adj.items():
        ei1, _ = dropout_edge(edge_index, p=drop_rate_1)
        ei2, _ = dropout_edge(edge_index, p=drop_rate_2)
        motif_adj_1[motif] = ei1
        motif_adj_2[motif] = ei2
    return motif_adj_1, motif_adj_2


def train_model_multi_dataset(datasets,
                              num_epochs: int,
                              lr: float,
                              device,
                              num_samples: int,
                              sampling_method: str,
                              n_hop: int,
                              shared_dim: int,
                              hidden_channels: int,
                              out_channels: int,
                              tau: float,
                              steps_per_dataset: int):
    all_motifs = ["edge", "triangle", "4-cycle", "4-clique"]

    # Load all sources once.  All features are 4096-d Qwen3-Embedding-8B vectors.
    dataset_data = {}
    feature_cache = {}
    source_manifest = {}
    in_dim = None
    for name in datasets.keys():
        x, edge_index, manifest = load_stage1_data(name, feature_cache=feature_cache)
        # Keep full embeddings and graph topology on CPU.  Only sampled subgraphs
        # move to the GPU, and relation-pair features share the same cache entry.
        dataset_data[name] = (x, edge_index)
        source_manifest[name] = manifest
        if in_dim is None:
            in_dim = x.size(1)
        else:
            assert in_dim == x.size(1), (
                f"feature dim mismatch: {name}={x.size(1)} vs {in_dim}")
        print(f"  [load] {name:<12s}  N={x.size(0):>7d}  "
              f"E={edge_index.size(1):>8d}  dim={x.size(1)}")
    print(f"[init] shared feature dim = {in_dim}")
    print(f"[init] datasets = {list(datasets.keys())}")

    model = MotifGNN(
        in_dim=in_dim,
        shared_dim=shared_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        motif_names=all_motifs,
        tau=tau,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        update_count = 0
        for name in datasets.keys():
            x, edge_index = dataset_data[name]
            cpu_data = Data(x=x, edge_index=edge_index)
            for _ in range(steps_per_dataset):
                optimizer.zero_grad(set_to_none=True)
                sampled_data, _, _ = graph_sampling(
                    cpu_data, min(num_samples, x.size(0)),
                    method=sampling_method, n_hop=n_hop)
                sampled_data = Data(
                    x=sampled_data.x.float().to(device),
                    edge_index=sampled_data.edge_index.to(device),
                )

                motif_adj = compute_motifs_torch(
                    sampled_data.edge_index, sampled_data.num_nodes)
                for m in all_motifs:
                    if m not in motif_adj:
                        motif_adj[m] = torch.zeros(
                            (2, 0), device=device, dtype=torch.long)
                    else:
                        motif_adj[m] = motif_adj[m].to(device)

                motif_adj_1, motif_adj_2 = generate_motif_views(
                    motif_adj, drop_rate_1=0.4, drop_rate_2=0.2)
                x1 = drop_feature(sampled_data.x, 0.3)
                x2 = drop_feature(sampled_data.x, 0.4)
                data1 = Data(x=x1, edge_index=sampled_data.edge_index)
                data2 = Data(x=x2, edge_index=sampled_data.edge_index)

                z1 = model(data1, motif_adj_1)
                z2 = model(data2, motif_adj_2)
                loss = model.loss(z1, z2, batch_size=0)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite Stage-1 loss for {name}")
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                update_count += 1

        avg = total_loss / max(1, update_count)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch + 1:>4d}/{num_epochs}  "
                  f"avg_loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}  avg_loss={avg:.4f}")

    return model, source_manifest


def process_new_dataset(model, new_dataset_name, device):
    x, edge_index = load_data(new_dataset_name)
    x = x.to(device)
    edge_index = edge_index.to(device)

    num_nodes = x.size(0)
    num_partitions = max(10, num_nodes // 20000)
    motif_adj = compute_motifs_subgraph(edge_index, num_nodes,
                                        num_partitions=num_partitions,
                                        device=device)
    for m in model.motif_names:
        if m not in motif_adj:
            motif_adj[m] = torch.zeros((2, 0), device=device, dtype=torch.long)
        else:
            motif_adj[m] = motif_adj[m].to(device)

    data = Data(x=x, edge_index=edge_index).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model(data, motif_adj)
    return embeddings


def parse_args():
    p = argparse.ArgumentParser(description="Train cross-dataset motif GNN (Stage 1)")
    p.add_argument('--datasets', nargs='+',
                   default=list(STAGE1_SOURCE_DATASETS),
                   help='Stage-1 source datasets (default: Yelp/Amazon review relation graphs).')
    p.add_argument('--steps-per-dataset', type=int, default=1,
                   help='Optimizer updates per dataset per epoch (default: 1).')
    p.add_argument('--samples-per-dataset', type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument('--num-epochs',  type=int, default=300)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--num-samples', type=int, default=2000,
                   help='Number of nodes to subsample per dataset per step.')
    p.add_argument('--sampling-method', type=str, default='n-hop',
                   choices=['n-hop', 'probability', 'random-walk'])
    p.add_argument('--n-hop',          type=int, default=2)
    p.add_argument('--shared-dim',      type=int, default=256)
    p.add_argument('--hidden-channels', type=int, default=256)
    p.add_argument('--out-channels',    type=int, default=128)
    p.add_argument('--tau',  type=float, default=0.4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--model-save-path', type=str,
                   default=os.path.join(CHECKPOINT_ROOT, 'structure_learner_qwen3.pth'))
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.samples_per_dataset is not None:
        print("[deprecated] --samples-per-dataset now aliases --steps-per-dataset")
        args.steps_per_dataset = args.samples_per_dataset
    if args.steps_per_dataset <= 0:
        raise ValueError("--steps-per-dataset must be positive")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu'
    print(f"[init] device={device}")

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    datasets = {name: args.steps_per_dataset for name in args.datasets}
    model, source_manifest = train_model_multi_dataset(
        datasets=datasets,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        num_samples=args.num_samples,
        sampling_method=args.sampling_method,
        n_hop=args.n_hop,
        shared_dim=args.shared_dim,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        tau=args.tau,
        steps_per_dataset=args.steps_per_dataset,
    )

    torch.save({'model_state_dict': model.state_dict(),
                'args': vars(args),
                'source_manifest': source_manifest,
                'training_scope': 'review_graph_pretraining_without_heldout_yelp'},
               args.model_save_path)
    print(f"[save] -> {args.model_save_path}")
