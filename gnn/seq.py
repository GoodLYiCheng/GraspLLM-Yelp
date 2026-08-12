import numpy as np
import json
import argparse
import os
import random
import sys
import time
from collections import Counter
from collections import defaultdict

import torch
import torch.nn.functional as F


from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ``scripts/stage2_generate_seqs.sh`` executes this file as ``python seq.py``.
# In that form its module name is ``__main__``; without this alias, gen.py's
# ``from seq import ...`` imports a second copy and its large-graph dispatcher
# keeps an independent, disabled state.  Keep both names bound to this exact
# module before importing gen.py.  Normal ``import seq`` behaviour is unchanged.
sys.modules.setdefault("seq", sys.modules[__name__])

from gnn import MotifGNN  
from get_matrix import load_data, compute_motifs_for_subgraph, convert_edge_index_to_adj_list, load_labels, \
    compute_motifs_torch, compute_motifs_sparse, compute_motifs_subgraph
from utils.paths import (dataset_dir, processed_data_path, 
                         CHECKPOINT_ROOT)

def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1, vec2)
    norm_a = torch.norm(vec1)
    norm_b = torch.norm(vec2)
    return dot_product / (norm_a * norm_b + 1e-8) 


def sigmoid(x):
    return 1 / (1 + torch.exp(-x)) 


def cosine_similarity_batch(vec1, vec2):
    vec1_norm = vec1 / vec1.norm(dim=-1, keepdim=True)  
    vec2_norm = vec2 / vec2.norm(dim=-1, keepdim=True) 
    return torch.matmul(vec1_norm, vec2_norm.T)

def greedy_search_no_revisit(
        start_node, embeddings, adjacency_dict, threshold, max_steps, device, center_node, beta=0.55
):
    sequence = []
    visited = set()
    S = set()

    center_vec = embeddings[center_node].unsqueeze(0)  

    current_node = start_node
    steps = 0

    while steps < max_steps:
        sequence.append(current_node)
        visited.add(current_node)
        S.add(current_node)
        steps += 1

        candidate_nodes = set()
        for s in S:
            if s in adjacency_dict:
                candidate_nodes.update(adjacency_dict[s])
        candidate_nodes.difference_update(visited)

        if not candidate_nodes:
            break

        candidate_nodes = list(candidate_nodes)
        candidate_tensor = torch.tensor(candidate_nodes, device=device)
        candidate_vecs = embeddings[candidate_tensor]  # [num_cand, dim]

        # ΔRel
        rel_scores = torch.clamp(
            torch.nn.functional.cosine_similarity(candidate_vecs, center_vec),
            min=0
        )  # [num_cand]

        # ΔCoh
        max_neighbors = 10
        struct_scores = torch.zeros(len(candidate_nodes), device=device)

        for i, node in enumerate(candidate_nodes):
            neighbors = list(adjacency_dict.get(node, set()))

            if len(neighbors) > max_neighbors:
                neighbors = random.sample(neighbors, max_neighbors) 

            if len(neighbors) == 0:
                continue

            neighbor_vecs = embeddings[torch.tensor(neighbors, device=device)]  # [s, dim]
            node_vec = candidate_vecs[i].unsqueeze(0).expand(len(neighbors), -1)  # [s, dim]

            cos_sim = torch.clamp(
                torch.nn.functional.cosine_similarity(node_vec, neighbor_vecs),
                min=0
            )
            struct_scores[i] = torch.sum(cos_sim) * (len(neighbors) / max_neighbors if max_neighbors > 0 else 1)

        # -------- η(v|S) --------
        contextual_scores = beta * struct_scores + (1 - beta) * rel_scores
        shifted_scores = contextual_scores - torch.max(contextual_scores)
        exp_scores = torch.exp(shifted_scores)
        eta_scores = exp_scores / (torch.sum(exp_scores) + 1e-10)

        valid_mask = eta_scores > threshold
        if not valid_mask.any():
            break

        best_index = torch.argmax(eta_scores * valid_mask.float())
        current_node = candidate_nodes[best_index.item()]

    sequence.extend([-500] * (max_steps - len(sequence)))

    return sequence

def build_adjacency_dict(edge_list):
    adjacency_dict = {}

    if isinstance(edge_list, torch.Tensor):
        u_list = edge_list[0].tolist()
        v_list = edge_list[1].tolist()
        edges = zip(u_list, v_list)
    else:
        edges = edge_list

    for u, v in edges:
        if u not in adjacency_dict:
            adjacency_dict[u] = set()
        if v not in adjacency_dict:
            adjacency_dict[v] = set()
        adjacency_dict[u].add(v)
        adjacency_dict[v].add(u)

    return adjacency_dict


def get_neighbors(node, adjacency_dict):
    return list(adjacency_dict.get(node, set()))


def get_neighbors_from_edge_list(node, edge_list, device):
    neighbors = []

    if isinstance(edge_list, torch.Tensor):
        src_mask = edge_list[0] == node
        tgt_mask = edge_list[1] == node

        neighbors_from_src = edge_list[1][src_mask].tolist()
        neighbors_from_tgt = edge_list[0][tgt_mask].tolist()

        neighbors = neighbors_from_src + neighbors_from_tgt
    else:
        for u, v in edge_list:
            if u == node:
                neighbors.append(v)
            elif v == node:
                neighbors.append(u)

    return neighbors

def _generate_final_sequence_default(center_node, embeddings, adjacency_dict, threshold=0.3, max_steps=10, num_neighbors=9,
                           max_elements=111, device="cuda", beta=0.55):
    sequence = [center_node]

    neighbors = get_neighbors(center_node, adjacency_dict)
    neighbors_tensor = torch.tensor(neighbors, device=device)

    if len(neighbors_tensor) > num_neighbors:
        selected_indices = torch.randperm(len(neighbors_tensor))[:num_neighbors]
        neighbors = neighbors_tensor[selected_indices].tolist()
    else:
        neighbors = neighbors_tensor.tolist()

    sequence.extend(neighbors)
    sequence.extend([-500] * (11 - len(sequence)))  
    for neighbor in neighbors:
        neighbor_sequence = greedy_search_no_revisit(neighbor, embeddings, adjacency_dict, threshold, max_steps, device, center_node, beta=0.55)
        sequence.extend(neighbor_sequence)

    sequence.extend([-500] * (max_elements - len(sequence)))

    return sequence


# Large-graph dispatcher: when --large-graph is set, route generate_final_sequence
# through the GPU-CSR + batched-walks kernel in seq_largegraph.py (default unchanged).
_LARGE_GRAPH_STATE = {"enabled": False}


def enable_large_graph_mode(embeddings, edge_index, *, fp16=True,
                            device=None, seed=42, compile_kernel=False):
    """Switch generate_final_sequence to the large-graph implementation (call once)."""
    from seq_largegraph import build_csr_gpu
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    emb = (embeddings.half() if fp16 else embeddings.float()).to(device)
    nb_flat, ptr, deg = build_csr_gpu(edge_index, emb.shape[0], device)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    B = 9
    _LARGE_GRAPH_STATE.update({
        "enabled":        True,
        "device":         device,
        "embeddings":     emb,
        "nb_flat":        nb_flat,
        "ptr":            ptr,
        "deg":            deg,
        "N":              emb.shape[0],
        "gen":            torch.Generator(device=device).manual_seed(seed),
        "buf_visited":    torch.zeros(B, emb.shape[0], dtype=torch.bool, device=device),
        "buf_cand":       torch.zeros(B, emb.shape[0], dtype=torch.bool, device=device),
        "compile_kernel": compile_kernel,
    })
    print(f"[seq] large-graph mode ENABLED  N={emb.shape[0]:,}  "
          f"emb_dtype={emb.dtype}  device={device}  "
          f"compile={compile_kernel}")


def disable_large_graph_mode():
    _LARGE_GRAPH_STATE.clear()
    _LARGE_GRAPH_STATE["enabled"] = False


def generate_final_sequence(center_node, embeddings, adjacency_dict,
                            threshold=0.3, max_steps=10, num_neighbors=9,
                            max_elements=111, device="cuda", beta=0.55):
    """OCS sampling entry point used by gen.py; routes to large-graph kernel if enabled."""
    if _LARGE_GRAPH_STATE.get("enabled"):
        from seq_largegraph import generate_ocs_sequence  # local import
        s = _LARGE_GRAPH_STATE
        return generate_ocs_sequence(
            int(center_node), s["embeddings"], s["nb_flat"], s["ptr"], s["deg"],
            s["N"],
            threshold=threshold, max_steps=max_steps,
            num_neighbors=num_neighbors, max_elements=max_elements,
            beta=beta, device=s["device"], gen=s["gen"],
            buf_visited=s["buf_visited"], buf_cand=s["buf_cand"],
            compile_kernel=s.get("compile_kernel", False),
        )
    return _generate_final_sequence_default(
        center_node, embeddings, adjacency_dict,
        threshold=threshold, max_steps=max_steps,
        num_neighbors=num_neighbors, max_elements=max_elements,
        device=device, beta=beta,
    )



def merge_motif_adjacency(motif_adjs, num_nodes):
    merged_adj = defaultdict(set)

    for motif, adj_matrix in motif_adjs.items():
        for src, tgt in zip(adj_matrix[0], adj_matrix[1]):
            src = src.item()
            tgt = tgt.item()
            merged_adj[src].add(tgt)
            merged_adj[tgt].add(src)

    final_adj = defaultdict(set, {i: merged_adj[i] for i in range(num_nodes)})

    return final_adj


def compute_wasserstein_distance(dist1, dist2):
    import ot

    dist1 = dist1.cpu().numpy()
    dist2 = dist2.cpu().numpy()

    M = ot.dist(dist1, dist2)

    a = np.ones(len(dist1)) / len(dist1)
    b = np.ones(len(dist2)) / len(dist2)

    return ot.emd2(a, b, M)


def find_closest_dataset(new_features, reference_features):
    distances = {}
    for dataset_name, features in reference_features.items():
        dist = compute_wasserstein_distance(new_features, features)
        distances[dataset_name] = dist

    return min(distances.items(), key=lambda x: x[1])[0]


def find_closest_dataset_with_sampling(new_features, reference_features, sample_size, num_samples):
    votes = []

    for _ in range(num_samples):
        sampled_indices = torch.randperm(len(new_features))[:sample_size]
        sampled_new_features = new_features[sampled_indices]

        distances = {}
        for dataset_name, features in reference_features.items():
            sampled_ref_indices = torch.randperm(len(features))[:sample_size]
            sampled_ref_features = features[sampled_ref_indices]
            dist = compute_wasserstein_distance(sampled_new_features, sampled_ref_features)
            distances[dataset_name] = dist

        closest_dataset = min(distances.items(), key=lambda x: x[1])[0]
        votes.append(closest_dataset)

    most_common_dataset = Counter(votes).most_common(1)[0][0]
    return most_common_dataset

config = {
    # Stage-1 sources (see gnn/train.sh for rationale).
    "datasets": ["arxiv", "pubmed", "computer", "history", "reddit"],
    "samples_per_dataset": 60,
    "num_epochs": 1,
    "learning_rate": 0.0001,
    "num_samples": 2000,
    "sampling_method": "n-hop",
    "n_hop": 2,
    "shared_dim": 256,
    "hidden_channels": 256,
    "out_channels": 128,
    "tau": 0.4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": os.path.join(CHECKPOINT_ROOT,
                                    "structure_learner_qwen3.pth"),
}


def process_new_dataset(model, new_dataset_name, device, *,
                        motif_parallel_gpus=None, gnn_chunk_size=8192):
    """Run the trained shared GNN on a new dataset.

    Qwen3-Embedding-8B gives a unified feature space across datasets, so we
    simply apply the trained model directly.
    """
    node_features, edge_index = load_data(new_dataset_name)  # qwen3_emb fp32, edge_index
    use_motif_parallel = motif_parallel_gpus is not None
    if not use_motif_parallel:
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)

    num_nodes = node_features.size(0)
    processed = torch.load(
        processed_data_path(new_dataset_name), map_location="cpu", weights_only=False
    )
    precomputed = getattr(processed, "motif_adj", None)
    if precomputed is not None:
        required = set(model.motif_names)
        if set(precomputed) != required:
            raise ValueError(
                f"{new_dataset_name} precomputed motif channels differ: "
                f"got={sorted(precomputed)}, expected={sorted(required)}"
            )
        motif_adj = {
            name: value.cpu() if use_motif_parallel else value.to(device)
            for name, value in precomputed.items()
        }
        print(f"[{new_dataset_name}] loaded precomputed sparse motif adjacency")
    elif new_dataset_name == "cora":
        motif_adj = compute_motifs_torch(edge_index, num_nodes)
    else:
        num_partitions = max(10, num_nodes // 20000)
        motif_adj = compute_motifs_subgraph(edge_index, num_nodes,
                                            num_partitions=num_partitions,
                                            device=device)
    motif_adj = {
        k: v.cpu() if use_motif_parallel else v.to(device)
        for k, v in motif_adj.items()
    }
    for motif in model.motif_names:
        if motif not in motif_adj:
            motif_adj[motif] = torch.zeros(
                (2, 0), device="cpu" if use_motif_parallel else device, dtype=torch.long
            )

    model.eval()
    with torch.no_grad():
        if use_motif_parallel:
            from multi_gpu_inference import motif_parallel_inference
            print(f"[{new_dataset_name}] motif-parallel GNN inference on logical GPUs "
                  f"{motif_parallel_gpus}; target chunk={gnn_chunk_size}")
            embeddings = motif_parallel_inference(
                model, node_features.cpu(), motif_adj,
                gpus=motif_parallel_gpus, chunk_size=gnn_chunk_size,
            )
        else:
            data = Data(x=node_features, edge_index=edge_index).to(device)
            embeddings = model(data, motif_adj)
    return embeddings, motif_adj


def load_model(cfg):
    """Load the shared MotifGNN checkpoint trained in Stage 1."""
    motif_names = ["edge", "triangle", "4-cycle", "4-clique"]
    model = MotifGNN(
        in_dim=4096,                # Qwen3-Embedding-8B
        shared_dim=cfg["shared_dim"],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        motif_names=motif_names,
        tau=cfg["tau"],
    )
    ckpt = torch.load(cfg["model_save_path"],
                      map_location=cfg["device"], weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg["device"])
    model.eval()
    return model


def merge_motif_adjacency(motif_adjs, num_nodes):
    merged_adj = defaultdict(set)

    for motif, adj_matrix in motif_adjs.items():
        for src, tgt in zip(adj_matrix[0], adj_matrix[1]):
            src = src.item()
            tgt = tgt.item()
            merged_adj[src].add(tgt)
            merged_adj[tgt].add(src)

    final_adj = defaultdict(set, {i: merged_adj[i] for i in range(num_nodes)})

    return final_adj

TRAIN_DATASETS = {"arxiv", "computer", "reddit"}

def generate_data(new_dataset, threshold=0.1, beta=0.55,
                  large_graph=False, fp16=True, compile_kernel=False,
                  motif_parallel_gpus=None, gnn_chunk_size=8192):
    model = load_model(config)
    embeddings, motif_adj = process_new_dataset(
        model, new_dataset, device=config["device"],
        motif_parallel_gpus=motif_parallel_gpus,
        gnn_chunk_size=gnn_chunk_size)

    print(f"New dataset {new_dataset} processed.")
    print(f"Generated embeddings shape: {embeddings.shape}")

    node_embeddings = embeddings

    node_features, edge_index = load_data(new_dataset)

    adj_lists = edge_index
    adj_lists_dropped, _ = dropout_edge(edge_index, p=0.2)

    if large_graph:
        # Build GPU CSR + per-walk buffers once; reused across all centers.
        enable_large_graph_mode(node_embeddings, edge_index, fp16=fp16,
                                compile_kernel=compile_kernel)

    ds_dir = dataset_dir(new_dataset)
    train_path = os.path.join(ds_dir, 'ocs_train.jsonl')
    val_path   = os.path.join(ds_dir, 'ocs_val.jsonl')
    test_path  = os.path.join(ds_dir, 'ocs_test.jsonl')

    tens = torch.load(processed_data_path(new_dataset), weights_only=False)
    labels = tens.y
    metadata = getattr(tens, "metadata", {})
    if new_dataset in ("yelpzip_rur", "yelpzip_rbr"):
        yelp_seed = int(metadata.get("seed", 42))
        random.seed(yelp_seed)
        np.random.seed(yelp_seed)
        torch.manual_seed(yelp_seed)
    train_indices = torch.where(tens.train_mask)[0].numpy()
    val_mask = getattr(tens, "val_mask", None)
    val_indices = (
        torch.where(val_mask)[0].numpy()
        if val_mask is not None else np.empty(0, dtype=np.int64)
    )
    test_indices  = torch.where(tens.test_mask)[0].numpy()

    # Import only after generate_final_sequence is fully defined.  Importing
    # gen.py at module load time would re-enter this file while it is only
    # partially initialised when Stage 2 is launched as ``python seq.py``.
    import gen

    # Dispatch table: dataset name -> *_input function (defined in gen.py).
    DISPATCH = {
        "cora":       gen.cora_input,
        "citeseer":   gen.citeseer_input,
        "pubmed":     gen.pubmed_input,
        "arxiv":      gen.arxiv_input,
        "history":    gen.history_input,
        "computer":   gen.computer_input,
        "photo":      gen.photo_input,
        "wikics":     gen.wikics_input,
        "instagram":  gen.instagram_input,
        "reddit":     gen.reddit_input,
        "cornell":    gen.cornell_input,
        "texas":      gen.texas_input,
        "wisconsin":  gen.wisconsin_input,
        "washington": gen.washington_input,
        "bookchild":  gen.bookchild_input,
        "sportsfit":  gen.sportsfit_input,
        "yelpzip_rur": gen.yelpzip_rur_input,
        "yelpzip_rbr": gen.yelpzip_rbr_input,
    }
    if new_dataset not in DISPATCH:
        raise ValueError(f"Unknown dataset: {new_dataset!r}.  "
                         f"Add a *_input() function in gen.py and a row in the "
                         f"DISPATCH table here.")
    fn = DISPATCH[new_dataset]
    common = (node_embeddings, adj_lists, adj_lists_dropped, labels)

    if new_dataset in TRAIN_DATASETS:
        print(f"  -> writing train ({len(train_indices)} samples) -> {train_path}")
        fn(train_indices, *common, train_path, threshold=threshold, beta=beta)
    else:
        print(f"  -> {new_dataset} is a test-only dataset; skipping train split")

    if len(val_indices):
        print(f"  -> writing val   ({len(val_indices)} samples) -> {val_path}")
        fn(val_indices, *common, val_path, threshold=threshold, beta=beta)

    print(f"  -> writing test  ({len(test_indices)} samples) -> {test_path}")
    fn(test_indices,  *common, test_path,  threshold=threshold, beta=beta)

    if new_dataset in ("yelpzip_rur", "yelpzip_rbr"):
        ocs_manifest = {
            "dataset": new_dataset,
            "relation": metadata.get("relation"),
            "protocol": "static_transductive_zero_shot",
            "gnn_training_uses_yelp": False,
            "labels_used_for_ocs": False,
            "seed": int(metadata.get("seed", 42)),
            "threshold": float(threshold),
            "beta": float(beta),
            "max_sequence_length": 111,
            "validation_rows": int(len(val_indices)),
            "test_rows": int(len(test_indices)),
            "review_id_hash": metadata.get("review_id_hash"),
            "text_hash": metadata.get("text_hash"),
            "mask_hash": metadata.get("mask_hash"),
            "graph": {
                "nodes": metadata.get("nodes"),
                "directed_edges": metadata.get("directed_edges"),
                "max_degree": metadata.get("max_degree"),
                "isolated_nodes": metadata.get("isolated_nodes"),
            },
            "gnn_checkpoint": str(config["model_save_path"]),
            "gnn_inference": {
                "mode": "motif_parallel" if motif_parallel_gpus is not None else "original_single_gpu",
                "logical_gpus": motif_parallel_gpus,
                "target_chunk_size": int(gnn_chunk_size),
            },
        }
        with open(os.path.join(ds_dir, "ocs_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(ocs_manifest, handle, indent=2, ensure_ascii=False)

    print(f"  done: {ds_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate sequences for graph datasets')
    parser.add_argument('--dataset', type=str, default='history', help='Dataset name')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold value')
    parser.add_argument('--beta', type=float, default=0.55, help='Beta value')
    parser.add_argument('--force-train', action='store_true',
                        help='Force generating ocs_train.jsonl even if dataset is not in TRAIN_DATASETS '
                             '(useful for smoke tests on cora etc.)')
    parser.add_argument('--large-graph', action='store_true',
                        help='Enable the large-graph adaptation: GPU-resident CSR adjacency, '
                             'incremental candidate mask, batched 9-walks-per-center, fp16 '
                             'embeddings, and persistent buffers. Recommended when N > ~500k '
                             'or when the default path is too slow. See gnn/seq_largegraph.py.')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false', default=True,
                        help='With --large-graph: keep embeddings in fp32 instead of fp16. '
                             'Slower and uses 2x memory; only needed if you observe '
                             'numerical issues.')
    parser.add_argument('--compile', dest='compile_kernel', action='store_true',
                        help='With --large-graph: wrap the per-step kernel with '
                             '``torch.compile``. Pays a one-time JIT cost (~20-60 s) '
                             'and saves ~20-30%% per step. Disabled by default.')
    parser.add_argument('--shared-emb', action='store_true',
                        help='Multi-GPU only: share one embedding replica across GPU '
                             'workers via CUDA IPC, reducing per-GPU memory ~70%%. '
                             'Use the multi-GPU driver compute_ocs_sequences_multi_gpu '
                             '(see gnn/seq_largegraph.py); ignored in single-process mode.')
    parser.add_argument('--emb-shard', action='store_true',
                        help='Multi-GPU only (experimental): row-shard the embedding '
                             'across workers and use NCCL all-gather. Intended for '
                             'graphs that exceed single-GPU memory (N > ~5M). The '
                             'collective wiring lives in gnn/lg_emb_shard.py; the '
                             'inner loop integration is left for future work.')
    parser.add_argument('--center-batch', type=int, default=1,
                        help='Multi-GPU only: process this many centres per GPU '
                             'forward pass (default 1, i.e. disabled). Values 2-8 '
                             'often add a 1.5-2x speedup on launch-bound graphs at '
                             'the cost of extra GPU memory. Ignored in single-process '
                             'mode (the gen.py-driven CLI here).')
    parser.add_argument('--motif-parallel-gpus', type=str, default=None,
                        help='Opt-in exact multi-GPU GNN inference. Comma-separated logical '
                             'CUDA IDs after CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3. Motif '
                             'branches are assigned across these GPUs and GraphSAGE target '
                             'aggregation is chunked; the original single-GPU path is unchanged.')
    parser.add_argument('--gnn-chunk-size', type=int, default=8192,
                        help='Target-node chunk size for --motif-parallel-gpus (default: 8192). '
                             'Lower this to reduce per-GPU peak memory.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.force_train:
        TRAIN_DATASETS.add(args.dataset)
        print(f"[force-train] {args.dataset} added to TRAIN_DATASETS for this run")
    if (args.shared_emb or args.emb_shard or args.center_batch > 1) and not args.large_graph:
        print("[warn] --shared-emb / --emb-shard / --center-batch require --large-graph")
    if args.shared_emb or args.emb_shard or args.center_batch > 1:
        # Advanced multi-GPU features live in compute_ocs_sequences_multi_gpu().
        print("[note] --shared-emb / --emb-shard / --center-batch are not consumed by "
              "this single-process CLI; call compute_ocs_sequences_multi_gpu() in "
              "gnn/seq_largegraph.py to use them.")
    motif_parallel_gpus = None
    if args.motif_parallel_gpus:
        from multi_gpu_inference import parse_gpu_ids
        motif_parallel_gpus = parse_gpu_ids(args.motif_parallel_gpus)
    generate_data(args.dataset, threshold=args.threshold, beta=args.beta,
                  large_graph=args.large_graph, fp16=args.fp16,
                  compile_kernel=args.compile_kernel,
                  motif_parallel_gpus=motif_parallel_gpus,
                  gnn_chunk_size=args.gnn_chunk_size)

    

