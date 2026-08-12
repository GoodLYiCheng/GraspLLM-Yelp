"""Exact, memory-bounded inference for :class:`MotifGNN`.

The original ``MotifGNN.forward`` is intentionally left untouched.  It places
all four motif graphs and their intermediate GraphSAGE messages on one device,
which is not feasible for YelpZip on 32--40 GB GPUs.  This module is an opt-in
inference-only alternative: motif branches are independent, so they can be
assigned to different GPUs; each branch aggregates destination-node chunks.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F


def parse_gpu_ids(value: str | Sequence[int]) -> list[int]:
    """Parse logical CUDA IDs (after ``CUDA_VISIBLE_DEVICES`` is applied)."""
    if isinstance(value, str):
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        ids = [int(item) for item in value]
    if not ids:
        raise ValueError("--motif-parallel-gpus must contain at least one GPU ID")
    if len(set(ids)) != len(ids):
        raise ValueError("--motif-parallel-gpus contains duplicate GPU IDs")
    available = torch.cuda.device_count()
    if not available:
        raise RuntimeError("motif-parallel inference requires CUDA")
    invalid = [item for item in ids if item < 0 or item >= available]
    if invalid:
        raise ValueError(
            f"invalid logical GPU IDs {invalid}; CUDA_VISIBLE_DEVICES exposes "
            f"0..{available - 1}"
        )
    return ids


def _sorted_by_destination(edge_index: torch.Tensor) -> torch.Tensor:
    """Return a CPU edge index sorted by target node for chunk slicing."""
    edge_index = edge_index.detach().cpu().long().contiguous()
    if edge_index.numel() == 0:
        return edge_index
    order = torch.argsort(edge_index[1], stable=True)
    return edge_index.index_select(1, order).contiguous()


def _sage_mean_chunked(conv, x_cpu: torch.Tensor, edge_index_cpu: torch.Tensor,
                       *, device: torch.device, chunk_size: int) -> torch.Tensor:
    """Run one standard ``SAGEConv`` exactly, without materialising E x D data.

    ``SAGEConv`` used by ``MotifGNN`` is the default mean aggregator.  We retain
    its linear weights/root transform and evaluate only a contiguous target-node
    range per step.  This bounds temporary message storage by
    ``max_degree * chunk_size * feature_dim`` rather than the full graph.
    """
    if getattr(conv, "project", False):
        raise NotImplementedError("chunked inference currently supports SAGEConv(project=False)")
    if getattr(conv, "aggr", "mean") != "mean":
        raise NotImplementedError("chunked inference currently supports SAGEConv(aggr='mean')")
    if chunk_size <= 0:
        raise ValueError("gnn_chunk_size must be positive")

    num_nodes = int(x_cpu.size(0))
    out_dim = int(conv.out_channels)
    x = x_cpu.to(device, non_blocking=True)
    edge = edge_index_cpu.to(device, non_blocking=True)
    src_all, dst_all = edge[0], edge[1]
    result = torch.empty((num_nodes, out_dim), dtype=x_cpu.dtype, device="cpu")

    for begin in range(0, num_nodes, chunk_size):
        end = min(begin + chunk_size, num_nodes)
        # dst_all is sorted, so this does not allocate a full-edge boolean mask.
        left = int(torch.searchsorted(dst_all, begin).item())
        right = int(torch.searchsorted(dst_all, end).item())
        width = end - begin
        aggregate = torch.zeros((width, x.size(1)), dtype=x.dtype, device=device)
        if right > left:
            src = src_all[left:right]
            local_dst = dst_all[left:right] - begin
            aggregate.index_add_(0, local_dst, x.index_select(0, src))
            degree = torch.bincount(local_dst, minlength=width).to(dtype=x.dtype)
            aggregate.div_(degree.clamp_min_(1).unsqueeze(1))

        out = F.linear(aggregate, conv.lin_l.weight, conv.lin_l.bias)
        if conv.root_weight:
            out = out + F.linear(x[begin:end], conv.lin_r.weight, conv.lin_r.bias)
        if conv.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        result[begin:end].copy_(out.cpu())

    del edge, x
    return result


def _load_mmap(path: str) -> torch.Tensor:
    """Use the page cache for worker inputs; keep a compatibility fallback."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # older PyTorch without mmap support
        return torch.load(path, map_location="cpu", weights_only=False)


def _motif_worker(gpu_id: int, motif_names: list[str], model, h0_path: str,
                  edge_paths: Dict[str, str], output_dir: str, chunk_size: int,
                  queue) -> None:
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        h0 = _load_mmap(h0_path)
        for motif in motif_names:
            conv1 = model.conv1s[model.motif_names.index(motif)].to(device).eval()
            conv2 = model.conv2s[model.motif_names.index(motif)].to(device).eval()
            batch_norm = model.batch_norm[model.motif_names.index(motif)].to(device).eval()
            edge = _load_mmap(edge_paths[motif])
            h1 = _sage_mean_chunked(conv1, h0, edge, device=device, chunk_size=chunk_size)
            # BatchNorm is in eval mode, therefore it is valid to apply per chunk.
            h1 = F.relu(batch_norm(h1.to(device)).cpu())
            h2 = _sage_mean_chunked(conv2, h1, edge, device=device, chunk_size=chunk_size)
            torch.save(h2, os.path.join(output_dir, f"{motif}.pt"))
            del edge, h1, h2, conv1, conv2, batch_norm
            torch.cuda.empty_cache()
        queue.put((gpu_id, None))
    except Exception:
        queue.put((gpu_id, traceback.format_exc()))


def motif_parallel_inference(model, node_features: torch.Tensor,
                             motif_adj: Dict[str, torch.Tensor], *,
                             gpus: Iterable[int], chunk_size: int = 8192) -> torch.Tensor:
    """Return the original MotifGNN output using motif-parallel CUDA workers.

    The output is CPU-resident so the caller can decide how to use it next.
    The method does not change weights, random state, labels, or graph edges.
    """
    gpu_ids = parse_gpu_ids(list(gpus))
    if set(motif_adj) != set(model.motif_names):
        raise ValueError("motif adjacency keys must exactly match model.motif_names")
    if node_features.device.type != "cpu":
        node_features = node_features.cpu()
    node_features = node_features.float().contiguous()
    model = model.cpu().eval()

    # The 4096-D input is the only large dense tensor.  Project it in chunks on
    # one GPU before branch parallelism; this prevents a 9+ GB feature replica
    # from appearing on every worker.
    projection_device = torch.device(f"cuda:{gpu_ids[0]}")
    projector = model.shared_proj.to(projection_device).eval()
    shared_dim = int(model.conv1s[0].in_channels)
    out_dim = int(model.conv2s[0].out_channels)
    h0 = torch.empty((node_features.size(0), shared_dim), dtype=torch.float32)
    with torch.no_grad():
        for begin in range(0, node_features.size(0), chunk_size):
            end = min(begin + chunk_size, node_features.size(0))
            h0[begin:end].copy_(projector(node_features[begin:end].to(projection_device)).cpu())
    projector.cpu()
    torch.cuda.empty_cache()

    temp_dir = Path(tempfile.mkdtemp(prefix="graspllm_motif_parallel_"))
    try:
        h0_path = temp_dir / "shared_projection.pt"
        torch.save(h0, h0_path)
        edge_paths: Dict[str, str] = {}
        for motif in model.motif_names:
            path = temp_dir / f"{motif}.edges.pt"
            torch.save(_sorted_by_destination(motif_adj[motif]), path)
            edge_paths[motif] = str(path)

        assignments = {gpu: [] for gpu in gpu_ids}
        for index, motif in enumerate(model.motif_names):
            assignments[gpu_ids[index % len(gpu_ids)]].append(motif)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        workers = []
        for gpu, motifs in assignments.items():
            if not motifs:
                continue
            process = ctx.Process(
                target=_motif_worker,
                args=(gpu, motifs, model, str(h0_path), edge_paths, str(temp_dir), chunk_size, queue),
            )
            process.start()
            workers.append(process)

        errors = []
        for _ in workers:
            _, error = queue.get()
            if error:
                errors.append(error)
        for process in workers:
            process.join()
            if process.exitcode != 0:
                errors.append(f"motif worker exited with code {process.exitcode}")
        if errors:
            raise RuntimeError("motif-parallel inference failed:\n" + "\n".join(errors))

        output = torch.zeros((node_features.size(0), out_dim), dtype=torch.float32)
        for index, motif in enumerate(model.motif_names):
            branch = torch.load(temp_dir / f"{motif}.pt", map_location="cpu", weights_only=False)
            output.add_(branch * float(model.motif_weights[index].detach().cpu()))
        return output
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
