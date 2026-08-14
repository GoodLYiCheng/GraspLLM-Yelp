from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from . import PROTOCOL_NAME
from .artifacts import sha256_lines, write_json
from .motifs import MOTIF_NAMES, compute_motifs, offset_channels


YEARS = tuple(range(2014, 2023))
REQUIRED_EVALUATION_YEARS = frozenset((2019, 2020, 2021, 2022))


def validate_snapshot_years(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Validate an explicit, ordered FiGraph snapshot selection."""
    snapshot_years = tuple(values)
    if not snapshot_years:
        raise ValueError("at least one FiGraph snapshot year is required")
    if tuple(sorted(snapshot_years)) != snapshot_years or len(set(snapshot_years)) != len(snapshot_years):
        raise ValueError("snapshot years must be unique and in ascending order")
    unsupported = set(snapshot_years) - set(YEARS)
    if unsupported:
        raise ValueError(f"unsupported FiGraph snapshot years: {sorted(unsupported)}")
    missing = sorted(REQUIRED_EVALUATION_YEARS - set(snapshot_years))
    if missing:
        raise ValueError(f"snapshot selection must retain 2019--2022; missing={missing}")
    return snapshot_years


def _read_mda(year_dir: Path, year: int) -> pd.DataFrame:
    paths = sorted(year_dir.glob(f"MDA_{year}*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"no MDA workbook found for {year}")
    frames = [
        pd.read_excel(
            path,
            usecols=["nodeID", "Year", "Label", "ManaDiscAnal"],
            dtype={"nodeID": str},
        )
        for path in paths
    ]
    return pd.concat(frames, ignore_index=True)


def _read_year(raw_root: Path, year: int):
    year_dir = raw_root / str(year)
    feature_path = year_dir / f"ListedCompanyFeatures772_{year}.csv"
    edge_path = year_dir / f"edges{year}.csv"
    features = pd.read_csv(
        feature_path,
        usecols=["nodeID", "Year", "Label"],
        dtype={"nodeID": str},
    )
    mda = _read_mda(year_dir, year)
    if features["nodeID"].duplicated().any() or mda["nodeID"].duplicated().any():
        raise ValueError(f"{year}: duplicate company nodeID")
    if set(features["nodeID"]) != set(mda["nodeID"]):
        missing = sorted(set(features["nodeID"]) - set(mda["nodeID"]))[:5]
        extra = sorted(set(mda["nodeID"]) - set(features["nodeID"]))[:5]
        raise ValueError(f"{year}: MDA/company mismatch; missing={missing}, extra={extra}")
    indexed = mda.set_index("nodeID").loc[features["nodeID"]].reset_index()
    if not np.array_equal(features["Year"].to_numpy(), indexed["Year"].to_numpy()):
        raise ValueError(f"{year}: Year differs between feature and MDA tables")
    if not np.array_equal(features["Label"].to_numpy(), indexed["Label"].to_numpy()):
        raise ValueError(f"{year}: Label differs between feature and MDA tables")
    if set(features["Label"].astype(int).unique()) - {0, 1}:
        raise ValueError(f"{year}: labels must be 0/1")

    # FiGraph edge CSVs are headerless. Reading with header=0 silently loses edge 1.
    raw_edges = pd.read_csv(
        edge_path,
        header=None,
        names=["src", "dst", "relation"],
        dtype=str,
    )
    company_ids = set(features["nodeID"])
    direct = raw_edges[
        raw_edges["src"].isin(company_ids) & raw_edges["dst"].isin(company_ids)
    ].copy()
    direct = direct[direct["src"] != direct["dst"]]
    direct["u"] = direct[["src", "dst"]].min(axis=1)
    direct["v"] = direct[["src", "dst"]].max(axis=1)
    relation_counts = Counter(direct["relation"].tolist())
    pairs = direct[["u", "v"]].drop_duplicates().sort_values(["u", "v"], kind="stable")

    texts = indexed["ManaDiscAnal"].fillna("").astype(str)
    present = texts.str.strip().ne("").to_numpy(dtype=bool)
    audit = {
        "year": year,
        "company_rows": int(len(features)),
        "fraud": int(features["Label"].astype(int).sum()),
        "fraud_rate": float(features["Label"].astype(int).mean()),
        "mda_present": int(present.sum()),
        "mda_missing": int((~present).sum()),
        "raw_edge_rows": int(len(raw_edges)),
        "direct_company_edge_rows": int(len(direct)),
        "unique_direct_company_edges": int(len(pairs)),
        "direct_relation_rows": dict(sorted(relation_counts.items())),
    }
    return features, texts.tolist(), present, pairs, audit


def prepare(
    raw_root: Path,
    output_dir: Path,
    *,
    motif_mode: str,
    snapshot_years: tuple[int, ...] = YEARS,
) -> dict:
    raw_root = raw_root.resolve()
    snapshot_years = validate_snapshot_years(snapshot_years)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_keys: list[str] = []
    company_ids: list[str] = []
    years: list[int] = []
    labels: list[int] = []
    raw_texts: list[str] = []
    mda_present: list[bool] = []
    edge_chunks: list[torch.Tensor] = []
    motif_chunks = {name: [] for name in MOTIF_NAMES}
    annual_audit = {}
    offset = 0

    for year in snapshot_years:
        features, texts, present, pairs, audit = _read_year(raw_root, year)
        ids = features["nodeID"].astype(str).tolist()
        mapping = {node_id: index for index, node_id in enumerate(ids)}
        local_pairs = np.asarray(
            [(mapping[u], mapping[v]) for u, v in pairs.itertuples(index=False, name=None)],
            dtype=np.int64,
        ).reshape(-1, 2)
        motif_result = compute_motifs(local_pairs, len(ids), mode=motif_mode)
        local_directed = motif_result.channels["edge"]
        edge_chunks.append(local_directed + offset if local_directed.numel() else local_directed)
        for name, value in offset_channels(motif_result.channels, offset).items():
            motif_chunks[name].append(value)

        node_keys.extend(f"{year}:{node_id}" for node_id in ids)
        company_ids.extend(ids)
        years.extend([year] * len(ids))
        labels.extend(features["Label"].astype(int).tolist())
        raw_texts.extend(texts)
        mda_present.extend(present.tolist())
        audit["motifs"] = motif_result.audit
        audit["node_offset"] = offset
        annual_audit[str(year)] = audit
        offset += len(ids)

    edge_index = torch.cat(edge_chunks, dim=1) if edge_chunks else torch.empty((2, 0), dtype=torch.long)
    motif_adj = {
        name: torch.cat(values, dim=1) if values else torch.empty((2, 0), dtype=torch.long)
        for name, values in motif_chunks.items()
    }
    year_tensor = torch.tensor(years, dtype=torch.int16)
    present_tensor = torch.tensor(mda_present, dtype=torch.bool)
    support_mask = (year_tensor == 2019) & present_tensor
    val_mask = (year_tensor == 2020) & present_tensor
    test_mask = ((year_tensor == 2021) | (year_tensor == 2022)) & present_tensor
    structure_mask = year_tensor <= 2018
    train_mask = torch.zeros(len(node_keys), dtype=torch.bool)
    node_order_hash = sha256_lines(node_keys)
    text_hash = sha256_lines(raw_texts)
    if len(set(node_keys)) != len(node_keys):
        raise RuntimeError("(year,nodeID) keys are not unique")
    if bool((support_mask & val_mask).any() or (support_mask & test_mask).any() or (val_mask & test_mask).any()):
        raise RuntimeError("support/validation/test masks overlap")
    if edge_index.numel() and not torch.equal(year_tensor[edge_index[0]], year_tensor[edge_index[1]]):
        raise RuntimeError("cross-year edge detected in annual disjoint union")
    if bool(((support_mask | val_mask | test_mask) & ~present_tensor).any()):
        raise RuntimeError("missing-MDA row entered the main cohort")
    metadata = {
        "protocol": PROTOCOL_NAME,
        "raw_root": str(raw_root),
        "node_definition": "company-year",
        "node_order": "year ascending; feature-table row order",
        "node_order_hash": node_order_hash,
        "text_hash": text_hash,
        "graph": "annual direct-company disjoint union",
        "snapshot_years": list(snapshot_years),
        "snapshot_count": len(snapshot_years),
        "excluded_snapshot_years": [year for year in YEARS if year not in snapshot_years],
        "cross_year_edges": False,
        "edge_types_collapsed": True,
        "background_projection": False,
        "motif_mode": motif_mode,
        "support_year": 2019,
        "validation_year": 2020,
        "test_years": [2021, 2022],
        "main_cohort_requires_mda": True,
        "missing_mda_embedding": "all-zero",
        "labels_used_for_graph": False,
        "labels_used_for_structure_training": False,
        "verified_invariants": {
            "unique_company_year_keys": True,
            "no_cross_year_edges": True,
            "split_masks_disjoint": True,
            "main_cohort_has_mda": True,
            "missing_mda_nodes_retained": True,
        },
        "counts": {
            "nodes": len(node_keys),
            "directed_edges": int(edge_index.shape[1]),
            "support": int(support_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
    }
    data = Data(
        x=torch.zeros((len(node_keys), 1), dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=len(node_keys),
    )
    data.support_mask = support_mask
    data.structure_mask = structure_mask
    data.mda_present = present_tensor
    data.years = year_tensor
    data.node_keys = node_keys
    data.company_ids = company_ids
    data.raw_texts = raw_texts
    data.label_texts = ["Normal", "Fraud"]
    data.motif_adj = motif_adj
    data.metadata = metadata
    torch.save(data, output_dir / "processed_data.pt")
    # Deliberately graph-free artifact for raw-text baselines.  Its consumers
    # never deserialize edge_index, motif tensors, GNNs, or projectors.
    torch.save(
        {
            "node_keys": node_keys,
            "years": year_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "raw_texts": raw_texts,
            "mda_present": present_tensor,
            "support_mask": support_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "metadata": {
                "protocol": PROTOCOL_NAME,
                "graph_free": True,
                "node_order_hash": node_order_hash,
                "text_hash": text_hash,
            },
        },
        output_dir / "text_cohort.pt",
    )
    write_json(output_dir / "data_audit.json", {"annual": annual_audit, "metadata": metadata})
    write_json(
        output_dir / "motif_audit.json",
        {year: value["motifs"] for year, value in annual_audit.items()},
    )
    split_manifest = {
        "protocol": PROTOCOL_NAME,
        "support": {"year": 2019, "rows": int(support_mask.sum())},
        "validation": {"year": 2020, "rows": int(val_mask.sum())},
        "test": {"years": [2021, 2022], "rows": int(test_mask.sum())},
        "excluded_missing_mda": {
            str(year): int((~present_tensor & (year_tensor == year)).sum()) for year in snapshot_years
        },
        "node_order_hash": node_order_hash,
    }
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "run_manifest.json", metadata)
    return {"status": "COMPLETED", "output_dir": str(output_dir.resolve()), **metadata["counts"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare strict-transfer FiGraph direct-company TAG")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--motif-mode",
        choices=("grasp_legacy", "exact_edge_membership"),
        default="grasp_legacy",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(YEARS),
        help="FiGraph snapshot years, ascending; must include 2019 2020 2021 2022",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.raw_root,
                args.output_dir,
                motif_mode=args.motif_mode,
                snapshot_years=tuple(args.years),
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
