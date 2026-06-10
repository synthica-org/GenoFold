"""
genofold.represent.graph
------------------------
Convert preprocessed Hi-C and NGS data into a PyTorch Geometric graph:

  Nodes  — genomic bins (one per valid locus)
  Edges  — Hi-C contacts above a contact threshold
  Node features — NGS coverage + sequence embeddings + metadata
  Edge features — normalized contact weight + log genomic distance

Output: torch_geometric.data.Data object.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_edge_index(
    contact_matrix: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 0.0,
    top_k: Optional[int] = None,
    self_loops: bool = False,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """
    Build a sparse edge index from a contact matrix.

    Only bins in `valid_mask` become nodes. The edge index uses
    re-mapped node IDs (0..M-1 where M = valid_mask.sum()).

    Parameters
    ----------
    contact_matrix:
        Preprocessed Hi-C matrix (N, N).
    valid_mask:
        Boolean array (N,) indicating valid bins.
    threshold:
        Minimum contact weight for an edge to be included.
    top_k:
        If set, retain only the top-k contacts per node (further sparsification).
    self_loops:
        Whether to include self-loop edges.

    Returns
    -------
    (edge_index, edge_attr)
        edge_index : (2, E) LongTensor — source/target node indices.
        edge_attr  : (E, 2) FloatTensor — [contact_weight, log_genomic_distance].
    """
    import torch
    # Remap original bin indices to valid-only node indices
    n = len(valid_mask)
    remap = np.full(n, -1, dtype=np.int64)
    valid_indices = np.where(valid_mask)[0]
    remap[valid_indices] = np.arange(len(valid_indices))

    mat = contact_matrix.copy()

    # Apply top-k sparsification per row
    if top_k is not None:
        for i in valid_indices:
            row = mat[i].copy()
            if not self_loops:
                row[i] = 0.0
            threshold_val = np.sort(row)[-top_k] if top_k < (row > 0).sum() else 0.0
            mat[i, row < threshold_val] = 0.0

    # Collect edges
    rows, cols = np.where(mat > threshold)

    # Filter to valid bins only (both endpoints must be valid)
    valid_set = set(valid_indices.tolist())
    keep = np.array([
        r in valid_set and c in valid_set and (self_loops or r != c)
        for r, c in zip(rows, cols)
    ], dtype=bool)

    rows, cols = rows[keep], cols[keep]
    weights = mat[rows, cols]

    # Remap to compressed node IDs
    src = remap[rows]
    dst = remap[cols]

    # Genomic distance feature: log(|i - j| + 1) normalized
    genomic_dist = np.abs(rows.astype(np.float32) - cols.astype(np.float32))
    log_dist = np.log1p(genomic_dist)
    # Normalize to [0, 1]
    if log_dist.max() > 0:
        log_dist /= log_dist.max()

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(
        np.stack([weights, log_dist], axis=1), dtype=torch.float32
    )

    logger.info(
        f"Graph edges: {edge_index.shape[1]:,} edges from "
        f"{len(valid_indices)} nodes (threshold={threshold})"
    )
    return edge_index, edge_attr


def build_node_positions(
    valid_mask: np.ndarray,
    bin_size: int,
    chrom_start: int = 0,
) -> torch.Tensor:
    """
    Compute the genomic coordinate (center of each bin) for valid nodes.

    Used as a positional feature and for spatial distance calculations.

    Parameters
    ----------
    valid_mask:
        Boolean array (N,) indicating valid bins.
    bin_size:
        Bin size in base pairs.
    chrom_start:
        Start coordinate of the first bin (default 0).

    Returns
    -------
    torch.Tensor
        (M, 1) float32 tensor of normalized bin center positions.
    """
    import torch
    valid_indices = np.where(valid_mask)[0]
    centers = chrom_start + (valid_indices + 0.5) * bin_size
    # Normalize to [0, 1] within this chromosome arm
    if centers.max() > centers.min():
        centers = (centers - centers.min()) / (centers.max() - centers.min())
    return torch.tensor(centers.reshape(-1, 1), dtype=torch.float32)


def assemble_node_features(
    ngs_features: np.ndarray,
    sequence_embeddings: np.ndarray,
    metadata_features: np.ndarray,
    valid_mask: np.ndarray,
    bin_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Concatenate all per-bin features into a single node feature matrix.

    All inputs must be aligned to the same bin indexing (N bins).
    Only bins where valid_mask=True are included in the output.

    Parameters
    ----------
    ngs_features:
        (N, F_ngs) float32 — RPKM-normalized coverage tracks.
    sequence_embeddings:
        (N, F_seq) float32 — k-mer or DNA-BERT embeddings.
    metadata_features:
        (N, F_meta) float32 — normalized GC, mappability, chromatin state.
    valid_mask:
        Boolean (N,) — only valid bins are kept.
    bin_positions:
        Optional (M, 1) float32 — normalized genomic position.

    Returns
    -------
    torch.Tensor
        (M, F_total) float32 node feature matrix,
        where M = valid_mask.sum() and F_total = sum of all feature dims.
    """
    import torch
    # Validate shapes
    n = len(valid_mask)
    for name, arr in [("ngs_features", ngs_features),
                      ("sequence_embeddings", sequence_embeddings),
                      ("metadata_features", metadata_features)]:
        if arr.shape[0] != n:
            raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")

    parts = [
        torch.tensor(ngs_features[valid_mask], dtype=torch.float32),
        torch.tensor(sequence_embeddings[valid_mask], dtype=torch.float32),
        torch.tensor(metadata_features[valid_mask], dtype=torch.float32),
    ]

    if bin_positions is not None:
        parts.append(bin_positions)

    x = torch.cat(parts, dim=1)
    logger.info(f"Node feature matrix: {x.shape} (M={valid_mask.sum()}, F={x.shape[1]})")
    return x
