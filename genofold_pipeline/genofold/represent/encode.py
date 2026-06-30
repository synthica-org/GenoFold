"""
genofold.represent.encode
--------------------------
Final assembly of all stage-2 components into a PyTorch Geometric Data object.

This is the last step before the GNN encoder (stage 3).
The Data object carries:

  data.x            — (M, F) node feature matrix
  data.edge_index   — (2, E) edge connectivity
  data.edge_attr    — (E, 2) edge features [contact_weight, log_dist]
  data.species_ids  — (M,) integer species labels per node
  data.pos          — (M, 1) normalized genomic position
  data.num_nodes    — M (int)
  data.bin_size     — bin resolution in bp
  data.valid_mask   — (N,) original-space validity mask (for mapping back)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .graph import assemble_node_features, build_edge_index, build_node_positions

logger = logging.getLogger(__name__)


def build_pyg_data(
    contact_matrix: np.ndarray,
    valid_mask: np.ndarray,
    ngs_features: np.ndarray,
    sequence_embeddings: np.ndarray,
    metadata_features: np.ndarray,
    species_ids: np.ndarray,
    bin_size: int,
    chrom_start: int = 0,
    contact_threshold: float = 0.0,
    top_k_edges: Optional[int] = None,
    self_loops: bool = False,
):
    """
    Assemble all preprocessed data into a PyTorch Geometric Data object.

    Parameters
    ----------
    contact_matrix:
        Preprocessed Hi-C matrix (N, N) from stage 1.
    valid_mask:
        Boolean (N,) — which bins are valid.
    ngs_features:
        (N, F_ngs) NGS feature matrix from stage 1.
    sequence_embeddings:
        (N, F_seq) sequence embedding matrix from stage 2.
    metadata_features:
        (N, F_meta) metadata feature matrix from stage 1.
    species_ids:
        (N,) integer species labels.
    bin_size:
        Genomic bin resolution in base pairs.
    chrom_start:
        Chromosome start coordinate for position encoding.
    contact_threshold:
        Minimum contact weight to include an edge.
    top_k_edges:
        Keep only top-k contacts per node (None = keep all above threshold).
    self_loops:
        Whether to add self-loop edges.

    Returns
    -------
    torch_geometric.data.Data
        Fully assembled graph ready for GNN processing.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "Install PyTorch Geometric: pip install torch-geometric"
        )

    logger.info("Assembling PyG Data object...")

    # Build edges
    edge_index, edge_attr = build_edge_index(
        contact_matrix,
        valid_mask,
        threshold=contact_threshold,
        top_k=top_k_edges,
        self_loops=self_loops,
    )

    # Build positional encoding
    pos = build_node_positions(valid_mask, bin_size, chrom_start)

    # Assemble node features
    x = assemble_node_features(
        ngs_features=ngs_features,
        sequence_embeddings=sequence_embeddings,
        metadata_features=metadata_features,
        valid_mask=valid_mask,
        bin_positions=pos,
    )

    # Species IDs for valid nodes only
    species_tensor = torch.tensor(species_ids[valid_mask], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        species_ids=species_tensor,
        num_nodes=int(valid_mask.sum()),
    )

    # Store metadata as attributes (not part of PyG's standard fields)
    data.bin_size = bin_size
    data.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)

    logger.info(
        f"PyG Data assembled:\n"
        f"  Nodes:         {data.num_nodes}\n"
        f"  Edges:         {data.edge_index.shape[1]:,}\n"
        f"  Node features: {data.x.shape[1]}\n"
        f"  Edge features: {data.edge_attr.shape[1]}\n"
        f"  Bin size:      {bin_size:,} bp"
    )
    return data


def save_data(data, path: str) -> None:
    """Save a PyG Data object to disk."""
    import torch
    torch.save(data, path)
    logger.info(f"Saved PyG Data to {path}")


def load_data(path: str):
    """Load a PyG Data object from disk."""
    import torch
    data = torch.load(path, weights_only=False)
    logger.info(f"Loaded PyG Data from {path}")
    return data
