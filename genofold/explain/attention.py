"""
genofold.explain.attention
-----------------------------
Attention weight extraction and influential loci detection.

Two attention sources are analyzed:
  1. GAT attention (stage 3 GNN) — which Hi-C contacts did the model
     weight most heavily when aggregating neighbor information?
  2. Transformer attention (stage 3 Transformer) — which bins attended
     most strongly to which other bins, regardless of Hi-C contact?

"Influential loci" are bins that receive disproportionately high
attention from many other bins — these are candidate structural
hubs (e.g. potential CTCF anchor sites or compartment cores).
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_gat_attention(
    gnn_encoder: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract attention weights from the first GAT layer of the GNN encoder.

    Re-runs the input projection + first GAT layer with
    return_attention_weights=True to capture the learned attention
    coefficients on each edge.

    Parameters
    ----------
    gnn_encoder:
        Trained GNNEncoder module from stage 3.
    x:
        (M, input_dim) node features.
    edge_index:
        (2, E) edge connectivity.
    edge_attr:
        (E, edge_dim) edge features.

    Returns
    -------
    (edge_index_out, attention_weights)
        edge_index_out : (2, E') — may include self-loops added by GAT.
        attention_weights : (E', heads) — per-edge, per-head attention.
    """
    with torch.no_grad():
        x_proj = gnn_encoder.input_proj(x)
        edge_proj = gnn_encoder.edge_proj(edge_attr)

        # First GAT layer's underlying GATv2Conv supports attention return
        conv = gnn_encoder.gat_layers[0].conv
        out, (edge_index_out, attn_weights) = conv(
            x_proj, edge_index, edge_attr=edge_proj,
            return_attention_weights=True,
        )

    logger.info(
        f"GAT attention extracted: {attn_weights.shape[0]} edges, "
        f"{attn_weights.shape[1]} heads"
    )
    return edge_index_out, attn_weights


def extract_transformer_attention(
    transformer: nn.Module,
    gnn_embeddings: torch.Tensor,
    bin_positions: torch.Tensor,
    species_ids: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    Extract self-attention weights from a Transformer layer.

    Parameters
    ----------
    transformer:
        Trained GenomicTransformer module from stage 3.
    gnn_embeddings:
        (M, hidden_dim) GNN output (input to the Transformer).
    bin_positions:
        (M, 1) normalized genomic positions.
    species_ids:
        (M,) species labels.
    layer_idx:
        Which Transformer layer to extract attention from
        (-1 = last layer, which usually captures the most global context).

    Returns
    -------
    torch.Tensor
        (M, M) attention weight matrix, averaged across heads.
    """
    with torch.no_grad():
        x = transformer.pos_encoding(gnn_embeddings, bin_positions)
        x = x + transformer.species_embedding(species_ids)
        x = x.unsqueeze(0)   # (1, M, hidden_dim)

        layer = transformer.layers[layer_idx]
        normed = layer.norm1(x)

        # MultiheadAttention supports returning weights
        _, attn_weights = layer.attn(
            normed, normed, normed,
            need_weights=True,
            average_attn_weights=True,  # Average across heads → (1, M, M)
        )

    attn_matrix = attn_weights.squeeze(0)   # (M, M)
    logger.info(f"Transformer attention extracted: layer={layer_idx}, shape={attn_matrix.shape}")
    return attn_matrix


def detect_influential_loci(
    attention_matrix: torch.Tensor,
    top_k: int = 10,
) -> torch.Tensor:
    """
    Detect influential genomic bins (structural hubs) from an
    attention matrix, by measuring total incoming attention.

    A bin that receives high attention from many other bins is
    likely a structural anchor point (e.g. CTCF site, compartment core).

    Parameters
    ----------
    attention_matrix:
        (M, M) attention weights, attention_matrix[i, j] = how much
        bin i attends to bin j.
    top_k:
        Number of top influential bins to highlight.

    Returns
    -------
    torch.Tensor
        (M,) influence score per bin (column-sum of attention,
        i.e. total attention RECEIVED by each bin), normalized to [0, 1].
    """
    # Column sum: total attention received by each bin (as a "key")
    incoming_attention = attention_matrix.sum(dim=0)   # (M,)

    max_val = incoming_attention.max().clamp(min=1e-8)
    influence_score = incoming_attention / max_val

    top_indices = torch.topk(influence_score, k=min(top_k, len(influence_score))).indices
    logger.info(
        f"Influential loci detected: top-{top_k} bins = {top_indices.tolist()}"
    )
    return influence_score
