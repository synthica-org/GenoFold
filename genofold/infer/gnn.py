"""
genofold.infer.gnn
------------------
Graph Attention Network (GAT) encoder for genomic bin representations.

Architecture:
  - 4 layers of multi-head Graph Attention (GATv2)
  - Edge features (contact weight + genomic distance) injected at each layer
  - Residual connections between layers to prevent vanishing gradients
  - Layer normalization for training stability
  - Dropout for regularization

Input:  PyG Data object from stage 2
Output: (M, hidden_dim) latent node embeddings — one vector per genomic bin

Why GAT over vanilla GCN?
  - Attention weights let the model learn WHICH Hi-C contacts matter most
  - Different bins can have different numbers of contacts (variable degree)
    and GAT handles this naturally
  - GATv2 fixes the static attention problem of original GAT, making it
    strictly more expressive
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge feature projection
# ---------------------------------------------------------------------------

class EdgeFeatureProjection(nn.Module):
    """
    Projects edge features into the attention computation.

    Hi-C contact weight and genomic distance are both informative —
    this module learns how to combine them with node features
    during attention score calculation.
    """

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.proj(edge_attr)


# ---------------------------------------------------------------------------
# Single GAT layer with edge features
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Single GATv2 layer with edge feature integration.

    Uses multi-head attention where each head independently learns
    which neighboring bins to attend to based on both node features
    and edge (Hi-C contact) features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError:
            raise ImportError("Install torch-geometric: pip install torch-geometric")

        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat,
        )
        self.norm = nn.LayerNorm(out_dim * heads if concat else out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Full GNN Encoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """
    Multi-layer GAT encoder with residual connections.

    Architecture (4 layers):
      Layer 0: input_dim  → hidden_dim  (8 heads, concat → hidden_dim * 8)
      Layer 1: hidden*8   → hidden_dim  (8 heads, concat → hidden_dim * 8)
      Layer 2: hidden*8   → hidden_dim  (8 heads, concat → hidden_dim * 8)
      Layer 3: hidden*8   → hidden_dim  (4 heads, mean   → hidden_dim)

    Residual projections are added when dimensions change between layers.

    Parameters
    ----------
    input_dim:
        Dimension of input node features (F from stage 2, e.g. 155).
    hidden_dim:
        Base hidden dimension per attention head (default 64).
    edge_dim:
        Dimension of edge features (2 from stage 2).
    dropout:
        Dropout rate applied after each layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection: raw features → hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.LayerNorm(hidden_dim * 8),
        )

        # Edge feature projection
        self.edge_proj = EdgeFeatureProjection(edge_dim, edge_dim)

        # 4 GAT layers
        # Layers 0-2: 8 heads concat → output = hidden_dim * 8
        # Layer 3:    4 heads mean   → output = hidden_dim
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim * 8, hidden_dim, edge_dim, heads=8, dropout=dropout, concat=True),
            GATLayer(hidden_dim * 8, hidden_dim, edge_dim, heads=8, dropout=dropout, concat=True),
            GATLayer(hidden_dim * 8, hidden_dim, edge_dim, heads=8, dropout=dropout, concat=True),
            GATLayer(hidden_dim * 8, hidden_dim, edge_dim, heads=4, dropout=dropout, concat=False),
        ])

        # Residual projection for final layer (hidden*8 → hidden)
        self.residual_proj = nn.Linear(hidden_dim * 8, hidden_dim)

        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)

        self._log_architecture(input_dim, hidden_dim, edge_dim)

    def _log_architecture(self, input_dim, hidden_dim, edge_dim):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"GNNEncoder initialized:\n"
            f"  input_dim={input_dim}, hidden_dim={hidden_dim}, edge_dim={edge_dim}\n"
            f"  Layers: 4x GATv2 (3x concat-8head + 1x mean-4head)\n"
            f"  Trainable parameters: {total_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the GNN encoder.

        Parameters
        ----------
        x:
            Node feature matrix (M, input_dim).
        edge_index:
            Edge connectivity (2, E).
        edge_attr:
            Edge features (E, edge_dim).

        Returns
        -------
        torch.Tensor
            Node embeddings (M, hidden_dim).
        """
        # Project input features and edge features
        x = self.input_proj(x)           # (M, hidden_dim * 8)
        edge_attr = self.edge_proj(edge_attr)  # (E, edge_dim)

        # Layers 0, 1, 2 — residual connections (same dim in/out)
        for i in range(3):
            residual = x
            x = self.gat_layers[i](x, edge_index, edge_attr)
            x = x + residual             # (M, hidden_dim * 8)

        # Layer 3 — dimension reduction with residual projection
        residual = self.residual_proj(x) # (M, hidden_dim)
        x = self.gat_layers[3](x, edge_index, edge_attr)  # (M, hidden_dim)
        x = x + residual

        x = self.output_norm(x)
        return x                         # (M, hidden_dim)
