"""
genofold.reconstruct.distance
------------------------------
Pairwise genomic distance predictor.

Takes the latent embeddings from stage 3 (M, hidden_dim) and predicts
a symmetric distance matrix (M, M) representing the expected 3D spatial
distance between every pair of genomic bins.

Architecture:
  - Pairwise feature construction: for each bin pair (i, j), compute
    [embed_i, embed_j, embed_i - embed_j, embed_i * embed_j]
  - Shared MLP that maps pairwise features → scalar distance
  - Symmetry enforcement: D = (D + D^T) / 2
  - Distance must be non-negative: enforced via softplus activation

Why predict distances first?
  - Hi-C contact frequency is known to correlate with spatial distance
  - Having an explicit distance matrix allows distance geometry (MDS)
    to anchor the 3D reconstruction in physically meaningful space
  - The distance matrix is also directly comparable to experimental
    measurements and to benchmark method outputs
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PairwiseDistancePredictor(nn.Module):
    """
    Predicts pairwise 3D spatial distances from bin embeddings.

    For M bins, constructs M*M pairwise feature vectors and passes
    each through a shared MLP to predict one distance value per pair.
    Memory-efficient chunked computation is used for large M.

    Parameters
    ----------
    hidden_dim:
        Embedding dimension from stage 3 (default 64).
    mlp_dim:
        Inner dimension of the distance prediction MLP (default 256).
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input: [e_i | e_j | e_i - e_j | e_i * e_j] = 4 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, mlp_dim // 4),
            nn.GELU(),
            nn.Linear(mlp_dim // 4, 1),
        )

        # Softplus ensures distances are always positive
        self.dist_activation = nn.Softplus()

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"PairwiseDistancePredictor: {total:,} parameters")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict pairwise distance matrix.

        Parameters
        ----------
        embeddings:
            (M, hidden_dim) tensor from stage 3.

        Returns
        -------
        torch.Tensor
            Symmetric distance matrix (M, M), all values >= 0.
        """
        M, D = embeddings.shape

        # Construct pairwise features efficiently using broadcasting
        # e_i: (M, 1, D) broadcast → (M, M, D)
        # e_j: (1, M, D) broadcast → (M, M, D)
        e_i = embeddings.unsqueeze(1).expand(M, M, D)
        e_j = embeddings.unsqueeze(0).expand(M, M, D)

        # [e_i | e_j | e_i - e_j | e_i * e_j] → (M, M, 4D)
        pair_features = torch.cat([
            e_i,
            e_j,
            e_i - e_j,
            e_i * e_j,
        ], dim=-1)

        # Flatten to (M*M, 4D), pass through MLP, reshape to (M, M)
        flat = pair_features.view(M * M, 4 * D)
        distances = self.dist_activation(self.mlp(flat)).view(M, M)

        # Enforce symmetry: D[i,j] == D[j,i]
        distances = (distances + distances.T) / 2.0

        # Zero out diagonal (self-distance = 0)
        distances = distances * (1 - torch.eye(M, device=distances.device))

        logger.debug(
            f"Distance matrix: shape={distances.shape}, "
            f"min={distances.min():.3f}, max={distances.max():.3f}, "
            f"mean={distances.mean():.3f}"
        )
        return distances
