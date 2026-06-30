"""
genofold.reconstruct.decoder
-----------------------------
Autoencoder-style decoder for 3D coordinate prediction.

Takes the latent embeddings from stage 3 (M, hidden_dim) and directly
decodes them into 3D Cartesian coordinates (M, 3) representing the
spatial position of each genomic bin in 3D nuclear space.

Architecture:
  - Progressive upsampling MLP: hidden_dim → 128 → 256 → 128 → 3
  - Batch normalization at each layer for training stability
  - Final tanh activation scales coordinates to a unit sphere,
    which can then be rescaled to real nuclear dimensions
  - Coordinate refinement: uses predicted distances as a geometric
    constraint to push coordinates into physically consistent positions

Why a decoder alongside distance geometry?
  - The decoder learns a direct mapping from learned features to space
  - Distance geometry (MDS in geometry.py) optimizes from distances
  - Fusing both gives coordinates that are consistent with BOTH the
    learned representations AND the predicted pairwise distances
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CoordinateDecoder(nn.Module):
    """
    Decodes latent bin embeddings into 3D Cartesian coordinates.

    Parameters
    ----------
    hidden_dim:
        Embedding dimension from stage 3 (default 64).
    dropout:
        Dropout rate.
    scale:
        Output coordinate scale in arbitrary units (default 1.0).
        Multiply by nuclear radius (~5 μm) to get real-space units.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale

        self.decoder = nn.Sequential(
            # Layer 1: hidden_dim → 128
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 2: 128 → 256
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 3: 256 → 256
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 4: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Output: 128 → 3 (X, Y, Z)
            nn.Linear(128, 3),
            nn.Tanh(),  # Constrains to [-1, 1] cube
        )

        # Initialize output layer with small weights for stability
        nn.init.xavier_uniform_(self.decoder[-2].weight, gain=0.1)
        nn.init.zeros_(self.decoder[-2].bias)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"CoordinateDecoder: {total:,} parameters | scale={scale}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to 3D coordinates.

        Parameters
        ----------
        embeddings:
            (M, hidden_dim) latent bin representations from stage 3.

        Returns
        -------
        torch.Tensor
            (M, 3) float32 coordinates (X, Y, Z) scaled by self.scale.
        """
        coords = self.decoder(embeddings)   # (M, 3) in [-1, 1]
        coords = coords * self.scale        # rescale to desired range
        logger.debug(f"Decoded coordinates: shape={coords.shape}, "
                     f"range=[{coords.min():.3f}, {coords.max():.3f}]")
        return coords


class CoordinateRefinementLayer(nn.Module):
    """
    Refines raw decoded coordinates using predicted distance constraints.

    Iteratively adjusts coordinates so that the implied pairwise
    distances (from coordinates) better match the predicted distance
    matrix. This acts as a learned geometric consistency layer.

    Parameters
    ----------
    n_iterations:
        Number of refinement steps (default 3).
    step_size:
        Gradient step size for coordinate adjustment (default 0.01).
    """

    def __init__(self, n_iterations: int = 3, step_size: float = 0.01):
        super().__init__()
        self.n_iterations = n_iterations
        self.step_size = step_size
        # Learnable per-iteration weights
        self.iter_weights = nn.Parameter(
            torch.ones(n_iterations) * step_size
        )

    def forward(
        self,
        coords: torch.Tensor,
        target_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine coordinates to be consistent with target distance matrix.

        Parameters
        ----------
        coords:
            Initial (M, 3) coordinate tensor from CoordinateDecoder.
        target_distances:
            (M, M) predicted distance matrix from PairwiseDistancePredictor.

        Returns
        -------
        torch.Tensor
            Refined (M, 3) coordinate tensor.
        """
        refined = coords.clone()
        M = refined.shape[0]

        for i in range(self.n_iterations):
            # Compute current implied distances from coordinates
            diff = refined.unsqueeze(1) - refined.unsqueeze(0)   # (M, M, 3)
            current_dists = torch.norm(diff, dim=-1)              # (M, M)

            # Distance error: how far off are we from target?
            dist_error = current_dists - target_distances         # (M, M)

            # Gradient: push each bin toward/away from others
            # to reduce distance error
            safe_dists = current_dists.clamp(min=1e-8)
            direction = diff / safe_dists.unsqueeze(-1)           # (M, M, 3)

            # Force on each bin = sum of corrections from all pairs
            force = (dist_error.unsqueeze(-1) * direction).sum(dim=1)  # (M, 3)

            # Apply learned step
            step = torch.abs(self.iter_weights[i])
            refined = refined - step * force

        logger.debug(f"Coordinate refinement: {self.n_iterations} iterations done")
        return refined
