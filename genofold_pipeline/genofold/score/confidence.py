"""
genofold.score.confidence
---------------------------
Per-bin confidence scoring.

Predicts a reliability score in [0, 1] for every genomic bin, indicating
how confident the model is in that bin's predicted 3D position.

Confidence is derived from multiple signals:
  1. Embedding norm — bins with extreme or unusual embeddings tend to
     be less reliably positioned (learned via MLP on embeddings)
  2. Distance consistency — bins whose predicted distances to neighbors
     are internally consistent (low variance) are more trustworthy
  3. Graph degree — bins with very few Hi-C contacts (low degree) have
     less evidence supporting their position

These signals are combined through a learned scoring head.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConfidenceScorer(nn.Module):
    """
    Predicts per-bin confidence scores from embeddings and structural signals.

    Parameters
    ----------
    hidden_dim:
        Embedding dimension from stage 3 (default 64).
    mlp_dim:
        Inner dimension of the scoring MLP (default 128).
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        mlp_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input: [embedding, distance_consistency, graph_degree] = hidden_dim + 2
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + 2, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, 1),
            nn.Sigmoid(),   # Output in [0, 1]
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"ConfidenceScorer: {total:,} parameters")

    def compute_distance_consistency(
        self,
        distance_matrix: torch.Tensor,
        coordinates: torch.Tensor,
        top_k: int = 10,
    ) -> torch.Tensor:
        """
        Measure how consistent each bin's predicted distances are with
        the distances implied by its final 3D coordinates.

        Low discrepancy = high consistency = high confidence signal.

        Parameters
        ----------
        distance_matrix:
            (M, M) predicted pairwise distances from stage 4.
        coordinates:
            (M, 3) final 3D coordinates from stage 4.
        top_k:
            Number of nearest neighbors to consider per bin.

        Returns
        -------
        torch.Tensor
            (M,) consistency score, higher = more consistent (less error).
        """
        M = distance_matrix.shape[0]
        k = min(top_k, M - 1)

        # Implied distances from final coordinates
        diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)  # (M, M, 3)
        implied_dist = torch.norm(diff, dim=-1)                      # (M, M)

        # For each bin, look at its k nearest neighbors (by predicted distance)
        # and compute the discrepancy between predicted and implied distance
        _, nn_idx = torch.topk(-distance_matrix, k=k + 1, dim=1)  # +1 to skip self
        nn_idx = nn_idx[:, 1:]  # remove self

        row_idx = torch.arange(M, device=distance_matrix.device).unsqueeze(1).expand(-1, k)
        pred_d = distance_matrix[row_idx, nn_idx]      # (M, k)
        impl_d = implied_dist[row_idx, nn_idx]          # (M, k)

        discrepancy = (pred_d - impl_d).abs().mean(dim=1)  # (M,)

        # Convert to consistency score: lower discrepancy → higher consistency
        max_disc = discrepancy.max().clamp(min=1e-8)
        consistency = 1.0 - (discrepancy / max_disc)
        return consistency

    def compute_graph_degree_signal(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute normalized graph degree per bin as a confidence signal.

        Bins with more Hi-C contacts have more evidence supporting
        their structural placement.

        Parameters
        ----------
        edge_index:
            (2, E) edge connectivity from stage 2.
        num_nodes:
            Number of bins (M).

        Returns
        -------
        torch.Tensor
            (M,) normalized degree in [0, 1].
        """
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree.scatter_add_(
            0, edge_index[0],
            torch.ones(edge_index.shape[1], device=edge_index.device),
        )
        max_deg = degree.max().clamp(min=1e-8)
        return degree / max_deg

    def forward(
        self,
        embeddings: torch.Tensor,
        distance_matrix: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict per-bin confidence scores.

        Parameters
        ----------
        embeddings:
            (M, hidden_dim) latent embeddings from stage 3.
        distance_matrix:
            (M, M) predicted distances from stage 4.
        coordinates:
            (M, 3) final coordinates from stage 4.
        edge_index:
            (2, E) Hi-C graph edges from stage 2.

        Returns
        -------
        torch.Tensor
            (M,) confidence scores in [0, 1].
        """
        M = embeddings.shape[0]

        consistency = self.compute_distance_consistency(distance_matrix, coordinates)
        degree_signal = self.compute_graph_degree_signal(edge_index, M)

        features = torch.cat([
            embeddings,
            consistency.unsqueeze(1),
            degree_signal.unsqueeze(1),
        ], dim=1)

        confidence = self.scorer(features).squeeze(-1)   # (M,)

        logger.info(
            f"Confidence scores: mean={confidence.mean():.3f}, "
            f"min={confidence.min():.3f}, max={confidence.max():.3f}"
        )
        return confidence
