"""
genofold.score.model
----------------------
GenoFold Stage 5: Full scoring model.

Combines three components:
  1. ConfidenceScorer    — per-bin confidence in [0, 1]
  2. Region detection    — groups bins into TAD-like domains
  3. VarianceHead + MC   — coordinate uncertainty (error bars)

Input:  InferenceOutput (stage 3) + ReconstructionOutput (stage 4)
Output: ScoringOutput with per-bin confidence, per-region confidence,
        and coordinate uncertainty bounds — ready for stage 6 (Explain)
        and for updating the PDB B-factor column with real confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .confidence import ConfidenceScorer
from .region import Region, detect_regions, aggregate_region_confidence, regions_to_dataframe
from .uncertainty import VarianceHead, UncertaintyOutput, estimate_uncertainty

logger = logging.getLogger(__name__)


@dataclass
class ScoringOutput:
    """
    Output of stage 5 scoring.

    Attributes
    ----------
    bin_confidence:
        (M,) per-bin confidence scores in [0, 1].
    regions:
        List of Region objects with per-region aggregated confidence.
    uncertainty:
        UncertaintyOutput with coordinate error bars.
    region_df:
        pandas DataFrame version of regions (for export/plotting).
    """
    bin_confidence: torch.Tensor
    regions: List[Region]
    uncertainty: UncertaintyOutput
    region_df: "pd.DataFrame" = None


class GenoFoldScoreModel(nn.Module):
    """
    Stage 5: Confidence and uncertainty scoring.

    Parameters
    ----------
    hidden_dim:
        Embedding dimension from stage 3 (default 64).
    confidence_mlp_dim:
        Inner dimension for ConfidenceScorer MLP.
    variance_mlp_dim:
        Inner dimension for VarianceHead MLP.
    min_region_size:
        Minimum bins per detected region.
    boundary_sensitivity:
        Region boundary detection sensitivity (higher = more regions).
    n_mc_samples:
        Number of Monte Carlo dropout samples for uncertainty.
    use_mc_dropout:
        Whether to run MC dropout (slower but more accurate uncertainty).
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        confidence_mlp_dim: int = 128,
        variance_mlp_dim: int = 128,
        min_region_size: int = 3,
        boundary_sensitivity: float = 1.0,
        n_mc_samples: int = 20,
        use_mc_dropout: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.min_region_size = min_region_size
        self.boundary_sensitivity = boundary_sensitivity
        self.n_mc_samples = n_mc_samples
        self.use_mc_dropout = use_mc_dropout

        self.confidence_scorer = ConfidenceScorer(
            hidden_dim=hidden_dim,
            mlp_dim=confidence_mlp_dim,
            dropout=dropout,
        )

        self.variance_head = VarianceHead(
            hidden_dim=hidden_dim,
            mlp_dim=variance_mlp_dim,
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"GenoFoldScoreModel ready | parameters: {total:,}")

    def forward(
        self,
        inference_output,
        reconstruction_output,
        edge_index: torch.Tensor,
        reconstruct_model: Optional[nn.Module] = None,
    ) -> ScoringOutput:
        """
        Run stage 5 scoring.

        Parameters
        ----------
        inference_output:
            InferenceOutput from stage 3.
        reconstruction_output:
            ReconstructionOutput from stage 4.
        edge_index:
            (2, E) Hi-C graph edges from stage 2 (data.edge_index).
        reconstruct_model:
            Stage 4 model instance, needed for MC dropout uncertainty.
            If None, MC dropout is skipped and only learned variance is used.

        Returns
        -------
        ScoringOutput
        """
        embeddings = inference_output.embeddings
        distance_matrix = reconstruction_output.distance_matrix
        coordinates = reconstruction_output.coordinates

        logger.info(f"Stage 5 scoring: {embeddings.shape[0]} bins")

        # ── 1. Per-bin confidence ────────────────────────────────────────────
        logger.info("  Computing per-bin confidence...")
        bin_confidence = self.confidence_scorer(
            embeddings, distance_matrix, coordinates, edge_index
        )

        # ── 2. Region detection + aggregation ───────────────────────────────
        logger.info("  Detecting structural regions...")
        region_ranges = detect_regions(
            distance_matrix,
            min_region_size=self.min_region_size,
            boundary_sensitivity=self.boundary_sensitivity,
        )
        regions = aggregate_region_confidence(bin_confidence, region_ranges)
        region_df = regions_to_dataframe(regions)

        # ── 3. Coordinate uncertainty ───────────────────────────────────────
        logger.info("  Estimating coordinate uncertainty...")
        use_mc = self.use_mc_dropout and reconstruct_model is not None
        uncertainty = estimate_uncertainty(
            self.variance_head,
            embeddings,
            reconstruct_model=reconstruct_model,
            inference_output=inference_output,
            n_mc_samples=self.n_mc_samples,
            use_mc_dropout=use_mc,
        )

        logger.info(
            f"Stage 5 complete ✓ | "
            f"mean confidence={bin_confidence.mean():.3f} | "
            f"{len(regions)} regions | "
            f"mean uncertainty radius={uncertainty.radius.mean():.4f}"
        )

        return ScoringOutput(
            bin_confidence=bin_confidence,
            regions=regions,
            uncertainty=uncertainty,
            region_df=region_df,
        )
