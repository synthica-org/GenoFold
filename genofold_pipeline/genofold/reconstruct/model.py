"""
genofold.reconstruct.model
---------------------------
GenoFold Stage 4: Full reconstruction model.

Combines three pathways:
  1. PairwiseDistancePredictor — predicts D[i,j] from embeddings
  2. CoordinateDecoder         — directly decodes embeddings → XYZ
  3. Distance geometry (MDS)  — converts predicted D → XYZ

Final coordinates are a weighted fusion of decoder + MDS outputs,
refined by the CoordinateRefinementLayer for geometric consistency.

Input:  InferenceOutput from stage 3
Output: ReconstructionOutput with coords, distances, and export paths
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .distance import PairwiseDistancePredictor
from .decoder import CoordinateDecoder, CoordinateRefinementLayer
from .geometry import classical_mds, stress_minimization_mds, fuse_coordinates
from .export import export_structure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionOutput:
    """
    Output of stage 4 reconstruction.

    Attributes
    ----------
    coordinates:
        (M, 3) final 3D coordinates per genomic bin.
    distance_matrix:
        (M, M) predicted pairwise distance matrix.
    decoder_coords:
        (M, 3) raw decoder coordinates (before fusion).
    mds_coords:
        (M, 3) MDS-derived coordinates (before fusion).
    export_paths:
        Dict of output file paths if export was requested.
    """
    coordinates: torch.Tensor
    distance_matrix: torch.Tensor
    decoder_coords: torch.Tensor
    mds_coords: torch.Tensor
    export_paths: Optional[dict] = None


# ---------------------------------------------------------------------------
# Full reconstruction model
# ---------------------------------------------------------------------------

class GenoFoldReconstructModel(nn.Module):
    """
    Stage 4: 3D genome structure reconstruction.

    Parameters
    ----------
    hidden_dim:
        Embedding dimension from stage 3 (must match InferenceOutput).
    mlp_dim:
        Inner dimension for distance prediction MLP.
    decoder_scale:
        Output scale for decoded coordinates.
    refinement_iters:
        Number of coordinate refinement steps.
    decoder_weight:
        Weight for decoder coords in fusion [0, 1].
        MDS weight = 1 - decoder_weight.
    use_stress_mds:
        If True, run stress-minimization MDS after classical MDS.
        More accurate but slower. Recommended for final outputs.
    mds_iter:
        Number of stress-MDS iterations (if use_stress_mds=True).
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        mlp_dim: int = 256,
        decoder_scale: float = 1.0,
        refinement_iters: int = 3,
        decoder_weight: float = 0.5,
        use_stress_mds: bool = True,
        mds_iter: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.decoder_weight = decoder_weight
        self.use_stress_mds = use_stress_mds
        self.mds_iter = mds_iter

        self.distance_predictor = PairwiseDistancePredictor(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.coordinate_decoder = CoordinateDecoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            scale=decoder_scale,
        )

        self.refinement = CoordinateRefinementLayer(
            n_iterations=refinement_iters,
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"GenoFoldReconstructModel ready | "
            f"parameters: {total:,} | "
            f"hidden_dim={hidden_dim} | "
            f"decoder_weight={decoder_weight}"
        )

    def forward(
        self,
        inference_output,
        export_dir: Optional[str] = None,
        export_prefix: str = "genofold_structure",
        bin_size: int = 40_000,
        chrom: str = "chr1",
        chrom_start: int = 0,
    ) -> ReconstructionOutput:
        """
        Run stage 4 reconstruction.

        Parameters
        ----------
        inference_output:
            InferenceOutput from stage 3 GenoFoldInferenceModel.
        export_dir:
            If provided, export structure files (.npy, .pdb, .tsv) here.
        export_prefix:
            Filename prefix for exported files.
        bin_size, chrom, chrom_start:
            Genomic metadata for export.

        Returns
        -------
        ReconstructionOutput
        """
        embeddings = inference_output.embeddings   # (M, hidden_dim)
        M = embeddings.shape[0]

        logger.info(f"Stage 4 reconstruction: {M} bins")

        # ── Pathway 1: Predict pairwise distances ────────────────────────────
        logger.info("  Predicting pairwise distance matrix...")
        dist_matrix = self.distance_predictor(embeddings)   # (M, M)

        # ── Pathway 2: Decode coordinates directly ───────────────────────────
        logger.info("  Decoding coordinates from embeddings...")
        decoder_coords = self.coordinate_decoder(embeddings)  # (M, 3)

        # ── Pathway 3: Distance geometry (MDS) ──────────────────────────────
        logger.info("  Running Classical MDS...")
        mds_coords = classical_mds(dist_matrix.detach(), n_components=3)

        if self.use_stress_mds:
            logger.info(f"  Running Stress MDS ({self.mds_iter} iterations)...")
            mds_coords = stress_minimization_mds(
                dist_matrix.detach(),
                init_coords=mds_coords,
                n_iter=self.mds_iter,
            )

        # ── Fusion: combine decoder + MDS ────────────────────────────────────
        logger.info("  Fusing decoder and MDS coordinates...")
        fused_coords = fuse_coordinates(
            decoder_coords.detach(),
            mds_coords,
            decoder_weight=self.decoder_weight,
        )

        # ── Refinement: enforce geometric consistency ─────────────────────────
        logger.info("  Refining coordinates...")
        final_coords = self.refinement(fused_coords, dist_matrix.detach())

        # ── Export ────────────────────────────────────────────────────────────
        export_paths = None
        if export_dir is not None:
            logger.info(f"  Exporting structure to {export_dir}...")
            export_paths = export_structure(
                coords=final_coords,
                output_dir=export_dir,
                prefix=export_prefix,
                bin_size=bin_size,
                chrom=chrom,
                chrom_start=chrom_start,
                confidence=None,   # Will be filled by stage 5 (scoring)
            )

        logger.info(
            f"Stage 4 complete ✓ | "
            f"coordinates shape: {final_coords.shape} | "
            f"coord range: [{final_coords.min():.3f}, {final_coords.max():.3f}]"
        )

        return ReconstructionOutput(
            coordinates=final_coords,
            distance_matrix=dist_matrix,
            decoder_coords=decoder_coords,
            mds_coords=mds_coords,
            export_paths=export_paths,
        )
