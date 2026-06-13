"""
pipeline.py
-----------
GenoFold end-to-end pipeline orchestrator.

Usage
-----
  from pipeline import GenoFoldPipeline

  pipeline = GenoFoldPipeline(bin_size=40_000)
  data = pipeline.run(
      contact_matrix=matrix,
      coverage=cov,
      sequences=seqs,
      metadata_df=df,
  )

Stages currently implemented:
  1. Preprocess  — Hi-C normalization, NGS correction, metadata encoding
  2. Represent   — graph construction, sequence embeddings, PyG Data assembly
  3. Infer       — GNN encoder + Transformer for latent embeddings

Stages 4-6 (Reconstruct, Score, Explain) will be added incrementally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from genofold.preprocess import preprocess_hic, preprocess_metadata, preprocess_ngs
from genofold.represent import build_pyg_data, compute_sequence_embeddings
from genofold.infer import GenoFoldInferenceModel, InferenceOutput, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("genofold.pipeline")


@dataclass
class PipelineConfig:
    """All hyperparameters for stages 1, 2, and 3."""

    # -- Resolution --
    bin_size: int = 40_000          # Hi-C resolution in bp

    # -- Hi-C preprocessing --
    zero_fraction_threshold: float = 0.8
    min_coverage: float = 0.0
    ice_max_iter: int = 100
    ice_tolerance: float = 1e-5
    ignore_diags: int = 2
    correct_distance_decay: bool = True
    denoise_sigma: float = 1.0
    sparse_threshold: float = 0.01
    apply_log: bool = True

    # -- NGS preprocessing --
    rpkm: bool = True
    zscore: bool = True
    min_mappability: float = 0.5
    extra_ngs_tracks: List[np.ndarray] = field(default_factory=list)

    # -- Sequence embeddings --
    kmer_k: int = 4
    use_dnabert: bool = False
    dnabert_model_path: Optional[str] = None
    dnabert_device: str = "cpu"

    # -- Graph construction --
    contact_threshold: float = 0.0
    top_k_edges: Optional[int] = 50
    self_loops: bool = False


class GenoFoldPipeline:
    """
    Orchestrates stages 1 (Preprocess) and 2 (Represent).

    Parameters
    ----------
    config:
        PipelineConfig with all hyperparameters.
        Can also pass bin_size and individual kwargs that override defaults.
    """

    def __init__(self, config: Optional[PipelineConfig] = None, **kwargs):
        if config is None:
            config = PipelineConfig(**{k: v for k, v in kwargs.items()
                                       if hasattr(PipelineConfig, k)})
        self.config = config
        logger.info(f"GenoFoldPipeline initialized | bin_size={config.bin_size:,} bp")

    def run(
        self,
        contact_matrix: np.ndarray,
        coverage: np.ndarray,
        sequences: List[str],
        metadata_df: pd.DataFrame,
        gc_content: Optional[np.ndarray] = None,
        mappability: Optional[np.ndarray] = None,
        chrom_start: int = 0,
    ):
        """
        Run stages 1 and 2 and return a PyG Data object.

        Parameters
        ----------
        contact_matrix:
            Raw Hi-C contact matrix (N, N).
        coverage:
            Per-bin NGS read counts (N,).
        sequences:
            Per-bin DNA sequences (N strings of length ~bin_size).
        metadata_df:
            DataFrame with 'species', 'bin_id', and optional feature columns.
        gc_content:
            Per-bin GC fraction (N,). Optional — used for NGS bias correction.
        mappability:
            Per-bin mappability (N,). Optional.
        chrom_start:
            Chromosome start coordinate for position encoding.

        Returns
        -------
        torch_geometric.data.Data
            Fully assembled graph ready for stage 3 (GNN encoder).
        """
        cfg = self.config
        n_bins = contact_matrix.shape[0]
        logger.info(f"Pipeline start: {n_bins} bins")

        # ── Stage 1: Preprocess ──────────────────────────────────────────────
        logger.info("── Stage 1: Preprocess ──")

        hic_matrix, valid_mask = preprocess_hic(
            matrix=contact_matrix,
            coverage=coverage,
            min_coverage=cfg.min_coverage,
            zero_fraction_threshold=cfg.zero_fraction_threshold,
            ice_max_iter=cfg.ice_max_iter,
            ice_tolerance=cfg.ice_tolerance,
            ignore_diags=cfg.ignore_diags,
            correct_distance_decay=cfg.correct_distance_decay,
            denoise_sigma=cfg.denoise_sigma,
            sparse_threshold=cfg.sparse_threshold,
            apply_log=cfg.apply_log,
        )

        ngs_features = preprocess_ngs(
            coverage=coverage,
            bin_size=cfg.bin_size,
            gc_content=gc_content,
            mappability=mappability,
            extra_tracks=cfg.extra_ngs_tracks if cfg.extra_ngs_tracks else None,
            rpkm=cfg.rpkm,
            zscore=cfg.zscore,
            min_mappability=cfg.min_mappability,
        )

        species_ids, metadata_features, species_to_id = preprocess_metadata(df=metadata_df)
        logger.info(f"Species vocabulary: {species_to_id}")

        # ── Stage 2: Represent ───────────────────────────────────────────────
        logger.info("── Stage 2: Represent ──")

        seq_embeddings = compute_sequence_embeddings(
            sequences=sequences,
            kmer_k=cfg.kmer_k,
            use_dnabert=cfg.use_dnabert,
            dnabert_model_path=cfg.dnabert_model_path,
            dnabert_device=cfg.dnabert_device,
        )

        data = build_pyg_data(
            contact_matrix=hic_matrix,
            valid_mask=valid_mask,
            ngs_features=ngs_features,
            sequence_embeddings=seq_embeddings,
            metadata_features=metadata_features,
            species_ids=species_ids,
            bin_size=cfg.bin_size,
            chrom_start=chrom_start,
            contact_threshold=cfg.contact_threshold,
            top_k_edges=cfg.top_k_edges,
            self_loops=cfg.self_loops,
        )

        logger.info("Pipeline stages 1-2 complete ✓")
        return data

    def infer(self, data, hidden_dim: int = 64):
        """
        Run stage 3 (Infer) on a PyG Data object from stage 2.

        Parameters
        ----------
        data:
            PyG Data object returned by pipeline.run().
        hidden_dim:
            Latent embedding dimension (default 64).

        Returns
        -------
        InferenceOutput
            Contains .embeddings (M, hidden_dim) and .gnn_embeddings (M, hidden_dim).
        """
        logger.info("── Stage 3: Infer ──")

        device = get_device()
        input_dim = data.x.shape[1]

        model = GenoFoldInferenceModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=data.edge_attr.shape[1],
        ).to(device)
        model.eval()

        # Move data to device
        data = data.to(device)

        with torch.no_grad():
            output = model(data)

        logger.info(
            f"Stage 3 complete ✓ | "
            f"embeddings shape: {output.embeddings.shape}"
        )
        return output, model
