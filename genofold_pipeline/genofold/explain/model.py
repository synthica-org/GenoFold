"""
genofold.explain.model
-------------------------
GenoFold Stage 6: Full explainability pipeline.

Combines:
  1. Integrated Gradients — per-feature attribution
  2. GAT + Transformer attention extraction — influential loci
  3. Static (PNG) + Interactive (HTML) visualization exports

This is the final stage. Its output is a complete explainability
report — the figures and statistics you'd present at a conference
to justify why GenoFold's predictions should be trusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .attribution import integrated_gradients, aggregate_feature_groups, default_feature_groups
from .attention import extract_gat_attention, extract_transformer_attention, detect_influential_loci
from .visualize import (
    plot_feature_importance_static, plot_feature_importance_interactive,
    plot_attention_heatmap_static, plot_attention_heatmap_interactive,
    plot_influential_loci_static, plot_influential_loci_interactive,
    plot_structure_confidence_static, plot_structure_confidence_interactive,
)

logger = logging.getLogger(__name__)


@dataclass
class ExplanationOutput:
    """
    Output of stage 6 explainability.

    Attributes
    ----------
    feature_attributions:
        (M, F) raw Integrated Gradients attribution scores.
    group_attributions:
        Dict mapping feature group name → (M,) summed |attribution|.
    transformer_attention:
        (M, M) Transformer self-attention matrix (last layer, averaged heads).
    gat_attention_weights:
        (E, heads) GAT attention weights per edge.
    influence_score:
        (M,) influential loci score per bin.
    figure_paths:
        Dict of all generated figure file paths (static + interactive).
    """
    feature_attributions: torch.Tensor
    group_attributions: Dict[str, torch.Tensor]
    transformer_attention: torch.Tensor
    gat_attention_weights: torch.Tensor
    influence_score: torch.Tensor
    figure_paths: Dict[str, str]


class GenoFoldExplainModel:
    """
    Stage 6: Explainability — feature attribution, attention analysis,
    and visualization.

    Unlike stages 3-5, this is not an nn.Module since it doesn't have
    its own trainable parameters — it analyzes the already-trained
    stage 3/4/5 models.

    Parameters
    ----------
    ig_steps:
        Number of Integrated Gradients interpolation steps (default 50).
    top_k_loci:
        Number of top influential bins to highlight (default 10).
    """

    def __init__(self, ig_steps: int = 50, top_k_loci: int = 10):
        self.ig_steps = ig_steps
        self.top_k_loci = top_k_loci
        logger.info(
            f"GenoFoldExplainModel ready | ig_steps={ig_steps}, top_k_loci={top_k_loci}"
        )

    def explain(
        self,
        infer_model: nn.Module,
        recon_model: nn.Module,
        data,
        inference_output,
        reconstruction_output,
        score_output=None,
        output_dir: str = "outputs/explain",
        feature_groups: Optional[Dict[str, tuple]] = None,
    ) -> ExplanationOutput:
        """
        Run the full stage 6 explainability pipeline.

        Parameters
        ----------
        infer_model:
            Trained GenoFoldInferenceModel from stage 3.
        recon_model:
            Trained GenoFoldReconstructModel from stage 4.
        data:
            PyG Data object from stage 2 (provides x, edge_index, edge_attr).
        inference_output:
            InferenceOutput from stage 3.
        reconstruction_output:
            ReconstructionOutput from stage 4.
        score_output:
            Optional ScoringOutput from stage 5 (for confidence overlay).
        output_dir:
            Directory to save all figures.
        feature_groups:
            Optional custom feature group ranges. Defaults to
            default_feature_groups() matching stage 2's assembly order.

        Returns
        -------
        ExplanationOutput
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Stage 6 explainability: output_dir={output_dir}")

        # ── 1. Integrated Gradients ──────────────────────────────────────────
        logger.info("  Computing Integrated Gradients...")

        def forward_fn(x_input):
            return infer_model.gnn(x_input, data.edge_index, data.edge_attr)

        def target_fn(gnn_output):
            return gnn_output.norm(dim=1).sum()

        attributions = integrated_gradients(
            forward_fn=forward_fn,
            inputs=data.x,
            target_fn=target_fn,
            n_steps=self.ig_steps,
        )

        if feature_groups is None:
            # Infer split sizes from total dim; assumes [ngs(1) | kmer | metadata | pos(1)]
            total_dim = data.x.shape[1]
            kmer_dim = 136  # default k=4 canonical vocab size
            metadata_dim = total_dim - 1 - kmer_dim - 1
            feature_groups = default_feature_groups(
                ngs_dim=1, kmer_dim=kmer_dim, metadata_dim=max(metadata_dim, 0), position_dim=1
            )

        group_attributions = aggregate_feature_groups(attributions, feature_groups)

        # ── 2. Attention extraction ──────────────────────────────────────────
        logger.info("  Extracting attention weights...")

        gat_edge_index, gat_attn = extract_gat_attention(
            infer_model.gnn, data.x, data.edge_index, data.edge_attr
        )

        transformer_attn = extract_transformer_attention(
            infer_model.transformer,
            inference_output.gnn_embeddings,
            data.pos,
            data.species_ids,
        )

        # ── 3. Influential loci ──────────────────────────────────────────────
        logger.info("  Detecting influential loci...")
        influence_score = detect_influential_loci(transformer_attn, top_k=self.top_k_loci)

        # ── 4. Visualization exports (static + interactive) ──────────────────
        logger.info("  Generating visualizations...")
        confidence = score_output.bin_confidence if score_output is not None else None

        figure_paths = {}

        figure_paths["feature_importance_png"] = plot_feature_importance_static(
            group_attributions, f"{output_dir}/feature_importance.png"
        )
        figure_paths["feature_importance_html"] = plot_feature_importance_interactive(
            group_attributions, f"{output_dir}/feature_importance.html"
        )

        figure_paths["attention_heatmap_png"] = plot_attention_heatmap_static(
            transformer_attn, f"{output_dir}/transformer_attention.png",
            title="Transformer Self-Attention (Last Layer)",
        )
        figure_paths["attention_heatmap_html"] = plot_attention_heatmap_interactive(
            transformer_attn, f"{output_dir}/transformer_attention.html",
            title="Transformer Self-Attention (Last Layer)",
        )

        figure_paths["influential_loci_png"] = plot_influential_loci_static(
            influence_score, confidence, f"{output_dir}/influential_loci.png",
            top_k=self.top_k_loci,
        )
        figure_paths["influential_loci_html"] = plot_influential_loci_interactive(
            influence_score, confidence, f"{output_dir}/influential_loci.html",
            top_k=self.top_k_loci,
        )

        if confidence is not None:
            figure_paths["structure_confidence_png"] = plot_structure_confidence_static(
                reconstruction_output.coordinates, confidence,
                f"{output_dir}/structure_confidence.png",
            )
            figure_paths["structure_confidence_html"] = plot_structure_confidence_interactive(
                reconstruction_output.coordinates, confidence,
                f"{output_dir}/structure_confidence.html",
            )

        logger.info(
            f"Stage 6 complete ✓ | {len(figure_paths)} figures saved to {output_dir}"
        )

        return ExplanationOutput(
            feature_attributions=attributions,
            group_attributions=group_attributions,
            transformer_attention=transformer_attn,
            gat_attention_weights=gat_attn,
            influence_score=influence_score,
            figure_paths=figure_paths,
        )
