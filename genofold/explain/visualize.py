"""
genofold.explain.visualize
-----------------------------
Visualization exports for stage 6 explainability.

Produces both:
  - Static PNG figures (matplotlib) — publication/poster-ready
  - Interactive HTML figures (plotly) — zoomable/hoverable for exploration

Visualizations produced:
  1. Feature importance bar chart (per feature group)
  2. Attention heatmap (GAT or Transformer)
  3. Influential loci track (1D plot along the genome)
  4. 3D structure colored by confidence (links stage 4/5 outputs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static (matplotlib) visualizations
# ---------------------------------------------------------------------------

def plot_feature_importance_static(
    group_attributions: Dict[str, torch.Tensor],
    output_path: str,
) -> str:
    """
    Bar chart of mean |attribution| per feature group, saved as PNG.

    Parameters
    ----------
    group_attributions:
        Dict from aggregate_feature_groups() — group name → (M,) tensor.
    output_path:
        Output .png file path.

    Returns
    -------
    str
        The output path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    names = list(group_attributions.keys())
    means = [group_attributions[n].mean().item() for n in names]
    stds = [group_attributions[n].std().item() for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
    ax.set_ylabel("Mean |Attribution| (Integrated Gradients)")
    ax.set_title("Feature Group Importance — GenoFold Stage 6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved feature importance plot: {output_path}")
    return output_path


def plot_attention_heatmap_static(
    attention_matrix: torch.Tensor,
    output_path: str,
    title: str = "Attention Heatmap",
) -> str:
    """
    Heatmap of an (M, M) attention matrix, saved as PNG.

    Parameters
    ----------
    attention_matrix:
        (M, M) attention weights (GAT or Transformer).
    output_path:
        Output .png file path.
    title:
        Plot title.

    Returns
    -------
    str
        The output path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    mat = attention_matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    ax.set_xlabel("Genomic bin (key)")
    ax.set_ylabel("Genomic bin (query)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved attention heatmap: {output_path}")
    return output_path


def plot_influential_loci_static(
    influence_score: torch.Tensor,
    confidence: Optional[torch.Tensor],
    output_path: str,
    top_k: int = 10,
) -> str:
    """
    1D track plot of influence score along the genome, with top-k
    influential bins highlighted. Optionally overlays confidence.

    Parameters
    ----------
    influence_score:
        (M,) influence score per bin from detect_influential_loci().
    confidence:
        Optional (M,) per-bin confidence scores from stage 5.
    output_path:
        Output .png file path.
    top_k:
        Number of top bins to highlight with markers.

    Returns
    -------
    str
        The output path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    score = influence_score.detach().cpu().numpy()
    M = len(score)
    x = np.arange(M)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(x, score, color="darkorange", linewidth=1.5, label="Influence score")
    ax1.fill_between(x, score, alpha=0.2, color="darkorange")

    top_idx = np.argsort(score)[-top_k:]
    ax1.scatter(top_idx, score[top_idx], color="red", zorder=5,
                s=40, label=f"Top-{top_k} influential bins")

    ax1.set_xlabel("Genomic bin index")
    ax1.set_ylabel("Influence score", color="darkorange")
    ax1.set_title("Influential Loci Along the Genome")

    if confidence is not None:
        ax2 = ax1.twinx()
        conf = confidence.detach().cpu().numpy()
        ax2.plot(x, conf, color="steelblue", linewidth=1.0, alpha=0.6, label="Confidence")
        ax2.set_ylabel("Confidence", color="steelblue")

    ax1.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved influential loci plot: {output_path}")
    return output_path


def plot_structure_confidence_static(
    coordinates: torch.Tensor,
    confidence: torch.Tensor,
    output_path: str,
) -> str:
    """
    3D scatter plot of the genome structure, colored by confidence.

    Parameters
    ----------
    coordinates:
        (M, 3) final XYZ coordinates from stage 4.
    confidence:
        (M,) confidence scores from stage 5.
    output_path:
        Output .png file path.

    Returns
    -------
    str
        The output path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    xyz = coordinates.detach().cpu().numpy()
    conf = confidence.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="gray", alpha=0.3, linewidth=1)

    scatter = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=conf, cmap="RdYlGn", s=40, vmin=0, vmax=1, edgecolor="black", linewidth=0.3,
    )
    fig.colorbar(scatter, ax=ax, label="Confidence", shrink=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Genome Structure — Colored by Confidence")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved 3D structure plot: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Interactive (plotly) visualizations
# ---------------------------------------------------------------------------

def plot_feature_importance_interactive(
    group_attributions: Dict[str, torch.Tensor],
    output_path: str,
) -> str:
    """Interactive bar chart of feature group importance (Plotly HTML)."""
    import plotly.graph_objects as go

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    names = list(group_attributions.keys())
    means = [group_attributions[n].mean().item() for n in names]
    stds = [group_attributions[n].std().item() for n in names]

    fig = go.Figure(data=[
        go.Bar(
            x=names, y=means,
            error_y=dict(type="data", array=stds),
            marker_color="mediumseagreen",
            text=[f"{m:.4f}" for m in means],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Feature Group Importance — GenoFold Stage 6",
        yaxis_title="Mean |Attribution| (Integrated Gradients)",
        template="plotly_white",
    )
    fig.write_html(output_path)

    logger.info(f"Saved interactive feature importance: {output_path}")
    return output_path


def plot_attention_heatmap_interactive(
    attention_matrix: torch.Tensor,
    output_path: str,
    title: str = "Attention Heatmap",
) -> str:
    """Interactive zoomable heatmap of attention weights (Plotly HTML)."""
    import plotly.graph_objects as go

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    mat = attention_matrix.detach().cpu().numpy()

    fig = go.Figure(data=go.Heatmap(
        z=mat, colorscale="Magma",
        colorbar=dict(title="Attention"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Genomic bin (key)",
        yaxis_title="Genomic bin (query)",
        template="plotly_white",
    )
    fig.write_html(output_path)

    logger.info(f"Saved interactive attention heatmap: {output_path}")
    return output_path


def plot_influential_loci_interactive(
    influence_score: torch.Tensor,
    confidence: Optional[torch.Tensor],
    output_path: str,
    top_k: int = 10,
) -> str:
    """Interactive influence track with hoverable top-k bin markers (Plotly HTML)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    score = influence_score.detach().cpu().numpy()
    M = len(score)
    x = list(range(M))
    top_idx = np.argsort(score)[-top_k:]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=x, y=score, mode="lines", name="Influence score",
        line=dict(color="darkorange"), fill="tozeroy",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=top_idx.tolist(), y=score[top_idx].tolist(), mode="markers",
        name=f"Top-{top_k} influential bins",
        marker=dict(color="red", size=10, symbol="star"),
    ), secondary_y=False)

    if confidence is not None:
        conf = confidence.detach().cpu().numpy()
        fig.add_trace(go.Scatter(
            x=x, y=conf, mode="lines", name="Confidence",
            line=dict(color="steelblue", width=1, dash="dot"),
        ), secondary_y=True)

    fig.update_layout(
        title="Influential Loci Along the Genome",
        xaxis_title="Genomic bin index",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Influence score", secondary_y=False)
    fig.update_yaxes(title_text="Confidence", secondary_y=True)
    fig.write_html(output_path)

    logger.info(f"Saved interactive influential loci plot: {output_path}")
    return output_path


def plot_structure_confidence_interactive(
    coordinates: torch.Tensor,
    confidence: torch.Tensor,
    output_path: str,
) -> str:
    """Interactive rotatable 3D structure colored by confidence (Plotly HTML)."""
    import plotly.graph_objects as go

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    xyz = coordinates.detach().cpu().numpy()
    conf = confidence.detach().cpu().numpy()

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="lines", line=dict(color="lightgray", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=5, color=conf, colorscale="RdYlGn", cmin=0, cmax=1,
            colorbar=dict(title="Confidence"),
            line=dict(color="black", width=0.5),
        ),
        text=[f"Bin {i}<br>Confidence: {c:.3f}" for i, c in enumerate(conf)],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title="3D Genome Structure — Colored by Confidence",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        template="plotly_white",
    )
    fig.write_html(output_path)

    logger.info(f"Saved interactive 3D structure: {output_path}")
    return output_path
