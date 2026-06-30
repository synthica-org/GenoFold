"""
genofold.explain.attribution
------------------------------
Feature importance via Integrated Gradients.

Determines which input features (NGS coverage, k-mer sequence content,
metadata) most influenced each bin's final predicted position.

Integrated Gradients (Sundararajan et al. 2017) works by:
  1. Defining a baseline input (all zeros — "no information")
  2. Interpolating linearly between baseline and actual input
  3. Accumulating gradients of the output w.r.t. input at each
     interpolation step
  4. The integral (approximated by a Riemann sum) gives attribution
     scores per input feature

This is preferred over simple gradients because it satisfies
"completeness" — attributions sum up to the actual output difference,
giving more trustworthy, theoretically grounded explanations.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


def integrated_gradients(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Compute Integrated Gradients attribution for a given forward function.

    Parameters
    ----------
    forward_fn:
        Function mapping input tensor → model output (e.g. embeddings
        or coordinates). Must be differentiable w.r.t. `inputs`.
    inputs:
        (M, F) input tensor to attribute (e.g. node features data.x).
    target_fn:
        Function mapping model output → scalar (or per-bin) target
        to explain (e.g. lambda out: out.norm(dim=1).sum() for
        "overall structural displacement").
    baseline:
        (M, F) reference input representing "no information".
        Defaults to all-zeros tensor (same shape as inputs).
    n_steps:
        Number of interpolation steps for the Riemann sum approximation.
        More steps = more accurate but slower.

    Returns
    -------
    torch.Tensor
        (M, F) attribution scores, same shape as inputs.
        Positive = feature pushed output up; negative = pushed down.
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    inputs = inputs.detach()
    baseline = baseline.detach()

    # Interpolation steps: alpha from 0 (baseline) to 1 (actual input)
    alphas = torch.linspace(0, 1, n_steps, device=inputs.device)

    total_gradients = torch.zeros_like(inputs)

    for alpha in alphas:
        interpolated = baseline + alpha * (inputs - baseline)
        interpolated = interpolated.requires_grad_(True)

        output = forward_fn(interpolated)
        target = target_fn(output)

        grad = torch.autograd.grad(
            target, interpolated,
            retain_graph=False, create_graph=False,
        )[0]

        total_gradients += grad.detach()

    avg_gradients = total_gradients / n_steps
    attributions = (inputs - baseline) * avg_gradients

    logger.info(
        f"Integrated Gradients: {n_steps} steps | "
        f"attribution range=[{attributions.min():.4f}, {attributions.max():.4f}]"
    )
    return attributions


def aggregate_feature_groups(
    attributions: torch.Tensor,
    feature_groups: dict,
) -> dict:
    """
    Aggregate per-feature attributions into named feature groups.

    Parameters
    ----------
    attributions:
        (M, F) attribution scores from integrated_gradients().
    feature_groups:
        Dict mapping group name → (start_idx, end_idx) column range.
        E.g. {'ngs': (0, 1), 'kmer': (1, 137), 'metadata': (137, 154),
              'position': (154, 155)}

    Returns
    -------
    dict
        Maps group name → (M,) tensor of summed absolute attribution
        per bin for that feature group.
    """
    result = {}
    for name, (start, end) in feature_groups.items():
        group_attr = attributions[:, start:end].abs().sum(dim=1)
        result[name] = group_attr
        logger.info(
            f"  Feature group '{name}' ({end - start} dims): "
            f"mean |attribution| = {group_attr.mean():.4f}"
        )
    return result


def default_feature_groups(
    ngs_dim: int = 1,
    kmer_dim: int = 136,
    metadata_dim: int = 17,
    position_dim: int = 1,
) -> dict:
    """
    Build the default feature group index ranges matching stage 2's
    node feature assembly order: [NGS | k-mer | metadata | position].

    Parameters
    ----------
    ngs_dim, kmer_dim, metadata_dim, position_dim:
        Dimensions of each feature block (must match your actual
        pipeline configuration — check data.x.shape[1] components).

    Returns
    -------
    dict
        Feature group name → (start_idx, end_idx).
    """
    groups = {}
    offset = 0
    for name, dim in [
        ("ngs", ngs_dim),
        ("kmer", kmer_dim),
        ("metadata", metadata_dim),
        ("position", position_dim),
    ]:
        groups[name] = (offset, offset + dim)
        offset += dim
    return groups
