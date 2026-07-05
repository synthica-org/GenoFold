"""
experiments.explain_trained
----------------------------
Integrated-Gradients feature-group importance on a TRAINED GenoFold model.

The reviewer noted that attention/attribution maps from an untrained network
carry no meaning. Once the network is trained (experiments.train), the same
Integrated Gradients computation attributes the reconstructed structure to the
input modalities in a way that is interpretable. This module aggregates IG
attributions into the four feature groups (NGS coverage, k-mer sequence,
metadata, position).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from experiments.train import FeatureLayout


@torch.no_grad()
def _forward_scalar(model, data, x):
    d = data.clone(); d.x = x
    coords, _ = model(d)
    # target = spread of the structure (mean pairwise distance); differentiable proxy
    c = coords - coords.mean(0)
    return (c ** 2).sum()


def feature_group_importance(model, data, layout: FeatureLayout,
                             steps: int = 32) -> Dict[str, float]:
    """
    Integrated Gradients of the reconstructed-structure scale w.r.t. node
    features, from a zero baseline, aggregated (absolute) by feature group and
    normalised to sum to 1.
    """
    model.eval()
    dev = next(model.parameters()).device
    x = data.x.to(dev)
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    for s in range(1, steps + 1):
        xi = (baseline + (s / steps) * (x - baseline)).clone().requires_grad_(True)
        d = data.clone(); d.x = xi
        coords, _ = model(d)
        c = coords - coords.mean(0)
        target = (c ** 2).sum()
        grad = torch.autograd.grad(target, xi)[0]
        total_grad += grad.detach()
    ig = ((x - baseline) * total_grad / steps).abs().sum(0).cpu().numpy()  # (F,)

    groups = {}
    for g in ("ngs", "seq", "meta", "pos"):
        a, b = layout.slice_of(g)
        groups[g] = float(ig[a:b].sum())
    tot = sum(groups.values()) + 1e-12
    return {k: v / tot for k, v in groups.items()}
