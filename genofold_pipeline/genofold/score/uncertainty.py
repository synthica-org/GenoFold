"""
genofold.score.uncertainty
-----------------------------
Coordinate uncertainty estimation.

Two complementary approaches:
  1. Learned variance head — predicts a (M, 3) uncertainty (std dev)
     for each coordinate directly from embeddings. Fast, single forward pass.

  2. Monte Carlo dropout — runs the reconstruction decoder multiple times
     with dropout active (stochastic forward passes) and measures the
     spread of resulting coordinates. Captures epistemic uncertainty
     that the learned variance head might miss.

Final uncertainty = combination of both signals, expressed as an
ellipsoid radius (a single scalar per bin, in the same units as
the coordinate space) for easy visualization, plus the full
3-axis breakdown for detailed analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyOutput:
    """
    Coordinate uncertainty estimates.

    Attributes
    ----------
    std_xyz:
        (M, 3) predicted standard deviation per axis (learned variance head).
    mc_std_xyz:
        (M, 3) standard deviation from Monte Carlo dropout sampling.
    combined_std_xyz:
        (M, 3) combined uncertainty (root-sum-square of both sources).
    radius:
        (M,) scalar uncertainty radius per bin (norm of combined_std_xyz),
        useful for drawing confidence spheres/ellipsoids in visualizations.
    """
    std_xyz: torch.Tensor
    mc_std_xyz: torch.Tensor
    combined_std_xyz: torch.Tensor
    radius: torch.Tensor


class VarianceHead(nn.Module):
    """
    Learned per-bin coordinate variance predictor.

    Predicts log-variance for numerical stability, then exponentiates
    to get standard deviation. This is the same trick used in
    probabilistic regression / heteroscedastic uncertainty models.

    Parameters
    ----------
    hidden_dim:
        Embedding dimension from stage 3 (default 64).
    mlp_dim:
        Inner MLP dimension (default 128).
    """

    def __init__(self, hidden_dim: int = 64, mlp_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, 3),   # log-variance per axis
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"VarianceHead: {total:,} parameters")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict per-bin coordinate standard deviation.

        Parameters
        ----------
        embeddings:
            (M, hidden_dim) latent embeddings.

        Returns
        -------
        torch.Tensor
            (M, 3) standard deviation per axis, always positive.
        """
        log_var = self.net(embeddings)            # (M, 3)
        log_var = log_var.clamp(min=-10, max=10)  # numerical stability
        std = torch.exp(0.5 * log_var)
        return std


def monte_carlo_dropout_sampling(
    reconstruct_model: nn.Module,
    inference_output,
    n_samples: int = 20,
) -> torch.Tensor:
    """
    Run the reconstruction model multiple times with dropout active
    to estimate epistemic uncertainty via prediction spread.

    Parameters
    ----------
    reconstruct_model:
        The GenoFoldReconstructModel from stage 4 (will be set to train()
        mode temporarily to activate dropout, then restored).
    inference_output:
        InferenceOutput from stage 3 (embeddings stay fixed across samples).
    n_samples:
        Number of stochastic forward passes.

    Returns
    -------
    torch.Tensor
        (n_samples, M, 3) stacked coordinate predictions across samples.
    """
    was_training = reconstruct_model.training
    reconstruct_model.train()   # Activate dropout

    samples = []
    with torch.no_grad():
        for i in range(n_samples):
            output = reconstruct_model(inference_output, export_dir=None)
            samples.append(output.coordinates.clone())

    if not was_training:
        reconstruct_model.eval()  # Restore original mode

    stacked = torch.stack(samples, dim=0)   # (n_samples, M, 3)
    logger.info(f"MC dropout sampling: {n_samples} samples collected")
    return stacked


def estimate_uncertainty(
    variance_head: VarianceHead,
    embeddings: torch.Tensor,
    reconstruct_model: nn.Module = None,
    inference_output=None,
    n_mc_samples: int = 20,
    use_mc_dropout: bool = True,
) -> UncertaintyOutput:
    """
    Full uncertainty estimation combining learned variance + MC dropout.

    Parameters
    ----------
    variance_head:
        Trained VarianceHead module.
    embeddings:
        (M, hidden_dim) latent embeddings from stage 3.
    reconstruct_model:
        Stage 4 model, required if use_mc_dropout=True.
    inference_output:
        InferenceOutput from stage 3, required if use_mc_dropout=True.
    n_mc_samples:
        Number of MC dropout samples.
    use_mc_dropout:
        Whether to run the (more expensive) MC dropout estimation.

    Returns
    -------
    UncertaintyOutput
    """
    M = embeddings.shape[0]
    device = embeddings.device

    # Learned variance head
    std_xyz = variance_head(embeddings)   # (M, 3)

    # Monte Carlo dropout
    if use_mc_dropout and reconstruct_model is not None and inference_output is not None:
        mc_samples = monte_carlo_dropout_sampling(
            reconstruct_model, inference_output, n_samples=n_mc_samples
        )
        mc_std_xyz = mc_samples.std(dim=0)   # (M, 3)
    else:
        mc_std_xyz = torch.zeros(M, 3, device=device)
        logger.info("Skipping MC dropout — using learned variance only")

    # Combine via root-sum-square (assumes independent error sources)
    combined_std_xyz = torch.sqrt(std_xyz ** 2 + mc_std_xyz ** 2)

    # Scalar radius per bin for easy visualization
    radius = torch.norm(combined_std_xyz, dim=1)   # (M,)

    logger.info(
        f"Uncertainty estimation complete: "
        f"mean radius={radius.mean():.4f}, max radius={radius.max():.4f}"
    )

    return UncertaintyOutput(
        std_xyz=std_xyz,
        mc_std_xyz=mc_std_xyz,
        combined_std_xyz=combined_std_xyz,
        radius=radius,
    )
