"""
genofold.reconstruct.geometry
------------------------------
Distance geometry: converts predicted pairwise distances into
3D Cartesian coordinates via Multidimensional Scaling (MDS).

MDS finds coordinates X such that ||X_i - X_j|| ≈ D[i,j] for all pairs.

Two variants are implemented:
  1. Classical MDS (cMDS) — closed-form solution via eigendecomposition.
     Fast, deterministic, works well when the distance matrix is
     approximately Euclidean.

  2. Stress-minimization MDS — iterative gradient-based refinement
     that minimizes the stress (discrepancy between target and implied
     distances). More accurate when the distance matrix has noise.

The output of geometry.py is fused with the decoder output in model.py
to produce the final coordinates.
"""

from __future__ import annotations

import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


def classical_mds(
    distance_matrix: torch.Tensor,
    n_components: int = 3,
) -> torch.Tensor:
    """
    Classical Multidimensional Scaling (cMDS).

    Converts a pairwise distance matrix to Cartesian coordinates
    using the eigendecomposition of the double-centered Gram matrix.

    Parameters
    ----------
    distance_matrix:
        (M, M) symmetric pairwise distance matrix (non-negative).
    n_components:
        Number of output dimensions (default 3 for 3D space).

    Returns
    -------
    torch.Tensor
        (M, n_components) float32 coordinate matrix.
    """
    D = distance_matrix.double()
    M = D.shape[0]

    # Square distances
    D_sq = D ** 2

    # Double centering: B = -0.5 * J * D^2 * J
    # where J = I - (1/M) * 11^T is the centering matrix
    ones = torch.ones(M, M, dtype=torch.float64, device=D.device)
    J = torch.eye(M, dtype=torch.float64, device=D.device) - ones / M
    B = -0.5 * J @ D_sq @ J

    # Eigendecomposition of the Gram matrix
    # B = V * Lambda * V^T  →  X = V * sqrt(Lambda)
    eigenvalues, eigenvectors = torch.linalg.eigh(B)

    # Sort by descending eigenvalue
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalues (clip small negatives from numerical noise)
    eigenvalues = eigenvalues[:n_components].clamp(min=0.0)
    eigenvectors = eigenvectors[:, :n_components]

    # Coordinates: X = V * sqrt(Lambda)
    coords = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)

    logger.info(
        f"Classical MDS: {M} bins → ({M}, {n_components}) coordinates | "
        f"top-3 eigenvalues: {eigenvalues[:3].tolist()}"
    )
    return coords.float()


def stress_minimization_mds(
    distance_matrix: torch.Tensor,
    init_coords: torch.Tensor = None,
    n_components: int = 3,
    n_iter: int = 200,
    lr: float = 0.01,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Stress-minimization MDS via gradient descent.

    Minimizes the normalized stress:
        stress = sqrt( sum_{i<j} (d_ij - D_ij)^2 / sum_{i<j} D_ij^2 )

    where d_ij = ||X_i - X_j|| are implied distances and D_ij are targets.

    Parameters
    ----------
    distance_matrix:
        (M, M) target pairwise distance matrix.
    init_coords:
        (M, 3) initial coordinates. If None, uses classical MDS output.
    n_components:
        Output dimensions (default 3).
    n_iter:
        Maximum gradient descent iterations.
    lr:
        Learning rate.
    tol:
        Convergence tolerance on stress improvement.

    Returns
    -------
    torch.Tensor
        (M, n_components) optimized coordinate matrix.
    """
    D = distance_matrix.detach()
    M = D.shape[0]

    # Initialize from classical MDS if not provided
    if init_coords is None:
        coords = classical_mds(D, n_components).detach().clone()
    else:
        coords = init_coords.detach().clone()

    coords = coords.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([coords], lr=lr)

    # Mask for upper triangle (avoid double-counting pairs)
    mask = torch.triu(torch.ones(M, M, dtype=torch.bool, device=D.device), diagonal=1)
    D_upper = D[mask]
    D_sq_sum = (D_upper ** 2).sum().clamp(min=1e-8)

    prev_stress = float("inf")

    with torch.enable_grad():
        for iteration in range(n_iter):
            optimizer.zero_grad()

            # Implied distances from current coordinates
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)        # (M, M, 3)
            implied = torch.norm(diff, dim=-1)                       # (M, M)

            # Normalized stress loss
            residuals = (implied[mask] - D_upper) ** 2
            stress = torch.sqrt(residuals.sum() / D_sq_sum)

            stress.backward()
            optimizer.step()

            stress_val = stress.item()
            improvement = abs(prev_stress - stress_val)

            if iteration % 50 == 0:
                logger.info(f"Stress MDS iter {iteration:4d}: stress={stress_val:.6f}")

            if improvement < tol and iteration > 10:
                logger.info(
                    f"Stress MDS converged at iteration {iteration} "
                    f"(stress={stress_val:.6f}, improvement={improvement:.2e})"
                )
                break

            prev_stress = stress_val

    return coords.detach()


def fuse_coordinates(
    decoder_coords: torch.Tensor,
    mds_coords: torch.Tensor,
    decoder_weight: float = 0.5,
) -> torch.Tensor:
    """
    Fuse decoder and MDS coordinates with learned weighting.

    Both reconstruction pathways contribute to the final structure:
    - Decoder coordinates carry learned feature information
    - MDS coordinates carry distance geometry constraints

    Parameters
    ----------
    decoder_coords:
        (M, 3) coordinates from CoordinateDecoder.
    mds_coords:
        (M, 3) coordinates from MDS.
    decoder_weight:
        Weight for decoder coordinates [0, 1].
        MDS weight = 1 - decoder_weight.

    Returns
    -------
    torch.Tensor
        (M, 3) fused coordinate tensor.
    """
    # Align MDS coordinates to decoder coordinate space via Procrustes
    mds_aligned = procrustes_align(mds_coords, decoder_coords)

    # Weighted fusion
    mds_weight = 1.0 - decoder_weight
    fused = decoder_weight * decoder_coords + mds_weight * mds_aligned

    logger.info(
        f"Coordinate fusion: decoder_weight={decoder_weight:.2f}, "
        f"mds_weight={mds_weight:.2f}"
    )
    return fused


def procrustes_align(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Align source coordinates to target via Procrustes analysis.

    Finds the optimal rotation (and optionally scale) that minimizes
    ||target - source @ R||^2. This is needed because MDS coordinates
    are defined up to rotation/reflection, while decoder coordinates
    have a specific orientation.

    Parameters
    ----------
    source:
        (M, 3) coordinates to align (MDS output).
    target:
        (M, 3) reference coordinates (decoder output).

    Returns
    -------
    torch.Tensor
        (M, 3) rotated source coordinates.
    """
    # Center both
    src = source - source.mean(dim=0)
    tgt = target - target.mean(dim=0)

    # SVD of cross-covariance matrix
    H = src.T @ tgt                              # (3, 3)
    U, S, Vh = torch.linalg.svd(H)

    # Rotation matrix
    d = torch.det(Vh.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=source.device))
    R = Vh.T @ D @ U.T                          # (3, 3)

    # Apply rotation and re-center to target mean
    aligned = src @ R.T + target.mean(dim=0)
    return aligned
