"""
genofold.preprocess.hic
-----------------------
Hi-C contact matrix preprocessing:
  1. Filtering    — remove unmappable bins and diagonal artifacts
  2. Normalization — ICE (iterative correction & eigenvector decomposition)
  3. Denoising    — Gaussian smoothing + sparse threshold
  4. Distance correction — log-transform distance-dependent decay
  5. Output       — normalized float32 matrix ready for graph construction
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def filter_unmappable_bins(
    matrix: np.ndarray,
    coverage: Optional[np.ndarray] = None,
    min_coverage: float = 0.0,
    zero_fraction_threshold: float = 0.97,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify and mask unmappable or low-coverage bins.

    A bin is considered unmappable if:
      - More than `zero_fraction_threshold` of its row is zero, OR
      - Its NGS coverage is below `min_coverage` (if coverage is provided).

    Parameters
    ----------
    matrix:
        Raw Hi-C contact matrix (N, N).
    coverage:
        Optional per-bin NGS coverage array (N,).
    min_coverage:
        Minimum NGS reads per bin to be considered mappable.
    zero_fraction_threshold:
        Maximum fraction of zeros allowed in a row before it is masked.

    Returns
    -------
    (filtered_matrix, valid_mask)
        filtered_matrix : (N, N) matrix with masked bins zeroed out.
        valid_mask      : boolean array (N,), True = valid bin.
    """
    n = matrix.shape[0]
    zero_fractions = (matrix == 0).mean(axis=1)
    valid = zero_fractions < zero_fraction_threshold

    if coverage is not None:
        if len(coverage) != n:
            raise ValueError(
                f"Coverage length {len(coverage)} does not match matrix size {n}"
            )
        valid &= coverage >= min_coverage

    n_removed = (~valid).sum()
    logger.info(f"Bin filtering: {n_removed}/{n} bins masked ({n_removed/n:.1%})")

    filtered = matrix.copy()
    filtered[~valid, :] = 0.0
    filtered[:, ~valid] = 0.0
    return filtered, valid


def ice_normalization(
    matrix: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-5,
    ignore_diags: int = 2,
) -> np.ndarray:
    """
    Iterative Correction and Eigenvector decomposition (ICE) normalization.

    Removes systematic biases (GC content, mappability, fragment length)
    by iteratively normalizing rows and columns to equal marginals.

    Parameters
    ----------
    matrix:
        Raw contact matrix (N, N), symmetric.
    max_iter:
        Maximum number of correction iterations.
    tolerance:
        Convergence criterion on marginal variance.
    ignore_diags:
        Number of diagonals around the main diagonal to set to zero
        before normalization (removes self-ligation artifacts).

    Returns
    -------
    np.ndarray
        ICE-normalized matrix (N, N), dtype float32.
    """
    mat = matrix.copy().astype(np.float64)

    # Zero out near-diagonal artifacts
    for d in range(ignore_diags):
        np.fill_diagonal(mat[d:], 0.0)
        np.fill_diagonal(mat[:, d:], 0.0)

    # ICE iterative correction
    for iteration in range(max_iter):
        marginals = mat.sum(axis=1)
        # Avoid division by zero for empty bins
        marginals[marginals == 0] = 1.0
        bias = np.sqrt(marginals)
        mat /= bias[:, None]
        mat /= bias[None, :]

        variance = np.var(mat.sum(axis=1))
        if variance < tolerance:
            logger.info(f"ICE converged at iteration {iteration + 1} (variance={variance:.2e})")
            break
    else:
        logger.warning(f"ICE did not converge after {max_iter} iterations")

    # Rescale so mean non-zero contact is 1
    nonzero = mat[mat > 0]
    if len(nonzero) > 0:
        mat /= nonzero.mean()

    return mat.astype(np.float32)


def remove_distance_decay(
    matrix: np.ndarray,
    n_diags: int = 200,
) -> np.ndarray:
    """
    Remove the genomic-distance-dependent contact decay.

    Divides each diagonal by its mean, so that long-range contacts
    are on the same scale as short-range contacts. This isolates
    compartment and TAD structure from the trivial distance effect.

    Parameters
    ----------
    matrix:
        ICE-normalized contact matrix (N, N).
    n_diags:
        Number of off-diagonals to correct (corrects up to n_diags bins
        away from the diagonal). Diagonals beyond this are left unchanged.

    Returns
    -------
    np.ndarray
        Distance-corrected matrix, dtype float32.
    """
    mat = matrix.copy().astype(np.float64)
    n = mat.shape[0]

    for d in range(1, min(n_diags, n)):
        diag = np.diag(mat, d)
        mean_val = diag[diag > 0].mean() if (diag > 0).any() else 1.0
        if mean_val > 0:
            scale = 1.0 / mean_val
            idx = np.arange(n - d)
            mat[idx, idx + d] *= scale
            mat[idx + d, idx] *= scale

    return mat.astype(np.float32)


def denoise(
    matrix: np.ndarray,
    sigma: float = 1.0,
    sparse_threshold: float = 0.0,
) -> np.ndarray:
    """
    Smooth the contact matrix and threshold very low values.

    Parameters
    ----------
    matrix:
        Normalized contact matrix (N, N).
    sigma:
        Gaussian kernel standard deviation. Set to 0 to skip smoothing.
    sparse_threshold:
        Contacts below this value are set to zero (promotes sparsity
        and removes noise floor). Values in [0, 0.1] are typical.

    Returns
    -------
    np.ndarray
        Denoised matrix, dtype float32.
    """
    mat = matrix.copy()

    if sigma > 0:
        mat = gaussian_filter(mat, sigma=sigma).astype(np.float32)

    if sparse_threshold > 0:
        mat[mat < sparse_threshold] = 0.0

    # Preserve symmetry after smoothing
    mat = (mat + mat.T) / 2.0

    logger.info(
        f"Denoising: sigma={sigma}, threshold={sparse_threshold}, "
        f"sparsity={( mat == 0).mean():.1%}"
    )
    return mat


def log_transform(matrix: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    """
    Apply log1p transform: log(x + pseudocount).

    Compresses the dynamic range of contact frequencies, which tend
    to span several orders of magnitude.
    """
    return np.log1p(matrix + pseudocount - 1.0).astype(np.float32)


def preprocess_hic(
    matrix: np.ndarray,
    coverage: Optional[np.ndarray] = None,
    min_coverage: float = 0.0,
    zero_fraction_threshold: float = 0.97,
    ice_max_iter: int = 100,
    ice_tolerance: float = 1e-5,
    ignore_diags: int = 2,
    correct_distance_decay: bool = True,
    denoise_sigma: float = 1.0,
    sparse_threshold: float = 0.01,
    apply_log: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Hi-C preprocessing pipeline.

    Steps:
      1. Filter unmappable bins
      2. ICE normalization
      3. Distance decay correction (optional)
      4. Gaussian denoising + sparse threshold
      5. Log transform (optional)

    Parameters
    ----------
    matrix:
        Raw Hi-C contact matrix (N, N).
    coverage:
        Per-bin NGS coverage (N,). Used for additional bin filtering.
    ...all other params documented in individual functions above...

    Returns
    -------
    (processed_matrix, valid_mask)
        processed_matrix : (N, N) float32, ready for graph construction.
        valid_mask       : boolean (N,), True = valid bin.
    """
    logger.info(f"Hi-C preprocessing: input shape {matrix.shape}")

    mat, valid_mask = filter_unmappable_bins(
        matrix, coverage, min_coverage, zero_fraction_threshold
    )
    mat = ice_normalization(mat, max_iter=ice_max_iter, tolerance=ice_tolerance,
                            ignore_diags=ignore_diags)

    if correct_distance_decay:
        mat = remove_distance_decay(mat)

    mat = denoise(mat, sigma=denoise_sigma, sparse_threshold=sparse_threshold)

    if apply_log:
        mat = log_transform(mat)

    logger.info(
        f"Hi-C preprocessing complete: non-zero contacts = {(mat > 0).sum()}, "
        f"valid bins = {valid_mask.sum()}/{len(valid_mask)}"
    )
    return mat, valid_mask
