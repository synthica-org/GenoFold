"""
genofold.preprocess.ngs
-----------------------
NGS read preprocessing and per-bin feature extraction:
  1. Coverage normalization — RPKM and z-score scaling
  2. GC-content bias correction — regression-based
  3. Mappability correction
  4. Multi-track stacking — combine coverage across replicates or marks
  5. Output — per-bin feature matrix (N, F) ready for node feature assembly
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def normalize_rpkm(
    coverage: np.ndarray,
    bin_size: int,
    total_reads: Optional[float] = None,
) -> np.ndarray:
    """
    Reads Per Kilobase per Million mapped reads (RPKM) normalization.

    Accounts for sequencing depth and bin size so that coverage values
    are comparable across samples and experiments.

    Parameters
    ----------
    coverage:
        Raw per-bin read count array (N,).
    bin_size:
        Bin size in base pairs.
    total_reads:
        Total mapped reads. If None, uses sum of coverage array.

    Returns
    -------
    np.ndarray
        RPKM-normalized coverage array (N,), dtype float32.
    """
    if total_reads is None:
        total_reads = coverage.sum()
    if total_reads == 0:
        logger.warning("Total reads is zero — returning zero array")
        return np.zeros_like(coverage, dtype=np.float32)

    rpkm = (coverage / (bin_size / 1000.0)) / (total_reads / 1e6)
    return rpkm.astype(np.float32)


def zscore_normalize(coverage: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score standardize a coverage array.

    Useful when combining multiple NGS tracks with different scales.
    Bins with zero std (constant arrays) are returned as zeros.
    """
    mean = coverage.mean()
    std = coverage.std()
    if std < eps:
        return np.zeros_like(coverage, dtype=np.float32)
    return ((coverage - mean) / std).astype(np.float32)


def correct_gc_bias(
    coverage: np.ndarray,
    gc_content: np.ndarray,
) -> np.ndarray:
    """
    Remove GC-content-dependent coverage bias via linear regression.

    Fits a linear model: coverage ~ gc_content, then subtracts the
    fitted values to yield GC-corrected residuals.

    Parameters
    ----------
    coverage:
        Per-bin coverage array (N,).
    gc_content:
        Per-bin GC fraction in [0, 1] (N,).

    Returns
    -------
    np.ndarray
        GC-corrected coverage residuals (N,), dtype float32.
    """
    if len(gc_content) != len(coverage):
        raise ValueError("gc_content and coverage must have the same length")

    valid = coverage > 0
    if valid.sum() < 10:
        logger.warning("Too few non-zero bins for GC correction — skipping")
        return coverage.astype(np.float32)

    X = gc_content[valid].reshape(-1, 1)
    y = coverage[valid]
    model = LinearRegression().fit(X, y)

    fitted = model.predict(gc_content.reshape(-1, 1))
    corrected = coverage - fitted
    corrected = corrected - corrected.min()  # Shift to non-negative

    logger.info(f"GC bias correction R²={model.score(X, y):.3f}")
    return corrected.astype(np.float32)


def correct_mappability(
    coverage: np.ndarray,
    mappability: np.ndarray,
    min_mappability: float = 0.5,
) -> np.ndarray:
    """
    Correct coverage for mappability by dividing by the mappability score.

    Bins with mappability below `min_mappability` are set to zero
    (treated as unmappable).

    Parameters
    ----------
    coverage:
        Per-bin coverage array (N,).
    mappability:
        Per-bin mappability score in [0, 1] (N,).
    min_mappability:
        Bins below this threshold are zeroed out.

    Returns
    -------
    np.ndarray
        Mappability-corrected coverage (N,), dtype float32.
    """
    corrected = coverage.copy().astype(np.float64)
    mappable = mappability >= min_mappability

    corrected[~mappable] = 0.0
    corrected[mappable] /= np.maximum(mappability[mappable], 1e-8)

    n_masked = (~mappable).sum()
    logger.info(f"Mappability correction: {n_masked} bins masked (mappability < {min_mappability})")
    return corrected.astype(np.float32)


def stack_tracks(
    tracks: List[np.ndarray],
    normalize: bool = True,
) -> np.ndarray:
    """
    Stack multiple NGS coverage tracks into a per-bin feature matrix.

    Useful when multiple ChIP-seq marks, ATAC-seq, or replicate
    coverage arrays should all be included as separate node features.

    Parameters
    ----------
    tracks:
        List of coverage arrays, each of shape (N,).
    normalize:
        If True, z-score each track before stacking.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, len(tracks)), dtype float32.
    """
    if not tracks:
        raise ValueError("tracks list is empty")

    n = len(tracks[0])
    for i, t in enumerate(tracks):
        if len(t) != n:
            raise ValueError(f"Track {i} has length {len(t)}, expected {n}")

    processed = [zscore_normalize(t) if normalize else t.astype(np.float32) for t in tracks]
    return np.stack(processed, axis=1)  # (N, F)


def preprocess_ngs(
    coverage: np.ndarray,
    bin_size: int,
    gc_content: Optional[np.ndarray] = None,
    mappability: Optional[np.ndarray] = None,
    extra_tracks: Optional[List[np.ndarray]] = None,
    rpkm: bool = True,
    zscore: bool = True,
    min_mappability: float = 0.5,
) -> np.ndarray:
    """
    Full NGS preprocessing pipeline.

    Steps:
      1. RPKM normalization
      2. GC bias correction (if gc_content provided)
      3. Mappability correction (if mappability provided)
      4. Z-score normalization
      5. Stack with extra tracks (if provided)

    Parameters
    ----------
    coverage:
        Raw per-bin read count array (N,).
    bin_size:
        Bin size in base pairs (e.g. 40000 for 40 kb resolution).
    gc_content:
        Per-bin GC fraction (N,). Optional.
    mappability:
        Per-bin mappability score (N,). Optional.
    extra_tracks:
        Additional NGS tracks (e.g. ChIP-seq marks) to stack. Each (N,).
    rpkm:
        Whether to apply RPKM normalization.
    zscore:
        Whether to apply z-score normalization as final step.
    min_mappability:
        Mappability threshold for masking.

    Returns
    -------
    np.ndarray
        Per-bin feature matrix of shape (N, F), dtype float32.
        F = 1 + len(extra_tracks) if extra_tracks provided, else 1.
    """
    logger.info(f"NGS preprocessing: {len(coverage)} bins, bin_size={bin_size:,} bp")

    cov = coverage.copy().astype(np.float32)

    if rpkm:
        cov = normalize_rpkm(cov, bin_size)

    if gc_content is not None:
        cov = correct_gc_bias(cov, gc_content)

    if mappability is not None:
        cov = correct_mappability(cov, mappability, min_mappability)

    if zscore:
        cov = zscore_normalize(cov)

    # Stack main coverage + optional extra tracks
    all_tracks = [cov]
    if extra_tracks:
        for t in extra_tracks:
            proc = zscore_normalize(t)
            all_tracks.append(proc)

    features = np.stack(all_tracks, axis=1)  # (N, F)
    logger.info(f"NGS preprocessing complete: feature matrix shape {features.shape}")
    return features
