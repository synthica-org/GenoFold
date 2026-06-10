"""
genofold.utils.io
-----------------
Loaders for all supported input formats:
  - Hi-C: .cool / .mcool (via cooler), raw .npy contact matrices
  - NGS:  pre-computed coverage arrays (.npy), BAM coverage extraction
  - Metadata: .csv / .tsv species + chromatin feature tables
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hi-C loaders
# ---------------------------------------------------------------------------

def load_cool(
    path: Union[str, Path],
    resolution: Optional[int] = None,
    chrom: Optional[str] = None,
) -> np.ndarray:
    """
    Load a Hi-C contact matrix from a .cool or .mcool file.

    Parameters
    ----------
    path:
        Path to the .cool or .mcool file.
    resolution:
        Required for .mcool files — selects the resolution zoom level (bp).
    chrom:
        If provided, return only the intra-chromosomal submatrix for this
        chromosome (e.g. 'chr1'). If None, returns the full genome-wide matrix.

    Returns
    -------
    np.ndarray
        Dense contact matrix of shape (N, N), dtype float32.
    """
    try:
        import cooler
    except ImportError:
        raise ImportError("Install 'cooler' to load .cool files: pip install cooler")

    path = str(path)
    if path.endswith(".mcool"):
        if resolution is None:
            raise ValueError("resolution must be provided for .mcool files")
        path = f"{path}::/resolutions/{resolution}"

    c = cooler.Cooler(path)
    logger.info(f"Loaded cooler: {c.info['genome-assembly'] if 'genome-assembly' in c.info else path}")

    if chrom is not None:
        matrix = c.matrix(balance=False).fetch(chrom).astype(np.float32)
    else:
        matrix = c.matrix(balance=False).fetch().astype(np.float32)

    logger.info(f"Contact matrix shape: {matrix.shape}")
    return matrix


def load_contact_matrix_npy(path: Union[str, Path]) -> np.ndarray:
    """Load a raw contact matrix saved as a .npy file."""
    mat = np.load(path).astype(np.float32)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {mat.shape}")
    logger.info(f"Loaded contact matrix from {path}, shape: {mat.shape}")
    return mat


# ---------------------------------------------------------------------------
# NGS coverage loaders
# ---------------------------------------------------------------------------

def load_coverage_npy(path: Union[str, Path]) -> np.ndarray:
    """
    Load a pre-computed per-bin coverage array saved as .npy.

    Shape: (N,) — one coverage value per genomic bin.
    """
    cov = np.load(path).astype(np.float32)
    if cov.ndim != 1:
        raise ValueError(f"Coverage array must be 1D, got shape {cov.shape}")
    logger.info(f"Loaded coverage array: {cov.shape[0]} bins")
    return cov


def extract_coverage_from_bam(
    bam_path: Union[str, Path],
    bin_size: int,
    chrom: str,
    chrom_length: int,
) -> np.ndarray:
    """
    Extract per-bin read coverage from a BAM file.

    Parameters
    ----------
    bam_path:
        Path to sorted, indexed .bam file.
    bin_size:
        Bin size in base pairs (must match Hi-C resolution).
    chrom:
        Chromosome name (e.g. 'chr1').
    chrom_length:
        Length of the chromosome in base pairs.

    Returns
    -------
    np.ndarray
        Coverage array of shape (n_bins,), dtype float32.
    """
    try:
        import pysam
    except ImportError:
        raise ImportError("Install 'pysam' to read BAM files: pip install pysam")

    n_bins = int(np.ceil(chrom_length / bin_size))
    coverage = np.zeros(n_bins, dtype=np.float32)

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(chrom):
            if read.is_unmapped or read.is_duplicate:
                continue
            bin_idx = read.reference_start // bin_size
            if bin_idx < n_bins:
                coverage[bin_idx] += 1.0

    logger.info(f"Extracted coverage: {n_bins} bins, total reads = {coverage.sum():.0f}")
    return coverage


# ---------------------------------------------------------------------------
# Metadata loaders
# ---------------------------------------------------------------------------

def load_metadata(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load species / chromatin metadata from a CSV or TSV file.

    Expected columns (at minimum):
        - species       : str  — species identifier (e.g. 'human', 'mouse')
        - bin_id        : int  — genomic bin index
        - gc_content    : float
        - mappability   : float
        - chromatin_state : str (optional)

    Returns
    -------
    pd.DataFrame sorted by bin_id.
    """
    path = Path(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)

    required = {"species", "bin_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata file missing required columns: {missing}")

    df = df.sort_values("bin_id").reset_index(drop=True)
    logger.info(f"Loaded metadata: {len(df)} bins, species: {df['species'].unique().tolist()}")
    return df
