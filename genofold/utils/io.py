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


def load_hic(
    path: Union[str, Path],
    resolution: int,
    chrom: str,
    normalization: str = "KR",
) -> np.ndarray:
    """
    Load a Hi-C contact matrix from a Juicer .hic file.

    Requires the 'hic-straw' package (imports as 'hicstraw'):
        pip install hic-straw

    Parameters
    ----------
    path:
        Path to the .hic file (local path or URL).
    resolution:
        Resolution in bp. Must already exist in the .hic file
        (common values: 2500000, 1000000, 500000, 100000, 50000,
        25000, 10000, 5000).
    chrom:
        Chromosome name (e.g. 'chr1' or '1', depending on the file's
        naming convention — check with hicstraw.HiCFile(path).getChromosomes()).
    normalization:
        Normalization method: 'NONE', 'VC', 'VC_SQRT', 'KR', or 'SCALE'.
        'KR' (Knight-Ruiz) is the standard choice, equivalent to ICE.

    Returns
    -------
    np.ndarray
        Dense contact matrix of shape (N, N), dtype float32.
    """
    try:
        import hicstraw
    except ImportError:
        raise ImportError(
            "Install hic-straw to load .hic files: pip install hic-straw"
        )

    hic_file = hicstraw.HiCFile(str(path))
    available_chroms = [c.name for c in hic_file.getChromosomes()]
    logger.info(
        f"Loaded .hic file: genome={hic_file.getGenomeID()}, "
        f"chromosomes={available_chroms}"
    )

    # Auto-strip 'chr' prefix if the file doesn't use it (common in hg19 .hic files)
    if chrom not in available_chroms:
        stripped = chrom.replace("chr", "")
        if stripped in available_chroms:
            logger.warning(
                f"Chromosome '{chrom}' not found, but '{stripped}' is. "
                f"Using '{stripped}' instead. This file does not use 'chr' prefixes."
            )
            chrom = stripped
        else:
            raise ValueError(
                f"Chromosome '{chrom}' not found. Available: {available_chroms}"
            )

    chrom_obj = next(
        (c for c in hic_file.getChromosomes() if c.name == chrom), None
    )

    # Try normalizations in order of preference, falling back gracefully
    # if the requested one wasn't computed for this resolution
    norm_candidates = [normalization, "KR", "SCALE", "VC_SQRT", "VC", "NONE"]
    norm_candidates = list(dict.fromkeys(norm_candidates))  # de-duplicate, keep order

    matrix_zoom_data = None
    used_norm = None
    for norm in norm_candidates:
        try:
            matrix_zoom_data = hic_file.getMatrixZoomData(
                chrom, chrom, "observed", norm, "BP", resolution
            )
            used_norm = norm
            break
        except Exception as e:
            logger.warning(f"Normalization '{norm}' unavailable at {resolution:,} bp ({e}) — trying next...")

    if matrix_zoom_data is None:
        raise ValueError(
            f"No normalization vector available for chrom={chrom} at "
            f"resolution={resolution:,} bp. Try a different --res "
            f"(e.g. 50000, 250000, 1000000) or check available resolutions "
            f"with hicstraw.HiCFile(path).getResolutions()."
        )

    if used_norm != normalization:
        logger.warning(
            f"Requested normalization '{normalization}' was unavailable — "
            f"used '{used_norm}' instead."
        )

    n_bins = int(np.ceil(chrom_obj.length / resolution))

    # getRecordsAsMatrix returns a dense numpy array directly
    matrix = matrix_zoom_data.getRecordsAsMatrix(
        0, chrom_obj.length, 0, chrom_obj.length
    ).astype(np.float32)

    # Replace NaNs (common in sparse/low-coverage regions) with 0
    matrix = np.nan_to_num(matrix, nan=0.0)

    logger.info(
        f"Contact matrix loaded from .hic: shape={matrix.shape}, "
        f"resolution={resolution:,} bp, normalization={normalization}"
    )
    return matrix


def load_bigwig_coverage(
    path: Union[str, Path],
    chrom: str,
    chrom_start: int,
    n_bins: int,
    bin_size: int,
) -> np.ndarray:
    """
    Extract per-bin coverage from a bigWig signal file (e.g. ENCODE
    DNase-seq, ChIP-seq, or RNA-seq coverage tracks).

    Requires the 'pyBigWig' package:
        pip install pyBigWig

    Parameters
    ----------
    path:
        Path to the .bigWig file.
    chrom:
        Chromosome name matching the bigWig's naming convention
        (check with pyBigWig.open(path).chroms() if unsure — ENCODE
        bigWig files typically use 'chr22' style naming, matching UCSC).
    chrom_start:
        Start coordinate of the first bin in bp.
    n_bins:
        Number of bins to extract.
    bin_size:
        Bin size in bp (should match your Hi-C resolution).

    Returns
    -------
    np.ndarray
        (n_bins,) float32 array of mean signal per bin.
    """
    try:
        import pyBigWig
    except ImportError:
        raise ImportError(
            "Install pyBigWig to read bigWig coverage files: pip install pyBigWig"
        )

    bw = pyBigWig.open(str(path))

    available_chroms = list(bw.chroms().keys())
    if chrom not in available_chroms:
        # Try stripping/adding 'chr' prefix to match naming convention
        alt = chrom.replace("chr", "") if "chr" in chrom else f"chr{chrom}"
        if alt in available_chroms:
            logger.warning(
                f"Chromosome '{chrom}' not found in bigWig, using '{alt}' instead."
            )
            chrom = alt
        else:
            raise ValueError(
                f"Chromosome '{chrom}' not found in bigWig. "
                f"Available: {available_chroms[:10]}..."
            )

    chrom_len = bw.chroms()[chrom]
    coverage = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        start = chrom_start + i * bin_size
        end = min(start + bin_size, chrom_len)
        if start >= chrom_len:
            break
        mean_val = bw.stats(chrom, start, end, type="mean")[0]
        coverage[i] = mean_val if mean_val is not None else 0.0

    bw.close()

    logger.info(
        f"Loaded bigWig coverage: {n_bins} bins from {chrom}, "
        f"mean={coverage.mean():.3f}, nonzero={np.count_nonzero(coverage)}/{n_bins}"
    )
    return coverage


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
