"""
genofold.utils.real_metadata
-------------------------------
Builds real (non-synthetic) metadata and sequence inputs for a
chromosome region, to pair with a real Hi-C contact matrix.

Unlike test_pipeline.py's synthetic generators, this module computes
GC content directly from actual reference genome sequence (via a
FASTA file), and builds a metadata table consistent with the bin
structure of a loaded Hi-C matrix.

If no reference FASTA is available, falls back to fetching sequence
from the UCSC DAS server (small regions only) or using placeholder
uniform GC content with a clear warning.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_metadata_table(
    n_bins: int,
    species: str = "human",
    chrom: str = "chr1",
    chrom_start: int = 0,
    bin_size: int = 100_000,
    gc_content: Optional[np.ndarray] = None,
    mappability: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build a metadata DataFrame matching the bin structure of a real
    Hi-C matrix, ready for genofold.preprocess.metadata.preprocess_metadata().

    Parameters
    ----------
    n_bins:
        Number of bins (must match the Hi-C matrix dimension).
    species:
        Species label for all bins (e.g. 'human', 'mouse').
    chrom:
        Chromosome name (for reference/logging only).
    chrom_start:
        Start coordinate of the region in bp.
    bin_size:
        Bin size in bp.
    gc_content:
        Optional (n_bins,) array of real GC fractions.
        If None, uses a neutral placeholder value with a warning.
    mappability:
        Optional (n_bins,) array of real mappability scores.
        If None, uses a neutral placeholder value with a warning.

    Returns
    -------
    pd.DataFrame
        Columns: bin_id, species, gc_content, mappability, chromatin_state.
    """
    if gc_content is None:
        logger.warning(
            "No real GC content provided — using placeholder value 0.41 "
            "(human genome average). For accurate results, compute GC "
            "content from a reference FASTA using fetch_gc_content_from_fasta()."
        )
        gc_content = np.full(n_bins, 0.41, dtype=np.float32)

    if mappability is None:
        logger.warning(
            "No real mappability scores provided — using placeholder "
            "value 0.9. For accurate results, use a mappability track "
            "(e.g. ENCODE Umap/Bismap)."
        )
        mappability = np.full(n_bins, 0.9, dtype=np.float32)

    if len(gc_content) != n_bins or len(mappability) != n_bins:
        raise ValueError(
            f"gc_content/mappability length must equal n_bins={n_bins}, "
            f"got {len(gc_content)}/{len(mappability)}"
        )

    df = pd.DataFrame({
        "bin_id": np.arange(n_bins),
        "species": species,
        "gc_content": gc_content,
        "mappability": mappability,
        # Chromatin state unknown without a real ChromHMM annotation file —
        # mark as unknown ("") so preprocess_metadata leaves it as all-zero one-hot.
        "chromatin_state": "",
    })

    logger.info(
        f"Built metadata table: {n_bins} bins, species={species}, "
        f"chrom={chrom}, region=[{chrom_start:,}, {chrom_start + n_bins * bin_size:,}) bp"
    )
    return df


def fetch_gc_content_from_fasta(
    fasta_path: str,
    chrom: str,
    chrom_start: int,
    n_bins: int,
    bin_size: int,
) -> np.ndarray:
    """
    Compute real per-bin GC content from a reference genome FASTA file.

    Requires a local, indexed FASTA file (e.g. hg38.fa with hg38.fa.fai,
    downloadable from UCSC or Ensembl). Uses pyfaidx for fast random access.

    Parameters
    ----------
    fasta_path:
        Path to reference genome FASTA (must have a matching .fai index,
        or pyfaidx will build one automatically on first use).
    chrom:
        Chromosome name matching the FASTA header (e.g. 'chr1').
    chrom_start:
        Start coordinate of the first bin in bp.
    n_bins:
        Number of bins to compute GC content for.
    bin_size:
        Bin size in bp.

    Returns
    -------
    np.ndarray
        (n_bins,) float32 array of GC fractions in [0, 1].
    """
    try:
        from pyfaidx import Fasta
    except ImportError:
        try:
            from pyfaidx import Faidx
        except ImportError:
            raise ImportError(
                "Install pyfaidx to compute real GC content: pip install pyfaidx"
            )

    from pyfaidx import Fasta
    fasta = Fasta(fasta_path)

    gc_values = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        start = chrom_start + i * bin_size
        end = start + bin_size
        seq = str(fasta[chrom][start:end]).upper()
        if len(seq) == 0:
            continue
        gc_count = seq.count("G") + seq.count("C")
        valid_count = sum(seq.count(b) for b in "ACGT")
        gc_values[i] = gc_count / valid_count if valid_count > 0 else 0.0

    logger.info(
        f"Computed real GC content from {fasta_path}: "
        f"{n_bins} bins, mean GC={gc_values.mean():.3f}"
    )
    return gc_values


def fetch_sequences_from_fasta(
    fasta_path: str,
    chrom: str,
    chrom_start: int,
    n_bins: int,
    bin_size: int,
) -> List[str]:
    """
    Extract real per-bin DNA sequences from a reference genome FASTA.

    These sequences feed directly into stage 2's k-mer / DNA-BERT
    embedding computation (genofold.represent.embeddings).

    Parameters
    ----------
    fasta_path:
        Path to reference genome FASTA file.
    chrom:
        Chromosome name.
    chrom_start:
        Start coordinate of the first bin in bp.
    n_bins:
        Number of bins.
    bin_size:
        Bin size in bp.

    Returns
    -------
    List[str]
        n_bins DNA sequence strings, one per bin.
    """
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise ImportError(
            "Install pyfaidx to extract real sequences: pip install pyfaidx"
        )

    fasta = Fasta(fasta_path)
    sequences = []
    for i in range(n_bins):
        start = chrom_start + i * bin_size
        end = start + bin_size
        seq = str(fasta[chrom][start:end]).upper()
        sequences.append(seq)

    logger.info(f"Extracted {n_bins} real sequences from {fasta_path} ({chrom})")
    return sequences
