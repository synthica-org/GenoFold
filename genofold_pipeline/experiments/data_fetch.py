"""
experiments.data_fetch
----------------------
Acquire REAL genomic data for GenoFold experiments (no placeholders):

  * Hi-C contact matrices  — streamed per-chromosome from public .hic files
                             (Rao et al. 2014, GSE63525) via hic-straw.
  * Mappability            — real hg19 CRG 50-mer alignability track
                             (UCSC ENCODE) queried remotely via pyBigWig.
  * Reference sequence     — per-chromosome hg19 FASTA (UCSC), used for real
                             GC content and k-mer composition.

Everything is cached under data_cache/ so re-runs are offline and fast.

This module replaces the old synthetic / placeholder inputs (uniform 0.9
mappability, random contacts) with genuine experimental data, addressing the
reviewer's "swap placeholders for real data" and "real inputs for all
modalities" requirements.
"""

from __future__ import annotations

import gzip
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("genofold.data_fetch")

# ---------------------------------------------------------------------------
# Public data sources (hg19)
# ---------------------------------------------------------------------------

# Rao et al. 2014 in-situ Hi-C, MboI, combined replicates (GSE63525).
HIC_URLS: Dict[str, str] = {
    "GM12878": "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
    "IMR90":   "https://hicfiles.s3.amazonaws.com/hiseq/imr90/in-situ/combined.hic",
    "K562":    "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined.hic",
    "NHEK":    "https://hicfiles.s3.amazonaws.com/hiseq/nhek/in-situ/combined.hic",
    "HMEC":    "https://hicfiles.s3.amazonaws.com/hiseq/hmec/in-situ/combined.hic",
}

# CRG 50-mer alignability/mappability, hg19 (ENCODE / UCSC).
MAPPABILITY_URL = (
    "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/"
    "wgEncodeMapability/wgEncodeCrgMapabilityAlign50mer.bigWig"
)

# Per-chromosome hg19 reference FASTA (UCSC).
FASTA_URL_TMPL = "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/{chrom}.fa.gz"

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")


@dataclass
class ChromDataset:
    """Everything stage-1 of the GenoFold pipeline needs for one chromosome."""
    cell_line: str
    chrom: str            # e.g. "chr22"
    hic_chrom: str        # e.g. "22"  (name inside the .hic file)
    resolution: int
    contact_matrix: np.ndarray    # (N, N) KR-normalised observed contacts
    coverage: np.ndarray          # (N,) Hi-C 1D read depth per bin (NGS modality)
    gc_content: np.ndarray        # (N,) real GC fraction from reference
    mappability: np.ndarray       # (N,) real CRG 50-mer mappability
    sequences: List[str]          # (N,) per-bin reference DNA
    chrom_start: int = 0

    @property
    def n_bins(self) -> int:
        return self.contact_matrix.shape[0]


# ---------------------------------------------------------------------------
# Hi-C
# ---------------------------------------------------------------------------

def fetch_contact_matrix(
    cell_line: str,
    hic_chrom: str,
    resolution: int,
    normalization: str = "KR",
) -> np.ndarray:
    """
    Stream a single chromosome's intra-chromosomal contact matrix from the
    public .hic file and densify it to (N, N).

    Falls back through KR -> VC -> NONE if a normalisation is unavailable.
    """
    import hicstraw

    url = HIC_URLS[cell_line]
    logger.info(f"Opening .hic for {cell_line} (this reads the remote index)…")
    hic = hicstraw.HiCFile(url)
    chrom_len = {c.name: c.length for c in hic.getChromosomes()}[hic_chrom]
    n_bins = chrom_len // resolution + 1

    norms = [normalization, "VC", "NONE"]
    records = None
    used = None
    for norm in norms:
        try:
            mzd = hic.getMatrixZoomData(hic_chrom, hic_chrom, "observed", norm, "BP", resolution)
            records = mzd.getRecords(0, chrom_len, 0, chrom_len)
            used = norm
            break
        except Exception as e:  # noqa: BLE001
            logger.warning(f"normalization {norm} failed for chr{hic_chrom}: {e}")
    if records is None:
        raise RuntimeError(f"Could not read Hi-C for {cell_line} chr{hic_chrom}")

    mat = np.zeros((n_bins, n_bins), dtype=np.float32)
    for r in records:
        i = r.binX // resolution
        j = r.binY // resolution
        v = r.counts
        if not np.isfinite(v):
            continue
        mat[i, j] = v
        mat[j, i] = v
    logger.info(
        f"{cell_line} chr{hic_chrom} @ {resolution:,}bp: {n_bins} bins, "
        f"{np.count_nonzero(mat)} nonzero entries, norm={used}"
    )
    return mat


# ---------------------------------------------------------------------------
# Reference sequence + GC
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    logger.info(f"Downloading {url} -> {dest}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def load_reference_chrom(chrom: str) -> str:
    """
    Download (once) and return the full uppercase sequence of `chrom` (hg19).
    Held in memory as a single string; a chromosome is ~50 MB.
    """
    gz = _download(FASTA_URL_TMPL.format(chrom=chrom), os.path.join(CACHE_DIR, f"{chrom}.fa.gz"))
    cache_txt = os.path.join(CACHE_DIR, f"{chrom}.seq.txt")
    if os.path.exists(cache_txt) and os.path.getsize(cache_txt) > 0:
        with open(cache_txt) as fh:
            return fh.read()
    logger.info(f"Parsing {gz}…")
    parts: List[str] = []
    with gzip.open(gz, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            parts.append(line.strip())
    seq = "".join(parts).upper()
    with open(cache_txt, "w") as fh:
        fh.write(seq)
    return seq


def per_bin_sequences(seq: str, n_bins: int, resolution: int, chrom_start: int = 0) -> List[str]:
    out = []
    for i in range(n_bins):
        s = chrom_start + i * resolution
        out.append(seq[s: s + resolution])
    return out


def gc_from_sequences(sequences: List[str]) -> np.ndarray:
    gc = np.zeros(len(sequences), dtype=np.float32)
    for i, s in enumerate(sequences):
        g = s.count("G") + s.count("C")
        valid = g + s.count("A") + s.count("T")
        gc[i] = g / valid if valid > 0 else 0.0
    return gc


# ---------------------------------------------------------------------------
# Mappability
# ---------------------------------------------------------------------------

def fetch_mappability(chrom: str, n_bins: int, resolution: int, chrom_start: int = 0) -> np.ndarray:
    """
    Real per-bin mean mappability from the hg19 CRG 50-mer track, queried
    remotely (bigWig supports HTTP range requests — no full download).
    Unmappable / N regions return 0.
    """
    import pyBigWig

    bw = pyBigWig.open(MAPPABILITY_URL)
    if bw is None:
        raise RuntimeError("Could not open mappability bigWig")
    clen = bw.chroms().get(chrom)
    vals = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        s = chrom_start + i * resolution
        e = min(s + resolution, clen) if clen else s + resolution
        if clen is None or s >= clen:
            continue
        try:
            m = bw.stats(chrom, s, e, type="mean")[0]
        except Exception:  # noqa: BLE001
            m = None
        vals[i] = float(m) if m is not None else 0.0
    bw.close()
    logger.info(f"{chrom} mappability: mean={vals[vals>0].mean():.3f} over {int((vals>0).sum())} mappable bins")
    return vals


# ---------------------------------------------------------------------------
# Assembly + caching
# ---------------------------------------------------------------------------

def build_chrom_dataset(
    cell_line: str,
    chrom: str,
    resolution: int,
    use_cache: bool = True,
) -> ChromDataset:
    """
    Assemble a fully-real per-chromosome dataset, cached to data_cache/.
    `chrom` is UCSC style ("chr22"); the .hic name ("22") is derived.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"{cell_line}_{chrom}_{resolution}"
    npz = os.path.join(CACHE_DIR, f"{tag}.npz")
    seqf = os.path.join(CACHE_DIR, f"{tag}.seq.npy")
    if use_cache and os.path.exists(npz) and os.path.exists(seqf):
        d = np.load(npz)
        seqs = list(np.load(seqf, allow_pickle=True))
        logger.info(f"Loaded cached dataset {tag}")
        return ChromDataset(
            cell_line=cell_line, chrom=chrom, hic_chrom=chrom.replace("chr", ""),
            resolution=resolution, contact_matrix=d["contact_matrix"],
            coverage=d["coverage"], gc_content=d["gc_content"],
            mappability=d["mappability"], sequences=seqs,
            chrom_start=int(d["chrom_start"]),
        )

    hic_chrom = chrom.replace("chr", "")
    contact = fetch_contact_matrix(cell_line, hic_chrom, resolution)
    n_bins = contact.shape[0]
    # NGS coverage modality = Hi-C 1D read depth (row marginal of raw contacts).
    coverage = contact.sum(axis=1).astype(np.float32)

    seq = load_reference_chrom(chrom)
    sequences = per_bin_sequences(seq, n_bins, resolution)
    gc = gc_from_sequences(sequences)
    mappability = fetch_mappability(chrom, n_bins, resolution)

    ds = ChromDataset(
        cell_line=cell_line, chrom=chrom, hic_chrom=hic_chrom, resolution=resolution,
        contact_matrix=contact, coverage=coverage, gc_content=gc,
        mappability=mappability, sequences=sequences, chrom_start=0,
    )
    np.savez_compressed(
        npz, contact_matrix=contact, coverage=coverage, gc_content=gc,
        mappability=mappability, chrom_start=0,
    )
    np.save(seqf, np.array(sequences, dtype=object), allow_pickle=True)
    logger.info(f"Cached dataset {tag}")
    return ds


if __name__ == "__main__":  # ponytail: runnable self-check on real data
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ds = build_chrom_dataset("GM12878", "chr22", 50_000)
    assert ds.contact_matrix.shape[0] == ds.contact_matrix.shape[1]
    assert ds.contact_matrix.shape[0] == len(ds.sequences) == len(ds.gc_content) == len(ds.mappability)
    assert np.allclose(ds.contact_matrix, ds.contact_matrix.T), "contact matrix not symmetric"
    mappable = ds.mappability > 0
    assert mappable.sum() > 300, "too few mappable bins"
    assert 0.3 < ds.gc_content[mappable].mean() < 0.6, "GC content out of human range"
    print(f"OK  chr22 @50kb: {ds.n_bins} bins | "
          f"mappable={int(mappable.sum())} | "
          f"meanGC={ds.gc_content[mappable].mean():.3f} | "
          f"meanMap={ds.mappability[mappable].mean():.3f} | "
          f"contacts nz={np.count_nonzero(ds.contact_matrix)}")
