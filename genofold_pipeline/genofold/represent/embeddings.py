"""
genofold.represent.embeddings
------------------------------
DNA sequence feature extraction per genomic bin:

  1. K-mer frequency vectors — fast, no external model required
     Computes normalized frequency of all k-mers (default k=4, 256 dims)
     with reverse complement collapsing (128 canonical dims).

  2. DNA-BERT embeddings — optional, high-quality contextual features
     Requires the DNABERT-2 model from HuggingFace (offline loading supported).
     Set use_dnabert=True in compute_sequence_embeddings() to enable.

  3. GC content per bin — computed from raw sequence as a basic feature.

Input: list of DNA sequence strings (one per bin) OR path to FASTA file.
Output: (N, D) float32 embedding matrix.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# K-mer utilities
# ---------------------------------------------------------------------------

BASES = "ACGT"


def _canonical_kmer(kmer: str) -> str:
    """Return the lexicographically smaller of kmer and its reverse complement."""
    complement = str.maketrans("ACGT", "TGCA")
    rc = kmer.translate(complement)[::-1]
    return min(kmer, rc)


def build_kmer_vocabulary(k: int, collapse_rc: bool = True) -> List[str]:
    """
    Build sorted vocabulary of all k-mers (optionally collapsed by reverse complement).

    Parameters
    ----------
    k:
        K-mer length.
    collapse_rc:
        If True, each canonical k-mer represents itself and its reverse complement,
        reducing vocabulary from 4^k to 4^k / 2 (approximately).

    Returns
    -------
    List of canonical k-mer strings, sorted.
    """
    all_kmers = ["".join(b) for b in product(BASES, repeat=k)]
    if collapse_rc:
        canonical = sorted(set(_canonical_kmer(km) for km in all_kmers))
    else:
        canonical = sorted(all_kmers)
    return canonical


def kmer_frequency_vector(
    sequence: str,
    k: int,
    vocab: List[str],
    collapse_rc: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute normalized k-mer frequency vector for a single sequence.

    Parameters
    ----------
    sequence:
        DNA sequence string (ACGT alphabet, N/lowercase tolerated).
    k:
        K-mer length.
    vocab:
        Sorted k-mer vocabulary (from build_kmer_vocabulary).
    collapse_rc:
        Whether reverse complement k-mers were collapsed in vocab.
    normalize:
        If True, divide counts by total number of k-mers.

    Returns
    -------
    np.ndarray
        Float32 vector of length len(vocab).
    """
    vocab_index = {km: i for i, km in enumerate(vocab)}
    seq = sequence.upper().replace("N", "")  # Remove ambiguous bases
    vec = np.zeros(len(vocab), dtype=np.float32)

    if len(seq) < k:
        return vec

    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i: i + k]
        if any(b not in BASES for b in kmer):
            continue
        key = _canonical_kmer(kmer) if collapse_rc else kmer
        if key in vocab_index:
            vec[vocab_index[key]] += 1
            total += 1

    if normalize and total > 0:
        vec /= total

    return vec


def compute_kmer_embeddings(
    sequences: List[str],
    k: int = 4,
    collapse_rc: bool = True,
) -> np.ndarray:
    """
    Compute k-mer frequency vectors for all bins.

    Parameters
    ----------
    sequences:
        List of DNA sequences, one per bin (N sequences).
    k:
        K-mer length. k=4 → 128 canonical dims, k=6 → 512 dims.
    collapse_rc:
        Collapse reverse complement pairs (recommended).

    Returns
    -------
    np.ndarray
        (N, D) float32 matrix, D = len(canonical vocabulary).
    """
    vocab = build_kmer_vocabulary(k, collapse_rc)
    logger.info(
        f"K-mer embeddings: k={k}, vocab_size={len(vocab)}, "
        f"collapse_rc={collapse_rc}, n_bins={len(sequences)}"
    )

    embeddings = np.stack(
        [kmer_frequency_vector(seq, k, vocab, collapse_rc) for seq in sequences],
        axis=0,
    )
    logger.info(f"K-mer embedding matrix: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# GC content
# ---------------------------------------------------------------------------

def compute_gc_content(sequences: List[str]) -> np.ndarray:
    """
    Compute GC fraction for each bin sequence.

    Parameters
    ----------
    sequences:
        List of DNA sequences (N,).

    Returns
    -------
    np.ndarray
        (N,) float32 array of GC fractions in [0, 1].
    """
    gc = []
    for seq in sequences:
        seq = seq.upper()
        valid = sum(1 for b in seq if b in BASES)
        gc_count = sum(1 for b in seq if b in "GC")
        gc.append(gc_count / valid if valid > 0 else 0.0)
    return np.array(gc, dtype=np.float32)


# ---------------------------------------------------------------------------
# DNA-BERT hook (optional)
# ---------------------------------------------------------------------------

def compute_dnabert_embeddings(
    sequences: List[str],
    model_path: str,
    batch_size: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute contextual sequence embeddings using DNABERT-2.

    Requires transformers and a local DNABERT-2 model checkpoint.
    Download from: https://huggingface.co/zhihan1996/DNABERT-2-117M

    Parameters
    ----------
    sequences:
        List of DNA sequences, one per bin.
    model_path:
        Path to local DNABERT-2 model directory.
    batch_size:
        Number of sequences per forward pass.
    device:
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray
        (N, 768) float32 matrix of CLS token embeddings.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "Install 'transformers' to use DNA-BERT embeddings: "
            "pip install transformers"
        )

    logger.info(f"Loading DNABERT-2 from {model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = sequences[i: i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # CLS token embedding = first token of last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

            if (i // batch_size) % 10 == 0:
                logger.info(f"DNABERT-2: processed {min(i + batch_size, n)}/{n} sequences")

    result = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    logger.info(f"DNABERT-2 embedding matrix: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_sequence_embeddings(
    sequences: List[str],
    kmer_k: int = 4,
    use_dnabert: bool = False,
    dnabert_model_path: Optional[str] = None,
    dnabert_device: str = "cpu",
) -> np.ndarray:
    """
    Compute sequence embeddings for all genomic bins.

    Uses k-mer features by default. If use_dnabert=True and a model path
    is provided, DNABERT-2 embeddings are used instead (768 dims).
    Both can be concatenated by calling each separately and using np.concatenate.

    Parameters
    ----------
    sequences:
        List of per-bin DNA sequences (N sequences).
    kmer_k:
        K-mer length for frequency embeddings (default 4 → 128 dims).
    use_dnabert:
        If True, compute DNABERT-2 contextual embeddings.
    dnabert_model_path:
        Required if use_dnabert=True. Path to local model directory.
    dnabert_device:
        'cpu' or 'cuda' for DNABERT-2 inference.

    Returns
    -------
    np.ndarray
        (N, D) float32 embedding matrix.
    """
    if use_dnabert:
        if dnabert_model_path is None:
            raise ValueError("dnabert_model_path must be provided when use_dnabert=True")
        return compute_dnabert_embeddings(
            sequences, dnabert_model_path, device=dnabert_device
        )
    else:
        return compute_kmer_embeddings(sequences, k=kmer_k)
