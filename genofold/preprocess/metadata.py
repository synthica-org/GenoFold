"""
genofold.preprocess.metadata
-----------------------------
Species and chromatin metadata preprocessing:
  1. Species label encoding — maps species strings to integer IDs
  2. Continuous feature normalization — GC content, mappability, etc.
  3. Chromatin state encoding — one-hot categorical states
  4. Output — per-bin metadata feature matrix (N, M) ready for node assembly
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)

# Canonical species list for consistent encoding across datasets
CANONICAL_SPECIES = [
    "human",      # Homo sapiens
    "mouse",      # Mus musculus
    "rat",        # Rattus norvegicus
    "zebrafish",  # Danio rerio
    "drosophila", # Drosophila melanogaster
    "yeast",      # Saccharomyces cerevisiae
]

# Known chromatin states (from ChromHMM 15-state model)
CHROMATIN_STATES = [
    "TssA",       # Active TSS
    "TssAFlnk",   # Flanking active TSS
    "TxFlnk",     # Transcr. at gene 5' and 3'
    "Tx",         # Strong transcription
    "TxWk",       # Weak transcription
    "EnhG",       # Genic enhancers
    "Enh",        # Enhancers
    "ZNF/Rpts",   # ZNF genes & repeats
    "Het",        # Heterochromatin
    "TssBiv",     # Bivalent/poised TSS
    "BivFlnk",    # Flanking bivalent TSS/Enh
    "EnhBiv",     # Bivalent enhancer
    "ReprPC",     # Repressed PolyComb
    "ReprPCWk",   # Weak repressed PolyComb
    "Quies",      # Quiescent/low
]


def encode_species(
    species_series: pd.Series,
    canonical_species: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode species labels as integer IDs.

    Unknown species (not in canonical list) are assigned ID = -1.
    This allows the model to learn a species embedding lookup table
    with a fixed vocabulary.

    Parameters
    ----------
    species_series:
        Pandas Series of species name strings.
    canonical_species:
        Ordered list defining the integer encoding.
        Defaults to CANONICAL_SPECIES.

    Returns
    -------
    (encoded_array, species_to_id)
        encoded_array : (N,) int64 array of species IDs.
        species_to_id : dict mapping species name → integer ID.
    """
    if canonical_species is None:
        canonical_species = CANONICAL_SPECIES

    species_to_id = {s: i for i, s in enumerate(canonical_species)}

    encoded = species_series.str.lower().map(species_to_id).fillna(-1).astype(np.int64)
    unknown = (encoded == -1).sum()
    if unknown > 0:
        logger.warning(
            f"{unknown} bins have unknown species labels. "
            f"Known species: {canonical_species}"
        )

    logger.info(
        f"Species encoding: {species_series.nunique()} unique species, "
        f"{len(canonical_species)} in vocabulary"
    )
    return encoded.values, species_to_id


def normalize_continuous_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """
    MinMax-normalize selected continuous feature columns to [0, 1].

    Parameters
    ----------
    df:
        Metadata DataFrame.
    feature_cols:
        Names of numeric columns to normalize.

    Returns
    -------
    np.ndarray
        Normalized feature matrix (N, len(feature_cols)), dtype float32.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in metadata: {missing}")

    values = df[feature_cols].fillna(0.0).values.astype(np.float64)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(values)

    logger.info(f"Normalized {len(feature_cols)} continuous features: {feature_cols}")
    return normalized.astype(np.float32)


def encode_chromatin_states(
    state_series: pd.Series,
    state_vocabulary: Optional[List[str]] = None,
) -> np.ndarray:
    """
    One-hot encode chromatin state annotations.

    Parameters
    ----------
    state_series:
        Pandas Series of chromatin state strings. NaN = unknown state.
    state_vocabulary:
        Ordered list of state names. Defaults to CHROMATIN_STATES.

    Returns
    -------
    np.ndarray
        One-hot matrix (N, len(state_vocabulary)), dtype float32.
        Rows with unknown / NaN states are all-zero.
    """
    if state_vocabulary is None:
        state_vocabulary = CHROMATIN_STATES

    n = len(state_series)
    k = len(state_vocabulary)
    state_to_idx = {s: i for i, s in enumerate(state_vocabulary)}

    one_hot = np.zeros((n, k), dtype=np.float32)
    for row_idx, state in enumerate(state_series.fillna("").values):
        if state in state_to_idx:
            one_hot[row_idx, state_to_idx[state]] = 1.0

    known = (one_hot.sum(axis=1) > 0).sum()
    logger.info(
        f"Chromatin state encoding: {known}/{n} bins with known state, "
        f"{k} state classes"
    )
    return one_hot


def preprocess_metadata(
    df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
    include_chromatin_states: bool = True,
    canonical_species: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Full metadata preprocessing pipeline.

    Steps:
      1. Encode species → integer IDs
      2. Normalize continuous features (GC, mappability, etc.)
      3. One-hot encode chromatin states (optional)

    Parameters
    ----------
    df:
        Metadata DataFrame (sorted by bin_id). Must contain 'species' column.
    continuous_cols:
        Continuous columns to normalize. Defaults to
        ['gc_content', 'mappability'] if those columns are present.
    include_chromatin_states:
        Whether to include one-hot chromatin state features.
    canonical_species:
        Species vocabulary for encoding.

    Returns
    -------
    (species_ids, feature_matrix, species_to_id)
        species_ids    : (N,) int64 — species integer IDs per bin.
        feature_matrix : (N, M) float32 — combined continuous + state features.
        species_to_id  : dict mapping species name → integer.
    """
    logger.info(f"Metadata preprocessing: {len(df)} bins")

    # Species encoding
    species_ids, species_to_id = encode_species(df["species"], canonical_species)

    # Auto-detect continuous columns
    if continuous_cols is None:
        continuous_cols = [c for c in ["gc_content", "mappability"] if c in df.columns]

    feature_parts = []

    if continuous_cols:
        cont_features = normalize_continuous_features(df, continuous_cols)
        feature_parts.append(cont_features)

    if include_chromatin_states and "chromatin_state" in df.columns:
        state_features = encode_chromatin_states(df["chromatin_state"])
        feature_parts.append(state_features)

    if feature_parts:
        feature_matrix = np.concatenate(feature_parts, axis=1)
    else:
        # Fallback: return species one-hot if no other features
        n_species = len(canonical_species or CANONICAL_SPECIES)
        feature_matrix = np.zeros((len(df), n_species), dtype=np.float32)
        valid_mask = species_ids >= 0
        feature_matrix[valid_mask, species_ids[valid_mask]] = 1.0
        logger.warning("No continuous or chromatin features found — using species one-hot only")

    logger.info(
        f"Metadata preprocessing complete: "
        f"species_ids shape={species_ids.shape}, "
        f"feature_matrix shape={feature_matrix.shape}"
    )
    return species_ids, feature_matrix, species_to_id
