"""
genofold.score.region
-----------------------
Per-region confidence aggregation.

Aggregates per-bin confidence scores into TAD-like structural domains,
giving a region-level reliability score. This is more interpretable
for biologists who think in terms of domains/compartments rather than
individual bins, and is the natural unit for conference figures.

Region detection uses a simple but effective approach: bins are
grouped based on contact density changes along the diagonal of the
Hi-C matrix (a lightweight proxy for TAD boundary calling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """
    A structural domain (TAD-like region) with aggregated confidence.

    Attributes
    ----------
    start_bin, end_bin:
        Bin index range (inclusive) defining this region.
    mean_confidence:
        Average per-bin confidence within the region.
    min_confidence:
        Lowest per-bin confidence within the region (worst case).
    size:
        Number of bins in the region.
    """
    start_bin: int
    end_bin: int
    mean_confidence: float
    min_confidence: float
    size: int


def detect_regions(
    distance_matrix: torch.Tensor,
    min_region_size: int = 3,
    boundary_sensitivity: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Detect TAD-like structural domains from the predicted distance matrix.

    Uses a simple insulation-score approach: for each bin, compute
    the average distance to nearby bins on the left vs. right side.
    A local maximum in this "insulation" signal indicates a domain
    boundary (analogous to how TAD callers like TopDom work).

    Parameters
    ----------
    distance_matrix:
        (M, M) predicted pairwise distances from stage 4.
    min_region_size:
        Minimum bins per region (merges small regions into neighbors).
    boundary_sensitivity:
        Higher = more boundaries detected (smaller regions).

    Returns
    -------
    List[Tuple[int, int]]
        List of (start_bin, end_bin) ranges, inclusive, covering all bins.
    """
    M = distance_matrix.shape[0]
    D = distance_matrix.detach().cpu().numpy()

    window = max(2, M // 20)  # Adaptive window size

    # Insulation score: higher score = more "boundary-like"
    # Compares average distance within a local window vs across it
    insulation = np.zeros(M)
    for i in range(M):
        left_start = max(0, i - window)
        right_end = min(M, i + window)
        if right_end - left_start < 2:
            continue
        # Cross-boundary distances (left window to right window)
        left_idx = np.arange(left_start, i)
        right_idx = np.arange(i, right_end)
        if len(left_idx) == 0 or len(right_idx) == 0:
            continue
        cross_dists = D[np.ix_(left_idx, right_idx)]
        insulation[i] = cross_dists.mean()

    # Normalize
    if insulation.max() > insulation.min():
        insulation = (insulation - insulation.min()) / (insulation.max() - insulation.min())

    # Find local maxima as boundaries
    boundaries = [0]
    threshold = np.percentile(insulation, 100 - 30 * boundary_sensitivity)
    for i in range(1, M - 1):
        if (
            insulation[i] > threshold
            and insulation[i] >= insulation[i - 1]
            and insulation[i] >= insulation[i + 1]
        ):
            boundaries.append(i)
    boundaries.append(M)
    boundaries = sorted(set(boundaries))

    # Build regions from consecutive boundaries
    regions = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1] - 1
        if end >= start:
            regions.append((start, end))

    # Merge small regions into neighbors
    merged = []
    for start, end in regions:
        size = end - start + 1
        if merged and size < min_region_size:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    logger.info(f"Region detection: {len(merged)} regions from {M} bins")
    return merged


def aggregate_region_confidence(
    confidence: torch.Tensor,
    regions: List[Tuple[int, int]],
) -> List[Region]:
    """
    Aggregate per-bin confidence scores into per-region statistics.

    Parameters
    ----------
    confidence:
        (M,) per-bin confidence scores from ConfidenceScorer.
    regions:
        List of (start_bin, end_bin) tuples from detect_regions().

    Returns
    -------
    List[Region]
        One Region object per detected domain, with aggregated stats.
    """
    conf_np = confidence.detach().cpu().numpy()
    result = []

    for start, end in regions:
        segment = conf_np[start: end + 1]
        result.append(Region(
            start_bin=start,
            end_bin=end,
            mean_confidence=float(segment.mean()),
            min_confidence=float(segment.min()),
            size=len(segment),
        ))

    logger.info(
        f"Region confidence aggregated: {len(result)} regions, "
        f"mean of means={np.mean([r.mean_confidence for r in result]):.3f}"
    )
    return result


def regions_to_dataframe(regions: List[Region]):
    """
    Convert a list of Region objects to a pandas DataFrame for export/analysis.

    Parameters
    ----------
    regions:
        List of Region objects.

    Returns
    -------
    pd.DataFrame
        Columns: region_id, start_bin, end_bin, size, mean_confidence, min_confidence
    """
    import pandas as pd

    rows = [{
        "region_id": i,
        "start_bin": r.start_bin,
        "end_bin": r.end_bin,
        "size": r.size,
        "mean_confidence": r.mean_confidence,
        "min_confidence": r.min_confidence,
    } for i, r in enumerate(regions)]

    return pd.DataFrame(rows)
