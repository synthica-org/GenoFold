"""
genofold.utils.run_dir
-----------------------
Manages output directories for each pipeline run.

Every run gets its own folder named:
    outputs/<dataset>_<chrom>_<resolution>_<YYYY-MM-DD_HH-MM>/

This means:
  - No previous results ever get overwritten
  - You can always tell what a folder contains just from its name
  - Multiple runs on different chromosomes/resolutions stay organized
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def make_run_dir(
    dataset: str,
    chrom: str,
    bin_size: int,
    base_dir: str = "outputs",
) -> str:
    """
    Create and return a unique output directory for a pipeline run.

    Directory name format:
        <base_dir>/<dataset>_<chrom>_<resolution>_<YYYY-MM-DD_HH-MM>/

    Example:
        outputs/GM12878_chr1_100kb_2024-01-15_14-32/

    Parameters
    ----------
    dataset:
        Short name for the dataset (e.g. 'GM12878', 'IMR90').
    chrom:
        Chromosome being processed (e.g. 'chr1').
    bin_size:
        Resolution in bp (e.g. 100000 → '100kb', 40000 → '40kb').
    base_dir:
        Root outputs directory (default 'outputs').

    Returns
    -------
    str
        Path to the created run directory.
    """
    # Format resolution as human-readable string
    if bin_size >= 1_000_000:
        res_str = f"{bin_size // 1_000_000}mb"
    elif bin_size >= 1_000:
        res_str = f"{bin_size // 1_000}kb"
    else:
        res_str = f"{bin_size}bp"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = f"{dataset}_{chrom}_{res_str}_{timestamp}"
    run_dir = Path(base_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    explain_dir = run_dir / "explain"
    explain_dir.mkdir(exist_ok=True)

    logger.info(f"Run directory created: {run_dir}")
    return str(run_dir)


def get_explain_dir(run_dir: str) -> str:
    """Return the explain/ subdirectory path within a run directory."""
    return str(Path(run_dir) / "explain")
