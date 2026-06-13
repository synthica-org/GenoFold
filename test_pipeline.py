"""
test_pipeline.py
----------------
Synthetic end-to-end smoke test for GenoFold stages 1 and 2.

Generates realistic synthetic data (Hi-C matrix, NGS coverage,
DNA sequences, metadata) and runs the full pipeline to confirm
that a valid PyG Data object is produced.

Run with:
    python test_pipeline.py
"""

import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("genofold.test")

# ── Synthetic data generation ────────────────────────────────────────────────

def make_synthetic_hic(n_bins: int, seed: int = 42) -> np.ndarray:
    """
    Create a synthetic Hi-C matrix with realistic properties:
    - Distance-dependent decay (contacts drop off with genomic distance)
    - TAD-like block structure
    - Poisson noise
    """
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_bins, n_bins), dtype=np.float32)

    # Distance decay: contact_freq ~ 1 / (1 + distance)
    for i in range(n_bins):
        for j in range(i, n_bins):
            dist = abs(i - j)
            base = 100.0 / (1.0 + dist * 0.5)
            count = rng.poisson(lam=max(base, 0.1))
            matrix[i, j] = count
            matrix[j, i] = count

    # Add 3 TAD-like blocks (elevated contact frequency within blocks)
    tad_boundaries = [0, n_bins // 3, 2 * n_bins // 3, n_bins]
    for start, end in zip(tad_boundaries[:-1], tad_boundaries[1:]):
        block = matrix[start:end, start:end]
        block += rng.poisson(lam=20, size=block.shape).astype(np.float32)

    # Symmetrize
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0)
    return matrix


def make_synthetic_coverage(n_bins: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Most bins have moderate coverage; some are low (unmappable)
    coverage = rng.negative_binomial(n=5, p=0.3, size=n_bins).astype(np.float32)
    # Zero out ~5% of bins to simulate unmappable regions
    zero_idx = rng.choice(n_bins, size=n_bins // 20, replace=False)
    coverage[zero_idx] = 0.0
    return coverage


def make_synthetic_sequences(n_bins: int, bin_size: int = 200, seed: int = 1) -> list:
    """Generate random DNA sequences (short for testing)."""
    rng = np.random.default_rng(seed)
    bases = list("ACGT")
    sequences = []
    for _ in range(n_bins):
        seq = "".join(rng.choice(bases, size=bin_size))
        sequences.append(seq)
    return sequences


def make_synthetic_metadata(n_bins: int) -> pd.DataFrame:
    species_pool = ["human", "mouse", "rat"]
    rows = []
    for i in range(n_bins):
        rows.append({
            "bin_id": i,
            "species": species_pool[i % len(species_pool)],
            "gc_content": 0.35 + 0.1 * np.sin(i / n_bins * np.pi),
            "mappability": 0.7 + 0.2 * (i % 5 != 0),  # Every 5th bin is lower
            "chromatin_state": "Tx" if i % 3 == 0 else ("Enh" if i % 3 == 1 else "Het"),
        })
    return pd.DataFrame(rows)


# ── Test ─────────────────────────────────────────────────────────────────────

def test_pipeline(n_bins: int = 50):
    logger.info(f"Generating synthetic data: {n_bins} bins")

    matrix   = make_synthetic_hic(n_bins)
    coverage = make_synthetic_coverage(n_bins)
    sequences = make_synthetic_sequences(n_bins)
    metadata_df = make_synthetic_metadata(n_bins)

    logger.info(f"  Hi-C matrix shape:   {matrix.shape}")
    logger.info(f"  Coverage shape:      {coverage.shape}")
    logger.info(f"  Sequences:           {len(sequences)} bins")
    logger.info(f"  Metadata shape:      {metadata_df.shape}")

    try:
        from pipeline import GenoFoldPipeline, PipelineConfig
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)

    config = PipelineConfig(
        bin_size=40_000,
        top_k_edges=10,        # Keep graphs manageable in tests
        denoise_sigma=0.5,
        sparse_threshold=0.05,
        kmer_k=4,
    )

    pipeline = GenoFoldPipeline(config=config)
    data = pipeline.run(
        contact_matrix=matrix,
        coverage=coverage,
        sequences=sequences,
        metadata_df=metadata_df,
    )

    # ── Assertions ────────────────────────────────────────────────────────────
    assert data.x is not None,                      "Node features must not be None"
    assert data.edge_index is not None,             "Edge index must not be None"
    assert data.x.shape[0] == data.num_nodes,       "Node count mismatch"
    assert data.edge_index.shape[0] == 2,           "edge_index must have shape (2, E)"
    assert data.edge_attr.shape[1] == 2,            "edge_attr must have 2 features"
    assert data.species_ids.shape[0] == data.num_nodes, "species_ids length mismatch"
    assert not data.x.isnan().any(),                "Node features contain NaN"
    assert not data.edge_attr.isnan().any(),        "Edge features contain NaN"

    logger.info("")
    logger.info("═" * 55)
    logger.info("  GenoFold stages 1-2: ALL ASSERTIONS PASSED ✓")
    logger.info("═" * 55)
    logger.info(f"  Nodes:           {data.num_nodes}")
    logger.info(f"  Edges:           {data.edge_index.shape[1]:,}")
    logger.info(f"  Node feat. dim:  {data.x.shape[1]}")
    logger.info(f"  Edge feat. dim:  {data.edge_attr.shape[1]}")
    logger.info(f"  Species IDs:     {data.species_ids.unique().tolist()}")
    logger.info("═" * 55)
    return data


if __name__ == "__main__":
    test_pipeline(n_bins=50)


def test_stage3(n_bins: int = 50):
    """Test stage 3 (Infer) — GNN + Transformer."""
    import torch
    logger.info("Running stage 3 test...")

    # Run stages 1-2 first to get the Data object
    data = test_pipeline(n_bins=n_bins)

    from pipeline import GenoFoldPipeline, PipelineConfig
    config = PipelineConfig(bin_size=40_000, top_k_edges=10)
    pipeline = GenoFoldPipeline(config=config)

    output, model = pipeline.infer(data, hidden_dim=64)

    # Assertions
    assert output.embeddings is not None,                        "Embeddings must not be None"
    assert output.embeddings.shape[0] == data.num_nodes,        "Node count mismatch"
    assert output.embeddings.shape[1] == 64,                    "Hidden dim mismatch"
    assert output.gnn_embeddings.shape == output.embeddings.shape, "GNN/final shape mismatch"
    assert not output.embeddings.isnan().any(),                  "Embeddings contain NaN"
    assert not output.gnn_embeddings.isnan().any(),              "GNN embeddings contain NaN"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("")
    logger.info("═" * 55)
    logger.info("  GenoFold stage 3: ALL ASSERTIONS PASSED ✓")
    logger.info("═" * 55)
    logger.info(f"  Embedding shape:   {tuple(output.embeddings.shape)}")
    logger.info(f"  GNN emb shape:     {tuple(output.gnn_embeddings.shape)}")
    logger.info(f"  Model parameters:  {total_params:,}")
    logger.info("═" * 55)
    return output, model


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "1-2":
        test_pipeline(n_bins=50)
    elif stage == "3":
        test_stage3(n_bins=50)
    else:
        test_stage3(n_bins=50)
