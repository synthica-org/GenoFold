"""
test_pipeline.py
----------------
Synthetic end-to-end smoke test for GenoFold stages 1-5.

Generates realistic synthetic data (Hi-C matrix, NGS coverage,
DNA sequences, metadata) and runs the full pipeline to confirm
that every stage produces valid, well-shaped outputs.

Run with:
    python3 test_pipeline.py          # runs all stages 1-5
    python3 test_pipeline.py 1-2      # runs only stages 1-2
    python3 test_pipeline.py 3        # runs stages 1-3
    python3 test_pipeline.py 4        # runs stages 1-4
    python3 test_pipeline.py 5        # runs stages 1-5
"""

import logging
import os
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

    for i in range(n_bins):
        for j in range(i, n_bins):
            dist = abs(i - j)
            base = 100.0 / (1.0 + dist * 0.5)
            count = rng.poisson(lam=max(base, 0.1))
            matrix[i, j] = count
            matrix[j, i] = count

    tad_boundaries = [0, n_bins // 3, 2 * n_bins // 3, n_bins]
    for start, end in zip(tad_boundaries[:-1], tad_boundaries[1:]):
        block = matrix[start:end, start:end]
        block += rng.poisson(lam=20, size=block.shape).astype(np.float32)

    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0)
    return matrix


def make_synthetic_coverage(n_bins: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coverage = rng.negative_binomial(n=5, p=0.3, size=n_bins).astype(np.float32)
    zero_idx = rng.choice(n_bins, size=n_bins // 20, replace=False)
    coverage[zero_idx] = 0.0
    return coverage


def make_synthetic_sequences(n_bins: int, bin_size: int = 200, seed: int = 1) -> list:
    rng = np.random.default_rng(seed)
    bases = list("ACGT")
    return ["".join(rng.choice(bases, size=bin_size)) for _ in range(n_bins)]


def make_synthetic_metadata(n_bins: int) -> pd.DataFrame:
    species_pool = ["human", "mouse", "rat"]
    rows = []
    for i in range(n_bins):
        rows.append({
            "bin_id": i,
            "species": species_pool[i % len(species_pool)],
            "gc_content": 0.35 + 0.1 * np.sin(i / n_bins * np.pi),
            "mappability": 0.7 + 0.2 * (i % 5 != 0),
            "chromatin_state": "Tx" if i % 3 == 0 else ("Enh" if i % 3 == 1 else "Het"),
        })
    return pd.DataFrame(rows)


# ── Stage 1-2: Preprocess + Represent ────────────────────────────────────────

def test_stages_1_2(n_bins: int = 50):
    logger.info(f"Generating synthetic data: {n_bins} bins")

    matrix = make_synthetic_hic(n_bins)
    coverage = make_synthetic_coverage(n_bins)
    sequences = make_synthetic_sequences(n_bins)
    metadata_df = make_synthetic_metadata(n_bins)

    logger.info(f"  Hi-C matrix shape:   {matrix.shape}")
    logger.info(f"  Coverage shape:      {coverage.shape}")
    logger.info(f"  Sequences:           {len(sequences)} bins")
    logger.info(f"  Metadata shape:      {metadata_df.shape}")

    from pipeline import GenoFoldPipeline, PipelineConfig

    config = PipelineConfig(
        bin_size=40_000,
        top_k_edges=10,
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

    assert data.x is not None
    assert data.edge_index is not None
    assert data.x.shape[0] == data.num_nodes
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == 2
    assert data.species_ids.shape[0] == data.num_nodes
    assert not data.x.isnan().any()
    assert not data.edge_attr.isnan().any()

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
    return data, pipeline


# ── Stage 3: Infer ───────────────────────────────────────────────────────────

def test_stage3(n_bins: int = 50):
    logger.info("Running stage 3 test...")
    data, pipeline = test_stages_1_2(n_bins=n_bins)

    output, model = pipeline.infer(data, hidden_dim=64)

    assert output.embeddings is not None
    assert output.embeddings.shape[0] == data.num_nodes
    assert output.embeddings.shape[1] == 64
    assert output.gnn_embeddings.shape == output.embeddings.shape
    assert not output.embeddings.isnan().any()
    assert not output.gnn_embeddings.isnan().any()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("")
    logger.info("═" * 55)
    logger.info("  GenoFold stage 3: ALL ASSERTIONS PASSED ✓")
    logger.info("═" * 55)
    logger.info(f"  Embedding shape:   {tuple(output.embeddings.shape)}")
    logger.info(f"  GNN emb shape:     {tuple(output.gnn_embeddings.shape)}")
    logger.info(f"  Model parameters:  {total_params:,}")
    logger.info("═" * 55)
    return output, model, data, pipeline


# ── Stage 4: Reconstruct ─────────────────────────────────────────────────────

def test_stage4(n_bins: int = 50):
    logger.info("Running stage 4 test...")
    inference_output, infer_model, data, pipeline = test_stage3(n_bins=n_bins)

    export_dir = "outputs/test"
    recon_output, recon_model = pipeline.reconstruct(
        inference_output,
        hidden_dim=64,
        export_dir=export_dir,
        chrom="chr1",
    )

    M = inference_output.num_nodes
    assert recon_output.coordinates is not None
    assert recon_output.coordinates.shape == (M, 3)
    assert recon_output.distance_matrix.shape == (M, M)
    assert recon_output.decoder_coords.shape == (M, 3)
    assert recon_output.mds_coords.shape == (M, 3)
    assert not recon_output.coordinates.isnan().any()
    assert not recon_output.distance_matrix.isnan().any()
    assert (recon_output.distance_matrix >= 0).all()
    assert os.path.exists(recon_output.export_paths["npy"])
    assert os.path.exists(recon_output.export_paths["pdb"])
    assert os.path.exists(recon_output.export_paths["tsv"])

    total_params = sum(p.numel() for p in recon_model.parameters() if p.requires_grad)

    logger.info("")
    logger.info("═" * 55)
    logger.info("  GenoFold stage 4: ALL ASSERTIONS PASSED ✓")
    logger.info("═" * 55)
    logger.info(f"  Coordinates shape:   {tuple(recon_output.coordinates.shape)}")
    logger.info(f"  Distance matrix:     {tuple(recon_output.distance_matrix.shape)}")
    logger.info(f"  Coord range:         [{recon_output.coordinates.min():.3f}, {recon_output.coordinates.max():.3f}]")
    logger.info(f"  Model parameters:    {total_params:,}")
    logger.info(f"  Output .npy:         {recon_output.export_paths['npy']}")
    logger.info(f"  Output .pdb:         {recon_output.export_paths['pdb']}")
    logger.info(f"  Output .tsv:         {recon_output.export_paths['tsv']}")
    logger.info("═" * 55)
    return recon_output, recon_model, inference_output, infer_model, data, pipeline


# ── Stage 5: Score ───────────────────────────────────────────────────────────

def test_stage5(n_bins: int = 50):
    logger.info("Running stage 5 test...")
    (recon_output, recon_model, inference_output,
     infer_model, data, pipeline) = test_stage4(n_bins=n_bins)

    score_output, score_model = pipeline.score(
        inference_output,
        recon_output,
        data.edge_index,
        reconstruct_model=recon_model,
        hidden_dim=64,
        n_mc_samples=5,   # Small for fast testing
        use_mc_dropout=True,
    )

    M = inference_output.num_nodes
    assert score_output.bin_confidence.shape == (M,)
    assert (score_output.bin_confidence >= 0).all()
    assert (score_output.bin_confidence <= 1).all()
    assert not score_output.bin_confidence.isnan().any()
    assert len(score_output.regions) > 0
    assert score_output.uncertainty.radius.shape == (M,)
    assert not score_output.uncertainty.radius.isnan().any()
    assert (score_output.uncertainty.radius >= 0).all()

    # Re-export structure with REAL confidence scores in the PDB B-factor column
    from genofold.reconstruct.export import export_structure
    export_paths = export_structure(
        coords=recon_output.coordinates,
        output_dir="outputs/test",
        prefix="genofold_structure_scored",
        bin_size=40_000,
        chrom="chr1",
        confidence=score_output.bin_confidence,
    )

    logger.info("")
    logger.info("═" * 55)
    logger.info("  GenoFold stage 5: ALL ASSERTIONS PASSED ✓")
    logger.info("═" * 55)
    logger.info(f"  Bin confidence shape:  {tuple(score_output.bin_confidence.shape)}")
    logger.info(f"  Mean confidence:       {score_output.bin_confidence.mean():.3f}")
    logger.info(f"  Regions detected:      {len(score_output.regions)}")
    logger.info(f"  Mean uncertainty:      {score_output.uncertainty.radius.mean():.4f}")
    logger.info(f"  Max uncertainty:       {score_output.uncertainty.radius.max():.4f}")
    logger.info(f"  Scored .pdb:           {export_paths['pdb']}")
    logger.info("═" * 55)
    return score_output, score_model


# ── Stage 6: Explain ──────────────────────────────────────────────────────────

def test_stage6(n_bins: int = 50):
    logger.info("Running stage 6 test...")
    (recon_output, recon_model, inference_output,
     infer_model, data, pipeline) = test_stage4(n_bins=n_bins)

    score_output, score_model = pipeline.score(
        inference_output,
        recon_output,
        data.edge_index,
        reconstruct_model=recon_model,
        hidden_dim=64,
        n_mc_samples=5,
        use_mc_dropout=True,
    )

    explain_output, explainer = pipeline.explain(
        infer_model,
        recon_model,
        data,
        inference_output,
        recon_output,
        score_output=score_output,
        output_dir="outputs/explain",
        ig_steps=20,   # Small for fast testing
        top_k_loci=10,
    )

    M = inference_output.num_nodes
    assert explain_output.feature_attributions.shape == data.x.shape
    assert not explain_output.feature_attributions.isnan().any()
    assert len(explain_output.group_attributions) > 0
    assert explain_output.transformer_attention.shape == (M, M)
    assert explain_output.influence_score.shape == (M,)
    assert not explain_output.influence_score.isnan().any()
    assert len(explain_output.figure_paths) > 0
    for name, path in explain_output.figure_paths.items():
        assert os.path.exists(path), f"Missing figure: {name} -> {path}"

    logger.info("")
    logger.info("═" * 55)
    logger.info("  GenoFold stage 6: ALL ASSERTIONS PASSED ✓")
    logger.info("═" * 55)
    logger.info(f"  Attribution shape:     {tuple(explain_output.feature_attributions.shape)}")
    logger.info(f"  Feature groups:        {list(explain_output.group_attributions.keys())}")
    logger.info(f"  Transformer attn:      {tuple(explain_output.transformer_attention.shape)}")
    logger.info(f"  Figures generated:     {len(explain_output.figure_paths)}")
    for name, path in explain_output.figure_paths.items():
        logger.info(f"    {name}: {path}")
    logger.info("═" * 55)
    return explain_output, explainer


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "6"

    if stage == "1-2":
        test_stages_1_2(n_bins=50)
    elif stage == "3":
        test_stage3(n_bins=50)
    elif stage == "4":
        test_stage4(n_bins=50)
    elif stage == "5":
        test_stage5(n_bins=50)
    elif stage == "6":
        test_stage6(n_bins=50)
    else:
        test_stage6(n_bins=50)
