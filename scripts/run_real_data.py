"""
scripts/run_real_data.py
------------------------
Runs the full GenoFold pipeline (stages 1-6) on a real Hi-C dataset.

Default: GM12878 in-situ Hi-C from GEO (GSE63525, Rao et al. 2014, Cell)
         — one of the most cited Hi-C papers ever published.
         No login/account required to download.

Usage
-----
    # First, download the data:
    bash scripts/download_real_data.sh

    # Install the .hic reader:
    pip install hic-straw

    # Then run the full pipeline:
    python3 scripts/run_real_data.py

    # Or customize:
    python3 scripts/run_real_data.py \\
        --hic   data/real/GM12878_insitu_hic.hic \\
        --format hic \\
        --chrom chr1 \\
        --res   100000 \\
        --bins  500

Output
------
Each run creates a unique folder under outputs/ named:
    outputs/GM12878_chr1_100kb_YYYY-MM-DD_HH-MM/

containing:
    genofold_structure.npy           — 3D coordinates (numpy)
    genofold_structure.pdb           — 3D structure (PyMOL/ChimeraX)
    genofold_structure.tsv           — coordinates + confidence table
    genofold_structure_scored.pdb    — same PDB with confidence in B-factor
    explain/
        feature_importance.png/.html — which features drove predictions
        transformer_attention.png/.html
        influential_loci.png/.html   — structural hub detection
        structure_confidence.png/.html
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-35s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("genofold.run_real")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GenoFold pipeline on real Hi-C data"
    )
    parser.add_argument(
        "--hic",
        default="data/real/GM12878_insitu_hic.hic",
        help="Path to Hi-C file (default: GM12878 .hic from GEO)",
    )
    parser.add_argument(
        "--format",
        choices=["cool", "hic"],
        default="hic",
        help="Hi-C file format: 'cool' for .cool/.mcool (cooler), "
             "'hic' for Juicer .hic files (default: hic)",
    )
    parser.add_argument(
        "--normalization",
        default="KR",
        help="Normalization for .hic files: NONE, VC, VC_SQRT, KR, SCALE "
             "(default: KR, equivalent to ICE). Ignored for .cool files.",
    )
    parser.add_argument(
        "--chrom",
        default="chr1",
        help="Chromosome to process (default: chr1)",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=100_000,
        help="Resolution in bp (default: 100000 = 100kb)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=500,
        help="Max bins to process from the start of the chromosome "
             "(default: 500 = first 50Mb at 100kb). Increase for more "
             "coverage, decrease for faster runs.",
    )
    parser.add_argument(
        "--dataset",
        default="GM12878",
        help="Dataset name for output folder naming (default: GM12878)",
    )
    parser.add_argument(
        "--fasta",
        default=None,
        help="Optional: path to reference genome FASTA for real GC "
             "content and sequences. If not provided, placeholder values "
             "are used (acceptable for pipeline testing).",
    )
    parser.add_argument(
        "--bigwig",
        default=None,
        help="Optional: path to a bigWig signal file (e.g. ENCODE "
             "DNase-seq) for real NGS coverage. If not provided, the "
             "Hi-C row-sum is used as a coverage proxy.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Model hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10,
        help="Monte Carlo dropout samples for uncertainty (default: 10)",
    )
    parser.add_argument(
        "--zero-fraction-threshold",
        type=float,
        default=0.97,
        help="Max fraction of zeros allowed in a bin's row before it's "
             "masked as unmappable (default: 0.97). Real Hi-C data is "
             "very sparse, especially over small regions — lower values "
             "(like the old 0.8 default) will mask out every bin. Raise "
             "toward 1.0 if bins are still being over-filtered; lower it "
             "if you want stricter mappability filtering.",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=0.0,
        help="Hi-C denoising sparsity threshold (default: 0.0, i.e. off). "
             "Real Hi-C data can have a different value scale than "
             "synthetic data — start at 0.0 and increase only if needed.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.0,
        help="Minimum contact weight to create a graph edge (default: 0.0, "
             "i.e. any nonzero contact becomes an edge).",
    )
    parser.add_argument(
        "--stages",
        default="1-6",
        help="Which stages to run: '1-2', '1-4', or '1-6' (default: 1-6)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("═" * 60)
    logger.info("  GenoFold — Real Data Pipeline")
    logger.info("═" * 60)
    logger.info(f"  Hi-C file:   {args.hic}")
    logger.info(f"  Chromosome:  {args.chrom}")
    logger.info(f"  Resolution:  {args.res:,} bp")
    logger.info(f"  Max bins:    {args.bins}")
    logger.info(f"  Stages:      {args.stages}")
    logger.info("═" * 60)

    # ── Check Hi-C file exists ────────────────────────────────────────────────
    if not Path(args.hic).exists():
        logger.error(
            f"\nHi-C file not found: {args.hic}\n"
            f"Run this first:\n"
            f"    bash scripts/download_real_data.sh\n"
        )
        sys.exit(1)

    # ── Create timestamped output directory ───────────────────────────────────
    from genofold.utils.run_dir import make_run_dir, get_explain_dir
    run_dir = make_run_dir(
        dataset=args.dataset,
        chrom=args.chrom,
        bin_size=args.res,
    )
    explain_dir = get_explain_dir(run_dir)
    logger.info(f"Output directory: {run_dir}")

    # ── Load Hi-C contact matrix ──────────────────────────────────────────────
    logger.info("\n── Loading Hi-C data ──")
    import numpy as np

    if args.format == "hic":
        from genofold.utils.io import load_hic
        matrix = load_hic(
            args.hic, resolution=args.res, chrom=args.chrom,
            normalization=args.normalization,
        )
    else:
        from genofold.utils.io import load_cool
        matrix = load_cool(args.hic, resolution=args.res, chrom=args.chrom)

    # Crop to the requested number of bins
    n_bins = min(args.bins, matrix.shape[0])
    matrix = matrix[:n_bins, :n_bins]
    logger.info(f"Contact matrix loaded and cropped: {matrix.shape}")

    # ── Build per-bin NGS coverage ────────────────────────────────────────────
    logger.info("\n── Loading NGS coverage ──")
    if args.bigwig:
        from genofold.utils.io import load_bigwig_coverage
        coverage = load_bigwig_coverage(
            args.bigwig, args.chrom, 0, n_bins, args.res
        )
        logger.info(f"Real NGS coverage loaded from {args.bigwig}")
    else:
        # Without a bigWig/BAM file, use the row-sum of the Hi-C matrix
        # as a coverage proxy. This is a standard approximation used in
        # several published methods, but is NOT real NGS data.
        logger.warning(
            "No bigWig provided — using Hi-C row-sum as a coverage proxy. "
            "Pass --bigwig path/to/signal.bigWig for real NGS coverage."
        )
        coverage = matrix.sum(axis=1).astype(np.float32)

    # ── Build sequences ───────────────────────────────────────────────────────
    logger.info("\n── Building sequences ──")
    if args.fasta:
        from genofold.utils.real_metadata import fetch_sequences_from_fasta
        sequences = fetch_sequences_from_fasta(
            args.fasta, args.chrom, 0, n_bins, args.res
        )
        logger.info(f"Real sequences loaded from {args.fasta}")
    else:
        # Generate placeholder sequences using bin index as seed
        # This is only for pipeline testing — real sequences from a
        # FASTA file will produce more accurate k-mer features
        logger.warning(
            "No FASTA provided — using placeholder sequences. "
            "Pass --fasta path/to/genome.fa for real sequence features."
        )
        rng = np.random.default_rng(42)
        bases = list("ACGT")
        seq_len = min(args.res, 1000)  # Cap at 1000bp for speed
        sequences = [
            "".join(rng.choice(bases, size=seq_len))
            for _ in range(n_bins)
        ]

    # ── Build metadata table ──────────────────────────────────────────────────
    logger.info("\n── Building metadata ──")
    from genofold.utils.real_metadata import build_metadata_table

    gc_content = None
    if args.fasta:
        from genofold.utils.real_metadata import fetch_gc_content_from_fasta
        gc_content = fetch_gc_content_from_fasta(
            args.fasta, args.chrom, 0, n_bins, args.res
        )

    metadata_df = build_metadata_table(
        n_bins=n_bins,
        species="human",
        chrom=args.chrom,
        chrom_start=0,
        bin_size=args.res,
        gc_content=gc_content,
    )

    # ── Run pipeline stages 1-2 ───────────────────────────────────────────────
    from pipeline import GenoFoldPipeline, PipelineConfig

    config = PipelineConfig(
        bin_size=args.res,
        top_k_edges=50,
        kmer_k=4,
        denoise_sigma=1.0,
        sparse_threshold=args.sparse_threshold,
        contact_threshold=args.contact_threshold,
        zero_fraction_threshold=args.zero_fraction_threshold,
    )
    pipeline = GenoFoldPipeline(config=config)

    logger.info(
        f"Hi-C matrix stats before preprocessing: "
        f"min={matrix.min():.4f}, max={matrix.max():.4f}, "
        f"mean={matrix.mean():.4f}, nonzero={np.count_nonzero(matrix):,}/{matrix.size:,}"
    )

    logger.info("\n── Running stages 1-2: Preprocess + Represent ──")
    data = pipeline.run(
        contact_matrix=matrix,
        coverage=coverage,
        sequences=sequences,
        metadata_df=metadata_df,
    )

    if args.stages == "1-2":
        logger.info("\nStopping at stages 1-2 as requested.")
        _print_summary(run_dir, data=data)
        return

    # ── Stage 3: Infer ────────────────────────────────────────────────────────
    logger.info("\n── Running stage 3: Infer ──")
    inference_output, infer_model = pipeline.infer(data, hidden_dim=args.hidden_dim)

    if args.stages == "1-3":
        _print_summary(run_dir, data=data, inference_output=inference_output)
        return

    # ── Stage 4: Reconstruct ──────────────────────────────────────────────────
    logger.info("\n── Running stage 4: Reconstruct ──")
    recon_output, recon_model = pipeline.reconstruct(
        inference_output,
        hidden_dim=args.hidden_dim,
        export_dir=run_dir,
        chrom=args.chrom,
    )

    if args.stages == "1-4":
        _print_summary(run_dir, data=data, inference_output=inference_output,
                       recon_output=recon_output)
        return

    # ── Stage 5: Score ────────────────────────────────────────────────────────
    logger.info("\n── Running stage 5: Score ──")
    score_output, score_model = pipeline.score(
        inference_output,
        recon_output,
        data.edge_index,
        reconstruct_model=recon_model,
        hidden_dim=args.hidden_dim,
        n_mc_samples=args.mc_samples,
        use_mc_dropout=True,
    )

    # Re-export PDB with real confidence scores in the B-factor column
    from genofold.reconstruct.export import export_structure
    export_structure(
        coords=recon_output.coordinates,
        output_dir=run_dir,
        prefix="genofold_structure_scored",
        bin_size=args.res,
        chrom=args.chrom,
        confidence=score_output.bin_confidence,
    )

    if args.stages == "1-5":
        _print_summary(run_dir, data=data, inference_output=inference_output,
                       recon_output=recon_output, score_output=score_output)
        return

    # ── Stage 6: Explain ──────────────────────────────────────────────────────
    logger.info("\n── Running stage 6: Explain ──")
    explain_output, explainer = pipeline.explain(
        infer_model,
        recon_model,
        data,
        inference_output,
        recon_output,
        score_output=score_output,
        output_dir=explain_dir,
        ig_steps=30,
        top_k_loci=10,
    )

    _print_summary(
        run_dir,
        data=data,
        inference_output=inference_output,
        recon_output=recon_output,
        score_output=score_output,
        explain_output=explain_output,
    )


def _print_summary(
    run_dir,
    data=None,
    inference_output=None,
    recon_output=None,
    score_output=None,
    explain_output=None,
):
    logger.info("")
    logger.info("═" * 60)
    logger.info("  GenoFold — Run Complete")
    logger.info("═" * 60)
    logger.info(f"  Output directory: {run_dir}")
    if data is not None:
        logger.info(f"  Nodes (valid bins):  {data.num_nodes}")
        logger.info(f"  Edges (Hi-C contacts): {data.edge_index.shape[1]:,}")
        logger.info(f"  Node feature dim:    {data.x.shape[1]}")
    if inference_output is not None:
        logger.info(f"  Embedding dim:       {inference_output.embeddings.shape[1]}")
    if recon_output is not None:
        logger.info(f"  3D coordinates:      {tuple(recon_output.coordinates.shape)}")
        logger.info(f"  Coord range:         [{recon_output.coordinates.min():.3f}, "
                    f"{recon_output.coordinates.max():.3f}]")
    if score_output is not None:
        logger.info(f"  Mean confidence:     {score_output.bin_confidence.mean():.3f}")
        logger.info(f"  Regions detected:    {len(score_output.regions)}")
        logger.info(f"  Mean uncertainty:    {score_output.uncertainty.radius.mean():.4f}")
    if explain_output is not None:
        logger.info(f"  Figures generated:   {len(explain_output.figure_paths)}")
    logger.info("")
    logger.info("  Output files:")
    from pathlib import Path
    for f in sorted(Path(run_dir).rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            logger.info(f"    {f.relative_to(run_dir)}  ({size_kb:.1f} KB)")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
