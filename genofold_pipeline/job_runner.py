"""
genofold_pipeline.job_runner
------------------------------
Runs the real GenoFold pipeline (stages 1-6) against user-uploaded files
and writes a JSON summary the frontend can render.

This is the bridge between the Flask web layer and the existing
pipeline.py / genofold package — no science logic lives here, only
orchestration, progress reporting, and result serialization.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("genofold.job_runner")

JOB_REGISTRY: dict = {}
_REGISTRY_LOCK = threading.Lock()


@dataclass
class JobState:
    job_id: str
    params: dict
    paths: dict
    results_dir: str
    status: str = "queued"          # queued | running | done | error
    stage: str = ""
    percent: int = 0
    message: str = "Queued"
    error_detail: Optional[str] = None


def _set_progress(job: JobState, stage: str, percent: int, message: str):
    job.status = "running"
    job.stage = stage
    job.percent = percent
    job.message = message
    logger.info(f"[{job.job_id}] {stage} ({percent}%) — {message}")


def run_pipeline_job(job: JobState):
    Path(job.results_dir).mkdir(parents=True, exist_ok=True)

    _set_progress(job, "load", 5, "Reading uploaded files")

    from genofold.utils.io import load_hic, load_cool, load_bigwig_coverage
    from genofold.utils.real_metadata import (
        build_metadata_table, fetch_gc_content_from_fasta, fetch_sequences_from_fasta,
    )
    from pipeline import GenoFoldPipeline, PipelineConfig

    chrom = job.params["chrom"]
    res = job.params["resolution"]
    max_bins = job.params["max_bins"]

    # ── Load Hi-C ─────────────────────────────────────────────────────────────
    hic_path = job.paths["hic_path"]
    hic_format = job.paths.get("hic_format", "hic")

    if hic_format == "hic":
        matrix = load_hic(hic_path, resolution=res, chrom=chrom)
    else:
        matrix = load_cool(hic_path, resolution=res, chrom=chrom)

    n_bins = min(max_bins, matrix.shape[0])
    matrix = matrix[:n_bins, :n_bins]

    _set_progress(job, "load", 15, f"Loaded {n_bins} bins of Hi-C contact data")

    # ── NGS coverage ──────────────────────────────────────────────────────────
    used_real_ngs = False
    if "ngs_path" in job.paths:
        ngs_path = job.paths["ngs_path"]
        if ngs_path.lower().endswith((".bigwig", ".bw")):
            coverage = load_bigwig_coverage(ngs_path, chrom, 0, n_bins, res)
            used_real_ngs = True
        elif ngs_path.lower().endswith(".npy"):
            coverage = np.load(ngs_path).astype(np.float32)[:n_bins]
            used_real_ngs = True
        else:
            coverage = matrix.sum(axis=1).astype(np.float32)
    else:
        coverage = matrix.sum(axis=1).astype(np.float32)

    _set_progress(job, "load", 25, "NGS coverage ready")

    # ── Sequences + GC content ───────────────────────────────────────────────
    used_real_fasta = False
    gc_content = None
    if "fasta_path" in job.paths:
        fasta_path = job.paths["fasta_path"]
        sequences = fetch_sequences_from_fasta(fasta_path, chrom, 0, n_bins, res)
        gc_content = fetch_gc_content_from_fasta(fasta_path, chrom, 0, n_bins, res)
        used_real_fasta = True
    else:
        rng = np.random.default_rng(42)
        bases = list("ACGT")
        seq_len = min(res, 1000)
        sequences = ["".join(rng.choice(bases, size=seq_len)) for _ in range(n_bins)]

    _set_progress(job, "load", 32, "Sequence data ready")

    # ── Metadata ──────────────────────────────────────────────────────────────
    if "metadata_path" in job.paths:
        import pandas as pd
        metadata_df = pd.read_csv(job.paths["metadata_path"])
    else:
        metadata_df = build_metadata_table(
            n_bins=n_bins, species=job.params["species"], chrom=chrom,
            chrom_start=0, bin_size=res, gc_content=gc_content,
        )

    # ── Run stages 1-2 ────────────────────────────────────────────────────────
    _set_progress(job, "preprocess", 38, "Normalizing Hi-C contacts (ICE)")

    config = PipelineConfig(
        bin_size=res, top_k_edges=50, kmer_k=4, denoise_sigma=1.0,
        sparse_threshold=0.0, contact_threshold=0.0,
        zero_fraction_threshold=0.97,
    )
    pipeline = GenoFoldPipeline(config=config)
    data = pipeline.run(
        contact_matrix=matrix, coverage=coverage,
        sequences=sequences, metadata_df=metadata_df,
    )

    _set_progress(job, "represent", 48, f"Built contact graph: {data.num_nodes} bins, {data.edge_index.shape[1]} edges")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    _set_progress(job, "infer", 55, "Running graph attention + transformer encoder")
    inference_output, infer_model = pipeline.infer(data, hidden_dim=64)

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    _set_progress(job, "reconstruct", 68, "Reconstructing 3D coordinates")
    recon_output, recon_model = pipeline.reconstruct(
        inference_output, hidden_dim=64, export_dir=job.results_dir, chrom=chrom,
    )

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    _set_progress(job, "score", 80, "Estimating confidence and uncertainty")
    score_output, score_model = pipeline.score(
        inference_output, recon_output, data.edge_index,
        reconstruct_model=recon_model, hidden_dim=64,
        n_mc_samples=8, use_mc_dropout=True,
    )

    from genofold.reconstruct.export import export_structure
    export_structure(
        coords=recon_output.coordinates, output_dir=job.results_dir,
        prefix="genofold_structure_scored", bin_size=res, chrom=chrom,
        confidence=score_output.bin_confidence,
    )

    # ── Stage 6 ───────────────────────────────────────────────────────────────
    _set_progress(job, "explain", 90, "Generating explainability figures")
    explain_dir = str(Path(job.results_dir) / "explain")
    explain_output, explainer = pipeline.explain(
        infer_model, recon_model, data, inference_output, recon_output,
        score_output=score_output, output_dir=explain_dir, ig_steps=20, top_k_loci=10,
    )

    # ── Build summary.json for the frontend ──────────────────────────────────
    _set_progress(job, "finalize", 97, "Writing results")

    coords = recon_output.coordinates.detach().cpu().numpy()
    confidence = score_output.bin_confidence.detach().cpu().numpy()
    uncertainty = score_output.uncertainty.radius.detach().cpu().numpy()
    influence = explain_output.influence_score.detach().cpu().numpy()

    bins_payload = []
    for i in range(coords.shape[0]):
        bins_payload.append({
            "i": i,
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "confidence": round(float(confidence[i]), 4),
            "uncertainty": round(float(uncertainty[i]), 4),
            "influence": round(float(influence[i]), 4),
        })

    regions_payload = [
        {
            "start_bin": r.start_bin, "end_bin": r.end_bin, "size": r.size,
            "mean_confidence": round(r.mean_confidence, 4),
            "min_confidence": round(r.min_confidence, 4),
        }
        for r in score_output.regions
    ]

    group_attr = {
        name: round(float(t.mean().item()), 6)
        for name, t in explain_output.group_attributions.items()
    }

    top_loci = sorted(
        range(len(influence)), key=lambda i: influence[i], reverse=True
    )[:10]

    summary = {
        "job_id": job.job_id,
        "dataset_name": job.params["dataset_name"],
        "chrom": chrom,
        "resolution": res,
        "num_bins": int(coords.shape[0]),
        "num_edges": int(data.edge_index.shape[1]),
        "node_feature_dim": int(data.x.shape[1]),
        "used_real_ngs": used_real_ngs,
        "used_real_fasta": used_real_fasta,
        "mean_confidence": round(float(confidence.mean()), 4),
        "mean_uncertainty": round(float(uncertainty.mean()), 4),
        "num_regions": len(regions_payload),
        "bins": bins_payload,
        "regions": regions_payload,
        "feature_importance": group_attr,
        "top_influential_loci": top_loci,
        "figures": {
            "feature_importance_html": "explain/feature_importance.html",
            "attention_heatmap_html": "explain/transformer_attention.html",
            "influential_loci_html": "explain/influential_loci.html",
            "structure_confidence_html": "explain/structure_confidence.html",
        },
        "downloads": {
            "pdb": "genofold_structure_scored.pdb",
            "npy": "genofold_structure_scored.npy",
            "tsv": "genofold_structure_scored.tsv",
        },
    }

    with open(Path(job.results_dir) / "summary.json", "w") as f:
        json.dump(summary, f)

    job.status = "done"
    job.stage = "done"
    job.percent = 100
    job.message = "Structure ready"
    logger.info(f"[{job.job_id}] complete")
