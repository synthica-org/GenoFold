"""
GenoFold web application — Flask backend.

Routes
------
GET  /                      Upload page (the "lab bench")
POST /api/submit            Accepts uploaded files, starts a background job
GET  /api/status/<job_id>   Polls job progress (stage, percent, message)
GET  /results/<job_id>      Results page (3D viewer + explainability)
GET  /api/results/<job_id>  JSON payload backing the results page
GET  /outputs/<job_id>/<f>  Serves generated files (PDB, PNG, HTML figures)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, abort

sys.path.insert(0, str(Path(__file__).parent / "genofold_pipeline"))

from job_runner import run_pipeline_job, JOB_REGISTRY, JobState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("genofold.web")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    "hic": {".hic"},
    "cool": {".cool", ".mcool"},
    "ngs": {".bigwig", ".bw", ".npy", ".bam"},
    "fasta": {".fa", ".fasta", ".fa.gz"},
    "metadata": {".csv", ".tsv"},
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = None  # No Limit for File Uploads


def _ext_ok(filename: str, kinds) -> bool:
    name = filename.lower()
    allowed = set()
    for k in kinds:
        allowed |= ALLOWED_EXTENSIONS.get(k, set())
    return any(name.endswith(ext) for ext in allowed)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/submit", methods=["POST"])
def submit():
    job_id = uuid.uuid4().hex[:10]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    errors = []

    def resolve_input(field_file, field_path, kinds, label):
        uploaded = request.files.get(field_file)
        local_path = (request.form.get(field_path) or "").strip()
        if uploaded and uploaded.filename:
            if not _ext_ok(uploaded.filename, kinds):
                errors.append(f"{label}: unsupported file type.")
                return None
            p = job_dir / uploaded.filename
            uploaded.save(p)
            return str(p)
        elif local_path:
            p = Path(local_path).expanduser().resolve()
            if not p.exists():
                errors.append(f"{label}: path not found — {local_path}")
                return None
            if not _ext_ok(p.name, kinds):
                errors.append(f"{label}: unsupported file type at path.")
                return None
            return str(p)
        return None

    hic_path = resolve_input("hic_file", "hic_path", ["hic", "cool"], "Hi-C file")
    if not hic_path:
        if not any(e.startswith("Hi-C") for e in errors):
            errors.append("A Hi-C file or local path is required.")
    else:
        saved["hic_path"] = hic_path
        saved["hic_format"] = "hic" if hic_path.lower().endswith(".hic") else "cool"

    ngs_path = resolve_input("ngs_file", "ngs_path", ["ngs"], "NGS file")
    if ngs_path:
        saved["ngs_path"] = ngs_path

    fasta_path = resolve_input("fasta_file", "fasta_path", ["fasta"], "Reference genome")
    if fasta_path:
        saved["fasta_path"] = fasta_path

    metadata_path = resolve_input("metadata_file", "metadata_path", ["metadata"], "Metadata")
    if metadata_path:
        saved["metadata_path"] = metadata_path

    if errors:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"ok": False, "errors": errors}), 400

    params = {
        "species": request.form.get("species", "human"),
        "chrom": request.form.get("chrom", "chr22"),
        "resolution": int(request.form.get("resolution", 100000)),
        "max_bins": int(request.form.get("max_bins", 200)),
        "dataset_name": request.form.get("dataset_name", "Upload"),
    }

    job = JobState(job_id=job_id, params=params, paths=saved, results_dir=str(RESULTS_DIR / job_id))
    JOB_REGISTRY[job_id] = job

    thread = threading.Thread(target=_run_job_safe, args=(job_id,), daemon=True)
    thread.start()

    return jsonify({"ok": True, "job_id": job_id})


def _run_job_safe(job_id: str):
    job = JOB_REGISTRY[job_id]
    try:
        run_pipeline_job(job)
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job.status = "error"
        job.message = str(e)
        job.error_detail = traceback.format_exc()


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOB_REGISTRY.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    return jsonify({
        "ok": True,
        "status": job.status,
        "stage": job.stage,
        "percent": job.percent,
        "message": job.message,
        "error_detail": job.error_detail if job.status == "error" else None,
    })


@app.route("/results/<job_id>")
def results_page(job_id):
    job = JOB_REGISTRY.get(job_id)
    if job is None:
        abort(404)
    return render_template("results.html", job_id=job_id)


@app.route("/api/results/<job_id>")
def results_json(job_id):
    job = JOB_REGISTRY.get(job_id)
    if job is None or job.status != "done":
        return jsonify({"ok": False, "error": "Results not ready"}), 404
    summary_path = Path(job.results_dir) / "summary.json"
    if not summary_path.exists():
        return jsonify({"ok": False, "error": "Summary missing"}), 500
    with open(summary_path) as f:
        summary = json.load(f)
    return jsonify({"ok": True, **summary})


@app.route("/outputs/<job_id>/<path:filename>")
def serve_output(job_id, filename):
    job_dir = RESULTS_DIR / job_id
    if not (job_dir / filename).resolve().is_relative_to(job_dir.resolve()):
        abort(403)
    return send_from_directory(job_dir, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050, threaded=True)
