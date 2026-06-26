<div align="center">

# GenoFold

```
  NGS · Hi-C · Metadata
         ╲   │   ╱
          ╲  │  ╱
      ┌─────────────┐
      │   GenoFold  │
      └─────────────┘
            │
     3D structure · confidence · insight
```

*Folding fragmented genomic data into structure you can read, trust, and explain.*

<br/>

**A confidence-aware, explainable, multi-modal AI framework**  
for reconstructing three-dimensional genome architecture

<br/>

**Ram Suhrut Seelamankalugari** (SYALIS) · **Quang Bui** (Synthica)

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

GenoFold is a research project developing an AI-driven, end-to-end pipeline for reconstructing three-dimensional genome architecture. By integrating Hi-C contact data, next-generation sequencing (NGS) reads, and genomic metadata across numerous species, GenoFold aims to improve the accuracy, robustness, and interpretability of 3D genome models compared with existing computational approaches.

---

## Research Question

> How can a confidence-aware, explainable, multi-modal AI framework integrating Hi-C, NGS, and genomic metadata of numerous species improve the accuracy, robustness, and interpretability of 3D genome reconstruction compared with existing approaches?

---

## Problem Statement

Current computational methods for 3D genome construction face several limitations:

| Challenge | Description |
|-----------|-------------|
| **Data constraints** | Heavy reliance on distance conversion and sensitivity to noise or resolution limits in Hi-C contact matrices |
| **Interpretability** | Models often function as black boxes, failing to integrate raw sequencing features or provide confidence-aware outputs |
| **Fragmentation** | Pipelines are disjointed, requiring multiple independent tools for preprocessing, assembly, and inference—reducing efficiency and reproducibility |

---

## Core Contributions & Novelty

GenoFold addresses these gaps through four key innovations:

### Multi-Modal Integration
Combines NGS reads, Hi-C interaction matrices, and genomic metadata rather than relying on Hi-C contact maps alone.

### AI-Enhanced Processing
Utilizes neural networks, transformer-based models, and autoencoders to clean sparse or noisy data and capture long-range genomic dependencies.

### Confidence & Transparency
Outputs reliability scores and uncertainty estimates, with explainable insights such as feature-importance heatmaps—making models transparent for biomedical and clinical applications.

### Unified Pipeline
Handles preprocessing, inference, scoring, and interpretability within a single, cohesive framework.

---

## Research Pipeline

GenoFold follows a six-stage end-to-end workflow:

```mermaid
flowchart LR
    A[1. Preprocess] --> B[2. Represent]
    B --> C[3. Infer]
    C --> D[4. Reconstruct]
    D --> E[5. Score]
    E --> F[6. Explain]
```

| Stage | Description |
|-------|-------------|
| **1. Preprocess** | Filter, correct, align, normalize, and denoise raw NGS and Hi-C data |
| **2. Represent** | Convert genomic interactions into graphs and generate sequence embeddings |
| **3. Infer** | Apply GNN encoders and transformer attention to capture relationships and predict early 3D representations |
| **4. Reconstruct** | Decode learned features into 3D coordinates, spatial distances, and structural domains |
| **5. Score** | Estimate prediction reliability and uncertainty on a per-region basis |
| **6. Explain** | Generate heatmaps and summaries highlighting influential loci and structural importance |

---

## Benchmark Comparisons

GenoFold will be evaluated against established methods across complementary dimensions:

| Method | Evaluation Focus |
|--------|-------------------|
| [**HiC-GNN**](pubmed.ncbi.nlm.nih) | Graph learning and cross-dataset transfer capabilities |
| [**FLAMINGO**](https://www.nature.com/articles/s41467-022-30270-2) | Robustness to sparse data and high-resolution reconstruction |
| [**3DMax**](https://github.com/BDM-Lab/3DMax) | Classic maximum-likelihood, distance-based baseline |

---

## Installation & Setup

### Requirements

- Python 3.10+
- pip
- ~50 GB of free disk space (real Hi-C datasets are large)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/GenoFold.git
cd GenoFold
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs the core stack (PyTorch, PyTorch Geometric, NumPy, pandas, SciPy, scikit-learn) along with genomics-specific packages used for real data: `cooler`, `hic-straw`, `pysam`, `pyfaidx`, and `pyBigWig`.

> **Note:** If you're on a machine without a GPU, install the CPU-only build of PyTorch first for a faster setup:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 3. Run the synthetic smoke test

Before touching any real data, confirm the full pipeline runs end-to-end on generated synthetic data:

```bash
python3 test_pipeline.py
```

This runs all 6 stages and should finish with `ALL ASSERTIONS PASSED` for each stage.

### 4. Download real data

GenoFold ships with scripts to fetch real, publicly available datasets — no account or login required.

**Hi-C contact data** (GM12878, Rao et al. 2014, via GEO — several GB):
```bash
bash scripts/download_real_data.sh
```

**Reference genome + real NGS coverage** (hg19 chr22 + GM12878 DNase-seq, via UCSC/ENCODE — a few MB):
```bash
bash scripts/download_reference_data.sh
```

### 5. Run the pipeline on real data

```bash
python3 scripts/run_real_data.py \
    --chrom chr22 \
    --bins 200 \
    --fasta data/reference/chr22.fa \
    --bigwig data/reference/GM12878_DNase.bigWig
```

Each run creates a uniquely named, timestamped folder under `outputs/` (e.g. `outputs/GM12878_chr22_100kb_2026-01-15_14-32/`) containing the predicted 3D structure (`.npy`, `.pdb`, `.tsv`) and a full set of static and interactive explainability figures.

Run `python3 scripts/run_real_data.py --help` to see all available options (chromosome, resolution, bin count, normalization method, and more).

---

## Authors

| Name | Affiliation |
|------|-------------|
| **Ram Suhrut Seelamankalugari** | SYALIS |
| **Quang Bui** | Synthica |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

*Citation information will be added upon publication.*
