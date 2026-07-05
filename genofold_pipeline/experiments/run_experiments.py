"""
experiments.run_experiments
----------------------------
Runs the full GenoFold empirical study and writes results/ + structures for
figures. Each block maps directly to a reviewer requirement:

  1. Main benchmark      — GenoFold vs 3DMax / HiC-GNN / FLAMINGO-style / ShRec3D
                           on GM12878 chr22 (dSCC, distance-Pearson, TAD-F1, runtime).
  2. Ablation            — full vs -kmer vs -metadata vs Hi-C-only, overall and
                           in sparse (low-coverage) regions.
  3. Scaling             — multiple whole chromosomes; bins, runtime, memory.
  4. Cross-cell-line     — GM12878-trained model transferred to IMR90 & K562.
  5. Uncertainty calib.  — MC-dropout uncertainty vs per-bin error.
  6. Resolution          — chr22 at 50kb and 25kb.

Usage:  python -m experiments.run_experiments [--fast]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import time

import numpy as np

from experiments.data_fetch import build_chrom_dataset
from experiments.prep import prepare
from experiments.train import train_genofold, evaluate_transfer
from experiments.baselines import BASELINES
from experiments import metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("genofold.experiments")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
STRUCT_DIR = os.path.join(RESULTS_DIR, "structures")
os.makedirs(STRUCT_DIR, exist_ok=True)

ALPHA = 0.25


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kB
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def save_json(name, obj):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    log.info(f"wrote {path}")


# ---------------------------------------------------------------------------

def run_main_benchmark(prep, epochs):
    log.info("=== BLOCK 1: main benchmark (GM12878 chr22) ===")
    C = prep.contact_valid
    obs = M.observed_pair_mask(C)
    rows = {}

    t0 = time.time()
    # Head-to-head: train on ALL observed contacts (transductive), scored over all
    # observed pairs — the same data regime the classical optimisers use.
    res = train_genofold(prep.data, C, epochs=epochs, eval_every=25, mc_samples=25,
                         layout=prep.layout, held_out=0.0, log_prefix="[GenoFold] ",
                         verbose=False, report_final=True)
    dscc_all = M.dSCC(res.coords, C, ALPHA, pair_mask=obs)
    rows["GenoFold"] = dict(
        dscc=dscc_all, pearson=M.distance_pearson(res.coords, C, ALPHA, pair_mask=obs),
        tad_f1=M.tad_boundary_f1(C, res.coords, alpha=ALPHA)["f1"],
        seconds=res.seconds, n_bins=int(C.shape[0]),
    )
    # Generalisation (GenoFold-only capability the optimisers lack): a second run
    # that holds out 20% of contact pairs, reporting train vs held-out dSCC.
    gen = train_genofold(prep.data, C, epochs=epochs, eval_every=25, mc_samples=1,
                        layout=prep.layout, held_out=0.2, log_prefix="[GenoFold/gen] ",
                        verbose=False)
    save_json("generalization.json", dict(dscc_train=gen.dscc_train, dscc_heldout=gen.dscc_test))
    log.info(f"  generalisation: train dSCC {gen.dscc_train:.4f} vs held-out {gen.dscc_test:.4f}")
    np.save(os.path.join(STRUCT_DIR, "GenoFold_chr22.npy"), res.coords)
    np.save(os.path.join(STRUCT_DIR, "GenoFold_chr22_uncertainty.npy"), res.uncertainty)
    np.save(os.path.join(STRUCT_DIR, "GenoFold_chr22_contact.npy"), C)  # aligned target matrix

    # Integrated-Gradients feature-group importance on the TRAINED model
    try:
        from experiments.explain_trained import feature_group_importance
        ig = feature_group_importance(res.model, prep.data, prep.layout)
        save_json("feature_importance.json", ig)
        log.info(f"  IG feature importance: {ig}")
    except Exception as e:  # noqa: BLE001
        log.warning(f"IG importance failed: {e}")

    for name, fn in BASELINES.items():
        t = time.time()
        coords = fn(C, alpha=ALPHA)
        rows[name] = dict(
            dscc=M.dSCC(coords, C, ALPHA), dscc_train=None,
            pearson=M.distance_pearson(coords, C, ALPHA),
            tad_f1=M.tad_boundary_f1(C, coords, alpha=ALPHA)["f1"],
            seconds=time.time() - t, n_bins=int(C.shape[0]),
        )
        np.save(os.path.join(STRUCT_DIR, f"{name}_chr22.npy"), coords)
        log.info(f"  {name}: dSCC={rows[name]['dscc']:.4f}")

    log.info(f"block 1 done in {time.time()-t0:.0f}s")
    save_json("main_benchmark.json", rows)
    return rows, res


def run_ablation(prep, epochs):
    log.info("=== BLOCK 2: multimodality ablation (chr22) ===")
    C = prep.contact_valid
    # sparse-region pairs: pairs where either endpoint is in the low-coverage tercile
    cov = C.sum(1)
    sparse_bins = cov < np.quantile(cov, 0.33)
    obs = M.observed_pair_mask(C)
    ii, jj = np.where(obs)
    sp = sparse_bins[ii] | sparse_bins[jj]
    sparse_mask = np.zeros_like(obs)
    sparse_mask[ii[sp], jj[sp]] = True

    configs = {
        "Full (multi-modal)": [],
        "− k-mer": ["seq"],
        "− metadata (GC/map)": ["meta"],
        "Hi-C only": ["ngs", "seq", "meta"],
    }
    rows = {}
    for label, drop in configs.items():
        res = train_genofold(prep.data, C, epochs=epochs, eval_every=25, mc_samples=1,
                             layout=prep.layout, drop_features=drop,
                             log_prefix=f"[{label}] ", verbose=False)
        rows[label] = dict(
            dscc=res.dscc_test,
            dscc_sparse=M.dSCC(res.coords, C, ALPHA, pair_mask=sparse_mask),
            seconds=res.seconds,
        )
        log.info(f"  {label}: dSCC={res.dscc_test:.4f} sparse={rows[label]['dscc_sparse']:.4f}")
    save_json("ablation.json", rows)
    return rows


def run_scaling(cell_line, chroms, resolution, epochs):
    log.info(f"=== BLOCK 3: scaling across chromosomes @ {resolution//1000}kb ===")
    rows = {}
    for ch in chroms:
        ds = build_chrom_dataset(cell_line, ch, resolution)
        prep = prepare(ds)
        C = prep.contact_valid
        m0 = peak_rss_mb(); t = time.time()
        res = train_genofold(prep.data, C, epochs=epochs, eval_every=50, mc_samples=1,
                             layout=prep.layout, log_prefix=f"[{ch}] ", verbose=False)
        rows[ch] = dict(n_bins=int(C.shape[0]), dscc=res.dscc_test,
                        seconds=time.time() - t, peak_rss_mb=peak_rss_mb(),
                        dense_matrix_mb=float(C.nbytes) / 1e6)
        log.info(f"  {ch}: {C.shape[0]} bins, dSCC={res.dscc_test:.4f}, "
                 f"{rows[ch]['seconds']:.0f}s, peakRSS={rows[ch]['peak_rss_mb']:.0f}MB")
    save_json(f"scaling_{resolution}.json", rows)
    return rows


def run_cross_cell(gm_prep, gm_res, targets, resolution, epochs):
    log.info("=== BLOCK 4: cross-cell-line generalisation ===")
    rows = {}
    C_gm = gm_prep.contact_valid
    rows["GM12878 (train)"] = dict(dscc=gm_res.dscc_test, mode="in-domain", n_bins=int(C_gm.shape[0]))
    for cl in targets:
        ds = build_chrom_dataset(cl, gm_prep.ds.chrom, resolution)
        prep = prepare(ds)
        C = prep.contact_valid
        # transfer: GM12878-trained model applied to target cell line, no retraining
        transfer = evaluate_transfer(gm_res.model, prep.data, C, alpha=ALPHA)
        # retrained from scratch on target (upper bound reference)
        res = train_genofold(prep.data, C, epochs=epochs, eval_every=50, mc_samples=1,
                             layout=prep.layout, log_prefix=f"[{cl}] ", verbose=False)
        rows[cl] = dict(dscc_transfer=transfer, dscc_retrained=res.dscc_test,
                        n_bins=int(C.shape[0]), mode="transfer")
        log.info(f"  {cl}: transfer dSCC={transfer:.4f}, retrained dSCC={res.dscc_test:.4f}")
    save_json("cross_cell.json", rows)
    return rows


def run_calibration(prep, gm_res):
    log.info("=== BLOCK 5: uncertainty calibration (chr22) ===")
    C = prep.contact_valid
    cal = M.calibration(gm_res.uncertainty, gm_res.coords, C, alpha=ALPHA)
    out = dict(spearman=cal.spearman, curve_x=cal.curve_x.tolist(), curve_y=cal.curve_y.tolist())
    np.save(os.path.join(STRUCT_DIR, "calibration_error.npy"), cal.per_bin_error)
    np.save(os.path.join(STRUCT_DIR, "calibration_uncertainty.npy"), cal.per_bin_uncertainty)
    log.info(f"  calibration Spearman(uncertainty, error) = {cal.spearman:.4f}")
    save_json("calibration.json", out)
    return out


def run_resolution(cell_line, chrom, epochs):
    log.info("=== BLOCK 6: resolution comparison (chr22 50kb vs 25kb) ===")
    rows = {}
    for res_bp in (50_000, 25_000):
        ds = build_chrom_dataset(cell_line, chrom, res_bp)
        prep = prepare(ds)
        C = prep.contact_valid
        r = train_genofold(prep.data, C, epochs=epochs, eval_every=50, mc_samples=1,
                           layout=prep.layout, log_prefix=f"[{res_bp}] ", verbose=False)
        rows[f"{res_bp//1000}kb"] = dict(n_bins=int(C.shape[0]), dscc=r.dscc_test, seconds=r.seconds)
        log.info(f"  {res_bp//1000}kb: {C.shape[0]} bins dSCC={r.dscc_test:.4f}")
    save_json("resolution.json", rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer chromosomes / targets for a quick pass")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs")
    ap.add_argument("--blocks", default="1,2,3,4,5,6")
    args = ap.parse_args()
    blocks = set(args.blocks.split(","))

    epochs = args.epochs if args.epochs is not None else (150 if args.fast else 300)
    scaling_chroms = ["chr22", "chr20"] if args.fast else ["chr22", "chr20", "chr16"]
    targets = ["IMR90"] if args.fast else ["IMR90", "K562"]

    summary = {"config": dict(epochs=epochs, alpha=ALPHA, fast=args.fast)}
    ds = build_chrom_dataset("GM12878", "chr22", 50_000)
    prep = prepare(ds)
    gm_res = None

    if "1" in blocks:
        summary["main"], gm_res = run_main_benchmark(prep, epochs)
    if "2" in blocks:
        summary["ablation"] = run_ablation(prep, epochs)
    if "3" in blocks:
        summary["scaling"] = run_scaling("GM12878", scaling_chroms, 50_000, epochs)
    if "4" in blocks:
        if gm_res is None:
            _, gm_res = run_main_benchmark(prep, epochs)
        summary["cross_cell"] = run_cross_cell(prep, gm_res, targets, 50_000, epochs)
    if "5" in blocks:
        if gm_res is None:
            _, gm_res = run_main_benchmark(prep, epochs)
        summary["calibration"] = run_calibration(prep, gm_res)
    if "6" in blocks:
        summary["resolution"] = run_resolution("GM12878", "chr22", epochs)

    save_json("summary.json", summary)
    log.info("ALL BLOCKS COMPLETE")


if __name__ == "__main__":
    main()
