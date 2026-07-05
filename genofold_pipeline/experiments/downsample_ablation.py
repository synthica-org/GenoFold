"""
experiments.downsample_ablation
--------------------------------
Honest multi-modality ablation UNDER DATA SPARSITY.

At the deep sequencing depth of GM12878, the Hi-C contact graph is so informative
that the node-feature modalities (k-mer, GC, mappability) add little to final
accuracy -- the graph alone suffices. The value of multi-modality should appear
when Hi-C is sparse. We simulate shallow sequencing by randomly keeping only a
fraction of observed contacts, then ask whether the multi-modal model recovers the
TRUE (full-depth) structure better than a Hi-C-graph-only model.

For each keep-fraction we train two models on the downsampled matrix:
  * Full     -- all node features,
  * Hi-C only -- position feature only (structure from the contact graph),
and score both by dSCC against the FULL-depth wish distances. A positive
Full - (Hi-C only) gap means the extra modalities buy robustness to sparsity.
"""

from __future__ import annotations

import copy
import json
import logging
import os

import numpy as np

from experiments.data_fetch import build_chrom_dataset
from experiments.prep import prepare
from experiments.train import train_genofold
from experiments import metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("genofold.downsample")

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ALPHA = 0.25


def downsample_contacts(C: np.ndarray, keep: float, seed: int = 0) -> np.ndarray:
    """Randomly keep a fraction of observed upper-triangle contacts (symmetric)."""
    n = C.shape[0]
    up = np.triu(np.ones((n, n), dtype=bool), 1) & (C > 0)
    ii, jj = np.where(up)
    rng = np.random.default_rng(seed)
    k = rng.random(len(ii)) < keep
    Cd = np.zeros_like(C)
    Cd[ii[k], jj[k]] = C[ii[k], jj[k]]
    Cd = Cd + Cd.T
    return Cd


def run(chrom="chr22", resolution=50_000, fractions=(0.5, 0.2, 0.1, 0.05), epochs=300):
    ds = build_chrom_dataset("GM12878", chrom, resolution)
    C_full = ds.contact_matrix.astype(np.float32).copy()
    np.fill_diagonal(C_full, 0.0)

    rows = {}
    for kf in fractions:
        Cd = downsample_contacts(C_full, kf, seed=0)
        dsd = copy.copy(ds)
        dsd.contact_matrix = Cd
        dsd.coverage = Cd.sum(1).astype(np.float32)   # 1D coverage also thins
        prep = prepare(dsd)
        vidx = prep.valid_idx
        # ground-truth target = FULL-depth contacts over the (downsampled) valid bins
        C_target = C_full[np.ix_(vidx, vidx)].astype(np.float32)
        np.fill_diagonal(C_target, 0.0)
        obs_full = M.observed_pair_mask(C_target)

        res_full = train_genofold(prep.data, prep.contact_valid, epochs=epochs, eval_every=50,
                                  mc_samples=1, layout=prep.layout, held_out=0.0,
                                  log_prefix=f"[kf={kf} full] ", verbose=False)
        res_hic = train_genofold(prep.data, prep.contact_valid, epochs=epochs, eval_every=50,
                                 mc_samples=1, layout=prep.layout, held_out=0.0,
                                 drop_features=["ngs", "seq", "meta"],
                                 log_prefix=f"[kf={kf} hic] ", verbose=False)
        # score both against the FULL-depth structure
        d_full = M.dSCC(res_full.coords, C_target, ALPHA, pair_mask=obs_full)
        d_hic = M.dSCC(res_hic.coords, C_target, ALPHA, pair_mask=obs_full)
        n_edges = int((Cd > 0).sum() // 2)
        rows[f"{int(kf*100)}%"] = dict(keep=kf, n_bins=int(prep.contact_valid.shape[0]),
                                       n_edges=n_edges, dscc_full=d_full, dscc_hic=d_hic,
                                       gain=d_full - d_hic)
        log.info(f"keep={kf:.2f}: Full={d_full:.4f}  Hi-C-only={d_hic:.4f}  "
                 f"gain={d_full-d_hic:+.4f}  ({prep.contact_valid.shape[0]} bins, {n_edges} edges)")

    with open(os.path.join(RESULTS, "downsample_ablation.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    log.info("wrote downsample_ablation.json")
    return rows


if __name__ == "__main__":
    run()
