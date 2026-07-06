"""
experiments.fish_sparse  (Option 1: sparse-Hi-C accuracy against FISH)
--------------------------------------------------------------------
The full-depth FISH check (experiments.fish_validation) shows GenoFold ties the
best distance optimizer. The interesting regime is SPARSE Hi-C: as contacts are
thinned, a contact-only optimizer has less to solve from, while GenoFold can lean
on its node features and a backbone-continuity prior. We test whether GenoFold's
reconstructions stay closer to the MEASURED single-molecule structure (Bintu 2018
FISH) than the baselines as the input Hi-C is downsampled.

For each cell line and keep-fraction we downsample the 10 kb window contacts,
reconstruct with every method, coarse-grain to the 30 kb imaging grid, and score
Spearman against the (fixed, full-depth) FISH median-distance matrix. The raw-Hi-C
anchor is the sparse contact matrix itself, converted to distances. Downsampling
seed is fixed so every method sees the identical sparse input; GenoFold is averaged
over training seeds.

  cd genofold_pipeline && GENOFOLD_DEVICE=cpu python -m experiments.fish_sparse
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

from experiments.data_fetch import ChromDataset
from experiments.downsample_ablation import downsample_contacts
from experiments.fish_validation import (imaging_median_distance, _window_dataset,
                                         _coarse_grain, _seg_distance, _score,
                                         hic_wish_distance_30kb, CELLS, ALPHA, MIN_DETECT)
from experiments import metrics as M

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# GenoFold training config. Contact-dropout / backbone-prior augmentations were
# evaluated on held-out GM12878 chr22 (see train.py flags) and did not beat the base
# model, which is already sparse-robust; the probe therefore uses the base config.
GF = dict()
KEEP = [1.0, 0.25, 0.1, 0.05]
SEEDS = [0, 1, 2]


def _sparse_window(ds, C_window, keep, seed=0):
    """Downsample the 10 kb window contacts and rebuild a prepared graph."""
    Cd = C_window if keep >= 1.0 else downsample_contacts(C_window, keep, seed=seed)
    dsd = copy.copy(ds)
    dsd.contact_matrix = Cd
    dsd.coverage = Cd.sum(1).astype(np.float32)   # 1D coverage thins with the matrix
    from experiments.prep import prepare
    return prepare(dsd), Cd


def run_cell(cell):
    from experiments.train import train_genofold
    from experiments.baselines import shrec3d, threedmax

    Dimg, ndet, K = imaging_median_distance(cell)
    Dimg_r = np.where(ndet >= MIN_DETECT, Dimg, np.nan)
    ds, C_window = _window_dataset(K, cell)
    print(f"\n== {cell}: sparse-Hi-C vs FISH ({K} segments) ==")
    print(f"{'keep':>6} {'GenoFold':>16} {'3DMax':>9} {'ShRec3D':>9} {'RawHiC':>9} {'segs':>5}")

    out = {}
    for kf in KEEP:
        prep, Cd = _sparse_window(ds, C_window, kf, seed=0)
        Cv = prep.contact_valid
        vidx = prep.valid_idx.astype(np.int64)

        gf_cen = [_coarse_grain(
            train_genofold(prep.data, Cv, epochs=300, eval_every=100, mc_samples=1,
                           layout=prep.layout, held_out=0.0, report_final=True, seed=s,
                           log_prefix=f"[GF {cell} kf={kf} s{s}] ", verbose=False, **GF).coords,
            vidx, K) for s in SEEDS]
        base_cen = {name: _coarse_grain(fn(Cv, alpha=ALPHA), vidx, K)
                    for name, fn in [("3DMax", threedmax), ("ShRec3D", shrec3d)]}

        seg_ok = np.ones(K, bool)
        for c in list(base_cen.values()) + gf_cen:
            seg_ok &= np.isfinite(c).all(1)

        gf = [_score(_seg_distance(c), Dimg_r, seg_ok)[0] for c in gf_cen]
        row = dict(keep=kf, n_segments=int(seg_ok.sum()),
                   GenoFold=float(np.mean(gf)), GenoFold_std=float(np.std(gf)))
        for name, cen in base_cen.items():
            row[name] = _score(_seg_distance(cen), Dimg_r, seg_ok)[0]
        # raw sparse Hi-C anchor: the thinned contacts themselves as distances
        row["HiC_wish"] = _score(hic_wish_distance_30kb(Cd, K), Dimg_r, seg_ok)[0]
        out[f"{int(kf*100)}%"] = row
        print(f"{kf:>6} {row['GenoFold']:>10.3f}+/-{row['GenoFold_std']:<4.3f} "
              f"{row['3DMax']:>9.3f} {row['ShRec3D']:>9.3f} {row['HiC_wish']:>9.3f} "
              f"{row['n_segments']:>5}")
    return out


def main(cells=CELLS):
    out = {cell: run_cell(cell) for cell in cells}
    out["_meta"] = dict(keep_fractions=KEEP, seeds=len(SEEDS), genofold_config=GF,
                        metric="Spearman(model_dist, FISH_median_dist) recovering full-depth "
                               "FISH structure from downsampled Hi-C")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "fish_sparse.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/fish_sparse.json ({', '.join(cells)})")
    return out


if __name__ == "__main__":
    main()
