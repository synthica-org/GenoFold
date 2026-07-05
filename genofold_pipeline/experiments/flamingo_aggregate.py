"""
experiments.flamingo_aggregate
------------------------------
Score one or more REAL FLAMINGO coordinate files (from run_flamingo_chr22.R) on
the shared GM12878 chr22 valid-bin matrix, reporting global dSCC AND short-range
(<=1 Mb) dSCC, mean +/- std across seeds. FLAMINGO is a multi-scale method built
for 1-5 kb; run here at 50 kb for comparability, its coarse 1 Mb inter-domain
backbone makes long-range pairs its weak point, so both numbers are reported.

Usage: python -m experiments.flamingo_aggregate ../flamingo_run/coords_seed*.tsv
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from experiments.data_fetch import build_chrom_dataset
from experiments.prep import prepare
from experiments import metrics as M

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ALPHA = 0.25
SHORT_BINS = 20   # <=20 bins @50kb = <=1 Mb (FLAMINGO domain scale)


def score(coords_file, prep):
    d = np.loadtxt(coords_file)
    bp, xyz = d[:, 0], d[:, 1:4].astype(np.float32)
    pos = {b: i for i, b in enumerate(prep.valid_idx)}
    bins = (bp / prep.ds.resolution).round().astype(int)
    keep = [i for i, b in enumerate(bins) if b in pos]
    rows = np.array([pos[bins[i]] for i in keep])
    coords = xyz[keep]
    C = prep.contact_valid[np.ix_(rows, rows)].astype(np.float32)
    np.fill_diagonal(C, 0.0)
    n = C.shape[0]
    g = M.dSCC(coords, C, ALPHA)
    tad = M.tad_boundary_f1(C, coords, alpha=ALPHA)["f1"]
    up = np.triu(np.ones((n, n), bool), 1) & (C > 0)
    ii, jj = np.where(up); sep = np.abs(ii - jj)
    sm = np.zeros((n, n), bool); sm[ii[sep <= SHORT_BINS], jj[sep <= SHORT_BINS]] = True
    s = M.dSCC(coords, C, ALPHA, pair_mask=sm)
    return g, s, tad, len(rows)


def main():
    files = []
    for a in sys.argv[1:]:
        files.extend(glob.glob(a))
    files = sorted(set(files))
    if not files:
        raise SystemExit("no coordinate files given")
    ds = build_chrom_dataset("GM12878", "chr22", 50_000)
    prep = prepare(ds)
    g_all, s_all, tad_all = [], [], []
    for f in files:
        g, s, tad, n = score(f, prep)
        g_all.append(g); s_all.append(s); tad_all.append(tad)
        print(f"{os.path.basename(f)}: global dSCC={g:.4f}  short-range(<=1Mb) dSCC={s:.4f}  TAD_F1={tad:.3f}  ({n} bins)")
    out = dict(
        dscc_mean=float(np.mean(g_all)), dscc_std=float(np.std(g_all)),
        dscc_short_mean=float(np.mean(s_all)), dscc_short_std=float(np.std(s_all)),
        tad_mean=float(np.mean(tad_all)), tad_std=float(np.std(tad_all)),
        n_seeds=len(files), values=g_all,
    )
    json.dump(out, open(os.path.join(RESULTS, "flamingo_real.json"), "w"), indent=2)
    print(f"\nFLAMINGO (real, {len(files)} seeds): global {out['dscc_mean']:.3f}+/-{out['dscc_std']:.3f}  "
          f"short-range {out['dscc_short_mean']:.3f}+/-{out['dscc_short_std']:.3f}")
    print("wrote results/flamingo_real.json")


if __name__ == "__main__":
    main()
