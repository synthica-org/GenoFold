"""
experiments.flamingo_real
-------------------------
Score a structure produced by the ORIGINAL FLAMINGO (Wang et al. 2022,
github.com/wangjr03/FLAMINGO) with GenoFold's dSCC on the SAME valid-bin contact
matrix as every other method, so the real FLAMINGO number can replace the
FLAMINGO-style re-implementation in the benchmark table.

Run FLAMINGO in R at 50 kb fragment resolution on the same GM12878 chr22 .hic,
export its coordinates, then:

    python -m experiments.flamingo_real /path/to/flamingo_chr22.tsv

The coordinate file is whitespace/tab/comma-separated with either
  (genomic_start_bp, x, y, z)   [default]  or   (x, y, z)  in 50 kb bin order (--binorder).
Rows FLAMINGO left unmodeled (all-NaN / missing) are dropped; dSCC is computed on
the bins present in BOTH FLAMINGO's output and GenoFold's valid set.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from experiments.data_fetch import build_chrom_dataset
from experiments.prep import prepare
from experiments import metrics as M

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_coords(path, resolution, binorder):
    rows = np.genfromtxt(path, delimiter=None if _sniff(path) is None else _sniff(path))
    rows = np.atleast_2d(rows)
    if binorder or rows.shape[1] == 3:
        xyz = rows[:, -3:]
        bin_idx = np.arange(len(xyz))
    else:  # (start_bp, x, y, z)
        bin_idx = (rows[:, 0] / resolution).round().astype(int)
        xyz = rows[:, 1:4]
    keep = np.isfinite(xyz).all(1)
    return bin_idx[keep], xyz[keep]


def _sniff(path):
    with open(path) as fh:
        line = fh.readline()
    return "," if "," in line else ("\t" if "\t" in line else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coords", help="FLAMINGO coordinate file (see module docstring)")
    ap.add_argument("--cell", default="GM12878")
    ap.add_argument("--chrom", default="chr22")
    ap.add_argument("--resolution", type=int, default=50_000)
    ap.add_argument("--binorder", action="store_true",
                    help="coords are (x,y,z) in bin order rather than (start_bp,x,y,z)")
    ap.add_argument("--alpha", type=float, default=0.25)
    args = ap.parse_args()

    ds = build_chrom_dataset(args.cell, args.chrom, args.resolution)
    prep = prepare(ds)
    valid = prep.valid_idx                                    # original-bin indices GenoFold scored on
    pos_of_valid = {b: i for i, b in enumerate(valid)}        # original bin -> row in contact_valid

    fbin, fxyz = load_coords(args.coords, args.resolution, args.binorder)
    # intersect FLAMINGO bins with GenoFold's valid bins
    rows_in_valid, coords_sel = [], []
    for b, xyz in zip(fbin, fxyz):
        if b in pos_of_valid:
            rows_in_valid.append(pos_of_valid[b]); coords_sel.append(xyz)
    rows_in_valid = np.array(rows_in_valid)
    coords_sel = np.array(coords_sel, dtype=np.float32)
    if len(rows_in_valid) < 20:
        raise SystemExit(f"Only {len(rows_in_valid)} bins overlap GenoFold's valid set — "
                         "check the resolution / coordinate-file format.")

    # restrict the shared contact matrix to the overlapping bins and score
    C = prep.contact_valid[np.ix_(rows_in_valid, rows_in_valid)].astype(np.float32)
    np.fill_diagonal(C, 0.0)
    dscc = M.dSCC(coords_sel, C, args.alpha)
    tad = M.tad_boundary_f1(C, coords_sel, alpha=args.alpha)["f1"]
    print(f"FLAMINGO (real): dSCC={dscc:.4f}  TAD_F1={tad:.3f}  "
          f"on {len(rows_in_valid)}/{len(valid)} shared valid bins")

    # write so make_tables can pick it up (add a 'FLAMINGO' entry alongside the re-impl)
    out = os.path.join(RESULTS, "flamingo_real.json")
    json.dump(dict(dscc=dscc, tad_f1=tad, n_bins=int(len(rows_in_valid))), open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
