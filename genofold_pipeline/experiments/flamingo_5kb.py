"""
experiments.flamingo_5kb
------------------------
Run/score FLAMINGO at its NATIVE 5 kb resolution on the mappable arm of GM12878
chr22 (16-51.3 Mb, re-based to 0), and score every method on the same 5 kb KR
contact matrix by dSCC (global + short-range <=1 Mb).

  python -m experiments.flamingo_5kb export     # write sparse contacts + norm vecs for FLAMINGO
  python -m experiments.flamingo_5kb score <coords.tsv> [<coords2.tsv> ...]

The dense 5 kb region matrix is cached at flamingo_run/chr22_5kb.npz by fetch5kb.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))
RUN = os.path.join(BASE, "..", "flamingo_run")
RESULTS = os.path.join(BASE, "results")
NPZ = os.path.join(RUN, "chr22_5kb.npz")
RES = 5000
ALPHA = 0.25
SHORT_BP = 1_000_000     # short-range threshold (<=1 Mb)


def _load_matrix():
    d = np.load(NPZ)
    M = d["M"].astype(np.float32)
    M = (M + M.T) / 2.0      # ensure symmetric
    np.fill_diagonal(M, 0.0)
    return M, int(d["start"])


def export():
    M, start = _load_matrix()
    n = M.shape[0]
    ii, jj = np.where(M > 0)          # both triangles -> symmetric list, re-based bp = bin*RES
    vals = M[ii, jj]
    np.savetxt(os.path.join(RUN, "chr22_5kb.contacts.txt"),
               np.column_stack([ii * RES, jj * RES, vals]),
               fmt=["%d", "%d", "%.6f"], delimiter="\t")
    import math
    chr_size = n * RES
    nh = math.ceil(chr_size / RES); nl = math.ceil(chr_size / 1e6)
    open(os.path.join(RUN, "ones_high_5kb.txt"), "w").write("\n".join(["1"] * nh) + "\n")
    open(os.path.join(RUN, "ones_low_5kb.txt"), "w").write("\n".join(["1"] * nl) + "\n")
    print(f"exported {len(vals)} contacts; n_bins={n}; chr_size(rebased)={chr_size}; "
          f"norm high={nh} low={nl}")


def _dscc(coords, C, mask=None):
    from experiments import metrics as M_
    return M_.dSCC(coords, C, ALPHA, pair_mask=mask)


def score(files):
    from experiments import metrics as M_
    M, start = _load_matrix()
    n = M.shape[0]
    up = np.triu(np.ones((n, n), bool), 1) & (M > 0)
    ii, jj = np.where(up); sep = np.abs(ii - jj) * RES
    short_mask = np.zeros((n, n), bool); short_mask[ii[sep <= SHORT_BP], jj[sep <= SHORT_BP]] = True
    rows = {}
    for f in files:
        d = np.loadtxt(f)
        bp, xyz = d[:, 0], d[:, 1:4].astype(np.float32)
        bins = (bp / RES).round().astype(int)
        keep = (bins >= 0) & (bins < n)
        bins, xyz = bins[keep], xyz[keep]
        # place into full-length coord array (nan for missing), then restrict to present bins
        idx = bins
        C = M[np.ix_(idx, idx)].astype(np.float32); np.fill_diagonal(C, 0.0)
        # rebuild masks for the present-bin submatrix
        m = C.shape[0]
        upm = np.triu(np.ones((m, m), bool), 1) & (C > 0)
        a, b = np.where(upm); s = np.abs(a - b) * RES
        sm = np.zeros((m, m), bool); sm[a[s <= SHORT_BP], b[s <= SHORT_BP]] = True
        g = _dscc(xyz, C); sh = _dscc(xyz, C, sm)
        rows[os.path.basename(f)] = dict(dscc=float(g), dscc_short=float(sh), n_bins=int(m))
        print(f"{os.path.basename(f):28s} global dSCC={g:.4f}  short(<=1Mb)={sh:.4f}  ({m} bins)")
    json.dump(rows, open(os.path.join(RESULTS, "flamingo_5kb.json"), "w"), indent=2)
    print("wrote results/flamingo_5kb.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    if cmd == "export":
        export()
    elif cmd == "score":
        fs = []
        for a in sys.argv[2:]:
            fs.extend(glob.glob(a))
        score(sorted(set(fs)))
    else:
        raise SystemExit("usage: export | score <coords.tsv...>")
