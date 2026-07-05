"""
experiments.run_5kb_comparison
------------------------------
Fair head-to-head at FLAMINGO's NATIVE 5 kb resolution on a tractable 10 Mb window
of GM12878 chr22 (20-30 Mb, 2000 bins; the full arm OOMs on this 16 GB machine).
Every method reconstructs from the SAME 5 kb KR contact submatrix over the same
valid bins and is scored by dSCC (global + short-range <=1 Mb).

  python -m experiments.run_5kb_comparison run     # GenoFold + 3DMax + ShRec3D
  python -m experiments.run_5kb_comparison flamingo <coords_5kb.tsv ...>   # add FLAMINGO

FLAMINGO is run separately in R (flamingo_run/run_flamingo_5kb.R) and scored here
on the identical valid bins.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from experiments.data_fetch import (ChromDataset, per_bin_sequences, gc_from_sequences,
                                    fetch_mappability, load_reference_chrom, CACHE_DIR)
from experiments import metrics as M

BASE = os.path.dirname(os.path.dirname(__file__))
RUN = os.path.join(BASE, "..", "flamingo_run")
RESULTS = os.path.join(BASE, "results")
NPZ = os.path.join(RUN, "chr22_5kb.npz")
RES = 5000
ALPHA = 0.25
SHORT_BP = 1_000_000


def _build_window_dataset():
    d = np.load(NPZ)
    M_ = ((d["M"] + d["M"].T) / 2.0).astype(np.float32); np.fill_diagonal(M_, 0.0)
    start = int(d["start"]); n = M_.shape[0]
    seq = load_reference_chrom("chr22")
    sequences = per_bin_sequences(seq, n, RES, chrom_start=start)
    gc = gc_from_sequences(sequences)
    mapp = fetch_mappability("chr22", n, RES, chrom_start=start)
    return ChromDataset(cell_line="GM12878", chrom="chr22", hic_chrom="22", resolution=RES,
                        contact_matrix=M_, coverage=M_.sum(1).astype(np.float32),
                        gc_content=gc, mappability=mapp, sequences=sequences,
                        chrom_start=start)


def _short_mask(C, valid_idx):
    n = C.shape[0]
    up = np.triu(np.ones((n, n), bool), 1) & (C > 0)
    ii, jj = np.where(up)
    sep = np.abs(valid_idx[ii] - valid_idx[jj]) * RES
    m = np.zeros((n, n), bool); m[ii[sep <= SHORT_BP], jj[sep <= SHORT_BP]] = True
    return m


def run():
    from experiments.prep import prepare
    from experiments.train import train_genofold
    from experiments.baselines import shrec3d, threedmax
    ds = _build_window_dataset()
    prep = prepare(ds)
    C = prep.contact_valid
    vidx = prep.valid_idx.astype(np.int64)
    np.save(os.path.join(RESULTS, "w5kb_valididx.npy"), vidx)
    np.save(os.path.join(RESULTS, "w5kb_contact.npy"), C)
    sm = _short_mask(C, vidx)
    obs = M.observed_pair_mask(C)
    rows = {}

    res = train_genofold(prep.data, C, epochs=300, eval_every=50, mc_samples=1,
                         layout=prep.layout, held_out=0.0, report_final=True,
                         log_prefix="[GenoFold-5kb] ", verbose=False)
    rows["GenoFold"] = dict(dscc=M.dSCC(res.coords, C, ALPHA, pair_mask=obs),
                            dscc_short=M.dSCC(res.coords, C, ALPHA, pair_mask=sm))
    for name, fn in [("3DMax", threedmax), ("ShRec3D", shrec3d)]:
        coords = fn(C, alpha=ALPHA)
        rows[name] = dict(dscc=M.dSCC(coords, C, ALPHA, pair_mask=obs),
                          dscc_short=M.dSCC(coords, C, ALPHA, pair_mask=sm))
    for k, v in rows.items():
        print(f"{k:12s} global dSCC={v['dscc']:.4f}  short(<=1Mb)={v['dscc_short']:.4f}")
    rows["_meta"] = dict(n_bins=int(C.shape[0]), window="chr22:20-30Mb", resolution=RES)
    json.dump(rows, open(os.path.join(RESULTS, "cmp5kb.json"), "w"), indent=2)
    print("wrote results/cmp5kb.json")


def flamingo(files):
    C = np.load(os.path.join(RESULTS, "w5kb_contact.npy"))
    vidx = np.load(os.path.join(RESULTS, "w5kb_valididx.npy"))
    pos = {int(b): i for i, b in enumerate(vidx)}
    sm = _short_mask(C, vidx); obs = M.observed_pair_mask(C)
    cmp = json.load(open(os.path.join(RESULTS, "cmp5kb.json")))
    g_all, s_all = [], []
    for f in files:
        d = np.loadtxt(f); bp, xyz = d[:, 0], d[:, 1:4].astype(np.float32)
        bins = (bp / RES).round().astype(int)
        keep = [i for i, b in enumerate(bins) if int(b) in pos]
        rows = np.array([pos[int(bins[i])] for i in keep]); coords = xyz[keep]
        Cc = C[np.ix_(rows, rows)].astype(np.float32); np.fill_diagonal(Cc, 0.0)
        m = Cc.shape[0]
        smc = _short_mask(Cc, vidx[rows]); obsc = M.observed_pair_mask(Cc)
        g_all.append(M.dSCC(coords, Cc, ALPHA, pair_mask=obsc))
        s_all.append(M.dSCC(coords, Cc, ALPHA, pair_mask=smc))
        print(f"{os.path.basename(f)}: global={g_all[-1]:.4f} short={s_all[-1]:.4f} ({m}/{len(vidx)} bins)")
    cmp["FLAMINGO"] = dict(dscc=float(np.mean(g_all)), dscc_std=float(np.std(g_all)),
                           dscc_short=float(np.mean(s_all)), dscc_short_std=float(np.std(s_all)),
                           n_seeds=len(files))
    json.dump(cmp, open(os.path.join(RESULTS, "cmp5kb.json"), "w"), indent=2)
    print(f"\nFLAMINGO (5kb, {len(files)} seeds): global {cmp['FLAMINGO']['dscc']:.3f}  "
          f"short {cmp['FLAMINGO']['dscc_short']:.3f}")
    print("updated results/cmp5kb.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "flamingo":
        fs = []
        for a in sys.argv[2:]:
            fs.extend(glob.glob(a))
        flamingo(sorted(set(fs)))
