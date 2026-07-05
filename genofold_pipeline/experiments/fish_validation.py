"""
experiments.fish_validation
---------------------------
Orthogonal validation of GenoFold against experimentally MEASURED 3D distances
from super-resolution chromatin tracing (Bintu et al. 2018, Science 362:eaau1783).

Everything else in this pipeline scores dSCC = agreement between a reconstruction
and Hi-C-DERIVED "wish" distances, i.e. self-consistency with the same data the
model trained on. This script instead compares GenoFold's predicted pairwise
distances to independent single-molecule FISH distances, turning dSCC agreement
into an absolute accuracy claim (the reviewer's requested check).

Recipe (field standard; cf. FLAMINGO, Wang et al. 2022 Nat Commun):
  * Bintu image consecutive 30 kb segments of chr21:28-29.95 Mb (65 segments, hg38)
    in ~4,900 single IMR90 chromosome copies; each row is a segment centroid (nm).
    The region is lifted to hg19 (+1.37 Mb) to match Rao 2014 Hi-C coordinates.
  * Per copy -> pairwise Euclidean distances between segment centroids
    (translation-invariant, so the absolute stage offset cancels);
    median across copies -> the 65x65 median imaging distance matrix.
  * GenoFold (and 3DMax / ShRec3D) reconstruct the SAME region from IMR90 Rao 2014
    Hi-C at 10 kb; model coords are coarse-grained 3x10kb -> the 30 kb imaging grid.
  * Metric: Spearman between model pairwise distances and imaging median distances
    over all segment pairs -- rank-based, hence invariant to the unknown
    nm<->model-unit scale and to the contact->distance exponent.
  * Anchor: raw Hi-C wish distance vs imaging (how well Hi-C ITSELF predicts FISH)
    -- the ceiling a faithful reconstruction can hope to match.

  cd genofold_pipeline && GENOFOLD_DEVICE=cpu python -m experiments.fish_validation
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from experiments.data_fetch import build_chrom_dataset, ChromDataset, CACHE_DIR, _download
from experiments import metrics as M

BASE = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(BASE, "results")

# Bintu image chr21:28-30Mb in *hg38* (30 kb segments) across several cell lines.
# Rao 2014 Hi-C is hg19, so we window the Hi-C at the lifted-over coordinate:
# hg38 28,000,000 -> hg19 29,372,319 (UCSC liftOver, near-constant +1.37 Mb over
# this 2 Mb region). //HIC_RES lands on the hg19 29.37 Mb bin, which an offset scan
# confirms maximises raw Hi-C/FISH concordance (Spearman 0.97, cf. Bintu's ~0.92);
# the raw hg38 coordinate mis-registers the Hi-C by 1.37 Mb (anchor drops to 0.67).
CELLS = ["IMR90", "K562"]   # both have Bintu tracing AND matched Rao 2014 Hi-C
REGION_START = 29_372_319   # hg19, lifted from hg38 chr21:28,000,000
SEG_BP = 30_000
HIC_RES = 10_000            # 3 x 10kb == one 30kb imaged segment
CHROM = "chr21"
ALPHA = 0.25
MIN_DETECT = 100            # require >= this many copies detecting BOTH loci for a
                           # reliable median distance (out of ~4,900)


def _bintu_paths(cell):
    """Cached CSV path + Bintu-repo URL of the per-cell tracing coordinates."""
    csv = os.path.join(CACHE_DIR, "bintu", f"{cell}_chr21-28-30Mb.csv")
    url = ("https://raw.githubusercontent.com/BogdanBintu/ChromatinImaging/"
           f"master/Data/{cell}_chr21-28-30Mb.csv")
    return csv, url


# ---------------------------------------------------------------------------
# Imaging: median single-cell pairwise-distance matrix
# ---------------------------------------------------------------------------

def imaging_median_distance(cell):
    """(K,K) median pairwise centroid distance (nm) over single copies, plus an
    integer (K,K) count of copies detecting both loci. Segment index is 1-based."""
    csv, url = _bintu_paths(cell)
    _download(url, csv)                 # fetch once from the Bintu repo, then cache
    df = pd.read_csv(csv, skiprows=1)  # skip the "(in nm)" comment line
    df.columns = [c.strip() for c in df.columns]
    K = int(df["Segment index"].max())
    copies = sorted(df["Chromosome index"].unique())
    # (C, K, 3) with NaN for undetected segments
    A = np.full((len(copies), K, 3), np.nan, dtype=np.float64)
    cidx = {c: i for i, c in enumerate(copies)}
    ci = df["Chromosome index"].map(cidx).to_numpy()
    si = df["Segment index"].to_numpy() - 1
    A[ci, si, 0] = df["X"].to_numpy()
    A[ci, si, 1] = df["Y"].to_numpy()
    A[ci, si, 2] = df["Z"].to_numpy()

    Dimg = np.full((K, K), np.nan)
    ndet = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(i + 1, K):
            d = np.sqrt(((A[:, i] - A[:, j]) ** 2).sum(1))  # (C,), NaN if either missing
            finite = np.isfinite(d)
            ndet[i, j] = ndet[j, i] = int(finite.sum())
            if finite.any():
                Dimg[i, j] = Dimg[j, i] = float(np.median(d[finite]))
    return Dimg, ndet, K


# ---------------------------------------------------------------------------
# Model: reconstruct the region from Hi-C, coarse-grain to the 30 kb grid
# ---------------------------------------------------------------------------

def _window_dataset(K, cell):
    """`cell` chr21 Hi-C at 10 kb, windowed to the imaged region (3*K bins)."""
    full = build_chrom_dataset(cell, CHROM, HIC_RES)      # cached
    w0 = REGION_START // HIC_RES
    w1 = w0 + K * (SEG_BP // HIC_RES)                      # 3 bins per segment
    sl = slice(w0, w1)
    C = full.contact_matrix[sl, sl].astype(np.float32).copy()
    np.fill_diagonal(C, 0.0)
    ds = ChromDataset(cell_line=cell, chrom=CHROM, hic_chrom="21", resolution=HIC_RES,
                      contact_matrix=C, coverage=C.sum(1).astype(np.float32),
                      gc_content=full.gc_content[sl], mappability=full.mappability[sl],
                      sequences=list(full.sequences[sl]), chrom_start=w0 * HIC_RES)
    return ds, C


def _coarse_grain(coords, valid_idx, K):
    """Average the (<=3) valid 10kb-bin coords inside each 30kb segment -> (K,3)
    segment centroids; NaN row for segments with no valid bin."""
    pos = {int(v): i for i, v in enumerate(valid_idx)}    # window-bin -> coords row
    out = np.full((K, 3), np.nan)
    for k in range(K):
        rows = [pos[p] for p in (3 * k, 3 * k + 1, 3 * k + 2) if p in pos]
        if rows:
            out[k] = coords[rows].mean(0)
    return out


def _seg_distance(centroids):
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def hic_wish_distance_30kb(C_window, K):
    """Aggregate the 10kb window contacts to the 30kb grid and convert to wish
    distances -- the raw-Hi-C prediction of FISH distance (no reconstruction)."""
    Cagg = C_window.reshape(K, 3, K, 3).sum((1, 3))
    return M.wish_distance(Cagg, ALPHA)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(model_D, Dimg, seg_ok):
    """Spearman + Pearson(log) between model and imaging distances over the
    upper triangle of the common valid segments."""
    K = Dimg.shape[0]
    ii, jj = np.triu_indices(K, 1)
    m = seg_ok[ii] & seg_ok[jj] & np.isfinite(model_D[ii, jj]) & np.isfinite(Dimg[ii, jj])
    a, b = model_D[ii, jj][m], Dimg[ii, jj][m]
    rho = spearmanr(a, b).correlation
    pr = pearsonr(np.log(a + 1e-9), np.log(b + 1e-9))[0]
    return float(rho), float(pr), int(m.sum())


def _run_cell(cell, seeds=(0, 1, 2)):
    """Full FISH validation for one cell line -> per-method result dict."""
    from experiments.prep import prepare
    from experiments.train import train_genofold
    from experiments.baselines import shrec3d, threedmax

    print(f"\n== FISH validation: {cell} chr21:28-30Mb ==")
    Dimg, ndet, K = imaging_median_distance(cell)
    reliable = ndet >= MIN_DETECT
    print(f"imaging: {K} segments; up to {int(ndet.max())} single-copy detections/pair")

    ds, C_window = _window_dataset(K, cell)
    prep = prepare(ds)
    Cv = prep.contact_valid
    vidx = prep.valid_idx.astype(np.int64)
    print(f"Hi-C window: {C_window.shape[0]} bins @10kb -> {Cv.shape[0]} valid bins")

    # GenoFold over several seeds (mean +/- std); baselines are deterministic /
    # best-of-restarts single runs (as in the 5 kb table).
    gf_cen = [_coarse_grain(
        train_genofold(prep.data, Cv, epochs=300, eval_every=50, mc_samples=1,
                       layout=prep.layout, held_out=0.0, report_final=True, seed=s,
                       log_prefix=f"[GenoFold-FISH {cell} s{s}] ", verbose=False).coords,
        vidx, K) for s in seeds]
    base_cen = {name: _coarse_grain(fn(Cv, alpha=ALPHA), vidx, K)
                for name, fn in [("3DMax", threedmax), ("ShRec3D", shrec3d)]}

    seg_ok = np.ones(K, bool)               # score a segment only if every method has it
    for c in list(base_cen.values()) + gf_cen:
        seg_ok &= np.isfinite(c).all(1)
    Dimg_r = np.where(reliable, Dimg, np.nan)   # reliable-median pairs only

    out = {"_meta": dict(region_hg38="chr21:28.00-29.95Mb", region_hic_hg19="chr21:29.37-31.32Mb",
                         cell=cell, n_segments=int(K), n_segments_scored=int(seg_ok.sum()),
                         hic_res=HIC_RES, min_detections=MIN_DETECT,
                         metric="Spearman(model_dist, FISH_median_dist)",
                         source=f"Bintu 2018 Science; Rao 2014 GSE63525 {cell}")}
    print(f"{'method':12s} {'FISH-Spearman':>16s} {'FISH-Pearson(log)':>18s} {'pairs':>7s}")

    gf = [_score(_seg_distance(c), Dimg_r, seg_ok) for c in gf_cen]
    gf_rho = [x[0] for x in gf]; gf_pr = [x[1] for x in gf]; npairs = gf[0][2]
    out["GenoFold"] = dict(fish_spearman=float(np.mean(gf_rho)), fish_spearman_std=float(np.std(gf_rho)),
                           fish_pearson_log=float(np.mean(gf_pr)), n_pairs=npairs, n_seeds=len(seeds))
    print(f"{'GenoFold':12s} {np.mean(gf_rho):10.3f} +/-{np.std(gf_rho):5.3f} "
          f"{np.mean(gf_pr):18.3f} {npairs:7d}")
    for name, cen in base_cen.items():
        rho, pr, np_ = _score(_seg_distance(cen), Dimg_r, seg_ok)
        out[name] = dict(fish_spearman=rho, fish_pearson_log=pr, n_pairs=np_)
        print(f"{name:12s} {rho:16.3f} {pr:18.3f} {np_:7d}")

    rho, pr, npairs = _score(hic_wish_distance_30kb(C_window, K), Dimg_r, seg_ok)
    out["HiC_wish"] = dict(fish_spearman=rho, fish_pearson_log=pr, n_pairs=npairs)
    print(f"{'HiC-wish*':12s} {rho:14.3f} {pr:18.3f} {npairs:7d}   (*raw Hi-C, no reconstruction)")
    return out


def main(cells=CELLS):
    out = {cell: _run_cell(cell) for cell in cells}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "fish_validation.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/fish_validation.json ({', '.join(cells)})")
    return out


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:  # ponytail: scoring logic validated on a known structure
        rng = np.random.default_rng(0)
        gt = np.cumsum(rng.normal(size=(40, 3)), axis=0)          # ground-truth chain
        Dtrue = _seg_distance(gt)
        seg_ok = np.ones(40, bool)
        rho, pr, n = _score(Dtrue, Dtrue, seg_ok)
        assert rho > 0.999, f"identical matrices must give Spearman ~1, got {rho}"
        # a coarse-grain of exact 3x-replicated bins must recover the structure
        coords3 = np.repeat(gt, 3, axis=0)                         # 120 "10kb" bins
        cen = _coarse_grain(coords3, np.arange(120), 40)
        assert np.allclose(cen, gt), "coarse-grain of replicated bins must equal originals"
        rho2, _, _ = _score(_seg_distance(cen), Dtrue, seg_ok)
        assert rho2 > 0.999, f"coarse-grained recon must match, got {rho2}"
        print(f"OK selfcheck  identical rho={rho:.3f} pearson={pr:.3f} pairs={n}  cg_rho={rho2:.3f}")
    else:
        main()
