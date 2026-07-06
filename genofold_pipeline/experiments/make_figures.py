"""
experiments.make_figures
-------------------------
Render every paper figure from results/*.json + results/structures/*.npy.
Matplotlib only (self-contained PNGs for the LaTeX manuscript).

  fig_benchmark      — dSCC + TAD-F1 across methods
  fig_structure      — 3D GenoFold chr22 structure coloured by uncertainty
  fig_contact        — real Hi-C vs reconstructed pseudo-contact heatmaps
  fig_calibration    — per-bin uncertainty vs reconstruction error
  fig_ablation       — modality ablation (overall + sparse regions)
  fig_scaling        — bins vs runtime across chromosomes
  fig_cross_cell     — cross-cell-line transfer vs retrained dSCC
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(BASE, "results")
STRUCT = os.path.join(RESULTS, "structures")
OUT = os.path.join(BASE, "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, GREY, ORANGE, GREEN = "#2563eb", "#94a3b8", "#ea580c", "#16a34a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_benchmark():
    ms = _load("multiseed_benchmark.json")
    fl = _load("flamingo_real.json")
    d = _load("main_benchmark.json")
    if ms:
        data = {m: ms[m] for m in ms if m != "FLAMINGO-style"}
        if fl:
            data["FLAMINGO"] = dict(dscc_mean=fl["dscc_mean"], dscc_std=fl.get("dscc_std", 0.0),
                                    tad_mean=fl["tad_mean"], tad_std=fl.get("tad_std", 0.0))
        methods = sorted(data.keys(), key=lambda m: data[m]["dscc_mean"])
        dscc = [data[m]["dscc_mean"] for m in methods]
        err = [data[m]["dscc_std"] for m in methods]
        tad = [data[m]["tad_mean"] for m in methods]
        terr = [data[m]["tad_std"] for m in methods]
    elif d:
        methods = sorted(d.keys(), key=lambda m: d[m]["dscc"])
        dscc = [d[m]["dscc"] for m in methods]; err = [0] * len(methods)
        tad = [d[m]["tad_f1"] for m in methods]; terr = [0] * len(methods)
    else:
        return
    colors = [BLUE if m == "GenoFold" else GREY for m in methods]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].barh(methods, dscc, xerr=err, color=colors, capsize=3,
               error_kw=dict(ecolor="#334155", lw=1))
    ax[0].set_xlabel("dSCC"); ax[0].set_xlim(0, 1.0)
    ax[0].set_title("Distance Spearman correlation")
    for i, v in enumerate(dscc):
        ax[0].text(v + err[i] + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    ax[1].barh(methods, tad, xerr=terr, color=colors, capsize=3,
               error_kw=dict(ecolor="#334155", lw=1))
    ax[1].set_xlabel("TAD boundary F1"); ax[1].set_xlim(0, 0.8)
    ax[1].set_title("TAD boundary reproducibility")
    for i, v in enumerate(tad):
        ax[1].text(v + terr[i] + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_benchmark.png")); plt.close(fig)


def fig_structure():
    f = os.path.join(STRUCT, "GenoFold_chr22.npy")
    u = os.path.join(STRUCT, "GenoFold_chr22_uncertainty.npy")
    if not os.path.exists(f):
        return
    coords = np.load(f); unc = np.load(u) if os.path.exists(u) else np.zeros(len(coords))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=GREY, lw=0.6, alpha=0.5)
    p = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=unc, cmap="viridis", s=14)
    cb = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1); cb.set_label("MC-dropout uncertainty")
    ax.set_title("GenoFold reconstruction — GM12878 chr22")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.savefig(os.path.join(OUT, "fig_structure.png")); plt.close(fig)


def fig_contact():
    f = os.path.join(STRUCT, "GenoFold_chr22.npy")
    cf = os.path.join(STRUCT, "GenoFold_chr22_contact.npy")
    if not (os.path.exists(f) and os.path.exists(cf)):
        return
    from experiments.metrics import recon_pseudo_contact
    coords = np.load(f)
    Cv = np.load(cf)   # exact valid-bin contact matrix aligned to the structure
    rec = recon_pseudo_contact(coords, alpha=0.25)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(np.log1p(Cv), cmap="Reds"); ax[0].set_title("Observed Hi-C (KR)")
    ax[1].imshow(np.log1p(rec), cmap="Reds"); ax[1].set_title("GenoFold reconstruction")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(os.path.join(OUT, "fig_contact.png")); plt.close(fig)


def fig_calibration():
    d = _load("calibration.json")
    ef = os.path.join(STRUCT, "calibration_error.npy")
    uf = os.path.join(STRUCT, "calibration_uncertainty.npy")
    if not (d and os.path.exists(ef)):
        return
    err = np.load(ef); unc = np.load(uf)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].scatter(unc, err, s=8, alpha=0.4, color=BLUE)
    ax[0].set_xlabel("MC-dropout uncertainty"); ax[0].set_ylabel("per-bin reconstruction error")
    ax[0].set_title(f"Spearman r = {d['spearman']:.2f}")
    ax[1].plot(d["curve_x"], d["curve_y"], "o-", color=ORANGE)
    ax[1].set_xlabel("uncertainty (quantile mean)"); ax[1].set_ylabel("mean error")
    ax[1].set_title("Calibration curve")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_calibration.png")); plt.close(fig)


def fig_ablation():
    d = _load("ablation.json")
    if not d:
        return
    labels = list(d.keys())
    overall = [d[k]["dscc"] for k in labels]
    sparse = [d[k]["dscc_sparse"] for k in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.bar(x - w / 2, overall, w, label="all pairs", color=BLUE)
    ax.bar(x + w / 2, sparse, w, label="sparse regions", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("dSCC"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Multi-modal ablation")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ablation.png")); plt.close(fig)


def fig_scaling():
    d = _load("scaling_50000.json")
    if not d:
        return
    chroms = list(d.keys())
    bins = [d[c]["n_bins"] for c in chroms]
    secs = [d[c]["seconds"] for c in chroms]
    dscc = [d[c]["dscc"] for c in chroms]
    order = np.argsort(bins)
    bins = np.array(bins)[order]; secs = np.array(secs)[order]
    dscc = np.array(dscc)[order]; chroms = np.array(chroms)[order]
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.plot(bins, secs, "o-", color=BLUE)
    for b, s, c, dd in zip(bins, secs, chroms, dscc):
        ax.annotate(f"{c}\n(dSCC {dd:.2f})", (b, s), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("genomic bins (nodes)"); ax.set_ylabel("training runtime (s)")
    ax.set_title("Scaling across whole chromosomes")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scaling.png")); plt.close(fig)


def fig_cross_cell():
    d = _load("cross_cell.json")
    if not d:
        return
    cells, trans, retr = [], [], []
    for k, v in d.items():
        if "dscc_transfer" in v:
            cells.append(k); trans.append(v["dscc_transfer"]); retr.append(v["dscc_retrained"])
        elif "dscc" in v:  # the training cell line
            cells.append(k); trans.append(v["dscc"]); retr.append(v["dscc"])
    x = np.arange(len(cells)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.bar(x - w / 2, trans, w, label="GM12878→target (transfer)", color=BLUE)
    ax.bar(x + w / 2, retr, w, label="retrained on target", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(cells, rotation=10)
    ax.set_ylabel("dSCC"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("Cross-cell-line generalisation (chr22)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cross_cell.png")); plt.close(fig)


def fig_downsample():
    d = _load("downsample_ablation.json")
    if not d:
        return
    items = sorted(d.values(), key=lambda v: v["keep"])
    keep = [v["keep"] * 100 for v in items]
    full = [v["dscc_full"] for v in items]
    hic = [v["dscc_hic"] for v in items]
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.plot(keep, full, "o-", color=BLUE, label="Full (multi-modal)")
    ax.plot(keep, hic, "s--", color=ORANGE, label="Hi-C graph only")
    ax.set_xscale("log")
    ax.set_xlabel("% of Hi-C contacts retained (log)")
    ax.set_ylabel("dSCC vs full-depth structure")
    ax.invert_xaxis()
    ax.legend(); ax.set_title("Multi-modality under Hi-C sparsity")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_downsample.png")); plt.close(fig)


def fig_importance():
    d = _load("feature_importance.json")
    if not d:
        return
    names = {"ngs": "NGS coverage", "seq": "k-mer sequence",
             "meta": "metadata (GC/map)", "pos": "position"}
    keys = sorted(d, key=lambda k: d[k])
    vals = [d[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.barh([names[k] for k in keys], vals, color=BLUE)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Integrated-Gradients importance (normalised)")
    ax.set_title("Feature-group importance (trained model)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_importance.png")); plt.close(fig)


def fig_fish_sparse():
    d = _load("fish_sparse.json")
    if not d:
        return
    cells = [c for c in ["IMR90", "K562"] if c in d]
    series = [("GenoFold", BLUE, "o-"), ("3DMax", ORANGE, "s--"),
              ("ShRec3D", GREEN, "^--"), ("HiC_wish", GREY, "d:")]
    labels = {"GenoFold": "GenoFold", "3DMax": "3DMax", "ShRec3D": "ShRec3D",
              "HiC_wish": "raw Hi-C"}
    fig, axes = plt.subplots(1, len(cells), figsize=(7.2, 2.5), sharey=True)
    if len(cells) == 1:
        axes = [axes]
    for ax, cell in zip(axes, cells):
        rows = sorted(d[cell].values(), key=lambda v: v["keep"])
        keep = [v["keep"] * 100 for v in rows]
        for name, col, style in series:
            ax.plot(keep, [v[name] for v in rows], style, color=col,
                    label=labels[name], markersize=5)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("% Hi-C contacts retained (log)")
        ax.set_title(cell)
    axes[0].set_ylabel("FISH agreement (Spearman)")
    axes[-1].legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fish_sparse.png")); plt.close(fig)


def main():
    for fn in [fig_benchmark, fig_structure, fig_contact, fig_calibration,
               fig_ablation, fig_scaling, fig_cross_cell, fig_importance, fig_downsample,
               fig_fish_sparse]:
        try:
            fn(); print(f"ok  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"skip {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
