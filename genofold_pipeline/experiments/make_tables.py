"""
experiments.make_tables
------------------------
Emit LaTeX table fragments straight from results/*.json so every number in the
paper is generated from the actual run (no hand transcription). Writes .tex
files into paper/tables/ that the manuscript \\input's.
"""

from __future__ import annotations

import json
import os

BASE = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "..", "paper", "tables")
os.makedirs(OUT, exist_ok=True)


def _load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _tex(s: str) -> str:
    """Make a label safe for pdfLaTeX (ASCII minus, escaped chars)."""
    return s.replace("−", "$-$").replace("−", "$-$").replace("&", "\\&").replace("_", "\\_")


def _write(name, text):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(text)
    print("wrote", name)


def _tabular(colspec, header, rows, tabcolsep=None):
    """Wrap rows in a complete booktabs tabular (input'd as a whole block, which
    avoids the \\input-inside-open-alignment breakage)."""
    pre = "\\renewcommand{\\arraystretch}{1.2}\n"
    if tabcolsep:
        pre += f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n"
    body = "\n".join(rows)
    return (f"{pre}\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n"
            f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")


def _bold_max(vals, i, fmt="{:.3f}"):
    return ("\\textbf{" + fmt.format(vals[i]) + "}") if vals[i] == max(vals) else fmt.format(vals[i])


def table_benchmark():
    ms = _load("multiseed_benchmark.json")
    fl = _load("flamingo_real.json")   # real FLAMINGO (replaces the -style proxy)
    order = ["ShRec3D", "3DMax", "HiC-GNN", "FLAMINGO", "GenoFold"]
    if ms:
        # merge: use multiseed for all but FLAMINGO; use flamingo_real.json for real FLAMINGO
        data = {m: ms[m] for m in ms if m != "FLAMINGO-style"}
        if fl:
            data["FLAMINGO"] = dict(dscc_mean=fl["dscc_mean"], dscc_std=fl.get("dscc_std", 0.0),
                                    tad_mean=fl["tad_mean"], tad_std=fl.get("tad_std", 0.0),
                                    n_seeds=fl.get("n_seeds", 1), seconds=float("nan"))
        order = [m for m in order if m in data]
        means = [data[m]["dscc_mean"] for m in order]
        lines = []
        for m in order:
            name = m + ("$^\\ddagger$" if m == "FLAMINGO" else "")
            mean, std, n_s = data[m]["dscc_mean"], data[m]["dscc_std"], data[m]["n_seeds"]
            tmn, tsd = data[m]["tad_mean"], data[m]["tad_std"]
            dcell = (f"\\textbf{{{mean:.3f}}}" if mean == max(means) else f"{mean:.3f}")
            dcell += (f" $\\pm$ {std:.3f}" if n_s > 1 else "")
            tcell = f"{tmn:.2f}" + (f" $\\pm$ {tsd:.2f}" if n_s > 1 else "")
            sec = data[m]["seconds"]
            scell = "---" if (sec != sec) else f"{sec:.1f}"   # nan -> em dash
            lines.append(f"{name} & {dcell} & {tcell} & {scell} \\\\")
        header = ("\\textbf{Method} & \\textbf{dSCC} & \\textbf{TAD $F_1$} & \\textbf{Time (s)}")
        _write("tab_benchmark.tex", _tabular("@{}lccc@{}", header, lines, tabcolsep="4pt"))
        print(f"  (benchmark; real FLAMINGO {'merged' if fl else 'MISSING'})")
        return
    # fallback: single-seed
    d = _load("main_benchmark.json")
    if not d:
        return
    order = [m for m in order if m in d]
    dscc = [d[m]["dscc"] for m in order]
    pear = [d[m]["pearson"] for m in order]
    tad = [d[m]["tad_f1"] for m in order]
    sec = [d[m]["seconds"] for m in order]
    lines = []
    for k, m in enumerate(order):
        name = m.replace("FLAMINGO-style", "FLAMINGO-style$^\\dagger$")
        row = f"{name} & {_bold_max(dscc,k)} & {_bold_max(pear,k)} & {_bold_max(tad,k,'{:.2f}')} & {sec[k]:.1f} \\\\"
        lines.append(row)
    header = "\\textbf{Method} & \\textbf{dSCC} & \\textbf{Pearson} & \\textbf{TAD $F_1$} & \\textbf{Time (s)}"
    _write("tab_benchmark.tex", _tabular("@{}lcccc@{}", header, lines, tabcolsep="4pt"))
    print(f"  (single-seed benchmark on {d[order[0]]['n_bins']} bins)")


def table_ablation():
    d = _load("ablation.json")
    if not d:
        return
    lines = []
    full = d.get("Full (multi-modal)", {})
    for label, v in d.items():
        dd = f"{v['dscc']:.3f}"; ds = f"{v['dscc_sparse']:.3f}"
        if label != "Full (multi-modal)" and full:
            dd += f" ({v['dscc']-full['dscc']:+.3f})"
            ds += f" ({v['dscc_sparse']-full['dscc_sparse']:+.3f})"
        lines.append(f"{_tex(label)} & {dd} & {ds} \\\\")
    header = "\\textbf{Configuration} & \\textbf{dSCC (all)} & \\textbf{dSCC (sparse)}"
    _write("tab_ablation.tex", _tabular("@{}lcc@{}", header, lines))


def table_scaling():
    d = _load("scaling_50000.json")
    if not d:
        return
    rows = sorted(d.items(), key=lambda kv: kv[1]["n_bins"])
    lines = [f"{ch} & {v['n_bins']} & {v['dscc']:.3f} & {v['seconds']:.0f} & {v['peak_rss_mb']:.0f} \\\\"
             for ch, v in rows]
    header = "\\textbf{Chrom.} & \\textbf{Bins} & \\textbf{dSCC} & \\textbf{Time (s)} & \\textbf{Peak RSS (MB)}"
    _write("tab_scaling.tex", _tabular("@{}lcccc@{}", header, lines))


def table_cross_cell():
    d = _load("cross_cell.json")
    if not d:
        return
    lines = []
    for k, v in d.items():
        if "dscc_transfer" in v:
            lines.append(f"{_tex(k)} & {v['dscc_transfer']:.3f} & {v['dscc_retrained']:.3f} \\\\")
        elif "dscc" in v:
            lines.append(f"{_tex(k)} & \\multicolumn{{2}}{{c}}{{{v['dscc']:.3f} (in-domain)}} \\\\")
    header = "\\textbf{Cell line} & \\textbf{Transfer dSCC} & \\textbf{Retrained dSCC}"
    _write("tab_cross_cell.tex", _tabular("@{}lcc@{}", header, lines))


def macros():
    """Single-number \\newcommand macros for inline use in prose."""
    m = []
    ms = _load("multiseed_benchmark.json")
    d = _load("main_benchmark.json")
    if ms and "GenoFold" in ms:
        m.append(f"\\renewcommand{{\\gfdscc}}{{{ms['GenoFold']['dscc_mean']:.3f}}}")
        m.append(f"\\renewcommand{{\\gfdsccstd}}{{{ms['GenoFold']['dscc_std']:.3f}}}")
        m.append(f"\\renewcommand{{\\gfnseeds}}{{{ms['GenoFold']['n_seeds']}}}")
    elif d and "GenoFold" in d:
        m.append(f"\\renewcommand{{\\gfdscc}}{{{d['GenoFold']['dscc']:.3f}}}")
    if d and "GenoFold" in d:
        m.append(f"\\renewcommand{{\\gfbins}}{{{d['GenoFold']['n_bins']}}}")
    g = _load("generalization.json")
    if g:
        m.append(f"\\renewcommand{{\\gftrain}}{{{g['dscc_train']:.3f}}}")
        m.append(f"\\renewcommand{{\\gfheldout}}{{{g['dscc_heldout']:.3f}}}")
    c = _load("calibration.json")
    if c:
        m.append(f"\\renewcommand{{\\gfcalib}}{{{c['spearman']:.2f}}}")
    cc = _load("cross_cell.json")
    if cc:
        tr = [v["dscc_transfer"] for v in cc.values() if "dscc_transfer" in v]
        rt = [v["dscc_retrained"] for v in cc.values() if "dscc_retrained" in v]
        if tr:
            m.append(f"\\renewcommand{{\\gftransfermin}}{{{min(tr):.3f}}}")
            m.append(f"\\renewcommand{{\\gftransfermax}}{{{max(tr):.3f}}}")
        if rt:
            m.append(f"\\renewcommand{{\\gfretrainmin}}{{{min(rt):.3f}}}")
            m.append(f"\\renewcommand{{\\gfretrainmax}}{{{max(rt):.3f}}}")
    sc = _load("scaling_50000.json")
    if sc:
        nb = max(v["n_bins"] for v in sc.values())
        m.append(f"\\renewcommand{{\\gfmaxbins}}{{{nb}}}")
        m.append(f"\\renewcommand{{\\gfnchrom}}{{{len(sc)}}}")
    _write("macros.tex", "\n".join(m) + "\n")


def main():
    for fn in [table_benchmark, table_ablation, table_scaling, table_cross_cell, macros]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"skip {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
