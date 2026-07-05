"""
experiments.multiseed_benchmark
-------------------------------
Head-to-head benchmark with error bars.

Single-seed dSCC on this task has run-to-run variance of order 0.01-0.02 for the
learned/gradient methods (GNN training, random restarts), which is comparable to
the gaps between methods. Reporting a single seed would make the ranking unstable
and could over- or under-state GenoFold. This module runs every method over
several seeds and reports mean +/- std so the comparison is honest.

All methods consume the same valid-bin contact matrix and are scored by dSCC over
all observed pairs. GenoFold uses its final-epoch structure (no best-of-checkpoints
rescue), matching the final-epoch reporting of the GNN/coordinate baselines.
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from experiments.data_fetch import build_chrom_dataset
from experiments.prep import prepare
from experiments.train import train_genofold
from experiments.baselines import shrec3d, threedmax, hicgnn, flamingo_style
from experiments import metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("genofold.multiseed")

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ALPHA = 0.25


def run(chrom="chr22", resolution=50_000, seeds=(0, 1, 2), epochs=300):
    ds = build_chrom_dataset("GM12878", chrom, resolution)
    prep = prepare(ds)
    C = prep.contact_valid
    obs = M.observed_pair_mask(C)

    def score(coords):
        return M.dSCC(coords, C, ALPHA, pair_mask=obs), M.tad_boundary_f1(C, coords, alpha=ALPHA)["f1"]

    methods = {}

    # ShRec3D is deterministic (closed form) -> single value.
    t = time.time(); coords = shrec3d(C, ALPHA); d, f = score(coords)
    methods["ShRec3D"] = dict(dscc=[d], tad=[f], seconds=time.time() - t)
    log.info(f"ShRec3D: dSCC={d:.4f} (deterministic)")

    # Stochastic methods over seeds.
    for name, fn in [("3DMax", threedmax), ("HiC-GNN", hicgnn), ("FLAMINGO-style", flamingo_style)]:
        ds_list, f_list, t = [], [], time.time()
        for s in seeds:
            coords = fn(C, alpha=ALPHA, seed=s)
            d, f = score(coords)
            ds_list.append(d); f_list.append(f)
        methods[name] = dict(dscc=ds_list, tad=f_list, seconds=(time.time() - t) / len(seeds))
        log.info(f"{name}: dSCC={np.mean(ds_list):.4f}+/-{np.std(ds_list):.4f} over {len(seeds)} seeds")

    # GenoFold: final-epoch structure per seed.
    ds_list, f_list, t = [], [], time.time()
    for s in seeds:
        res = train_genofold(prep.data, C, epochs=epochs, eval_every=50, mc_samples=1,
                             layout=prep.layout, held_out=0.0, report_final=True,
                             seed=s, log_prefix=f"[GenoFold s{s}] ", verbose=False)
        d, f = score(res.coords)
        ds_list.append(d); f_list.append(f)
    methods["GenoFold"] = dict(dscc=ds_list, tad=f_list, seconds=(time.time() - t) / len(seeds))
    log.info(f"GenoFold: dSCC={np.mean(ds_list):.4f}+/-{np.std(ds_list):.4f} over {len(seeds)} seeds")

    out = {}
    for k, v in methods.items():
        out[k] = dict(
            dscc_mean=float(np.mean(v["dscc"])), dscc_std=float(np.std(v["dscc"])),
            tad_mean=float(np.mean(v["tad"])), tad_std=float(np.std(v["tad"])),
            n_seeds=len(v["dscc"]), seconds=v["seconds"], values=v["dscc"],
        )
    with open(os.path.join(RESULTS, "multiseed_benchmark.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    log.info("wrote multiseed_benchmark.json")
    return out


if __name__ == "__main__":
    run()
