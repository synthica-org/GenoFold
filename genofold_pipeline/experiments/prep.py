"""
experiments.prep
----------------
Turn a real ChromDataset into the inputs every method shares:

  * a PyG graph (multimodal node features + Hi-C contact edges) for GenoFold,
  * the raw KR contact submatrix over the graph's valid bins (the common
    reconstruction target / evaluation matrix for GenoFold AND all baselines),
  * a FeatureLayout describing node-feature column groups (for ablations).

Using the SAME valid-bin set and SAME target matrix for every method is what
makes the benchmark fair.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline import GenoFoldPipeline, PipelineConfig
from experiments.data_fetch import ChromDataset
from experiments.train import FeatureLayout

logger = logging.getLogger("genofold.prep")


@dataclass
class PreparedChrom:
    data: object                 # PyG Data
    contact_valid: np.ndarray    # (M, M) raw KR contacts over valid bins
    valid_idx: np.ndarray        # (M,) indices into original N bins
    layout: FeatureLayout
    ds: ChromDataset


def prepare(ds: ChromDataset, kmer_k: int = 4, top_k_edges: int = 50) -> PreparedChrom:
    metadata_df = pd.DataFrame({
        "bin_id": np.arange(ds.n_bins),
        "species": "human",
        "gc_content": ds.gc_content,
        "mappability": ds.mappability,
    })

    cfg = PipelineConfig(bin_size=ds.resolution, kmer_k=kmer_k, top_k_edges=top_k_edges)
    pipe = GenoFoldPipeline(cfg)
    data = pipe.run(
        contact_matrix=ds.contact_matrix,
        coverage=ds.coverage,
        sequences=ds.sequences,
        metadata_df=metadata_df,
        gc_content=ds.gc_content,
        mappability=ds.mappability,
        chrom_start=ds.chrom_start,
    )

    valid_mask = data.valid_mask.cpu().numpy()
    valid_idx = np.where(valid_mask)[0]
    contact_valid = ds.contact_matrix[np.ix_(valid_idx, valid_idx)].astype(np.float32)
    np.fill_diagonal(contact_valid, 0.0)

    # Feature layout: [ngs | kmer | metadata(gc,mappability) | pos]
    n_seq = data.x.shape[1] - 1 - 2 - 1   # total - ngs(1) - meta(2) - pos(1)
    layout = FeatureLayout(n_ngs=1, n_seq=n_seq, n_meta=2, n_pos=1)
    logger.info(f"Prepared {ds.cell_line} {ds.chrom}: {len(valid_idx)} valid bins, "
                f"x dim {data.x.shape[1]}, layout ngs=1/seq={n_seq}/meta=2/pos=1")
    return PreparedChrom(data=data, contact_valid=contact_valid,
                         valid_idx=valid_idx, layout=layout, ds=ds)


if __name__ == "__main__":  # ponytail: smoke test — GenoFold trains and beats a random structure
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    from experiments.data_fetch import build_chrom_dataset
    from experiments.train import train_genofold
    from experiments.metrics import dSCC

    ds = build_chrom_dataset("GM12878", "chr22", 50_000)
    prep = prepare(ds)
    rng = np.random.default_rng(0)
    rand = rng.normal(size=(prep.contact_valid.shape[0], 3))
    baseline = dSCC(rand, prep.contact_valid)
    res = train_genofold(prep.data, prep.contact_valid, epochs=120, eval_every=20,
                         mc_samples=8, layout=prep.layout, verbose=True)
    print(f"\nOK  random dSCC={baseline:.3f}  ->  GenoFold held-out dSCC={res.dscc_test:.3f} "
          f"(train {res.dscc_train:.3f}, {res.seconds:.1f}s, {prep.contact_valid.shape[0]} bins)")
    assert res.dscc_test > baseline + 0.2, "GenoFold did not learn a meaningful structure"
