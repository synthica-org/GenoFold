"""
experiments.baselines
----------------------
Reference 3D-reconstruction methods, re-implemented in one unified framework so
they consume the SAME valid-bin contact submatrix and are scored by the SAME
dSCC metric as GenoFold — an apples-to-apples benchmark.

  * ShRec3D / classical-MDS  — shortest-path distance completion + eigen-MDS.
                               (Lesne et al. 2014; the standard distance baseline.)
  * 3DMax                    — gradient ascent that directly optimises 3D
                               coordinates to maximise distance/contact
                               agreement, with random restarts.
                               (Oluwadare et al. 2018.)
  * HiC-GNN                  — a GCN on Laplacian-eigenmap node features trained
                               with the distance-matching loss, Hi-C only.
                               (Wang et al. 2022.) Doubles as the
                               "GNN-only, Hi-C-only" ablation for GenoFold.
  * FLAMINGO-style           — low-rank distance-matrix completion + polymer
                               chain-continuity regulariser.
                               (Wang et al. 2022, Nat. Commun.) A faithful
                               re-implementation of the CORE low-rank + continuity
                               idea, not the full published pipeline — labelled
                               as such in all reporting.

Each function takes an (M, M) contact submatrix over valid bins and returns
(M, 3) coordinates as a numpy array.
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import wish_distance, observed_pair_mask, distance_matching_loss, dSCC

logger = logging.getLogger("genofold.baselines")


def _device() -> torch.device:
    import os
    forced = os.environ.get("GENOFOLD_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_pairs(contact: np.ndarray, alpha: float, device):
    D = wish_distance(contact, alpha)
    mask = observed_pair_mask(contact)
    ii, jj = np.where(mask)
    w = np.log(D[ii, jj] + 1e-6)
    wz = (w - w.mean()) / (w.std() + 1e-8)
    pair_idx = torch.tensor(np.stack([ii, jj]), dtype=torch.long, device=device)
    wish_z = torch.tensor(wz, dtype=torch.float32, device=device)
    return pair_idx, wish_z


# ---------------------------------------------------------------------------
# ShRec3D / classical MDS
# ---------------------------------------------------------------------------

def shrec3d(contact: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Shortest-path distance completion followed by classical MDS."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    D = wish_distance(contact, alpha)
    n = D.shape[0]
    W = np.where(np.isfinite(D) & (D > 0), D, 0.0)
    graph = csr_matrix(W)
    Dc = shortest_path(graph, method="D", directed=False)
    # any still-infinite pair -> max finite distance
    finite_max = Dc[np.isfinite(Dc)].max() if np.isfinite(Dc).any() else 1.0
    Dc[~np.isfinite(Dc)] = finite_max * 1.5
    np.fill_diagonal(Dc, 0.0)
    # classical MDS
    D2 = Dc ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    B = (B + B.T) / 2
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:3]
    L = np.clip(vals[idx], 0, None)
    coords = vecs[:, idx] * np.sqrt(L)[None, :]
    return coords.astype(np.float32)


# ---------------------------------------------------------------------------
# 3DMax-style gradient optimiser
# ---------------------------------------------------------------------------

def threedmax(contact: np.ndarray, alpha: float = 0.25, epochs: int = 800,
              restarts: int = 3, lr: float = 0.05, seed: int = 0) -> np.ndarray:
    """Direct coordinate optimisation to maximise distance agreement."""
    device = _device()
    pair_idx, wish_z = _train_pairs(contact, alpha, device)
    n = contact.shape[0]
    best_coords, best = None, -np.inf
    for r in range(restarts):
        g = torch.Generator(device="cpu").manual_seed(seed + r)
        X = (torch.randn(n, 3, generator=g) * 0.1).to(device).requires_grad_(True)
        opt = torch.optim.Adam([X], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = distance_matching_loss(X, wish_z, pair_idx)
            loss.backward()
            opt.step()
        d = dSCC(X, contact, alpha)
        if d > best:
            best, best_coords = d, X.detach().cpu().numpy().astype(np.float32)
    logger.info(f"3DMax: best dSCC over {restarts} restarts = {best:.4f}")
    return best_coords


# ---------------------------------------------------------------------------
# HiC-GNN-style
# ---------------------------------------------------------------------------

def _laplacian_eigenmaps(contact: np.ndarray, dim: int = 16) -> np.ndarray:
    """Node features = smallest non-trivial eigenvectors of the graph Laplacian."""
    C = contact.copy()
    np.fill_diagonal(C, 0.0)
    deg = C.sum(1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(C.shape[0]) - (d_inv_sqrt[:, None] * C * d_inv_sqrt[None, :])
    L = (L + L.T) / 2
    vals, vecs = np.linalg.eigh(L)
    feats = vecs[:, 1:dim + 1]
    return feats.astype(np.float32)


class _GCN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, x, edge_index, edge_weight):
        import torch.nn.functional as F
        h = F.gelu(self.c1(x, edge_index, edge_weight))
        h = F.gelu(self.c2(h, edge_index, edge_weight))
        return self.head(h)


def hicgnn(contact: np.ndarray, alpha: float = 0.25, epochs: int = 600,
           lr: float = 0.01, seed: int = 0) -> np.ndarray:
    """GCN on Laplacian eigenmaps trained with the distance-matching loss."""
    torch.manual_seed(seed)
    device = _device()
    n = contact.shape[0]
    feats = _laplacian_eigenmaps(contact, dim=16)
    x = torch.tensor(feats, device=device)
    ei = np.array(np.where((contact > 0) & ~np.eye(n, dtype=bool)))
    edge_index = torch.tensor(ei, dtype=torch.long, device=device)
    edge_weight = torch.tensor(contact[ei[0], ei[1]], dtype=torch.float32, device=device)
    edge_weight = edge_weight / (edge_weight.max() + 1e-9)
    pair_idx, wish_z = _train_pairs(contact, alpha, device)

    model = _GCN(feats.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        coords = model(x, edge_index, edge_weight)
        loss = distance_matching_loss(coords, wish_z, pair_idx)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        coords = model(x, edge_index, edge_weight).cpu().numpy().astype(np.float32)
    logger.info(f"HiC-GNN: final dSCC = {dSCC(coords, contact, alpha):.4f}")
    return coords


# ---------------------------------------------------------------------------
# FLAMINGO-style (low-rank completion + continuity)
# ---------------------------------------------------------------------------

def flamingo_style(contact: np.ndarray, alpha: float = 0.25, epochs: int = 800,
                   lr: float = 0.05, continuity: float = 0.3, seed: int = 0) -> np.ndarray:
    """
    Re-implementation of FLAMINGO's CORE idea: complete the sparse distance
    matrix with a low-rank (3D-embeddable) assumption and enforce polymer
    chain continuity. We optimise coordinates (rank-3 by construction) against
    observed wish distances plus a continuity penalty on consecutive bins.
    NOT the full published FLAMINGO pipeline — labelled as FLAMINGO-style.
    """
    device = _device()
    pair_idx, wish_z = _train_pairs(contact, alpha, device)
    n = contact.shape[0]
    torch.manual_seed(seed)
    X = (torch.randn(n, 3) * 0.1).to(device).requires_grad_(True)
    opt = torch.optim.Adam([X], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = distance_matching_loss(X, wish_z, pair_idx)
        # chain continuity: consecutive bins should stay close & even-spaced
        step = X[1:] - X[:-1]
        seglen = torch.sqrt((step ** 2).sum(-1) + 1e-9)
        cont = ((seglen - seglen.mean()) ** 2).mean()
        (loss + continuity * cont).backward()
        opt.step()
    coords = X.detach().cpu().numpy().astype(np.float32)
    logger.info(f"FLAMINGO-style: final dSCC = {dSCC(coords, contact, alpha):.4f}")
    return coords


BASELINES = {
    "ShRec3D": shrec3d,
    "3DMax": threedmax,
    "HiC-GNN": hicgnn,
    "FLAMINGO-style": flamingo_style,
}


if __name__ == "__main__":  # ponytail: self-check — every baseline beats random on a real structure
    logging.basicConfig(level=logging.WARNING)
    rng = np.random.default_rng(1)
    n = 150
    gt = np.cumsum(rng.normal(size=(n, 3)), axis=0)
    d = np.sqrt(((gt[:, None] - gt[None]) ** 2).sum(-1))
    with np.errstate(divide="ignore"):
        C = np.where(d > 0, d ** (-4.0), 0.0)
    C[d > np.quantile(d[d > 0], 0.5)] = 0.0   # sparsify: keep only near contacts
    np.fill_diagonal(C, 0.0)
    for name, fn in BASELINES.items():
        t = time.time()
        coords = fn(C)
        s = dSCC(coords, C)
        print(f"{name:16s} dSCC={s:.3f}  ({time.time()-t:.1f}s)")
        assert s > 0.3, f"{name} failed to recover structure (dSCC={s:.3f})"
    print("OK  all baselines recover structure from real-style sparse contacts")
