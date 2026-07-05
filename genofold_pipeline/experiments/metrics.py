"""
experiments.metrics
--------------------
Field-standard evaluation for Hi-C 3D reconstruction, plus the training
objective that GenoFold and the gradient baselines optimise.

Central quantity: the "wish distance" derived from the contact matrix.
Following 3DMax / HiC-GNN, an interaction frequency (IF) C_ij is mapped to a
target spatial distance  D_ij = C_ij^(-alpha)  (closer bins contact more).
A reconstruction is good when its Euclidean distances agree, in *rank*, with
these wish distances.

Primary metric: dSCC (distance Spearman correlation coefficient) — the Spearman
correlation between reconstructed pairwise distances and wish distances over
observed pairs. This is exactly the headline metric HiC-GNN reports, and it is
invariant to the conversion factor alpha (Spearman is rank-based), so it does
not depend on an arbitrary tuning choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr


# ---------------------------------------------------------------------------
# Wish distances and pair masks
# ---------------------------------------------------------------------------

def wish_distance(contact: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Convert an interaction-frequency matrix to target distances D = C^(-alpha).
    Zero contacts -> inf (unobserved, excluded downstream).
    """
    C = np.asarray(contact, dtype=np.float64)
    Cn = C / (C[C > 0].mean() + 1e-12)          # scale-normalise
    with np.errstate(divide="ignore"):
        D = np.where(Cn > 0, Cn ** (-alpha), np.inf)
    np.fill_diagonal(D, 0.0)
    return D


def observed_pair_mask(contact: np.ndarray) -> np.ndarray:
    """Upper-triangle boolean mask of observed (C>0), off-diagonal pairs."""
    n = contact.shape[0]
    up = np.triu(np.ones((n, n), dtype=bool), k=1)
    return up & (contact > 0)


def pairwise_distances(coords) -> torch.Tensor:
    """(M,3) coords -> (M,M) Euclidean distance matrix (torch, differentiable)."""
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-12)


# ---------------------------------------------------------------------------
# Training objective (differentiable)
# ---------------------------------------------------------------------------

def distance_matching_loss(
    coords: torch.Tensor,
    wish_z: torch.Tensor,
    pair_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Scale/shift-invariant distance-matching loss on a set of pairs.

    coords : (M,3) predicted coordinates.
    wish_z : (P,) z-scored log wish distances for the P training pairs.
    pair_idx : (2,P) long — row/col indices of the training pairs.
    Returns MSE between z-scored reconstructed log-distances and wish_z, which
    is minimised exactly when reconstructed and wish distances agree linearly
    (hence in rank), independent of overall scale.

    Distances are gathered only for the P training pairs (O(P), not O(M^2)),
    which keeps training cheap on large, sparse chromosomes.
    """
    a = coords[pair_idx[0]]
    b = coords[pair_idx[1]]
    dp = torch.sqrt(((a - b) ** 2).sum(-1) + 1e-12)
    logd = torch.log(dp + 1e-6)
    dz = (logd - logd.mean()) / (logd.std() + 1e-8)
    return torch.mean((dz - wish_z) ** 2)


def prepare_wish_targets(
    contact: np.ndarray, alpha: float = 0.25, device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (pair_idx, wish_z) for ALL observed pairs.
    wish_z = z-scored log wish distance (the regression target).
    """
    D = wish_distance(contact, alpha)
    mask = observed_pair_mask(contact)
    ii, jj = np.where(mask)
    w = np.log(D[ii, jj] + 1e-6)
    wz = (w - w.mean()) / (w.std() + 1e-8)
    pair_idx = torch.tensor(np.stack([ii, jj]), dtype=torch.long, device=device)
    wish_z = torch.tensor(wz, dtype=torch.float32, device=device)
    return pair_idx, wish_z


# ---------------------------------------------------------------------------
# Primary evaluation metrics
# ---------------------------------------------------------------------------

def dSCC(coords, contact: np.ndarray, alpha: float = 0.25,
         pair_mask: Optional[np.ndarray] = None) -> float:
    """
    Distance Spearman Correlation Coefficient (HiC-GNN's headline metric).
    Spearman between reconstructed distances and wish distances over observed
    (or a supplied held-out) set of pairs. Higher is better (max 1.0).
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    D = wish_distance(contact, alpha)
    mask = observed_pair_mask(contact) if pair_mask is None else pair_mask
    ii, jj = np.where(mask)
    diff = coords[ii] - coords[jj]
    recon = np.sqrt((diff ** 2).sum(-1))
    wish = D[ii, jj]
    finite = np.isfinite(wish) & np.isfinite(recon)
    if finite.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(recon[finite], wish[finite])
    return float(rho)


def distance_pearson(coords, contact: np.ndarray, alpha: float = 0.25,
                     pair_mask: Optional[np.ndarray] = None) -> float:
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    D = wish_distance(contact, alpha)
    mask = observed_pair_mask(contact) if pair_mask is None else pair_mask
    ii, jj = np.where(mask)
    diff = coords[ii] - coords[jj]
    recon = np.log(np.sqrt((diff ** 2).sum(-1)) + 1e-6)
    wish = np.log(D[ii, jj] + 1e-6)
    finite = np.isfinite(wish) & np.isfinite(recon)
    r, _ = pearsonr(recon[finite], wish[finite])
    return float(r)


def split_pairs(contact: np.ndarray, held_out: float = 0.2, seed: int = 0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split observed pairs into train / held-out boolean masks (upper triangle).
    Held-out pairs are never shown to the training loss, giving an honest
    generalisation estimate of dSCC rather than a memorisation score.
    """
    mask = observed_pair_mask(contact)
    ii, jj = np.where(mask)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ii))
    n_ho = int(len(ii) * held_out)
    ho = perm[:n_ho]
    tr = perm[n_ho:]
    n = contact.shape[0]
    train_mask = np.zeros((n, n), dtype=bool)
    test_mask = np.zeros((n, n), dtype=bool)
    train_mask[ii[tr], jj[tr]] = True
    test_mask[ii[ho], jj[ho]] = True
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# TAD boundary reproducibility (insulation score, Crane et al. 2015)
# ---------------------------------------------------------------------------

def insulation_score(contact: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Mean signal in a `window`x`window` square straddling the diagonal at each
    bin. Local minima are TAD boundaries.
    """
    n = contact.shape[0]
    ins = np.full(n, np.nan)
    for i in range(window, n - window):
        block = contact[i - window:i, i + 1:i + 1 + window]
        ins[i] = np.nanmean(block)
    m = np.nanmean(ins)
    ins = np.log2((ins + 1e-9) / (m + 1e-9))
    return ins


def call_boundaries(ins: np.ndarray, delta: float = 0.1, min_sep: int = 3) -> np.ndarray:
    """Return indices of local minima (boundaries) in an insulation track."""
    n = len(ins)
    cand = []
    for i in range(1, n - 1):
        if not np.isfinite(ins[i]):
            continue
        left = ins[i - 1] if np.isfinite(ins[i - 1]) else ins[i]
        right = ins[i + 1] if np.isfinite(ins[i + 1]) else ins[i]
        if ins[i] <= left and ins[i] <= right and (max(left, right) - ins[i]) > delta:
            cand.append(i)
    # enforce minimum separation
    out = []
    for c in cand:
        if not out or c - out[-1] >= min_sep:
            out.append(c)
    return np.array(out, dtype=int)


def recon_pseudo_contact(coords, alpha: float = 0.25) -> np.ndarray:
    """
    Turn a reconstructed structure into a pseudo contact matrix using the
    inverse of the wish-distance law (C = D^(-1/alpha)) so its distance decay
    matches real Hi-C, making the insulation comparison apples-to-apples.
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    c = (d + 1e-6) ** (-1.0 / alpha)
    np.fill_diagonal(c, 0.0)
    return c


def tad_boundary_f1(contact: np.ndarray, coords, window: int = 10,
                    tol: int = 1, alpha: float = 0.25) -> dict:
    """
    Compare TAD boundaries called from the reconstructed structure to those
    called from the real Hi-C contact matrix. A predicted boundary matches a
    true one if within `tol` bins. Returns precision / recall / F1 / Jaccard.
    """
    true_b = call_boundaries(insulation_score(contact, window))
    pred_b = call_boundaries(insulation_score(recon_pseudo_contact(coords, alpha), window))
    if len(true_b) == 0 or len(pred_b) == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, jaccard=0.0,
                    n_true=int(len(true_b)), n_pred=int(len(pred_b)))
    matched_true = sum(any(abs(t - p) <= tol for p in pred_b) for t in true_b)
    matched_pred = sum(any(abs(p - t) <= tol for t in true_b) for p in pred_b)
    recall = matched_true / len(true_b)
    precision = matched_pred / len(pred_b)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    union = len(true_b) + len(pred_b) - matched_true
    jaccard = matched_true / (union + 1e-9)
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard,
                n_true=int(len(true_b)), n_pred=int(len(pred_b)))


# ---------------------------------------------------------------------------
# Structural agreement (Procrustes RMSD)
# ---------------------------------------------------------------------------

def procrustes_rmsd(a, b) -> float:
    """RMSD between two structures after optimal similarity alignment."""
    A = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    B = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    A = A - A.mean(0); B = B - B.mean(0)
    A = A / (np.linalg.norm(A) + 1e-9)
    B = B / (np.linalg.norm(B) + 1e-9)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dg = np.diag([1, 1, d])
    R = Vt.T @ Dg @ U.T
    A_rot = A @ R.T
    return float(np.sqrt(((A_rot - B) ** 2).sum(1).mean()))


# ---------------------------------------------------------------------------
# Uncertainty calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    spearman: float           # corr(uncertainty, per-bin error) — want > 0
    curve_x: np.ndarray       # uncertainty quantile bin centres
    curve_y: np.ndarray       # mean per-bin error in each quantile
    per_bin_error: np.ndarray
    per_bin_uncertainty: np.ndarray


def per_bin_distance_error(coords, contact: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Per-bin reconstruction error = mean absolute rank-residual of that bin's
    observed pairs (reconstructed-distance rank vs wish-distance rank).
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    D = wish_distance(contact, alpha)
    n = contact.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    recon = np.sqrt((diff ** 2).sum(-1))
    mask = (contact > 0) & ~np.eye(n, dtype=bool)
    # global ranks
    ii, jj = np.where(np.triu(mask, 1))
    from scipy.stats import rankdata
    rr = rankdata(recon[ii, jj])
    rw = rankdata(D[ii, jj])
    resid = np.abs(rr - rw) / len(ii)
    err = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(err, ii, resid); np.add.at(err, jj, resid)
    np.add.at(cnt, ii, 1.0); np.add.at(cnt, jj, 1.0)
    return np.where(cnt > 0, err / np.maximum(cnt, 1), np.nan)


def calibration(uncertainty: np.ndarray, coords, contact: np.ndarray,
                alpha: float = 0.25, n_quantiles: int = 10) -> Calibration:
    unc = uncertainty.detach().cpu().numpy() if isinstance(uncertainty, torch.Tensor) else np.asarray(uncertainty)
    err = per_bin_distance_error(coords, contact, alpha)
    ok = np.isfinite(unc) & np.isfinite(err)
    unc, err = unc[ok], err[ok]
    rho, _ = spearmanr(unc, err)
    order = np.argsort(unc)
    xs, ys = [], []
    for q in np.array_split(order, n_quantiles):
        if len(q) == 0:
            continue
        xs.append(unc[q].mean()); ys.append(err[q].mean())
    return Calibration(spearman=float(rho), curve_x=np.array(xs), curve_y=np.array(ys),
                       per_bin_error=err, per_bin_uncertainty=unc)


if __name__ == "__main__":  # ponytail: self-check on a synthetic ground-truth structure
    rng = np.random.default_rng(0)
    n = 120
    # ground-truth 3D chain: random walk
    gt = np.cumsum(rng.normal(size=(n, 3)), axis=0)
    dgt = np.sqrt(((gt[:, None] - gt[None]) ** 2).sum(-1))
    with np.errstate(divide="ignore"):
        C = np.where(dgt > 0, dgt ** (-1.0 / 0.25), 0.0)   # perfect contacts from gt
    np.fill_diagonal(C, 0.0)
    d = dSCC(gt, C)
    assert d > 0.99, f"dSCC of ground-truth against its own contacts should be ~1, got {d}"
    # a random structure should score near 0
    rand = rng.normal(size=(n, 3))
    dr = dSCC(rand, C)
    assert dr < 0.3, f"random structure dSCC should be low, got {dr}"
    # TAD F1 of gt vs itself should be high-ish
    f = tad_boundary_f1(C, gt)
    print(f"OK  gt dSCC={d:.3f}  random dSCC={dr:.3f}  tad_f1(gt,gt)={f['f1']:.2f} "
          f"(n_true={f['n_true']})")
