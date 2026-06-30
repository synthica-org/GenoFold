"""
genofold.infer.model
---------------------
GenoFold Stage 3: Full inference model (GNN + Transformer).

Combines the GNNEncoder and GenomicTransformer into a single nn.Module
that takes a PyG Data object and outputs latent embeddings ready for
stage 4 (3D coordinate reconstruction).

Flow:
  PyG Data → GNNEncoder → (M, hidden_dim)
                        → GenomicTransformer → (M, hidden_dim)
                                             → InferenceOutput

The model also handles:
  - Automatic device detection (GPU if available, else CPU)
  - Checkpoint saving and loading
  - Parameter counting and model summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .gnn import GNNEncoder
from .transformer import GenomicTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class InferenceOutput:
    """
    Output of the stage 3 inference model.

    Attributes
    ----------
    embeddings:
        (M, hidden_dim) float32 — latent representation per bin.
        This is the input to stage 4 (reconstruction decoder).
    gnn_embeddings:
        (M, hidden_dim) float32 — intermediate GNN-only embeddings,
        useful for ablation studies and explainability.
    num_nodes:
        Number of valid genomic bins (M).
    hidden_dim:
        Embedding dimension.
    """
    embeddings: torch.Tensor
    gnn_embeddings: torch.Tensor
    num_nodes: int
    hidden_dim: int


# ---------------------------------------------------------------------------
# Full inference model
# ---------------------------------------------------------------------------

class GenoFoldInferenceModel(nn.Module):
    """
    Stage 3 inference model: GNN encoder + Transformer.

    Parameters
    ----------
    input_dim:
        Node feature dimension from stage 2 (e.g. 155).
    hidden_dim:
        Latent embedding dimension (default 64).
        All downstream stages use this dimension.
    edge_dim:
        Edge feature dimension from stage 2 (default 2).
    gnn_dropout:
        Dropout for GNN layers.
    transformer_layers:
        Number of Transformer encoder layers (default 6).
    transformer_heads:
        Number of attention heads (default 8).
    transformer_ffn_dim:
        Feedforward inner dimension in Transformer (default 512).
    transformer_dropout:
        Dropout for Transformer layers.
    num_species:
        Species vocabulary size (default 6).
    max_bins:
        Maximum bins per chromosome (default 5000).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        edge_dim: int = 2,
        gnn_dropout: float = 0.1,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 512,
        transformer_dropout: float = 0.1,
        num_species: int = 6,
        max_bins: int = 5000,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.gnn = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            dropout=gnn_dropout,
        )

        self.transformer = GenomicTransformer(
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            ffn_dim=transformer_ffn_dim,
            num_species=num_species,
            max_bins=max_bins,
            dropout=transformer_dropout,
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"GenoFoldInferenceModel ready | "
            f"total parameters: {total:,} | "
            f"input_dim={input_dim} → hidden_dim={hidden_dim}"
        )

    def forward(self, data) -> InferenceOutput:
        """
        Forward pass through stage 3.

        Parameters
        ----------
        data:
            PyG Data object from stage 2 pipeline.
            Must have: data.x, data.edge_index, data.edge_attr,
                       data.pos, data.species_ids

        Returns
        -------
        InferenceOutput
            Contains final embeddings and intermediate GNN embeddings.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        pos = data.pos
        species_ids = data.species_ids

        # Stage 3a: GNN — local graph attention over Hi-C contacts
        gnn_out = self.gnn(x, edge_index, edge_attr)    # (M, hidden_dim)

        # Stage 3b: Transformer — global self-attention over all bins
        final_out = self.transformer(gnn_out, pos, species_ids)  # (M, hidden_dim)

        return InferenceOutput(
            embeddings=final_out,
            gnn_embeddings=gnn_out,
            num_nodes=x.shape[0],
            hidden_dim=self.hidden_dim,
        )


# ---------------------------------------------------------------------------
# Device utility
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: GenoFoldInferenceModel,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save model weights and config to disk.

    Parameters
    ----------
    model:
        The inference model to save.
    path:
        Output file path (e.g. 'checkpoints/stage3.pt').
    metadata:
        Optional dict of extra info to store (epoch, loss, etc.).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_dim": model.hidden_dim,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: GenoFoldInferenceModel,
    path: str,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load model weights from a checkpoint file.

    Parameters
    ----------
    model:
        Model instance to load weights into (must match architecture).
    path:
        Path to checkpoint file.
    device:
        Device to map weights to. Auto-detected if None.

    Returns
    -------
    dict
        The metadata dict stored with the checkpoint.
    """
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    logger.info(f"Checkpoint loaded from {path}")
    return checkpoint.get("metadata", {})
