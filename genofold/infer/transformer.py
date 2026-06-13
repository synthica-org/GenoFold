"""
genofold.infer.transformer
---------------------------
Transformer encoder for capturing long-range genomic dependencies.

Why a Transformer after the GNN?
  - GNNs are inherently local — they aggregate information from
    graph neighbors (bins connected by Hi-C contacts)
  - Some structural features (e.g. compartment A/B identity, large
    topological domains) require global context across the whole chromosome
  - The Transformer applies self-attention over ALL bins simultaneously,
    regardless of whether they share a Hi-C contact

Architecture:
  - Sinusoidal positional encoding based on genomic bin position
  - N layers of standard multi-head self-attention + feedforward
  - Pre-norm (LayerNorm before attention) for training stability
  - Species embedding added to inject cross-species information
  - Output: same shape as input (M, hidden_dim) — enriched embeddings
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class GenomicPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding adapted for genomic bin positions.

    Unlike NLP where positions are discrete token indices, genomic bins
    have a continuous coordinate (chromosome position in bp). We use
    sinusoidal encoding over the normalized bin position [0, 1] so the
    model understands spatial order along the chromosome.

    Parameters
    ----------
    hidden_dim:
        Model dimension (must be even).
    max_len:
        Maximum number of bins supported.
    dropout:
        Dropout applied to the encoded positions.
    """

    def __init__(self, hidden_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Standard sinusoidal encoding table
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, bin_positions: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.

        Parameters
        ----------
        x:
            Input embeddings (M, hidden_dim) — note: no batch dim here,
            we treat the whole chromosome as one sequence.
        bin_positions:
            Normalized genomic positions (M, 1) in [0, 1].

        Returns
        -------
        torch.Tensor
            Position-encoded embeddings (M, hidden_dim).
        """
        M = x.shape[0]
        # Map continuous positions [0,1] to discrete indices [0, max_len-1]
        indices = (bin_positions.squeeze(-1) * (self.pe.shape[1] - 1)).long()
        indices = indices.clamp(0, self.pe.shape[1] - 1)
        pos_enc = self.pe[0, indices, :]  # (M, hidden_dim)
        return self.dropout(x + pos_enc)


# ---------------------------------------------------------------------------
# Species embedding
# ---------------------------------------------------------------------------

class SpeciesEmbedding(nn.Module):
    """
    Learnable embedding for species identity.

    Each species (human, mouse, rat, etc.) gets its own embedding vector
    that is added to bin representations. This allows the model to learn
    species-specific structural patterns — e.g. certain TAD boundaries
    may be conserved across species while others are species-specific.

    Parameters
    ----------
    num_species:
        Size of species vocabulary (default 6: human, mouse, rat,
        zebrafish, drosophila, yeast).
    hidden_dim:
        Embedding dimension (must match model hidden_dim).
    """

    def __init__(self, num_species: int = 6, hidden_dim: int = 64):
        super().__init__()
        # +1 for unknown species (ID = -1 mapped to 0)
        self.embedding = nn.Embedding(num_species + 1, hidden_dim, padding_idx=0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, species_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        species_ids:
            (M,) integer tensor. -1 values are shifted to 0 (unknown).

        Returns
        -------
        torch.Tensor
            Species embeddings (M, hidden_dim).
        """
        # Shift -1 → 0 (unknown species uses padding embedding)
        ids = (species_ids + 1).clamp(min=0)
        return self.norm(self.embedding(ids))


# ---------------------------------------------------------------------------
# Transformer encoder layer (pre-norm)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer.

    Pre-norm (applying LayerNorm BEFORE attention/FFN) is more stable
    than the original post-norm Transformer, especially for deep networks
    and smaller datasets — important for genomics where data can be limited.

    Architecture per layer:
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # Pre-norm feedforward with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full Transformer encoder
# ---------------------------------------------------------------------------

class GenomicTransformer(nn.Module):
    """
    Transformer encoder operating over all genomic bins.

    Takes GNN output embeddings and enriches them with global context
    via multi-head self-attention across the entire chromosome arm.

    Parameters
    ----------
    hidden_dim:
        Model dimension — must match GNNEncoder output (default 64).
    num_layers:
        Number of Transformer layers (default 6 for conference-grade depth).
    num_heads:
        Number of attention heads (default 8).
    ffn_dim:
        Feedforward network inner dimension (default 512).
    num_species:
        Species vocabulary size (default 6).
    max_bins:
        Maximum number of bins per chromosome (default 5000).
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 512,
        num_species: int = 6,
        max_bins: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pos_encoding = GenomicPositionalEncoding(hidden_dim, max_bins, dropout)
        self.species_embedding = SpeciesEmbedding(num_species, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"GenomicTransformer initialized:\n"
            f"  hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"num_heads={num_heads}, ffn_dim={ffn_dim}\n"
            f"  Trainable parameters: {total_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        bin_positions: torch.Tensor,
        species_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer encoder.

        Parameters
        ----------
        x:
            GNN output embeddings (M, hidden_dim).
        bin_positions:
            Normalized genomic positions (M, 1).
        species_ids:
            Integer species labels (M,).

        Returns
        -------
        torch.Tensor
            Enriched embeddings (M, hidden_dim).
        """
        # Add positional and species information
        x = self.pos_encoding(x, bin_positions)
        x = x + self.species_embedding(species_ids)

        # Transformer expects (batch, seq, dim) — treat chromosome as batch=1
        x = x.unsqueeze(0)  # (1, M, hidden_dim)

        for layer in self.layers:
            x = layer(x)

        x = x.squeeze(0)     # (M, hidden_dim)
        x = self.output_norm(x)

        logger.debug(f"Transformer output: {x.shape}")
        return x
