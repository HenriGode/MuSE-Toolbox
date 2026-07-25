import logging
import math
from typing import Any

import torch
import torch.nn as nn

from muse_toolbox.utils import STFTtransform
from .base_estimator import BaseSourceCountEstimator

log = logging.getLogger(__name__)


class FF(nn.Module):
    def __init__(self, in_chan: int, inner_size: int, dropout: float = 0.0):
        super().__init__()
        # PyTorch LayerNorm expects the last dimension to be normalized.
        # However, our input x is (B, in_chan, T).
        # We can use GroupNorm(1, in_chan) as a drop-in replacement for LayerNorm over channels
        # which is exactly what Asteroid's gLN does when not using global normalization across batch.
        self.norm = nn.LayerNorm(in_chan)
        self.ff = nn.Sequential(
            nn.Conv1d(in_chan, inner_size, 1),
            nn.ReLU(),  # Using ReLU as it is standard and avoids Asteroid dependencies
            nn.Dropout(dropout),
            nn.Conv1d(inner_size, in_chan, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, in_chan, T)
        normed = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.ff(normed) + x


class MHALayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, embed_dim, T)
        normed = self.norm(x.transpose(1, 2)).transpose(1, 2)
        
        # MultiheadAttention expects (T, B, embed_dim) if batch_first=False
        normed = normed.permute(2, 0, 1) # (T, B, embed_dim)
        
        # Create a strict causal mask (T, T) where future tokens are -inf
        T = normed.size(0)
        attn_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=normed.device), 
            diagonal=1
        )
        
        # In PyTorch, attn_mask is passed directly to the MHA
        out, _ = self.mha(normed, normed, normed, attn_mask=attn_mask)
        
        # Permute back to (B, embed_dim, T) and add residual
        return out.permute(1, 2, 0) + x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe shape: (max_len, d_model) -> (d_model, max_len) -> (1, d_model, max_len)
        pe = pe.transpose(0, 1).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, d_model, T)
        x = x + self.pe[..., : x.size(-1)]
        return self.dropout(x)


class TransformerEstimator(BaseSourceCountEstimator):
    """
    A purely causal Transformer-based detector for source activity estimation,
    adapting the OSDC (Cornell et al.) baseline to the MuSE-Toolbox online framework.
    """

    def __init__(
        self,
        input_dim: int,
        transform: STFTtransform,
        max_sources: int,
        chunk_size: int = 7,
        subsample: int = 1,
        embed_dim: int = 128,
        n_heads: int = 4,
        ff_inner: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 5000,
    ):
        super().__init__(input_dim, transform, max_sources)

        self.chunk_size = chunk_size
        self.subsample = subsample
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ff_inner = ff_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.norm_sc = nn.LayerNorm(input_dim)

        # Bottleneck reduces concatenated features back to embed_dim
        in_chan_cat = input_dim * (chunk_size + 1)
        self.bottleneck = nn.Sequential(nn.Conv1d(in_chan_cat, embed_dim, 1))
        
        self.pos_encs = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)
        
        # Output projection to max_sources + 1 classes
        self.output = nn.Conv1d(embed_dim, max_sources + 1, 1)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        MHALayer(embed_dim, n_heads, dropout=dropout),
                        FF(embed_dim, ff_inner, dropout=dropout),
                    ]
                )
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "max_sources": self.max_sources,
            "chunk_size": self.chunk_size,
            "subsample": self.subsample,
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "ff_inner": self.ff_inner,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "input_dim": self.input_dim,
        }

    def _verbose_parameters(self, indent: str = "") -> None:
        super()._verbose_parameters(indent)
        log.info(f"{indent}  Chunk Size (Past Context): {self.chunk_size * 2}")
        log.info(f"{indent}  Subsample Factor: {self.subsample}")
        log.info(f"{indent}  Embed Dim: {self.embed_dim}")
        log.info(f"{indent}  Num Heads: {self.n_heads}")
        log.info(f"{indent}  Num Layers: {self.n_layers}")

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes features through the Transformer.
        Args:
            features (torch.Tensor): (B, F, T)
        Returns:
            torch.Tensor: (B, T, C)
        """
        T_orig = features.size(-1)
        x = self.norm_sc(features.transpose(1, 2)).transpose(1, 2)

        # Causal Chunking (cat)
        if self.chunk_size > 0:
            # Pad purely in the past by chunk_size to make it strictly causal.
            # No padding in the future.
            x = torch.nn.functional.pad(x, (self.chunk_size, 0))
            # x is now (B, F, T + C)
            x = torch.nn.functional.unfold(
                x.unsqueeze(-1), # (B, F, T + C, 1)
                kernel_size=(self.chunk_size + 1, 1),
                padding=0,
                stride=(1, 1),
            )  # Output is (B, F*(C+1), T)

        # Subsampling (pool)
        if self.subsample > 1:
            x = x[..., :: self.subsample]

        x = self.bottleneck(x) # (B, embed_dim, T_sub)
        x = self.pos_encs(x)

        for layer_block in self.layers:
            mha = layer_block[0]
            ff = layer_block[1]
            x = mha(x)
            x = ff(x)

        # (B, C, T_sub)
        logits = self.output(x)

        # Restore original time resolution if subsampled
        if self.subsample > 1:
            logits = torch.nn.functional.interpolate(
                logits, size=T_orig, mode="nearest"
            )

        # Permute for output: (B, C, T) -> (B, T, C)
        return logits.permute(0, 2, 1)
