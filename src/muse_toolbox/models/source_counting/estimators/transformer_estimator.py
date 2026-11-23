import logging
import math
from typing import Any, Dict

import torch
import torch.nn as nn

from muse_toolbox.models.source_counting.estimators.base_estimator import (
    BaseSourceCountEstimator,
)
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class FF(nn.Module):
    """Position-wise feed-forward network block with pre-normalization and residual connection."""

    def __init__(self, in_chan: int, inner_size: int, dropout: float = 0.0) -> None:
        """Initializes the feed-forward block.

        Args:
            in_chan (int): Number of input and output feature channels.
            inner_size (int): Hidden dimensionality of the pointwise MLP expansion.
            dropout (float, optional): Dropout probability after ReLU activation. Defaults to 0.0.
        """
        super().__init__()
        self.norm = nn.LayerNorm(in_chan)
        self.ff = nn.Sequential(
            nn.Conv1d(in_chan, inner_size, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(inner_size, in_chan, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies LayerNorm, pointwise feed-forward projection, and residual addition.

        Args:
            x (torch.Tensor): Input feature tensor of shape ``(B, in_chan, T)``.

        Returns:
            torch.Tensor: Output feature tensor of shape ``(B, in_chan, T)``.
        """
        normed = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.ff(normed) + x


class MHALayer(nn.Module):
    """Multi-head self-attention block with pre-normalization and residual connection."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        """Initializes the multi-head self-attention layer.

        Args:
            embed_dim (int): Embedding feature dimension.
            num_heads (int): Number of parallel self-attention heads.
            dropout (float, optional): Dropout probability applied to attention weights
                and after the MHA output projection. Defaults to 0.0.
            causal (bool, optional): If True, applies an upper-triangular causal mask
                preventing attention to future frames. If False, attends bidirectionally
                across the entire sequence as in Cornell et al. Defaults to False.
        """
        super().__init__()
        self.causal = bool(causal)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies multi-head self-attention (causal or bidirectional) with residual connection.

        Args:
            x (torch.Tensor): Input sequence tensor of shape ``(B, embed_dim, T)``.

        Returns:
            torch.Tensor: Contextualized tensor of shape ``(B, embed_dim, T)``.
        """
        normed = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # MultiheadAttention expects (T, B, embed_dim) when batch_first=False
        normed = normed.permute(2, 0, 1)

        if self.causal:
            # Explicit upper-triangular boolean mask prevents attention to future frames (tau > t)
            T = normed.size(0)
            attn_mask = ~torch.tril(
                torch.ones((T, T), dtype=torch.bool, device=normed.device)
            )
            out, _ = self.mha(
                normed,
                normed,
                normed,
                need_weights=False,
                is_causal=True,
                attn_mask=attn_mask,
            )
        else:
            out, _ = self.mha(
                normed,
                normed,
                normed,
                need_weights=False,
                is_causal=False,
                attn_mask=None,
            )

        # Apply dropout after MHA output projection, permute back to (B, embed_dim, T), and add residual
        return self.dropout(out.permute(1, 2, 0)) + x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added along the temporal STFT frame axis."""

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        """Precomputes sinusoidal positional encodings up to ``max_len`` frames.

        Args:
            d_model (int): Embedding dimensionality matching transformer hidden width.
            dropout (float, optional): Dropout rate applied after adding encodings. Defaults to 0.1.
            max_len (int, optional): Maximum supported sequence length in frames. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Reshape pe: (max_len, d_model) -> (d_model, max_len) -> (1, d_model, max_len)
        pe = pe.transpose(0, 1).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encodings up to sequence length ``T`` and applies dropout.

        Args:
            x (torch.Tensor): Embedded tensor of shape ``(B, d_model, T)``.

        Returns:
            torch.Tensor: Position-encoded tensor of shape ``(B, d_model, T)``.
        """
        x = x + self.pe[..., : x.size(-1)]
        return self.dropout(x)


class TransformerEstimator(BaseSourceCountEstimator):
    """Configurable causal or non-causal Transformer estimator for sound source counting.

    Implements the OSDC Transformer architecture of Cornell et al. (2022) when ``causal=False``
    (symmetric bidirectional ``cat-pool`` context stacking, temporal subsampling, unmasked
    bidirectional self-attention, and sequence stretching), and its strictly causal online
    counterpart when ``causal=True`` (past-only context stacking, upper-triangular causal
    self-attention masking, and causal zero-order-hold upsampling).
    """

    def __init__(
        self,
        input_dim: int,
        transform: STFTtransform,
        max_sources: int,
        causal: bool = False,
        chunk_size: int = 7,
        subsample: int = 10,
        embed_dim: int = 384,
        n_heads: int = 4,
        ff_inner: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 5000,
    ) -> None:
        """Initializes the Transformer source count estimator.

        Args:
            input_dim (int): Feature dimensionality per STFT time frame.
            transform (STFTtransform): STFT transform instance configured for the pipeline.
            max_sources (int): Maximum number of simultaneous active sources.
            causal (bool, optional): If False, uses Cornell's non-causal bidirectional
                ``cat-pool`` (``+/-chunk_size`` frames), unmasked self-attention, and
                piecewise linear interpolation for temporal upsampling. If True, uses
                strictly causal past-only ``cat-pool`` (``chunk_size`` past frames),
                causal triangular self-attention masking, and causal zero-order-hold
                temporal upsampling. Defaults to False.
            chunk_size (int, optional): Context window parameter (``C`` in Cornell et al. when
                ``causal=False``; ``T_c`` past frames when ``causal=True``; 0 disables
                context concatenation). Defaults to 7.
            subsample (int, optional): Temporal subsampling factor (``S`` / ``T_sub >= 1``;
                1 disables temporal subsampling). Defaults to 10.
            embed_dim (int, optional): Transformer bottleneck embedding width (``H``).
                Defaults to 384.
            n_heads (int, optional): Number of heads in each self-attention layer.
                Defaults to 4.
            ff_inner (int, optional): Hidden width of feed-forward layers. Defaults to 1024.
            n_layers (int, optional): Number of stacked Transformer encoder blocks (``R``).
                Defaults to 4.
            dropout (float, optional): Dropout rate across positional encoding, MHA, and FFN.
                Defaults to 0.2.
            max_seq_len (int, optional): Maximum supported sequence length for positional
                encoding buffers. Defaults to 5000.

        Raises:
            ValueError: If ``chunk_size < 0`` or ``subsample < 1``.
        """
        super().__init__(input_dim, transform, max_sources)

        if chunk_size < 0:
            raise ValueError(f"chunk_size must be >= 0, got {chunk_size}")
        if subsample < 1:
            raise ValueError(f"subsample must be >= 1, got {subsample}")

        self.causal = bool(causal)
        self.chunk_size = chunk_size
        self.subsample = subsample
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ff_inner = ff_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # Context multiplier: (T_c + 1) when causal=True, (2*C + 1) when causal=False
        context_frames = (chunk_size + 1) if self.causal else (2 * chunk_size + 1)
        in_chan_cat = input_dim * context_frames
        self.norm_sc = nn.LayerNorm(in_chan_cat)
        self.bottleneck = nn.Sequential(nn.Conv1d(in_chan_cat, embed_dim, 1))

        self.pos_encs = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)

        # Output projection to max_sources + 1 classes (0 .. max_sources active sources)
        self.output = nn.Conv1d(embed_dim, max_sources + 1, 1)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        MHALayer(
                            embed_dim,
                            n_heads,
                            dropout=dropout,
                            causal=self.causal,
                        ),
                        FF(embed_dim, ff_inner, dropout=dropout),
                    ]
                )
            )

    def get_config(self) -> Dict[str, Any]:
        """Returns a dictionary of estimator hyperparameters for logging and checkpointing.

        Returns:
            Dict[str, Any]: Serialized configuration dictionary of the estimator.
        """
        return {
            "max_sources": self.max_sources,
            "causal": self.causal,
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
        """Logs a formatted summary of estimator hyperparameters.

        Args:
            indent (str, optional): Leading indentation string for log messages. Defaults to "".
        """
        super()._verbose_parameters(indent)
        log.info(f"{indent}  Causal Mode: {self.causal}")
        if self.causal:
            log.info(
                f"{indent}  Chunk Size (Past Context Frames T_c): {self.chunk_size} "
                f"(Total Stacked: {self.chunk_size + 1})"
            )
        else:
            log.info(
                f"{indent}  Chunk Size (Bidirectional Context +/-C): {self.chunk_size} "
                f"(Total Stacked: {2 * self.chunk_size + 1})"
            )
        log.info(f"{indent}  Subsample Factor: {self.subsample}")
        log.info(f"{indent}  Embed Dim: {self.embed_dim}")
        log.info(f"{indent}  Num Heads: {self.n_heads}")
        log.info(f"{indent}  Num Layers: {self.n_layers}")

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """Processes input STFT features through the Transformer (causal or Cornell baseline).

        Args:
            features (torch.Tensor): Input feature tensor of shape ``(B, input_dim, T)``.

        Returns:
            torch.Tensor: Class logits of shape ``(B, T, max_sources + 1)``.
        """
        T_orig = features.size(-1)
        x = features

        # Context concatenation (cat)
        if self.chunk_size > 0:
            if self.causal:
                # Pad strictly in the past by chunk_size frames (zero future look-ahead)
                x = torch.nn.functional.pad(x, (self.chunk_size, 0))
                kernel_len = self.chunk_size + 1
            else:
                # Pad symmetrically in past and future by chunk_size frames (Cornell cat-pool)
                x = torch.nn.functional.pad(x, (self.chunk_size, self.chunk_size))
                kernel_len = 2 * self.chunk_size + 1

            x = torch.nn.functional.unfold(
                x.unsqueeze(-1),
                kernel_size=(kernel_len, 1),
                padding=0,
                stride=(1, 1),
            )  # Output is (B, input_dim * kernel_len, T_orig)

        # Temporal subsampling (pool) at stride S / T_sub
        if self.subsample > 1:
            x = x[..., :: self.subsample]  # (B, in_chan_cat, T')

        x = self.norm_sc(x.transpose(1, 2)).transpose(1, 2)
        x = self.bottleneck(x)  # (B, embed_dim, T')
        x = self.pos_encs(x)

        for layer_block in self.layers:
            mha = layer_block[0]
            ff = layer_block[1]
            x = mha(x)
            x = ff(x)

        logits = self.output(x)  # (B, max_sources + 1, T')

        # Upsample by subsample factor back to native STFT frame resolution T_orig
        if self.subsample > 1:
            if self.causal:
                # Strictly causal zero-order hold: l_t = l_sub[floor(t / S)] (zero future look-ahead)
                logits = logits.repeat_interleave(self.subsample, dim=-1)[..., :T_orig]
            else:
                # Non-causal piecewise linear interpolation between subsampled anchor frames k * S
                t_sub = logits.size(-1)
                t_idx = torch.arange(T_orig, device=logits.device, dtype=torch.long)
                idx0 = (t_idx // self.subsample).clamp(max=t_sub - 1)
                idx1 = (idx0 + 1).clamp(max=t_sub - 1)
                alpha = (
                    (t_idx % self.subsample).to(logits.dtype) / float(self.subsample)
                ).view(1, 1, T_orig)
                logits = (1.0 - alpha) * logits[..., idx0] + alpha * logits[..., idx1]

        # Permute for output: (B, max_sources + 1, T_orig) -> (B, T_orig, max_sources + 1)
        return logits.permute(0, 2, 1)
