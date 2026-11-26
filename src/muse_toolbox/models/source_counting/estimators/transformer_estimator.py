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

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Applies multi-head self-attention (causal or bidirectional) with residual connection.

        Args:
            x (torch.Tensor): Input sequence tensor of shape ``(B, embed_dim, T)``.
            key_padding_mask (torch.Tensor | None, optional): Boolean mask of shape
                ``(B, T)`` where ``True`` marks right-padded tokens to ignore.
                Defaults to ``None``.

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
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=key_padding_mask is None,
                attn_mask=attn_mask,
            )
        else:
            out, _ = self.mha(
                normed,
                normed,
                normed,
                key_padding_mask=key_padding_mask,
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


class CatPoolNorm(nn.Module):
    """Context stacking (cat), temporal subsampling (pool), and frame-wise LayerNorm stem.

    Implements the Cornell et al. (2022) cat-pool + LayerNorm input stem for both
    bidirectional (``causal=False``) and strictly past-only (``causal=True``) modes.
    """

    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 7,
        subsample: int = 10,
        causal: bool = False,
    ) -> None:
        """Initializes the CatPoolNorm stem.

        Args:
            input_dim (int): Flattened per-frame feature dimension (``C * F``).
            chunk_size (int, optional): Context window parameter ``C`` (bidirectional
                ``+/-C`` when ``causal=False``, past-only ``T_c`` when ``causal=True``).
                Defaults to ``7``.
            subsample (int, optional): Temporal subsampling stride ``S`` / ``T_sub``.
                Defaults to ``10``.
            causal (bool, optional): Enables past-only context stacking when ``True``.
                Defaults to ``False``.

        Raises:
            ValueError: If ``chunk_size < 0`` or ``subsample < 1``.
        """
        super().__init__()
        if chunk_size < 0:
            raise ValueError(f"chunk_size must be >= 0, got {chunk_size}")
        if subsample < 1:
            raise ValueError(f"subsample must be >= 1, got {subsample}")

        self.input_dim = int(input_dim)
        self.chunk_size = int(chunk_size)
        self.subsample = int(subsample)
        self.causal = bool(causal)

        self.context_frames = (
            (self.chunk_size + 1) if self.causal else (2 * self.chunk_size + 1)
        )
        self.out_dim = self.input_dim * self.context_frames
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies context stacking, temporal subsampling, and frame-wise LayerNorm.

        Args:
            x (torch.Tensor): Flattened input feature sequence of shape
                ``(B, input_dim, T)``.

        Returns:
            torch.Tensor: Normalized subsampled context sequence of shape
                ``(B, out_dim, T')``.
        """
        if self.chunk_size > 0:
            if self.causal:
                x = torch.nn.functional.pad(x, (self.chunk_size, 0))
                kernel_len = self.chunk_size + 1
            else:
                x = torch.nn.functional.pad(x, (self.chunk_size, self.chunk_size))
                kernel_len = 2 * self.chunk_size + 1

            x = torch.nn.functional.unfold(
                x.unsqueeze(-1),
                kernel_size=(kernel_len, 1),
                padding=0,
                stride=(1, 1),
            )

        if self.subsample > 1:
            x = x[..., :: self.subsample]

        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer for late spatial conditioning.

    Implements Perez et al. (2018) / Cornell et al. (2022) linear affine modulation
    parameterized around the identity scale:

    .. math::
        \\mathrm{FiLM}(\\mathbf{h}, \\bar{\\mathbf{u}})
        = (\\mathbf{1} + \\Delta\\bm{\\gamma}(\\bar{\\mathbf{u}})) \\odot \\mathbf{h}
          + \\bm{\\beta}(\\bar{\\mathbf{u}})

    Initializes projection weights small (``std = 0.01 / sqrt(cond_dim)``) and biases
    to ``0.0`` so initial scales are tightly centered around ``1.0`` and shifts around
    ``0.0`` while decoupled weight decay regularizes cleanly toward identity.
    """

    def __init__(self, cond_dim: int, embed_dim: int) -> None:
        """Initializes the FiLM modulation projections.

        Args:
            cond_dim (int): Dimensionality of the normalized conditioning stream
                ``D_cat_spat``.
            embed_dim (int): Hidden embedding width ``D_emb`` to modulate.
        """
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.embed_dim = int(embed_dim)

        self.gamma_proj = nn.Conv1d(self.cond_dim, self.embed_dim, kernel_size=1)
        self.beta_proj = nn.Conv1d(self.cond_dim, self.embed_dim, kernel_size=1)

        weight_std = 0.01 / math.sqrt(max(1, self.cond_dim))
        nn.init.normal_(self.gamma_proj.weight, mean=0.0, std=weight_std)
        nn.init.normal_(self.beta_proj.weight, mean=0.0, std=weight_std)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Applies feature-wise scale and shift modulation to hidden states.

        Args:
            h (torch.Tensor): Hidden Transformer sequence of shape ``(B, embed_dim, T')``.
            cond (torch.Tensor): Normalized conditioning sequence of shape
                ``(B, cond_dim, T')``.

        Returns:
            torch.Tensor: Modulated sequence of shape ``(B, embed_dim, T')``.
        """
        scale = 1.0 + self.gamma_proj(cond)
        shift = self.beta_proj(cond)
        return scale * h + shift


def _prepare_valid_masks(
    t_orig: int,
    t_sub: int,
    subsample: int,
    valid_frames: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Builds frame zeroing masks and subsampled ``key_padding_mask`` for variable-length batches.

    Args:
        t_orig (int): Padded temporal STFT frame count ``T_orig``.
        t_sub (int): Subsampled token sequence length ``T'``.
        subsample (int): Temporal subsampling stride ``S`` / ``T_sub``.
        valid_frames (torch.Tensor | None): Per-item valid frame counts ``(B,)``.
        device (torch.device): Target tensor device.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None]: Tuple of
            ``(frame_valid_mask, key_padding_mask)`` where ``frame_valid_mask`` has shape
            ``(B, 1, T_orig)`` (``True`` on valid frames) and ``key_padding_mask`` has
            shape ``(B, T')`` (``True`` on padded tokens), or ``(None, None)`` when all
            batch items have full length ``t_orig``.
    """
    if valid_frames is None or not bool((valid_frames < t_orig).any()):
        return None, None

    vf = valid_frames.to(device=device, dtype=torch.long).clamp(min=1, max=t_orig)
    frame_valid_mask = (
        torch.arange(t_orig, device=device).unsqueeze(0) < vf.unsqueeze(1)
    ).unsqueeze(1)

    valid_sub = (torch.div(vf - 1, subsample, rounding_mode="floor") + 1).clamp(
        min=1, max=t_sub
    )
    key_padding_mask = (
        torch.arange(t_sub, device=device).unsqueeze(0) >= valid_sub.unsqueeze(1)
    )
    key_padding_mask[:, 0] = False
    return frame_valid_mask, key_padding_mask


def upsample_logits(
    logits: torch.Tensor,
    target_len: int,
    subsample: int,
    causal: bool,
    valid_frames: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restores subsampled frame logits back to native STFT frame resolution.

    Args:
        logits (torch.Tensor): Subsampled logit tensor of shape ``(B, num_classes, T')``.
        target_len (int): Target native STFT frame count ``T_orig``.
        subsample (int): Temporal subsampling stride ``S`` / ``T_sub``.
        causal (bool): If ``True``, applies strictly causal zero-order hold; if ``False``,
            applies non-causal piecewise linear interpolation between anchor frames.
        valid_frames (torch.Tensor | None, optional): Optional valid frame counts ``(B,)``
            so non-causal linear interpolation does not pull boundary logits toward
            right-padded tail tokens in variable-length batches. Defaults to ``None``.

    Returns:
        torch.Tensor: Full-resolution logit tensor of shape ``(B, num_classes, T_orig)``.
    """
    if subsample <= 1:
        return logits

    if causal:
        return logits.repeat_interleave(subsample, dim=-1)[..., :target_len]

    t_sub = logits.size(-1)
    t_idx = torch.arange(target_len, device=logits.device, dtype=torch.long)
    alpha = (
        (t_idx % subsample).to(logits.dtype) / float(subsample)
    ).view(1, 1, target_len)

    if valid_frames is not None and bool((valid_frames < target_len).any()):
        vf = valid_frames.to(device=logits.device, dtype=torch.long).clamp(
            min=1, max=target_len
        )
        valid_sub = (torch.div(vf - 1, subsample, rounding_mode="floor") + 1).clamp(
            min=1, max=t_sub
        )
        max_sub_idx = (valid_sub - 1).unsqueeze(1)  # (B, 1)
        base_idx0 = (t_idx // subsample).unsqueeze(0)  # (1, T)
        idx0 = torch.minimum(base_idx0, max_sub_idx)
        idx1 = torch.minimum(idx0 + 1, max_sub_idx)
        num_classes = logits.size(1)
        gather_idx0 = idx0.unsqueeze(1).expand(-1, num_classes, -1)
        gather_idx1 = idx1.unsqueeze(1).expand(-1, num_classes, -1)
        val0 = torch.gather(logits, dim=-1, index=gather_idx0)
        val1 = torch.gather(logits, dim=-1, index=gather_idx1)
        return (1.0 - alpha) * val0 + alpha * val1

    idx0 = (t_idx // subsample).clamp(max=t_sub - 1)
    idx1 = (idx0 + 1).clamp(max=t_sub - 1)
    return (1.0 - alpha) * logits[..., idx0] + alpha * logits[..., idx1]


class TransformerEstimator(BaseSourceCountEstimator):
    """Configurable causal or non-causal single-stream Transformer estimator.

    Implements the OSDC Transformer architecture of Cornell et al. (2022) when
    ``causal=False`` and its strictly causal online counterpart when ``causal=True``.
    Multi-channel inputs ``(B, C, F, T)`` are flattened in the input pre-stage to
    ``(B, C * F, T)`` immediately before ``CatPoolNorm``.
    """

    def __init__(
        self,
        input_dim: int,
        transform: STFTtransform,
        max_sources: int,
        num_input_channels: int = 1,
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
        """Initializes the single-stream Transformer source count estimator.

        Args:
            input_dim (int): Per-channel feature dimensionality ``F`` per STFT frame.
            transform (STFTtransform): STFT transform instance configured for the pipeline.
            max_sources (int): Maximum number of simultaneous active sources.
            num_input_channels (int, optional): Expected input channel count ``C``
                flattened at the input pre-stage (``C * F``). Defaults to ``1``.
            causal (bool, optional): Selects strictly causal mode when ``True`` or
                Cornell's non-causal baseline when ``False``. Defaults to ``False``.
            chunk_size (int, optional): Context window parameter ``C`` / ``T_c``.
                Defaults to ``7``.
            subsample (int, optional): Temporal subsampling factor ``S`` / ``T_sub``.
                Defaults to ``10``.
            embed_dim (int, optional): Transformer bottleneck embedding width ``H``.
                Defaults to ``384``.
            n_heads (int, optional): Number of attention heads. Defaults to ``4``.
            ff_inner (int, optional): Inner FFN expansion width. Defaults to ``1024``.
            n_layers (int, optional): Number of stacked TE blocks ``R``. Defaults to ``4``.
            dropout (float, optional): Dropout probability. Defaults to ``0.2``.
            max_seq_len (int, optional): Maximum supported positional encoding length.
                Defaults to ``5000``.
        """
        super().__init__(
            input_dim=input_dim,
            transform=transform,
            max_sources=max_sources,
            num_input_channels=num_input_channels,
        )

        self.causal = bool(causal)
        self.chunk_size = chunk_size
        self.subsample = subsample
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ff_inner = ff_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.cat_pool_norm = CatPoolNorm(
            input_dim=self.flat_input_dim,
            chunk_size=chunk_size,
            subsample=subsample,
            causal=self.causal,
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(self.cat_pool_norm.out_dim, embed_dim, 1)
        )
        self.pos_encs = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Conv1d(embed_dim, max_sources + 1, 1)

        self.layers = nn.ModuleList(
            [
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
                for _ in range(n_layers)
            ]
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
            "num_input_channels": self.num_input_channels,
        }

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs a formatted summary of estimator hyperparameters.

        Args:
            indent (str, optional): Leading indentation string for log messages. Defaults to ``""``.
        """
        super()._verbose_parameters(indent)
        log.info(
            f"{indent}  Input Channels x Feature Dim: "
            f"{self.num_input_channels} x {self.input_dim} = {self.flat_input_dim}"
        )
        log.info(f"{indent}  Causal Mode: {self.causal}")
        log.info(
            f"{indent}  Chunk Size: {self.chunk_size} "
            f"(Total Stacked: {self.cat_pool_norm.context_frames})"
        )
        log.info(f"{indent}  Subsample Factor: {self.subsample}")
        log.info(f"{indent}  Embed Dim: {self.embed_dim}")
        log.info(f"{indent}  Num Heads: {self.n_heads}")
        log.info(f"{indent}  Num Layers: {self.n_layers}")

    def forward(
        self,
        features: torch.Tensor,
        cond_features: torch.Tensor | None = None,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Processes input features with optional padding mask and returns logits.

        Args:
            features (torch.Tensor): Input features ``(B, C, F, T)`` or ``(B, C*F, T)``.
            cond_features (torch.Tensor | None, optional): Unused for single-stream.
            valid_frames (torch.Tensor | None, optional): Valid frame counts ``(B,)``.

        Returns:
            torch.Tensor: Class logits of shape ``(B, T, max_sources + 1)``.
        """
        if cond_features is not None:
            raise ValueError(
                f"{self.__class__.__name__} is a single-stream estimator and does not "
                "accept `cond_features`. Use FusionTransformerEstimator for dual-stream fusion."
            )
        flat_features = self._flatten_channel_features(
            features, self.num_input_channels, stream_name="features"
        )
        return self.forward_tensor(flat_features, valid_frames=valid_frames)

    def forward_tensor(
        self,
        features: torch.Tensor,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Processes flattened single-stream STFT features through the Transformer.

        Args:
            features (torch.Tensor): Flattened feature tensor of shape
                ``(B, C * F, T)``.
            valid_frames (torch.Tensor | None, optional): Valid frame counts ``(B,)``
                for right-padded variable-length batches. Defaults to ``None``.

        Returns:
            torch.Tensor: Class logits of shape ``(B, T, max_sources + 1)``.
        """
        t_orig = features.size(-1)
        t_sub = (t_orig - 1) // self.subsample + 1 if self.subsample > 1 else t_orig
        frame_valid_mask, key_padding_mask = _prepare_valid_masks(
            t_orig=t_orig,
            t_sub=t_sub,
            subsample=self.subsample,
            valid_frames=valid_frames,
            device=features.device,
        )
        if frame_valid_mask is not None:
            features = features.masked_fill(~frame_valid_mask, 0.0)

        x = self.cat_pool_norm(features)
        x = self.bottleneck(x)
        x = self.pos_encs(x)

        for layer_block in self.layers:
            mha = layer_block[0]
            ff = layer_block[1]
            x = mha(x, key_padding_mask=key_padding_mask)
            x = ff(x)

        x = self.out_norm(x.transpose(1, 2)).transpose(1, 2)
        logits = self.output(x)
        logits = upsample_logits(
            logits,
            target_len=t_orig,
            subsample=self.subsample,
            causal=self.causal,
            valid_frames=valid_frames,
        )
        return logits.permute(0, 2, 1)


class FusionTransformerEstimator(BaseSourceCountEstimator):
    """Dual-stream Transformer implementing Cornell et al. (2022) Early and Late FiLM fusion.

    Supports:
    - ``fusion_mode="early"`` (Cornell Fig. 3a): separate ``CatPoolNorm`` stems for
      the primary spectral stream and flattened spatial stream, concatenated prior
      to the bottleneck ``Conv1d(1x1)`` and sinusoidal positional encoding.
    - ``fusion_mode="film"`` (Cornell Fig. 3b): primary spectral stream mapped via
      ``CatPoolNorm -> Conv1d(1x1) -> PE``, while the separately cat-pooled and
      normalized spatial stream modulates hidden states via per-layer ``FiLMLayer``
      blocks immediately before every Transformer Encoder block.
    """

    def __init__(
        self,
        input_dim: int,
        cond_input_dim: int,
        transform: STFTtransform,
        max_sources: int,
        num_input_channels: int = 1,
        cond_num_channels: int = 1,
        fusion_mode: str = "early",
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
        """Initializes the dual-stream Early/Late-FiLM Transformer estimator.

        Args:
            input_dim (int): Per-channel feature dimension ``F_spec`` of the primary stream.
            cond_input_dim (int): Per-channel feature dimension ``F_spat`` of the spatial
                conditioning stream.
            transform (STFTtransform): STFT transform instance configured for the pipeline.
            max_sources (int): Maximum number of simultaneous active sources.
            num_input_channels (int, optional): Channel count ``C_spec`` of the primary
                stream flattened before ``CatPoolNorm``. Defaults to ``1``.
            cond_num_channels (int, optional): Channel count ``C_spat`` (e.g. ``P`` pairs)
                of the spatial stream flattened before ``CatPoolNorm``. Defaults to ``1``.
            fusion_mode (str, optional): Fusion scheme, either ``"early"`` (input
                concatenation after separate ``CatPoolNorm``) or ``"film"`` (layer-wise
                FiLM conditioning before each TE block). Defaults to ``"early"``.
            causal (bool, optional): Selects strictly causal mode when ``True`` or
                Cornell's non-causal baseline when ``False``. Defaults to ``False``.
            chunk_size (int, optional): Context window parameter ``C`` / ``T_c``.
                Defaults to ``7``.
            subsample (int, optional): Temporal subsampling factor ``S`` / ``T_sub``.
                Defaults to ``10``.
            embed_dim (int, optional): Transformer embedding width ``H``. Defaults to ``384``.
            n_heads (int, optional): Number of attention heads. Defaults to ``4``.
            ff_inner (int, optional): Inner FFN expansion width. Defaults to ``1024``.
            n_layers (int, optional): Number of stacked TE blocks ``R``. Defaults to ``4``.
            dropout (float, optional): Dropout probability. Defaults to ``0.2``.
            max_seq_len (int, optional): Maximum positional encoding length.
                Defaults to ``5000``.

        Raises:
            ValueError: If ``fusion_mode`` is not ``"early"`` or ``"film"``, or if
                ``cond_num_channels < 1``.
        """
        super().__init__(
            input_dim=input_dim,
            transform=transform,
            max_sources=max_sources,
            num_input_channels=num_input_channels,
        )
        mode = fusion_mode.lower()
        if mode not in ("early", "film"):
            raise ValueError(
                f"Invalid fusion_mode '{fusion_mode}'. Must be 'early' or 'film'."
            )
        if cond_num_channels < 1:
            raise ValueError(
                f"cond_num_channels must be >= 1, got {cond_num_channels}."
            )

        self.cond_input_dim = int(cond_input_dim)
        self.cond_num_channels = int(cond_num_channels)
        self.cond_flat_dim = self.cond_input_dim * self.cond_num_channels
        self.fusion_mode = mode
        self.causal = bool(causal)
        self.chunk_size = chunk_size
        self.subsample = subsample
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ff_inner = ff_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.spec_stem = CatPoolNorm(
            input_dim=self.flat_input_dim,
            chunk_size=chunk_size,
            subsample=subsample,
            causal=self.causal,
        )
        self.cond_stem = CatPoolNorm(
            input_dim=self.cond_flat_dim,
            chunk_size=chunk_size,
            subsample=subsample,
            causal=self.causal,
        )

        if self.fusion_mode == "early":
            self.bottleneck = nn.Sequential(
                nn.Conv1d(
                    self.spec_stem.out_dim + self.cond_stem.out_dim,
                    embed_dim,
                    1,
                )
            )
            self.film_layers: nn.ModuleList | None = None
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv1d(self.spec_stem.out_dim, embed_dim, 1)
            )
            self.film_layers = nn.ModuleList(
                [
                    FiLMLayer(cond_dim=self.cond_stem.out_dim, embed_dim=embed_dim)
                    for _ in range(n_layers)
                ]
            )

        self.pos_encs = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Conv1d(embed_dim, max_sources + 1, 1)
        self.layers = nn.ModuleList(
            [
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
                for _ in range(n_layers)
            ]
        )

    def get_config(self) -> Dict[str, Any]:
        """Returns a dictionary of fusion estimator hyperparameters.

        Returns:
            Dict[str, Any]: Serialized configuration dictionary.
        """
        return {
            "max_sources": self.max_sources,
            "fusion_mode": self.fusion_mode,
            "causal": self.causal,
            "chunk_size": self.chunk_size,
            "subsample": self.subsample,
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "ff_inner": self.ff_inner,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "input_dim": self.input_dim,
            "num_input_channels": self.num_input_channels,
            "cond_input_dim": self.cond_input_dim,
            "cond_num_channels": self.cond_num_channels,
        }

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs a formatted summary of fusion estimator hyperparameters.

        Args:
            indent (str, optional): Leading indentation string. Defaults to ``""``.
        """
        super()._verbose_parameters(indent)
        log.info(f"{indent}  Fusion Mode: {self.fusion_mode}")
        log.info(
            f"{indent}  Primary Stream: "
            f"{self.num_input_channels} x {self.input_dim} = {self.flat_input_dim}"
        )
        log.info(
            f"{indent}  Conditioning Stream: "
            f"{self.cond_num_channels} x {self.cond_input_dim} = {self.cond_flat_dim}"
        )
        log.info(f"{indent}  Causal Mode: {self.causal}")
        log.info(
            f"{indent}  Chunk Size: {self.chunk_size} "
            f"(Total Stacked: {self.spec_stem.context_frames})"
        )
        log.info(f"{indent}  Subsample Factor: {self.subsample}")
        log.info(f"{indent}  Embed Dim: {self.embed_dim}")
        log.info(f"{indent}  Num Heads: {self.n_heads}")
        log.info(f"{indent}  Num Layers: {self.n_layers}")

    def forward(
        self,
        features: torch.Tensor,
        cond_features: torch.Tensor | None = None,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flattens primary and conditioning streams and executes Early or Late FiLM fusion.

        Args:
            features (torch.Tensor): Primary stream tensor of shape
                ``(B, C_spec, F_spec, T)`` or ``(B, C_spec * F_spec, T)``.
            cond_features (torch.Tensor | None): Spatial conditioning stream tensor of
                shape ``(B, C_spat, F_spat, T)`` or ``(B, C_spat * F_spat, T)``.
            valid_frames (torch.Tensor | None, optional): Valid frame counts ``(B,)``
                for right-padded variable-length batches. Defaults to ``None``.

        Returns:
            torch.Tensor: Source activity logits of shape ``(B, T, max_sources + 1)``.

        Raises:
            ValueError: If ``cond_features`` is ``None`` or channel counts mismatch.
        """
        if cond_features is None:
            raise ValueError(
                "FusionTransformerEstimator requires `cond_features` in forward()."
            )
        x_spec = self._flatten_channel_features(
            features, self.num_input_channels, stream_name="features"
        )
        x_cond = self._flatten_channel_features(
            cond_features, self.cond_num_channels, stream_name="cond_features"
        )
        return self.forward_dual(x_spec, x_cond, valid_frames=valid_frames)

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """Not supported for dual-stream fusion; use ``forward(features, cond_features)``."""
        raise NotImplementedError(
            "FusionTransformerEstimator requires both primary and conditioning streams; "
            "call forward(features, cond_features) instead."
        )

    def forward_dual(
        self,
        x_spec: torch.Tensor,
        x_cond: torch.Tensor,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Processes flattened spectral and spatial streams through Early or Late FiLM fusion.

        Args:
            x_spec (torch.Tensor): Flattened primary stream ``(B, C_spec * F_spec, T)``.
            x_cond (torch.Tensor): Flattened conditioning stream ``(B, C_spat * F_spat, T)``.
            valid_frames (torch.Tensor | None, optional): Valid frame counts ``(B,)``.

        Returns:
            torch.Tensor: Source activity logits of shape ``(B, T, max_sources + 1)``.
        """
        t_orig = x_spec.size(-1)
        t_sub = (t_orig - 1) // self.subsample + 1 if self.subsample > 1 else t_orig
        frame_valid_mask, key_padding_mask = _prepare_valid_masks(
            t_orig=t_orig,
            t_sub=t_sub,
            subsample=self.subsample,
            valid_frames=valid_frames,
            device=x_spec.device,
        )
        if frame_valid_mask is not None:
            x_spec = x_spec.masked_fill(~frame_valid_mask, 0.0)
            x_cond = x_cond.masked_fill(~frame_valid_mask, 0.0)

        u_spec = self.spec_stem(x_spec)
        u_cond = self.cond_stem(x_cond)

        if self.fusion_mode == "early":
            h = self.bottleneck(torch.cat([u_spec, u_cond], dim=1))
            h = self.pos_encs(h)
            for layer_block in self.layers:
                mha = layer_block[0]
                ff = layer_block[1]
                h = ff(mha(h, key_padding_mask=key_padding_mask))
        else:
            assert self.film_layers is not None
            h = self.bottleneck(u_spec)
            h = self.pos_encs(h)
            for film, layer_block in zip(self.film_layers, self.layers):
                mha = layer_block[0]
                ff = layer_block[1]
                h = film(h, u_cond)
                h = ff(mha(h, key_padding_mask=key_padding_mask))

        h = self.out_norm(h.transpose(1, 2)).transpose(1, 2)
        logits = self.output(h)
        logits = upsample_logits(
            logits,
            target_len=t_orig,
            subsample=self.subsample,
            causal=self.causal,
            valid_frames=valid_frames,
        )
        return logits.permute(0, 2, 1)
