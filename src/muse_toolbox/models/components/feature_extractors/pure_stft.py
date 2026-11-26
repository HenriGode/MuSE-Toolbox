import logging
from typing import Any

import torch

from muse_toolbox.models.components.feature_extractors.base_feature import (
    BaseFeatureExtractor,
)
from muse_toolbox.utils import STFTtransform, get_real_dtype

log = logging.getLogger(__name__)


class PureSTFTFeatureExtractor(BaseFeatureExtractor):
    """Extracts pure STFT features without any learnable parameters.

    Supports both polar representation (`[magnitude, cos(phase), sin(phase)]`
    with output dimension `3 * F`) and backward-compatible Cartesian
    representation (`[real, imag]` with output dimension `2 * F`), combined
    with either reference-channel (`mode="ref"`, `C = 1`) or all-channel
    (`mode="all"`, `C = M`) selection.
    """

    def __init__(
        self,
        transform: STFTtransform,
        representation: str = "polar",
        mode: str = "all",
        ref_channel: int = 0,
        eps: float = 1e-8,
    ) -> None:
        """Initializes the pure STFT feature extractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            representation (str, optional): Output STFT format, either
                ``"polar"`` (``[|Y|, cos(phase), sin(phase)]``) or
                ``"cartesian"`` (``[real, imag]``). Defaults to ``"polar"``.
            mode (str, optional): Channel selection mode, either ``"ref"``
                (single reference channel) or ``"all"`` (all microphone
                channels). Defaults to ``"all"``.
            ref_channel (int, optional): Reference channel index used when
                ``mode="ref"``. Defaults to ``0``.
            eps (float, optional): Numerical floor applied to magnitudes when
                computing normalized polar phase components. Defaults to ``1e-8``.

        Raises:
            ValueError: If ``representation``, ``mode``, or ``eps`` is invalid.
        """
        super().__init__(transform=transform)
        if representation not in ("polar", "cartesian"):
            raise ValueError(
                f"Invalid representation '{representation}'. Must be 'polar' or 'cartesian'."
            )
        if mode not in ("ref", "all"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'ref' or 'all'.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.transform = transform
        self.representation = representation
        self.mode = mode
        self.ref_channel = ref_channel
        self.eps = eps

    def get_config(self) -> dict[str, Any]:
        """Returns the feature extractor configuration dictionary.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {
            "representation": self.representation,
            "mode": self.mode,
            "ref_channel": self.ref_channel,
            "eps": self.eps,
        }

    @property
    def is_trainable(self) -> bool:
        """Indicates that this feature extractor has no learnable parameters."""
        return False

    @property
    def is_single_channel_input(self) -> bool:
        """Returns ``True`` when configured for single-channel reference mode."""
        return self.mode == "ref"

    @property
    def precompute_type(self) -> str:
        """Returns the precomputation artifact type."""
        return "stft"

    @property
    def signature(self) -> str:
        """Returns a unique identifier string for caching and logging."""
        sign = (
            f"PureSTFT_{self.representation}_{self.mode}"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"
        return sign

    @property
    def feature_dim(self) -> int:
        """Returns the output feature dimension (``3*F`` for polar, ``2*F`` for cartesian)."""
        multiplier = 3 if self.representation == "polar" else 2
        return self.transform.num_freq_bins * multiplier

    def output_channels(self, num_mics: int | None = None) -> int:
        """Returns ``1`` in ``"ref"`` mode, or ``num_mics`` in ``"all"`` mode."""
        if self.mode == "ref":
            return 1
        return super().output_channels(num_mics)

    def forward_stft(
        self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extracts pure STFT features from a complex STFT tensor.

        Args:
            batch (torch.Tensor): Complex STFT tensor of shape ``(B, M, F, T)``.
            valid_mics (torch.Tensor | None, optional): Unused valid microphone
                counts preserved for interface compatibility. Defaults to ``None``.

        Returns:
            torch.Tensor: Real-valued STFT feature tensor of shape
                ``(B, C, 3*F, T)`` for ``"polar"`` or ``(B, C, 2*F, T)`` for
                ``"cartesian"``, where ``C=1`` in ``"ref"`` mode and ``C=M`` in
                ``"all"`` mode.

        Raises:
            IndexError: If ``mode="ref"`` and ``ref_channel`` is out of bounds.
        """
        num_channels = batch.shape[1]
        if self.mode == "ref":
            if num_channels == 1:
                x = batch[:, :1, :, :]
            else:
                if not (0 <= self.ref_channel < num_channels):
                    raise IndexError(
                        f"ref_channel={self.ref_channel} out of range for {num_channels} channels."
                    )
                x = batch[:, self.ref_channel : self.ref_channel + 1, :, :]
        else:
            x = batch

        if self.representation == "cartesian":
            return torch.cat([x.real, x.imag], dim=2)

        mag = torch.abs(x)
        mag_safe = mag.clamp(min=self.eps)
        cos_phi = torch.where(
            mag > self.eps,
            x.real / mag_safe,
            torch.zeros_like(mag),
        ).clamp(-1.0, 1.0)
        sin_phi = torch.where(
            mag > self.eps,
            x.imag / mag_safe,
            torch.zeros_like(mag),
        ).clamp(-1.0, 1.0)
        return torch.cat([mag, cos_phi, sin_phi], dim=2).to(dtype=get_real_dtype(batch))

    def get_valid_feature_mask(
        self, valid_mics_count: torch.Tensor | None, max_M: int
    ) -> torch.Tensor | None:
        """Returns a boolean validity mask aligned with the output channel dimension.

        Args:
            valid_mics_count (torch.Tensor | None): Tensor of shape ``(B,)``
                giving valid microphone counts per batch item.
            max_M (int): Maximum number of microphones in the batch.

        Returns:
            torch.Tensor | None: Boolean mask of shape ``(B, 1)`` for ``"ref"``
                mode, ``(B, max_M)`` for ``"all"`` mode, or ``None``.
        """
        if valid_mics_count is None:
            return None
        if self.mode == "ref":
            return torch.ones(
                (valid_mics_count.shape[0], 1),
                dtype=torch.bool,
                device=valid_mics_count.device,
            )
        return super().get_valid_feature_mask(valid_mics_count, max_M)

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs the feature extractor configuration parameters."""
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Representation: {self.representation}")
        log.info(f"{indent}  Mode: {self.mode}")
        if self.mode == "ref":
            log.info(f"{indent}  Reference Channel: {self.ref_channel}")
        log.info(f"{indent}  Epsilon: {self.eps}")
