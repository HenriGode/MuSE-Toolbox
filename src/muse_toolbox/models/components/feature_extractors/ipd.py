import logging
from typing import Any

import torch

from muse_toolbox.models.components.feature_extractors.base_feature import (
    BaseFeatureExtractor,
)
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class IPD_Feature_Extractor(BaseFeatureExtractor):
    """Extracts Inter-channel Phase Difference (IPD) features from an STFT.

    Outputs a tensor of shape ``(B, P, F, T)`` where ``P`` is the number of
    microphone pairs (``M - 1`` in ``"ref"`` mode or ``M * (M - 1) // 2`` in
    ``"all"`` mode).
    """

    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
    ) -> None:
        """Initializes the IPD Feature Extractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            mode (str, optional): Extraction mode, either ``"ref"`` (differences
                against ``ref_channel``) or ``"all"`` (all upper-triangular
                pairwise differences). Defaults to ``"ref"``.
            ref_channel (int, optional): Reference channel index if ``mode="ref"``.
                Defaults to ``0``.

        Raises:
            ValueError: If ``mode`` is not ``"ref"`` or ``"all"``.
        """
        super().__init__(transform=transform)
        self.transform = transform
        self.mode = mode
        self.ref_channel = ref_channel

        if mode not in ["ref", "all"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'ref' or 'all'.")

    def get_config(self) -> dict[str, Any]:
        """Returns the feature extractor configuration dictionary.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {
            "mode": self.mode,
            "ref_channel": self.ref_channel,
        }

    @property
    def is_trainable(self) -> bool:
        """Indicates that this feature extractor has no learnable parameters."""
        return False

    @property
    def signature(self) -> str:
        """Returns a unique identifier string for caching and logging."""
        sign = (
            f"IPD_Mdependent"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"
        return sign

    @property
    def feature_dim(self) -> int:
        """Returns the per-pair feature dimension ``F``."""
        return self.transform.num_freq_bins

    def output_channels(self, num_mics: int | None = None) -> int:
        """Returns the number of microphone pairs ``P`` for an ``M``-microphone array."""
        if num_mics is None:
            raise ValueError(
                f"{self.__class__.__name__} produces an M-dependent number of pairs (mode='{self.mode}'); "
                "set `num_mics` when pairing with a fixed-width flattening estimator."
            )
        if num_mics < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires at least 2 microphones, got num_mics={num_mics}."
            )
        if self.mode == "ref":
            return int(num_mics) - 1
        return int(num_mics) * (int(num_mics) - 1) // 2

    def forward_stft(
        self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Computes wrapped pairwise IPD values in ``[-pi, pi]``.

        Args:
            batch (torch.Tensor): Complex STFT tensor of shape ``(B, M, F, T)``.
            valid_mics (torch.Tensor | None, optional): Unused valid microphone
                counts preserved for interface compatibility. Defaults to ``None``.

        Returns:
            torch.Tensor: Pairwise IPD tensor of shape ``(B, P, F, T)``.
        """
        phase = torch.angle(batch)
        B, M, F, T = phase.shape

        if self.mode == "ref":
            ref_phase = phase[:, self.ref_channel : self.ref_channel + 1, :, :]
            diff = phase - ref_phase
            indices = [i for i in range(M) if i != self.ref_channel]
            if not indices:
                return torch.empty(B, 0, F, T, device=phase.device)
            diff = diff[:, indices, :, :]

        elif self.mode == "all":
            row, col = torch.triu_indices(M, M, offset=1, device=phase.device)
            diff = phase[:, row, :, :] - phase[:, col, :, :]
            if diff.shape[1] == 0:
                return torch.empty(B, 0, F, T, device=phase.device)

        else:
            raise ValueError(f"Invalid mode '{self.mode}'.")

        return torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi

    def get_valid_feature_mask(
        self, valid_mics_count: torch.Tensor | None, max_M: int
    ) -> torch.Tensor | None:
        """Returns a boolean mask of shape ``(B, P)`` marking valid microphone pairs."""
        if valid_mics_count is None:
            return None
        B = valid_mics_count.shape[0]
        device = valid_mics_count.device

        mic_mask = torch.arange(max_M, device=device).expand(
            B, max_M
        ) < valid_mics_count.unsqueeze(1)

        if self.mode == "ref":
            indices = [i for i in range(max_M) if i != self.ref_channel]
            return mic_mask[:, indices]

        if self.mode == "all":
            row, col = torch.triu_indices(max_M, max_M, offset=1, device=device)
            return mic_mask[:, row] & mic_mask[:, col]

        return None


class CSIPD_Feature_Extractor(IPD_Feature_Extractor):
    """Extracts block-concatenated Cosine and Sine Inter-channel Phase Difference (CSIPD) features."""

    @property
    def signature(self) -> str:
        """Returns a unique identifier string for caching and logging."""
        sign = (
            f"CSIPD_Mdependent"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"
        return sign

    @property
    def feature_dim(self) -> int:
        """Returns the per-pair feature dimension ``2 * F``."""
        return 2 * self.transform.num_freq_bins

    def forward_stft(
        self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extracts block-concatenated cosine and sine IPD features.

        Args:
            batch (torch.Tensor): Complex STFT tensor of shape ``(B, M, F, T)``.
            valid_mics (torch.Tensor | None, optional): Unused valid microphone
                counts preserved for interface compatibility. Defaults to ``None``.

        Returns:
            torch.Tensor: Block-concatenated CSIPD tensor ``[cos(ipd), sin(ipd)]``
                of shape ``(B, P, 2*F, T)``.
        """
        ipd = super().forward_stft(batch, valid_mics=valid_mics)
        c = torch.cos(ipd)
        s = torch.sin(ipd)
        return torch.cat((c, s), dim=2)
