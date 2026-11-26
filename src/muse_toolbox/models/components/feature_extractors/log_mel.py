import logging
from typing import Any

import torch
import torchaudio

from muse_toolbox.models.components.feature_extractors.base_feature import (
    BaseFeatureExtractor,
)
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class LogMel_Feature_Extractor(BaseFeatureExtractor):
    """
    Extracts Log-Mel spectrogram features from an STFT input.
    Outputs a tensor with an M-dependent feature dimension if mode is 'all'.
    """
    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
        log_offset: float = 1e-6,
    ) -> None:
        """
        Initializes the Log-Mel feature extractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            mode (str): Extraction mode. Either 'ref' (reference channel only) or 'all' (all channels flattened).
            ref_channel (int): Reference channel index if mode is 'ref'.
            n_mels (int): Number of Mel filterbank bands.
            f_min (float): Minimum frequency for the Mel scale.
            f_max (float | None): Maximum frequency for the Mel scale.
            log_offset (float): Small constant added to avoid log(0).
        """
        super().__init__(transform=transform)
        self.transform = transform
        self.mode = mode
        self.ref_channel = ref_channel
        self.n_mels = n_mels
        self.log_offset = log_offset

        if mode not in ["ref", "all"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'ref' or 'all'.")

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=int(transform.sampling_frequency),
            f_min=f_min,
            f_max=f_max,
            n_stft=transform.nfft // 2 + 1,
        )

    def get_config(self) -> dict[str, Any]:
        """
        Returns the feature extractor configuration.

        Returns:
            dict[str, Any]: The config dictionary.
        """
        return {
            "mode": self.mode,
            "ref_channel": self.ref_channel,
            "n_mels": self.n_mels,
            "f_min": self.mel_scale.f_min,
            "f_max": self.mel_scale.f_max,
            "log_offset": self.log_offset,
        }

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_single_channel_input(self) -> bool:
        """Returns ``True`` when configured for single-channel reference mode."""
        return self.mode == "ref"

    @property
    def signature(self) -> str:
        sign = (
            f"LogMel"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_nm{self.n_mels}_freq{self.mel_scale.f_min}-{self.mel_scale.f_max}_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"{self.ref_channel}"
        return sign

    @property
    def feature_dim(self) -> int:
        return self.n_mels

    def output_channels(self, num_mics: int | None = None) -> int:
        """Returns ``1`` in ``"ref"`` mode, or ``num_mics`` in ``"all"`` mode."""
        if self.mode == "ref":
            return 1
        return super().output_channels(num_mics)

    def forward_stft(self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None) -> torch.Tensor:
        # batch: (B, M, F, T)
        # Compute power spectrogram
        power_spec = batch.abs().pow(2)

        # Pad frequency dimension if DC/Nyquist were removed by STFT
        pad_bottom = 1 if self.transform.remove_Nyquist else 0
        pad_top = 1 if self.transform.remove_DC else 0
        if pad_top > 0 or pad_bottom > 0:
            # power_spec is (B, M, F, T), so F is the 2nd to last dim
            power_spec = torch.nn.functional.pad(
                power_spec, (0, 0, pad_top, pad_bottom)
            )

        # Apply MelScale: (B, M, n_mels, T)
        mel_spec = self.mel_scale(power_spec)

        # Log
        log_mel = torch.log(mel_spec + self.log_offset)

        if self.mode == "ref":
            # (B, 1, n_mels, T); if dataset pre-sliced to 1 channel, select index 0 safely
            if log_mel.shape[1] == 1:
                return log_mel[:, :1, :, :]
            return log_mel[:, self.ref_channel : self.ref_channel + 1, :, :]
        elif self.mode == "all":
            # (B, M, n_mels, T)
            return log_mel
        else:
            raise ValueError(f"Invalid mode '{self.mode}'.")

    def get_valid_feature_mask(
        self, valid_mics_count: torch.Tensor | None, max_M: int
    ) -> torch.Tensor | None:
        """Returns a boolean validity mask aligned with the output channel dimension.

        Args:
            valid_mics_count (torch.Tensor | None): Tensor of shape ``(B,)``
                with valid microphone counts per batch item.
            max_M (int): Maximum number of microphones in the batch.

        Returns:
            torch.Tensor | None: Boolean mask of shape ``(B, 1)`` when
                ``mode="ref"`` or ``(B, max_M)`` when ``mode="all"``.
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