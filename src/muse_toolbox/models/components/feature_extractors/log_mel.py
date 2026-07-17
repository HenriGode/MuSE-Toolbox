import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torchaudio

from muse_toolbox.models.components.nn_blocks.causal_conv1d import CausalConv1d
from muse_toolbox.utils import STFTtransform

from .base_feature import BaseFeatureExtractor

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
        if self.mode == "ref":
            return self.n_mels
        else:
            raise NotImplementedError(
                "feature_dim is M-dependent for LogMel features in 'all' mode. Use Condensed_LogMel_Feature_Extractor."
            )

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, F, M, T)
        # Compute power spectrogram
        power_spec = batch.abs().pow(2)

        # Permute to (B, M, F, T) for MelScale
        power_spec = power_spec.permute(0, 2, 1, 3)

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
            # (B, n_mels, T)
            return log_mel[:, self.ref_channel, :, :]
        elif self.mode == "all":
            # Flatten channels and mels -> (B, M*n_mels, T)
            B, M, N_mels, T = log_mel.shape
            return log_mel.reshape(B, M * N_mels, T)
        else:
            raise ValueError(f"Invalid mode '{self.mode}'.")


class Condensed_LogMel_Feature_Extractor(LogMel_Feature_Extractor):
    """
    Condenses the multi-channel Log-Mel features into a fixed-dimension representation
    either via a trainable CNN or simple mean over channels.
    """
    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
        condense_method: str = "conv",  # "conv" or "mean"
        max_channels: int = 6,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the Condensed Log-Mel feature extractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            mode (str): 'ref' or 'all'.
            ref_channel (int): Reference channel index.
            n_mels (int): Number of Mel bands.
            f_min (float): Min frequency.
            f_max (float | None): Max frequency.
            condense_method (str): Method to condense dimension ('conv' or 'mean').
            max_channels (int): Max expected microphone channels.
            num_layers (int): Number of CNN layers if condense_method is 'conv'.
            kernel_size (int): Kernel size for CNN.
            dropout (float): Dropout probability.
        """

        super().__init__(
            transform=transform,
            mode=mode,
            ref_channel=ref_channel,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.condense_method = condense_method
        self.max_channels = max_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout

        if self.condense_method == "conv" and self.mode == "all":
            self.models = nn.ModuleDict()
            for m in range(2, max_channels + 1):
                # Map (M*n_mels) -> n_mels
                self.models[str(m)] = self._build_model(input_dim=m * self.n_mels)

    def _build_model(self, input_dim: int) -> nn.Sequential:
        layers = []
        for i in range(self.num_layers):
            in_ch = input_dim if i == 0 else self.n_mels

            layers.append(CausalConv1d(in_ch, self.n_mels, self.kernel_size))
            layers.append(nn.BatchNorm1d(self.n_mels))
            layers.append(nn.ReLU())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    @property
    def is_trainable(self) -> bool:
        return self.condense_method == "conv" and self.mode == "all"

    @property
    def signature(self) -> str:
        sign = super().signature + f"_{self.condense_method}"
        if self.is_trainable:
            sign += f"_nl{self.num_layers}_ks{self.kernel_size}_do{self.dropout}"
        return sign

    @property
    def feature_dim(self) -> int:
        return self.n_mels

    @property
    def precompute_type(self) -> str:
        return "features"

    def precompute(
        self, batch: torch.Tensor, input_type: str = "raw_audio"
    ) -> dict[str, torch.Tensor]:
        precomputedict = defaultdict(torch.Tensor)

        if input_type == "raw_audio":
            stft = self.transform.encode(batch)
        elif input_type == "stft":
            stft = batch
        else:
            raise ValueError(f"Invalid input_type {input_type}")

        # Compute raw LogMel (using parent class)
        # This returns (B, n_mels, T) for ref, or (B, M*n_mels, T) for all

        if stft.dim() == 3:
            # (B, F, T) -> (B, F, 1, T)
            stft = stft.unsqueeze(0)
        else:
            raise ValueError(
                "Expected stft input to have 3 dimensions (F, M, T). No batch dimension in the precompute step allowed!"
            )

        raw_mel = super().forward_stft(stft)

        raw_mel = raw_mel.squeeze(0)  # (J, T)

        precomputedict["features"] = raw_mel
        return precomputedict

    def forward_precomputed_features(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, J, T)
        # If mode is ref, J = n_mels.
        # If mode is all, J = M * n_mels.

        if self.mode == "ref":
            return batch

        # mode == "all"
        mel_flat = batch
        B, J, T = mel_flat.shape
        M = J // self.n_mels

        if self.condense_method == "conv":
            if str(M) not in self.models:
                raise ValueError(
                    f"No model found for M={M} (max_channels={self.max_channels})"
                )
            return self.models[str(M)](mel_flat)

        elif self.condense_method == "mean":
            # Reshape to (B, M, n_mels, T)
            mel = mel_flat.reshape(B, M, self.n_mels, T)
            # Mean over channels
            return torch.mean(mel, dim=1)

        else:
            raise ValueError(f"Unknown condense_method: {self.condense_method}")

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        raw_mel = super().forward_stft(batch)
        return self.forward_precomputed_features(raw_mel)
