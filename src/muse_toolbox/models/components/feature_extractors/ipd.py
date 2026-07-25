import logging
import warnings
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from muse_toolbox.models.components.nn_blocks.causal_conv1d import CausalConv1d
from muse_toolbox.utils import STFTtransform

from .base_feature import BaseFeatureExtractor

log = logging.getLogger(__name__)


class IPD_Feature_Extractor(BaseFeatureExtractor):
    """
    Extracts Inter-channel Phase Difference (IPD) features from an STFT.
    Outputs a tensor with an M-dependent feature dimension.
    """
    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
    ) -> None:
        """
        Initializes the IPD Feature Extractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            mode (str): Mode of extraction. Either 'ref' (differences against a reference channel) 
                or 'all' (all pairwise differences).
            ref_channel (int): Index of the reference channel if mode is 'ref'.
        """
        super().__init__(transform=transform)
        self.transform = transform
        self.mode = mode
        self.ref_channel = ref_channel

        if mode not in ["ref", "all"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'ref' or 'all'.")

    def get_config(self) -> dict[str, Any]:
        """
        Returns the feature extractor configuration.

        Returns:
            dict[str, Any]: The config dictionary.
        """
        return {
            "mode": self.mode,
            "ref_channel": self.ref_channel,
        }

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def signature(self) -> str:
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
        return self.transform.num_freq_bins

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, M, F, T)
        phase = torch.angle(batch)
        B, M, F, T = phase.shape

        if self.mode == "ref":
            ref_phase = phase[:, self.ref_channel : self.ref_channel + 1, :, :]
            diff = phase - ref_phase
            # Remove the ref channel
            indices = [i for i in range(M) if i != self.ref_channel]
            if not indices:
                return torch.empty(B, 0, F, T, device=phase.device)
            diff = diff[:, indices, :, :]

        elif self.mode == "all":
            # Compute all pairs (i, j) where i < j
            diffs = []
            for i in range(M):
                for j in range(i + 1, M):
                    diffs.append(phase[:, i : i + 1, :, :] - phase[:, j : j + 1, :, :])
            if not diffs:
                return torch.empty(B, 0, F, T, device=phase.device)
            diff = torch.cat(diffs, dim=1)

        else:
            raise ValueError(f"Invalid mode '{self.mode}'.")

        # Wrap to [-pi, pi]
        ipd = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi

        return ipd


class CSIPD_Feature_Extractor(IPD_Feature_Extractor):
    """
    Extracts Cosine and Sine Inter-channel Phase Difference (CSIPD) features.
    Expands the IPD phase angles into their cos and sin components.
    """
    @property
    def signature(self) -> str:
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
        return 2 * self.transform.num_freq_bins

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # 1. Get standard IPD features: (B, P, F, T)
        ipd = super().forward_stft(batch)
        B, P, F, T = ipd.shape

        c = torch.cos(ipd)
        s = torch.sin(ipd)

        # Interleave cos and sin along frequency dim: (B, P, 2F, T)
        csipd = torch.stack((c, s), dim=3).reshape(B, P, 2 * F, T)

        return csipd


class Condensed_IPD_Feature_Extractor(IPD_Feature_Extractor):
    """
    .. deprecated:: 0.2.0
       This class is deprecated because channel combination logic has been decoupled 
       into the `ChannelCombinator` modules. Use `IPD_Feature_Extractor` with a 
       `ChannelCombinator` in the pipeline instead.

    Condenses the M-dependent IPD features into a fixed-dimension representation
    either via a trainable CNN or circular mean.
    """
    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
        condense_method: str = "conv",  # "conv" or "circular_mean"
        max_channels: int = 6,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the Condensed IPD feature extractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            mode (str): 'ref' or 'all' for pairwise differences.
            ref_channel (int): Reference channel index.
            condense_method (str): Method to condense the dimension ('conv' or 'circular_mean').
            max_channels (int): Max expected microphone channels.
            num_layers (int): Number of CNN layers if condense_method is 'conv'.
            kernel_size (int): Kernel size for CNN.
            dropout (float): Dropout probability.
        """
        warnings.warn("Use IPD_Feature_Extractor with a ChannelCombinator instead", DeprecationWarning)
        
        # Determine the number of condensed channels=None because feature_dim is now fixed (F)
        super().__init__(transform=transform, mode=mode, ref_channel=ref_channel)
        self.condense_method = condense_method
        self.max_channels = max_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.freq_dim = self.transform.num_freq_bins

        if self.condense_method == "conv":
            self.models = nn.ModuleDict()
            for m in range(2, max_channels + 1):
                # Calculate number of pairs P based on mode
                if self.mode == "ref":
                    p = m - 1
                else:
                    p = m * (m - 1) // 2

                # Map (P*F) -> F
                self.models[str(m)] = self._build_model(input_dim=p * self.freq_dim)

    def _build_model(self, input_dim: int) -> nn.Sequential:
        """Builds the causal CNN stack for a specific input dimension."""
        layers = []
        for i in range(self.num_layers):
            in_ch = input_dim if i == 0 else self.freq_dim

            # Use custom CausalConv1d
            layers.append(CausalConv1d(in_ch, self.freq_dim, self.kernel_size))
            layers.append(nn.BatchNorm1d(self.freq_dim))
            layers.append(nn.ReLU())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    @property
    def is_trainable(self) -> bool:
        return self.condense_method == "conv"

    @property
    def signature(self) -> str:
        sign = (
            f"IPD"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"

        sign += f"_{self.condense_method}"

        if self.condense_method == "conv":
            sign += f"_nl{self.num_layers}_ks{self.kernel_size}_do{self.dropout}"
        return sign

    @property
    def feature_dim(self) -> int:
        # Always returns F, independent of M
        return self.freq_dim

    @property
    def precompute_type(self) -> str:
        return "features"

    def precompute(
        self, batch: torch.Tensor, input_type: str = "raw_audio"
    ) -> dict[str, torch.Tensor]:
        precomputedict = defaultdict(torch.Tensor)

        # Get STFT first
        if input_type == "raw_audio":
            stft = self.transform.encode(batch)
        elif input_type == "stft":
            stft = batch
        else:
            raise ValueError(f"Invalid input_type {input_type}")

        # Compute raw IPD (using parent class)
        if stft.ndim == 3:
            stft = stft.unsqueeze(0)  # (1, F, M, T)
        else:
            raise ValueError(
                "Expected stft input to have 3 dimensions (F, M, T). No batch dimension in the precompute step allowed!"
            )

        raw_ipd = super().forward_stft(stft)

        raw_ipd = raw_ipd.squeeze(0)  # (J, T)

        precomputedict["features"] = raw_ipd
        return precomputedict

    def forward_precomputed_features(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, P, F, T)
        B, P, F_dim, T = batch.shape

        if self.condense_method == "conv":
            if self.mode == "ref":
                M = P + 1
            else:
                M = int((1 + (1 + 8 * P) ** 0.5) / 2)

            if str(M) not in self.models:
                raise ValueError(
                    f"No model found for M={M} (max_channels={self.max_channels})"
                )
            
            ipd_flat = batch.reshape(B, P * F_dim, T)
            return self.models[str(M)](ipd_flat)

        elif self.condense_method == "circular_mean":
            # Circular Mean over pairs P: atan2( sum(sin), sum(cos) )
            sin_sum = torch.sum(torch.sin(batch), dim=1)
            cos_sum = torch.sum(torch.cos(batch), dim=1)
            mean_ipd = torch.atan2(sin_sum, cos_sum)  # (B, F, T)

            return mean_ipd.unsqueeze(1)  # (B, 1, F, T)

        else:
            raise ValueError(f"Unknown condense_method: {self.condense_method}")

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # 1. Get standard IPD features: (B, P*F, T)
        ipd_flat = super().forward_stft(batch)
        return self.forward_precomputed_features(ipd_flat)


class Condensed_CSIPD_Feature_Extractor(CSIPD_Feature_Extractor):
    """
    .. deprecated:: 0.2.0
       This class is deprecated because channel combination logic has been decoupled 
       into the `ChannelCombinator` modules. Use `CSIPD_Feature_Extractor` with a 
       `ChannelCombinator` in the pipeline instead.

    Condenses the M-dependent CSIPD features into a fixed-dimension representation
    either via a trainable CNN or vector mean.
    """
    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
        condense_method: str = "conv",  # "conv" or "vector_mean"
        max_channels: int = 8,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the Condensed CSIPD feature extractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            mode (str): 'ref' or 'all' for pairwise differences.
            ref_channel (int): Reference channel index.
            condense_method (str): Method to condense the dimension ('conv' or 'vector_mean').
            max_channels (int): Max expected microphone channels.
            num_layers (int): Number of CNN layers if condense_method is 'conv'.
            kernel_size (int): Kernel size for CNN.
            dropout (float): Dropout probability.
        """
        warnings.warn("Use CSIPD_Feature_Extractor with a ChannelCombinator instead", DeprecationWarning)
        super().__init__(transform=transform, mode=mode, ref_channel=ref_channel)
        self.condense_method = condense_method
        self.max_channels = max_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.freq_dim = self.transform.num_freq_bins

        if self.condense_method == "conv":
            self.models = nn.ModuleDict()
            for m in range(2, max_channels + 1):
                if self.mode == "ref":
                    p = m - 1
                else:
                    p = m * (m - 1) // 2

                # Map (2*P*F) -> 2*F
                self.models[str(m)] = self._build_model(input_dim=2 * p * self.freq_dim)

    def _build_model(self, input_dim: int) -> nn.Sequential:
        """Builds the causal CNN stack for a specific input dimension."""
        layers = []
        for i in range(self.num_layers):
            in_ch = input_dim if i == 0 else 2 * self.freq_dim

            # Use custom CausalConv1d
            layers.append(CausalConv1d(in_ch, 2 * self.freq_dim, self.kernel_size))
            layers.append(nn.BatchNorm1d(2 * self.freq_dim))
            layers.append(nn.ReLU())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    @property
    def is_trainable(self) -> bool:
        return self.condense_method == "conv"

    @property
    def signature(self) -> str:
        sign = (
            f"CSIPD"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"

        sign += f"_{self.condense_method}"

        if self.condense_method == "conv":
            sign += f"_nl{self.num_layers}_ks{self.kernel_size}_do{self.dropout}"
        return sign

    @property
    def feature_dim(self) -> int:
        # Always returns 2*F
        return 2 * self.freq_dim

    @property
    def precompute_type(self) -> str:
        return "features"

    def precompute(
        self, batch: torch.Tensor, input_type: str = "raw_audio"
    ) -> dict[str, torch.Tensor]:
        precomputedict = defaultdict(torch.Tensor)

        # Get STFT first
        if input_type == "raw_audio":
            stft = self.transform.encode(batch)
        elif input_type == "stft":
            stft = batch
        else:
            raise ValueError(f"Invalid input_type {input_type}")

        # Compute raw CSIPD (using parent class)
        if stft.ndim == 3:
            stft = stft.unsqueeze(0)  # (1, F, M, T)
        else:
            raise ValueError(
                "Expected stft input to have 3 dimensions (F, M, T). No batch dimension in the precompute step allowed!"
            )

        raw_csipd = super().forward_stft(stft)

        raw_csipd = raw_csipd.squeeze(0)  # (J, T)

        precomputedict["features"] = raw_csipd
        return precomputedict

    def forward_precomputed_features(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, P, 2*F, T)
        B, P, F2, T = batch.shape

        if self.condense_method == "conv":
            # Infer M
            if self.mode == "ref":
                M = P + 1
            else:
                M = int((1 + (1 + 8 * P) ** 0.5) / 2)

            if str(M) not in self.models:
                raise ValueError(f"No model found for M={M}")
                
            csipd_flat = batch.reshape(B, P * F2, T)
            return self.models[str(M)](csipd_flat)

        elif self.condense_method == "vector_mean":
            # Reshape to (B, P, F, 2, T)
            csipd_unflat = batch.reshape(B, P, self.freq_dim, 2, T)

            # Average over P (dim 1) -> (B, F, 2, T)
            mean_csipd = torch.mean(csipd_unflat, dim=1)

            # Flatten F and 2 -> (B, 2*F, T) -> add Mproc -> (B, 1, 2F, T)
            return mean_csipd.reshape(B, 1, self.freq_dim * 2, T)

        else:
            raise ValueError(f"Unknown condense_method: {self.condense_method}")

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # 1. Get standard CSIPD features: (B, 2*P*F, T)
        csipd_flat = super().forward_stft(batch)
        return self.forward_precomputed_features(csipd_flat)

