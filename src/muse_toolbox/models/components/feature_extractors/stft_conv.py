import logging
from typing import Any

import torch
import torch.nn as nn

from muse_toolbox.models.components.nn_blocks.causal_conv1d import CausalConv1d
from muse_toolbox.utils import STFTtransform

from .base_feature import BaseFeatureExtractor

log = logging.getLogger(__name__)


class STFT_Conv_Feature_Encoder(BaseFeatureExtractor):
    """
    Extracts features from an STFT using a stack of causal 1D convolutions.
    It builds a distinct CNN model for every possible microphone count up to `max_channels`.
    """
    def __init__(
        self,
        transform: STFTtransform,
        out_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 3,
        dropout: float = 0.0,
        max_channels: int = 8,  # Maximum expected microphones
    ) -> None:
        """
        Initializes the STFT Conv Feature Encoder.

        Args:
            transform (STFTtransform): The STFT transformation config.
            out_channels (int): Output feature dimension (J).
            kernel_size (int): Kernel size for the causal 1D conv layers.
            num_layers (int): Number of CNN layers.
            dropout (float): Dropout probability.
            max_channels (int): Maximum supported microphone channel count.
        """
        super().__init__(transform=transform)
        self.transform = transform
        self.out_channels_dim = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_channels = max_channels

        # Input dim per microphone is F (freq bins)
        self.freq_dim = self.transform.num_freq_bins

        # Create a dictionary of models, one for each possible channel count M
        # Keys must be strings for nn.ModuleDict
        self.models = nn.ModuleDict()

        for m in range(1, max_channels + 1):
            # Input dim is M * F * 2 (Real + Imag)
            self.models[str(m)] = self._build_model(input_dim=m * self.freq_dim * 2)

    def _build_model(self, input_dim: int) -> nn.Sequential:
        """Builds the causal CNN stack for a specific input dimension."""
        layers = []
        for i in range(self.num_layers):
            in_ch = input_dim if i == 0 else self.out_channels_dim

            # Use custom CausalConv1d
            layers.append(CausalConv1d(in_ch, self.out_channels_dim, self.kernel_size))
            layers.append(nn.BatchNorm1d(self.out_channels_dim))
            layers.append(nn.ReLU())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the feature extractor configuration.

        Returns:
            dict[str, Any]: The config dictionary.
        """
        return {
            "out_channels": self.out_channels_dim,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "max_channels": self.max_channels,
        }

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def signature(self) -> str:
        return (
            f"STFT_Conv_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_ks{self.kernel_size}_oc{self.out_channels_dim}"
            f"_nl{self.num_layers}_mc{self.max_channels}_do{self.dropout}"
        )

    @property
    def feature_dim(self) -> int:
        return self.out_channels_dim

    def forward_stft(self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            batch: (B, M, F, T) Complex tensor
        Returns:
            features: (B, 1, J, T)
        """
        # batch is complex (B, M, F, T)
        B, M, F, T = batch.shape

        # 2. Check if we have a model for this M
        if str(M) not in self.models:
            raise ValueError(
                f"Received input with {M} channels, but STFT_Conv_Feature_Encoder "
                f"was initialized with max_channels={self.max_channels}. "
                f"Available models: {list(self.models.keys())}"
            )

        # 3. Prepare Input: Concatenate Real and Imaginary parts
        # We want to preserve the channel structure initially, then flatten.
        # (B, F, M, T) -> Real: (B, F, M, T), Imag: (B, F, M, T)
        # Concatenate along F dimension? Or create a new dimension?
        # Let's stack them to keep M distinct: (B, F, 2*M, T)
        # This way, when we flatten M, we get (B, F*2*M, T)

        real = batch.real
        imag = batch.imag

        # Concatenate along the frequency dimension (dim=2)
        # Result: (B, M, 2*F, T)
        x_complex = torch.cat([real, imag], dim=2)

        # 4. Flatten M and F dimensions -> (B, M * 2*F, T)
        x = x_complex.reshape(B, M * 2 * F, T)

        # 5. Select the specific model for this channel count
        model = self.models[str(M)]

        # 6. Forward pass (Causal Conv) -> (B, J, T)
        out = model(x)

        # 7. Reshape to (B, Mproc, Fproc, T) -> (B, 1, J, T)
        return out.unsqueeze(1)
