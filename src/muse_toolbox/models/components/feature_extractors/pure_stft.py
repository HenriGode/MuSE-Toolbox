import torch
from typing import Any

from muse_toolbox.utils import STFTtransform
from .base_feature import BaseFeatureExtractor


class PureSTFTFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts the pure STFT features without any learnable parameters.
    It simply concatenates the real and imaginary parts of the complex STFT 
    along the frequency dimension, preserving the microphone channel dimension.
    
    Output shape: (B, M, 2*F, T)
    where M is the number of microphones, and F is the number of frequency bins.
    """
    def __init__(self, transform: STFTtransform) -> None:
        super().__init__(transform=transform)
        self.transform = transform

    def get_config(self) -> dict[str, Any]:
        """
        Returns the feature extractor configuration.
        """
        return {}

    @property
    def is_trainable(self) -> bool:
        """This feature extractor has no learnable parameters."""
        return False

    @property
    def signature(self) -> str:
        return (
            f"PureSTFT_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
        )

    @property
    def feature_dim(self) -> int:
        """The output feature dimension is 2 * F (real and imaginary parts)."""
        return self.transform.num_freq_bins * 2

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the pure STFT feature extractor.
        
        Args:
            batch: (B, M, F, T) Complex tensor containing STFT values.
            
        Returns:
            features: (B, M, 2*F, T) Real tensor containing concatenated real/imag parts.
        """
        # Extract real and imaginary components
        real = batch.real
        imag = batch.imag

        # Concatenate along the frequency dimension (dim=2)
        # Shape: (B, M, 2*F, T)
        x = torch.cat([real, imag], dim=2)

        return x
