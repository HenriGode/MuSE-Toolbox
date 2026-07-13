from typing import Any
import logging
import torch

log = logging.getLogger(__name__)
from muse_toolbox.utils import (
    STFTtransform,
    smoothCovarianceMatrix,
    gmsc,
    regularize,
    get_real_dtype,
)
from .base_feature import BaseFeatureExtractor


class GMSC_Feature_Extractor(BaseFeatureExtractor):
    """
    Implements the Generalized Magnitude-Squared Coherence (GMSC) feature extractor.
    This is a simplified version of WGMSC without whitening and reverse features.
    """

    def __init__(
        self,
        transform: STFTtransform,
        smoothing_time_constant: float,
    ) -> None:
        """
        Initializes the GMSC feature extractor.

        Args:
            transform (STFTtransform): The STFT transformation applied to the audio.
            smoothing_time_constant (float): Time constant for smoothing the covariance matrix [s].
        """
        super().__init__(transform=transform)
        self.smoothing_time_constant = smoothing_time_constant
        self.transform = transform

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def signature(self) -> str:
        return (
            f"GMSC"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_stc{self.smoothing_time_constant}"
        )

    @property
    def feature_dim(self) -> int:
        return self.transform.num_freq_bins

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the feature extractor.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {
            "smoothing_time_constant": self.smoothing_time_constant,
        }

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch (torch.Tensor): STFT signal (B, F, M, T) complex.
        Returns:
            torch.Tensor: GMSC features (B, F, T).
        """
        # batch: (B, F, M, T)

        Ry = regularize(
            smoothCovarianceMatrix(
                batch.to(dtype=torch.complex128, device=batch.device),
                smoothing_factor=self.transform.timeConstant2smoothingFactor(
                    self.smoothing_time_constant
                ),
            ),
            reg_factor=1e-6,
        )

        # Compute GMSC: (B, F, T)
        gmsc_val = gmsc(Ry)[..., 0, 0].to(
            dtype=get_real_dtype(batch), device=batch.device
        )

        return gmsc_val

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs the feature extractor parameters."""
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Smoothing Time Constant: {self.smoothing_time_constant} s")
