import torch
from abc import ABC, abstractmethod
from typing import Any
import logging

log = logging.getLogger(__name__)


class BaseRTFestimator(torch.nn.Module, ABC):
    """Base class for RTF estimators."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        noise_cov: torch.Tensor,  # dimensions: (batch, freq, time, channels, channels)
        noisy_cov: torch.Tensor,  # dimensions: (batch, freq, time, channels, channels)
        old_rtfs: torch.Tensor,  # dimensions: (batch, freq, time, channels, num_sources)
        old_noisy_cov: torch.Tensor,  # dimensions: (batch, freq, time, channels, channels)
        **kwargs,
    ) -> torch.Tensor:
        """
        Wrapper for the forward_ method to maintain a consistent interface.

        Args:
            noise_cov (torch.Tensor): dimensions: (batch, freq, time, channels, channels)
            noisy_cov (torch.Tensor): dimensions: (batch, freq, time, channels, channels)
            old_rtfs (torch.Tensor): dimensions: (batch, freq, time, channels, num_sources)
            old_noisy_cov (torch.Tensor): dimensions: (batch, freq, time, channels, channels)

        Returns:
            torch.Tensor: estimated RTF vectors with dimensions (batch, freq, time, channels, 1)
        """
        return self.forward_(Rn=noise_cov, Ry=noisy_cov, G=old_rtfs, Rv=old_noisy_cov)

    @abstractmethod
    def forward_(
        self,
        Rn: torch.Tensor,  # dimensions: (batch, freq, time, channels, channels)
        Ry: torch.Tensor,  # dimensions: (batch, freq, time, channels, channels)
        G: torch.Tensor,  # dimensions: (batch, freq, time, channels, num_sources)
        Rv: torch.Tensor,  # dimensions: (batch, freq, time, channels, channels)
    ) -> torch.Tensor:
        """
        Args:
            Rn (torch.Tensor): dimensions: (batch, freq, time, channels, channels)
            Ry (torch.Tensor): dimensions: (batch, freq, time, channels, channels)
            G (torch.Tensor): dimensions: (batch, freq, time, channels, num_sources)
            Rv (torch.Tensor): dimensions: (batch, freq, time, channels, channels)

        Returns:
            torch.Tensor: estimated RTF vectors with dimensions (batch, freq, time, channels, 1)
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration dictionary.
        """
        pass
