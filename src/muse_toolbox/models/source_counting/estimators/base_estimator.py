import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class BaseSourceCountEstimator(nn.Module, ABC):
    """
    Abstract base class for all Source Count Estimators (Detectors) in the COSAD framework.

    This class defines the common interface for all detector implementations.
    Its primary role is to enforce a consistent input/output structure.

    Child classes must implement the `forward` method, which is responsible for
    processing input features and returning a dictionary containing the estimated
    source activity tensor.
    """

    def __init__(self, input_dim: int, transform: STFTtransform, max_sources: int):
        """
        Initializes the base detector.

        Args:
            input_dim (int): The dimension of the input feature vector (J).
            transform (STFTtransform): An STFT transformation object.
            max_sources (int): The maximum number of sources to consider. This
                defines the number of output classes (C = max_sources + 1).
        """
        super().__init__()
        self.input_dim = input_dim
        self.transform = transform
        self.max_sources = max_sources

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes the input features and returns the estimated source activity logits.

        Args:
            features (torch.Tensor): Input features of shape (B, 1, J, T) or (B, J, T), where
                B = batch size, J = feature dimension, T = time frames.
        Returns:
            torch.Tensor: Source activity logits of shape (B, T, C), where
                C = number of classes (max_sources + 1).
        """
        if features.dim() == 4:
            if features.size(1) != 1:
                raise ValueError(
                    f"Source count estimators expect exactly 1 channel, but got {features.size(1)} channels. "
                    "Make sure to use a ChannelCombinator to reduce the channel dimension to 1."
                )
            # Squeeze the channel dimension: (B, 1, F, T) -> (B, F, T)
            features = features.squeeze(1)
            
        return self.forward_tensor(features)



    @abstractmethod
    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to process a single feature tensor.

        Args:
            features (torch.Tensor): Input features (B, J, T).

        Returns:
            torch.Tensor: Source activity logits (B, T, C).
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration dictionary used to initialize this detector.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        pass

    def _verbose_parameters(self, indent: str = "") -> None:
        """
        Logs the parameters of the module in a structured, indented format.
        Child classes should extend this method to include their specific parameters.

        Args:
            indent (str, optional): A string to prepend to each line for indentation.
                                    Defaults to "".
        """
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Max Sources: {self.max_sources}")
