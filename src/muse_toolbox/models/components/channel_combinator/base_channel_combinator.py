import logging
from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class BaseChannelCombinator(nn.Module, ABC):
    """
    Abstract base class for all channel combinators in the COSAD framework.

    The Channel Combinator is responsible for transforming a variable number of 
    input channels (e.g., from different microphones or microphone pairs) into a 
    fixed-size output dimension, so the subsequent source count estimator (classifier) 
    can operate on a fixed shape.

    Input tensors to `forward` are strictly expected to have the shape:
        (Batch, Channels, Features, Time)  -> (B, C, F, T)
    where the channel dimension is at dim=1.
    
    The output must have the shape:
        (Batch, C_out, Features, Time)
    where C_out is independent of the input Channels C.
    """

    def __init__(self):
        """
        Initialize the base channel combinator.
        """
        super().__init__()

    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        """
        Indicates whether this channel combinator contains learnable parameters.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration dictionary used to initialize this channel combinator.
        """
        pass

    @abstractmethod
    def forward_(self, x: torch.Tensor, feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the channel combinator.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Features, Time)
            feature_mask (torch.Tensor | None): Valid feature mask (Batch, Channels)

        Returns:
            torch.Tensor: Output tensor of shape (Batch, C_out, Features, Time)
        """

    def forward(self, x: torch.Tensor, feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward dimensionality checker.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Features, Time) or (B, F, T)
            feature_mask (torch.Tensor | None): Valid feature mask (Batch, Channels)

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Channels, Features, Time) or (B, 1, F, T)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.forward_(x, feature_mask=feature_mask)
