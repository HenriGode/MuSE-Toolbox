import torch
import torch.nn as nn
from typing import Any
from .base_channel_combinator import BaseChannelCombinator


class MLPChannelCombinator(BaseChannelCombinator):
    """
    A learnable channel combinator that maps variable input channels
    to a fixed output channel size using a separate small MLP for each 
    possible number of input channels up to `max_channels`.
    """

    def __init__(self, max_channels: int = 28, hidden_dim: int = 16, out_channels: int = 1):
        super().__init__()
        self.max_channels = max_channels
        self.out_channels = out_channels
        
        # Create a dictionary of MLPs for each possible channel dimension
        # The key is the string representation of the number of channels
        self.networks = nn.ModuleDict()
        for c in range(1, max_channels + 1):
            self.networks[str(c)] = nn.Sequential(
                nn.Linear(c, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_channels)
            )

    @property
    def is_trainable(self) -> bool:
        return True

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "max_channels": self.max_channels,
            "hidden_dim": self.networks["2"][0].out_features,
            "out_channels": self.out_channels,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP channel combinator.
        
        Args:
            x (torch.Tensor): Shape (B, C, F, T)
            
        Returns:
            torch.Tensor: Shape (B, C_out, F, T)
        """
        b, c, f, t = x.shape
        
        if str(c) not in self.networks:
            raise ValueError(f"Input channel dimension {c} exceeds max_channels {self.max_channels} "
                             f"for MLPChannelCombinator.")
        
        # To apply a linear layer across channels, we need the channel dimension to be last
        # Shape: (B, C, F, T) -> (B, F, T, C)
        x_permuted = x.permute(0, 2, 3, 1)
        
        # Apply the specific network for this channel size
        # Output shape: (B, F, T, C_out)
        out_permuted = self.networks[str(c)](x_permuted)
        
        # Permute back to standard shape
        # Shape: (B, F, T, C_out) -> (B, C_out, F, T)
        out = out_permuted.permute(0, 3, 1, 2)
        
        return out
