import torch
from typing import Any
from .base_channel_combinator import BaseChannelCombinator


class IdentityChannelCombinator(BaseChannelCombinator):
    """
    A 'Do Nothing' channel combinator.
    
    This is useful for features that inherently combine channels during
    extraction (e.g. GMSC, wGMSC). It passes the input tensor through unchanged.
    Assumes the input already has a fixed channel size.
    """

    def __init__(self):
        super().__init__()

    @property
    def is_trainable(self) -> bool:
        return False

    def get_config(self) -> dict[str, Any]:
        return {"name": self.__class__.__name__}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns x unchanged.
        Shape: (B, C, F, T) -> (B, C, F, T)
        """
        return x


class AverageChannelCombinator(BaseChannelCombinator):
    """
    Averages across the channel dimension.
    
    Output will have exactly 1 channel.
    """

    def __init__(self):
        super().__init__()

    @property
    def is_trainable(self) -> bool:
        return False

    def get_config(self) -> dict[str, Any]:
        return {"name": self.__class__.__name__}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Averages across the channel dimension (dim=1) while keeping the dimension.
        Shape: (B, C, F, T) -> (B, 1, F, T)
        """
        return x.mean(dim=1, keepdim=True)


class SelectChannelCombinator(BaseChannelCombinator):
    """
    Selects a specific reference channel.
    
    Output will have exactly 1 channel.
    """

    def __init__(self, ref_channel: int = 0):
        super().__init__()
        self.ref_channel = ref_channel

    @property
    def is_trainable(self) -> bool:
        return False

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "ref_channel": self.ref_channel,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selects the specified reference channel.
        Shape: (B, C, F, T) -> (B, 1, F, T)
        """
        # slice to keep the channel dimension
        return x[:, self.ref_channel : self.ref_channel + 1, :, :]
