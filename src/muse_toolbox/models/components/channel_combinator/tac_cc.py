import torch
from typing import Any
from .base_channel_combinator import BaseChannelCombinator
from muse_toolbox.models.components.nn_blocks import TACBlock

class TACChannelCombinator(BaseChannelCombinator):
    """
    TAC Channel Combinator using the mathematical logic from the TAC paper.
    
    It first applies a TAC block to process the features and share information 
    permutation-invariantly across all channels. Then, it explicitly performs 
    global average pooling across the channel dimension to reduce the 
    representation to exactly 1 channel.
    """

    def __init__(self, input_feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        # Initialize the TAC block
        # If input_feature_dim is -1, it will lazily initialize its layers 
        # on the first forward pass.
        self.tac_block = TACBlock(input_feature_dim=input_feature_dim, hidden_dim=hidden_dim)

    @property
    def is_trainable(self) -> bool:
        return True

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "input_feature_dim": self.input_feature_dim,
            "hidden_dim": self.hidden_dim,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TAC channel combinator.
        
        Args:
            x (torch.Tensor): Shape (B, C, F, T)
            
        Returns:
            torch.Tensor: Shape (B, 1, F, T)
        """
        # 1. Apply TAC Block (maintains shape (B, C, F, T))
        h = self.tac_block(x)
        
        # 2. Average pooling across the channel dimension (dim=1)
        # Shape: (B, C, F, T) -> (B, 1, F, T)
        out = h.mean(dim=1, keepdim=True)
        
        return out
