import math
import torch
import torch.nn as nn
from typing import Any
from .base_channel_combinator import BaseChannelCombinator


class SelfAttentionChannelCombinator(BaseChannelCombinator):
    """
    Self-Attention Channel Combinator (SACC).
    
    Based on the literature (e.g., "Self-attention mechanism for multichannel ASR"),
    this module treats each channel as a token, calculates time-varying weights 
    for each channel using a self-attention mechanism on the frequency dimension, 
    and applies a weighted sum to reduce the representation to a single channel.
    
    Expected Input Shape: (B, C, F, T)
    Expected Output Shape: (B, 1, F, T)
    """

    def __init__(self, input_feature_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        
        # Linear layers to project from the Feature dimension (F) to Hidden dimension (D)
        # Note: The literature applies these independently of the Channel and Time dimensions.
        self.q_proj = nn.Linear(input_feature_dim, hidden_dim)
        self.k_proj = nn.Linear(input_feature_dim, hidden_dim)
        
        # The value projection contracts the entire frequency dimension down to 1 unit.
        # This ensures the final channel combinator weights are applied homogeneously 
        # across all frequency bins for a given channel at a given time frame.
        self.v_proj = nn.Linear(input_feature_dim, 1)

    @property
    def is_trainable(self) -> bool:
        return True

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "input_feature_dim": self.input_feature_dim,
            "hidden_dim": self.hidden_dim,
        }

    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, F, T)
            
        Returns:
            torch.Tensor: Weighted sum across channels, shape (B, 1, F, T)
        """
        b, c, f, t = x.shape
        
        # Permute to (B, T, C, F) to apply Linear layers over the F dimension
        x_perm = x.permute(0, 3, 1, 2)
        
        # Project Query, Key, Value
        # q, k shape: (B, T, C, D)
        q = self.q_proj(x_perm)
        k = self.k_proj(x_perm)
        
        # v shape: (B, T, C, 1)
        v = self.v_proj(x_perm)
        
        # Compute attention matrix (W^att)
        # q: (B, T, C, D), k^T: (B, T, D, C)
        k_t = k.transpose(-2, -1)
        
        # scores: (B, T, C, C)
        scores = torch.matmul(q, k_t) / math.sqrt(self.hidden_dim)
        
        # W^att: (B, T, C, C), softmax over the last dimension (Key channels)
        w_att = torch.softmax(scores, dim=-1)
        
        # Compute final combinator weights (w)
        # w_att: (B, T, C, C), v: (B, T, C, 1)
        # result: (B, T, C, 1)
        w_unnorm = torch.matmul(w_att, v)
        
        # Softmax over the channel dimension to ensure weights sum to 1
        w = torch.softmax(w_unnorm, dim=2)
        
        # Broadcast weights to all frequencies and apply to original input
        # w shape: (B, T, C, 1) -> (B, T, C, F)
        w_broadcast = w.expand(-1, -1, -1, f)
        
        # Element-wise multiplication with the permuted input (B, T, C, F)
        weighted_x = w_broadcast * x_perm
        
        # Sum across the channel dimension C (dim=2 in the permuted tensor)
        # Shape: (B, T, F)
        s_perm = weighted_x.sum(dim=2)
        
        # Reshape to standard output (B, 1, F, T)
        # s_perm is (B, T, F), we add the channel dim and permute T and F
        out = s_perm.unsqueeze(1).permute(0, 1, 3, 2)
        
        return out


class CrossAttentionChannelCombinator(BaseChannelCombinator):
    """
    Applies Global-to-Local Cross-Attention.
    
    The Query is derived from the mean across all channels (the global context).
    The Keys are derived from the individual channels.
    Attention scores between the global context and each channel determine the 
    weight of that channel in the final sum.
    
    Expected Input Shape: (B, C, F, T)
    Expected Output Shape: (B, 1, F, T)
    """

    def __init__(self, input_feature_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        
        # Project from Feature dimension (F) to Hidden dimension (D)
        self.q_proj = nn.Linear(input_feature_dim, hidden_dim)
        self.k_proj = nn.Linear(input_feature_dim, hidden_dim)

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
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, F, T)
            
        Returns:
            torch.Tensor: Weighted sum across channels, shape (B, 1, F, T)
        """
        b, c, f, t = x.shape
        
        # Permute to (B, T, C, F) to apply Linear layers over the F dimension
        x_perm = x.permute(0, 3, 1, 2)
        
        # Global query from mean of channels
        # Shape: (B, T, 1, F)
        q_input = x_perm.mean(dim=2, keepdim=True)
        
        # Project Query and Keys
        q = self.q_proj(q_input)      # (B, T, 1, D)
        k = self.k_proj(x_perm)       # (B, T, C, D)
        
        # Compute attention scores
        # q: (B, T, 1, D), k^T: (B, T, D, C)
        k_t = k.transpose(-2, -1)
        
        # scores: (B, T, 1, C)
        scores = torch.matmul(q, k_t) / math.sqrt(self.hidden_dim)
        
        # w_att: (B, T, 1, C) - softmax over the Keys (channels)
        w_att = torch.softmax(scores, dim=-1)
        
        # Transpose weights to align with channels: (B, T, C, 1)
        w = w_att.transpose(-2, -1)
        
        # Broadcast weights to all frequencies: (B, T, C, F)
        w_broadcast = w.expand(-1, -1, -1, f)
        
        # Element-wise multiplication with the permuted input (B, T, C, F)
        weighted_x = w_broadcast * x_perm
        
        # Sum across the channel dimension C (dim=2 in the permuted tensor)
        # Shape: (B, T, F)
        s_perm = weighted_x.sum(dim=2)
        
        # Reshape to standard output (B, 1, F, T)
        out = s_perm.unsqueeze(1).permute(0, 1, 3, 2)
        
        return out
