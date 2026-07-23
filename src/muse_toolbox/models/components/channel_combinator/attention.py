import math
import torch
import torch.nn as nn
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
    The Keys and Values are derived from the individual channels.
    This naturally outputs a representation of shape (Batch, 1, hidden_dim, F, T)
    which is then projected to C_out.
    """

    def __init__(self, hidden_dim: int = 16, out_channels: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        
        self.output_proj = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    @property
    def is_trainable(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        
        # Global query from mean of channels
        q_input = x.mean(dim=1, keepdim=True)  # (B, 1, F, T)
        
        # Project Q
        q = self.q_proj(q_input).view(b, 1, self.hidden_dim, f * t)
        
        # Project K, V for each channel independently
        x_reshaped = x.view(b * c, 1, f, t)
        k = self.k_proj(x_reshaped).view(b, c, self.hidden_dim, f * t)
        v = self.v_proj(x_reshaped).view(b, c, self.hidden_dim, f * t)
        
        # Permute for attention
        q = q.permute(0, 3, 1, 2)  # (B, F*T, 1, hidden_dim)
        k = k.permute(0, 3, 2, 1)  # (B, F*T, hidden_dim, C)
        
        # Compute attention scores
        # Shape: (B, F*T, 1, C)
        scores = torch.matmul(q, k) / math.sqrt(self.hidden_dim)
        attn = torch.softmax(scores, dim=-1)
        
        v = v.permute(0, 3, 1, 2)  # (B, F*T, C, hidden_dim)
        
        # Apply attention
        # Shape: (B, F*T, 1, hidden_dim)
        attended = torch.matmul(attn, v)
        
        # Reshape and remove the query sequence dimension
        attended = attended.permute(0, 2, 3, 1).view(b, self.hidden_dim, f, t)
        
        # Final projection
        out = self.output_proj(attended) # (B, C_out, F, T)
        
        return out
