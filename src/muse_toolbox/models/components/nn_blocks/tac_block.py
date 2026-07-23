import torch
import torch.nn as nn


class TACBlock(nn.Module):
    """
    Transform-Average-Concatenate (TAC) Block.
    
    Proposed in:
    Yi Luo, Zhuo Chen, Nima Mesgarani, and Takuya Yoshioka, "End-to-End Microphone Permutation 
    and Number Invariant Multi-Channel Speech Separation," in 2020 IEEE International Conference 
    on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 6394–6398.
    
    This module is highly lightweight and guarantees that the system is invariant to 
    both the number of microphones and their permutation. It outputs a tensor of the 
    exact same dimensions as the input.
    
    Expected Input Shape: (Batch, Channels, Features, Time)
    Expected Output Shape: (Batch, Channels, Features, Time)
    """

    def __init__(self, input_feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        
        # 1. Transform: P(.)
        self.transform = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.PReLU()
        )
        
        # 2. Average (Global Context): R(.)
        self.global_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU()
        )
        
        # 3. Concatenate (Fusion): S(.)
        # Takes the concatenated [local_features, global_features]
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_feature_dim),
            nn.PReLU()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TAC Block.
        
        Args:
            z (torch.Tensor): Input tensor of shape (B, C, F, T)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, F, T)
        """
        # Linear layers operate on the last dimension, so we must permute
        # (B, C, F, T) -> (B, C, T, F)
        z_perm = z.permute(0, 1, 3, 2)
        
        # 1. Transform (Independent Channel Processing)
        # f: (B, C, T, D)
        f = self.transform(z_perm)
        
        # 2. Average (Global Context Pooling)
        # mean over channels C (dim=1)
        # f_mean: (B, 1, T, D)
        f_mean = f.mean(dim=1, keepdim=True)
        # f_hat: (B, 1, T, D)
        f_hat = self.global_transform(f_mean)
        
        # Broadcast the global context to match local features
        # f_hat_broadcasted: (B, C, T, D)
        f_hat_broadcasted = f_hat.expand(-1, f.size(1), -1, -1)
        
        # 3. Concatenate
        # concatenated: (B, C, T, 2D)
        concatenated = torch.cat([f, f_hat_broadcasted], dim=-1)
        
        # g: (B, C, T, F)
        g = self.fusion(concatenated)
        
        # 4. Residual Connection
        z_hat_perm = z_perm + g
        
        # Permute back to original shape: (B, C, T, F) -> (B, C, F, T)
        z_hat = z_hat_perm.permute(0, 1, 3, 2)
        
        return z_hat
