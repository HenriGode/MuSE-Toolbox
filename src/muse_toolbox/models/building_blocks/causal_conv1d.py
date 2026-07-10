import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    A wrapper around nn.Conv1d that enforces causality by padding
    on the left side only.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # We handle padding manually
            dilation=dilation,
        )

    def forward(self, x):
        # x: (B, C, T)
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))  # Pad left side only
        return self.conv(x)
