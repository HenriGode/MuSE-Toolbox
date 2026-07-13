import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """
    A wrapper around nn.Conv1d that enforces causality by padding
    on the left side only.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ) -> None:
        """
        Initializes the CausalConv1d block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        """
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # We handle padding manually
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying causal padding followed by 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time).

        Returns:
            torch.Tensor: Output tensor after causal convolution.
        """
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))  # Pad left side only
        return self.conv(x)
