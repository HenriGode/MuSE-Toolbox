import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class cLN(nn.Module):
    """
    Cumulative Layer Normalization.
    """

    def __init__(self, dimension: int, eps: float = 1e-8, trainable: bool = True) -> None:
        """
        Initializes the cLN module.

        Args:
            dimension (int): The feature dimension.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.
            trainable (bool, optional): Whether gain and bias are trainable. Defaults to True.
        """
        super().__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cumulative Layer Normalization.

        Args:
            input (torch.Tensor): Input tensor of shape (Batch, Freq, Time).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt_tensor = torch.from_numpy(entry_cnt).type_as(input)
        entry_cnt_tensor = entry_cnt_tensor.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt_tensor  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt_tensor + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x) + self.bias.expand_as(x)


def repackage_hidden(h: torch.Tensor | tuple) -> torch.Tensor | tuple:
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    Args:
        h: The hidden state(s).

    Returns:
        Detached hidden state(s).
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class MultiRNN(nn.Module):
    """
    Container module for multiple stacked RNN layers.

    Args:
        rnn_type (str): Select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): Dimension of the input feature. The input should have shape
                          (batch, seq_len, input_size).
        hidden_size (int): Dimension of the hidden state. The corresponding output should
                           have shape (batch, seq_len, hidden_size).
        dropout (float, optional): Dropout probability. Defaults to 0.
        num_layers (int, optional): Number of stacked RNN layers. Defaults to 1.
        bidirectional (bool, optional): Whether the RNN layers are bidirectional. Defaults to False.
    """

    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(
        self, input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the MultiRNN.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            Tuple: The output tensor and the hidden states.
        """
        hidden = self.init_hidden(input.size(0))
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(
        self, batch_size: int
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden states for the RNN.

        Args:
            batch_size (int): The batch size.

        Returns:
            Hidden states, either a Tensor or a tuple of Tensors (for LSTM).
        """
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(
                    self.num_layers * self.num_direction,
                    batch_size,
                    self.hidden_size,
                ),
                weight.new_zeros(
                    self.num_layers * self.num_direction,
                    batch_size,
                    self.hidden_size,
                ),
            )
        else:
            return weight.new_zeros(
                self.num_layers * self.num_direction, batch_size, self.hidden_size
            )


class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.

    Args:
        input_size (int): Dimension of the input feature. The input should have shape
                          (batch, input_size).
        hidden_size (int): Dimension of the output. The corresponding output should
                           have shape (batch, hidden_size).
        bias (bool, optional): Whether to use bias. Defaults to True.
        nonlinearity (str, optional): The nonlinearity applied to the transformation. Defaults to None.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str | None = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None

        self.init_hidden()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FCLayer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)

    def init_hidden(self) -> None:
        """
        Initializes the fully-connected layer parameters.
        """
        initrange = 1.0 / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)


class DepthConv1d(nn.Module):
    """
    1D depthwise separable convolution block.
    """

    def __init__(
        self,
        input_channel: int,
        hidden_channel: int,
        kernel: int,
        padding: int,
        dilation: int = 1,
        skip: bool = True,
        causal: bool = False,
    ) -> None:
        """
        Initializes the DepthConv1d module.

        Args:
            input_channel (int): Number of input channels.
            hidden_channel (int): Number of hidden channels.
            kernel (int): Kernel size.
            padding (int): Padding size.
            dilation (int, optional): Dilation size. Defaults to 1.
            skip (bool, optional): Whether to use a skip connection. Defaults to True.
            causal (bool, optional): Whether to use causal convolution. Defaults to False.
        """
        super().__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the depthwise convolution.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            If skip is True, returns a tuple of (residual, skip).
            Else, returns only residual.
        """
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(
                self.nonlinearity2(self.dconv1d(output)[:, :, : -self.padding])
            )
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) module.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        BN_dim: int,
        hidden_dim: int,
        layer: int,
        stack: int,
        kernel: int = 3,
        skip: bool = True,
        causal: bool = False,
        dilated: bool = True,
    ) -> None:
        """
        Initializes the TCN module.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            BN_dim (int): Bottleneck dimension.
            hidden_dim (int): Hidden dimension for convolutional layers.
            layer (int): Number of layers per stack.
            stack (int): Number of stacks.
            kernel (int, optional): Kernel size. Defaults to 3.
            skip (bool, optional): Whether to use skip connections. Defaults to True.
            causal (bool, optional): Whether to use causal convolution. Defaults to False.
            dilated (bool, optional): Whether to use dilated convolution. Defaults to True.
        """
        super().__init__()

        # input is a sequence of features of shape (B, N, L)

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=2**i,
                            padding=2**i,
                            skip=skip,
                            causal=causal,
                        )
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=1,
                            padding=1,
                            skip=skip,
                            causal=causal,
                        )
                    )
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += kernel - 1

        log.info("Receptive field: %3d frames.", self.receptive_field)

        # output layer

        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        self.skip = skip

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TCN module.

        Args:
            input (torch.Tensor): Input tensor of shape (B, N, L).

        Returns:
            torch.Tensor: Output tensor.
        """
        # input shape: (B, N, L)

        # normalization
        output = self.BN(self.LN(input))

        # pass to TCN
        if self.skip:
            skip_connection = 0.0
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output
