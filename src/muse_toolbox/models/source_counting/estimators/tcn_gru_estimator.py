import torch
import torch.nn as nn
import logging
from typing import Any, Union

from .base_estimator import BaseSourceCountEstimator
from muse_toolbox.models.components.nn_blocks.conv_tasnet import TCN
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class TCN_GRU_estimator(BaseSourceCountEstimator):
    """
    A hybrid TCN-RNN detector for source activity estimation.

    This model uses a TCN front-end to extract robust local features from the
    coherence input, followed by a GRU back-end to model longer-term temporal
    dependencies. It remains fully causal.
    """

    def __init__(
        self,
        input_dim: int,
        transform: STFTtransform,
        max_sources: int,
        # TCN parameters
        tcn_BN_dim: int,
        tcn_hidden_dim: int,
        tcn_layer: int,
        tcn_stack: int,
        tcn_kernel: int,
        # GRU parameters
        gru_hidden_size: Union[int, float],
        gru_num_layers: int = 1,
        # Common parameters
        skip_connection: bool = False,
    ):
        """
        Initializes the Hybrid TCN-GRU detector.

        Args:
            input_dim (int): The dimension of the input feature vector.
            transform (STFTtransform): An STFT transformation object.
            max_sources (int): The maximum number of sources to consider.
            tcn_BN_dim (int): Bottleneck channels in the TCN.
            tcn_hidden_dim (int): Hidden channels in the TCN.
            tcn_layer (int): Number of layers in each TCN stack.
            tcn_stack (int): Number of TCN stacks.
            tcn_kernel (int, optional): TCN convolution kernel size. Defaults to 3.
            gru_hidden_size (Union[int, float]): The number of features in the GRU's hidden state.
            gru_num_layers (int, optional): Number of recurrent GRU layers. Defaults to 1.
            skip_connection (bool, optional): If True, adds a skip connection from input to GRU.
        """
        super().__init__(input_dim, transform, max_sources)
        # Store params
        self.skip_connection = skip_connection
        self.tcn_BN_dim = tcn_BN_dim
        self.tcn_hidden_dim = tcn_hidden_dim
        self.tcn_layer = tcn_layer
        self.tcn_stack = tcn_stack
        self.tcn_kernel = tcn_kernel
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        if skip_connection:
            # When using skip connection, the TCN output dimension matches the input dimension,
            # and the RNN input dimension is the sum of TCN output and input dimensions.
            tcn_output_dim = input_dim
            rnn_input_dim = tcn_output_dim + input_dim
        else:
            # Without skip connection, the TCN output dimension also matches the input dimension,
            # and the RNN input dimension matches the TCN output dimension.
            tcn_output_dim = input_dim
            rnn_input_dim = tcn_output_dim

        # 1. TCN Front-end
        self.tcn = TCN(
            input_dim=input_dim,
            output_dim=tcn_output_dim,
            BN_dim=tcn_BN_dim,
            hidden_dim=tcn_hidden_dim,
            layer=tcn_layer,
            stack=tcn_stack,
            kernel=tcn_kernel,
            skip=True,
            causal=True,
            dilated=True,
        )

        # Calculate the integer hidden size for the GRU, allowing for float input
        if isinstance(gru_hidden_size, float):
            assert (
                0 < gru_hidden_size <= 1
            ), "gru_hidden_size as float must be in (0, 1]."
            int_gru_hidden_size = max(1, int(rnn_input_dim * gru_hidden_size))
        else:
            int_gru_hidden_size = gru_hidden_size

        # 2. GRU Back-end
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=int_gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # 3. Final Output Layer
        self.fc = nn.Linear(int_gru_hidden_size, max_sources + 1)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the feature extractor configuration.

        Returns:
            dict[str, Any]: The config dictionary.
        """
        return {
            "max_sources": self.max_sources,
            "tcn_BN_dim": self.tcn_BN_dim,
            "tcn_hidden_dim": self.tcn_hidden_dim,
            "tcn_layer": self.tcn_layer,
            "tcn_stack": self.tcn_stack,
            "tcn_kernel": self.tcn_kernel,
            "gru_hidden_size": self.gru_hidden_size,
            "gru_num_layers": self.gru_num_layers,
            "skip_connection": self.skip_connection,
            "input_dim": self.input_dim,
        }

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs the specific parameters of this estimator."""
        super()._verbose_parameters(indent)
        log.info(f"{indent}  TCN BN Dim: {self.tcn_BN_dim}")
        log.info(f"{indent}  TCN Hidden Dim: {self.tcn_hidden_dim}")
        log.info(f"{indent}  TCN Layers/Stack: {self.tcn_layer}")
        log.info(f"{indent}  TCN Stacks: {self.tcn_stack}")
        log.info(f"{indent}  GRU Hidden Size: {self.gru_hidden_size}")
        log.info(f"{indent}  GRU Num Layers: {self.gru_num_layers}")
        log.info(f"{indent}  Skip Connection: {self.skip_connection}")

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes features through TCN then GRU.
        Args:
            features (torch.Tensor): (B, F, T)
        Returns:
            torch.Tensor: (B, T, C)
        """
        # --- Stage 1: TCN ---
        tcn_out = self.tcn(features)  # (B, F_tcn, T)

        # --- Stage 2: Prepare RNN Input ---
        tcn_out_permuted = tcn_out.permute(0, 2, 1)  # (B, T, F_tcn)

        if self.skip_connection:
            raw_input_permuted = features.permute(0, 2, 1)  # (B, T, F_raw)
            rnn_input = torch.cat([raw_input_permuted, tcn_out_permuted], dim=-1)
        else:
            rnn_input = tcn_out_permuted

        # --- Stage 3: GRU ---
        rnn_out, _ = self.rnn(rnn_input)  # (B, T, H_gru)

        # --- Stage 4: Output ---
        logits = self.fc(rnn_out)  # (B, T, C)
        return logits
