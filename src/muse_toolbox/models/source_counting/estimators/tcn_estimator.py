import logging
from typing import Any

import torch

from muse_toolbox.models.components.nn_blocks.conv_tasnet import TCN
from muse_toolbox.utils import STFTtransform

from .base_estimator import BaseSourceCountEstimator

log = logging.getLogger(__name__)


class TCN_estimator(BaseSourceCountEstimator):
    """
    A wrapper for the official TCN model from the Conv-TasNet repository.

    This class adapts the TCN model to the COSAD framework for causal source
    activity detection. It takes coherence features as input and produces
    frame-wise source count logits.
    """

    def __init__(
        self,
        input_dim: int,
        transform: STFTtransform,
        max_sources: int,
        BN_dim: int,
        hidden_dim: int,
        layer: int,
        stack: int,
        kernel: int = 3,
        skip: bool = True,
        dilated: bool = True,
    ):
        """
        Initializes the TCN detector wrapper.

        Args:
            input_dim (int): The dimension of the input feature vector.
            transform (STFTtransform): An STFT transformation object.
            max_sources (int): The maximum number of sources to consider (C-1).
            BN_dim (int): The number of channels in the bottleneck 1x1 conv layer.
            hidden_dim (int): The number of channels in the hidden layers.
            layer (int): The number of convolutional blocks in each stack.
            stack (int): The number of TCN stacks.
            kernel (int, optional): The kernel size in the TCN blocks. Defaults to 3.
            skip (bool, optional): Whether to use skip connections. Defaults to True.
            dilated (bool, optional): Whether to use dilated convolutions. Defaults to True.
        """
        super().__init__(input_dim, transform, max_sources)
        # Store params
        self.BN_dim = BN_dim
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.skip = skip
        self.dilated = dilated

        output_dim = max_sources + 1

        self.tcn = TCN(
            input_dim=input_dim,
            output_dim=output_dim,
            BN_dim=BN_dim,
            hidden_dim=hidden_dim,
            layer=layer,
            stack=stack,
            kernel=kernel,
            skip=skip,
            causal=True,  # Enforce causality as required
            dilated=dilated,
        )

    def get_config(self) -> dict[str, Any]:
        """
        Returns the feature extractor configuration.

        Returns:
            dict[str, Any]: The config dictionary.
        """
        return {
            "max_sources": self.max_sources,
            "BN_dim": self.BN_dim,
            "hidden_dim": self.hidden_dim,
            "layer": self.layer,
            "stack": self.stack,
            "kernel": self.kernel,
            "skip": self.skip,
            "dilated": self.dilated,
            "input_dim": self.input_dim,
        }

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs the specific parameters of this estimator."""
        super()._verbose_parameters(indent)
        log.info(f"{indent}  BN Dim: {self.BN_dim}")
        log.info(f"{indent}  Hidden Dim: {self.hidden_dim}")
        log.info(f"{indent}  Layers per Stack: {self.layer}")
        log.info(f"{indent}  Stacks: {self.stack}")
        log.info(f"{indent}  Kernel Size: {self.kernel}")

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes features through TCN.
        Args:
            features (torch.Tensor): (B, F, T)
        Returns:
            torch.Tensor: (B, T, C)
        """
        # TCN expects (B, N, L) -> (B, F, T) matches
        logits = self.tcn(features)  # (B, C, T)
        return logits.permute(0, 2, 1)  # (B, T, C)
