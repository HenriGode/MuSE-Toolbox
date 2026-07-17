import logging
from typing import Any

import torch
import torch.nn as nn

from muse_toolbox.utils import STFTtransform

from .base_estimator import BaseSourceCountEstimator

log = logging.getLogger(__name__)


class GRU_estimator(BaseSourceCountEstimator):
    """
    An RNN-based detector for source activity estimation.

    This model uses a GRU to process coherence features sequentially and predict
    the number of active sources for each time frame. It is fully causal.
    """

    def __init__(
        self,
        input_dim: int,
        transform: STFTtransform,
        max_sources: int,
        hidden_size: int | float,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Initializes the GRU detector.

        Args:
            input_dim (int): The dimension of the input feature vector.
            transform (STFTtransform): An STFT transformation object.
            max_sources (int): The maximum number of sources to consider.
            hidden_size (Union[int, float]): The number of features in the hidden
                state. If a float, it's interpreted as a fraction of the input size.
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            bias (bool, optional): If False, then the layer does not use bias weights.
                Defaults to True.
        """
        super().__init__(input_dim, transform, max_sources)
        self.hidden_size_param = hidden_size  # Store original param for config

        if isinstance(hidden_size, float):
            assert 0 < hidden_size <= 1, "hidden_size as float must be in (0, 1]."
            hidden_size = max(1, int(self.input_dim * hidden_size))

        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
            bias=bias,
            bidirectional=False,  # Unidirectional RNN to guarantee causality
        )
        self.fc = nn.Linear(hidden_size, max_sources + 1)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the feature extractor configuration.

        Returns:
            dict[str, Any]: The config dictionary.
        """
        return {
            "max_sources": self.max_sources,
            "hidden_size": self.hidden_size_param,
            "num_layers": self.rnn.num_layers,
            "dropout": self.rnn.dropout,
            "bias": self.rnn.bias,
            "input_dim": self.input_dim,
        }

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs the specific parameters of this estimator."""
        super()._verbose_parameters(indent)
        log.info(f"{indent}  Hidden Size: {self.hidden_size_param}")
        log.info(f"{indent}  Num Layers: {self.rnn.num_layers}")
        log.info(f"{indent}  Dropout: {self.rnn.dropout}")

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes coherence features through the RNN.
        Args:
            features (torch.Tensor): (B, F, T)
        Returns:
            torch.Tensor: (B, T, C)
        """
        #self.rnn.flatten_parameters()
        # Permute: (B, F, T) -> (B, T, F)
        rnn_out, _ = self.rnn(features.swapaxes(-1, -2))  # (B, F, T) -> (B, T, H)
        logits = self.fc(rnn_out)  # (B, T, H) -> (B, T, C)
        return logits
