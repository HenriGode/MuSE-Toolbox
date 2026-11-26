import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class BaseSourceCountEstimator(nn.Module, ABC):
    """
    Abstract base class for all Source Count Estimators (Detectors) in the OSC framework.

    This class defines the common interface for all detector implementations.
    Its primary role is to enforce a consistent input/output structure.

    Child classes must implement the `forward` method, which is responsible for
    processing input features and returning a dictionary containing the estimated
    source activity tensor.
    """

    def __init__(
        self,
        input_dim: int,
        transform: STFTtransform,
        max_sources: int,
        num_input_channels: int = 1,
    ) -> None:
        """Initializes the base source count estimator.

        Args:
            input_dim (int): Per-channel feature dimension (``F``).
            transform (STFTtransform): An STFT transformation object.
            max_sources (int): Maximum number of simultaneous active sources.
                Defines the number of output classes (``max_sources + 1``).
            num_input_channels (int, optional): Expected input channel count ``C``
                when flattening ``(B, C, F, T) -> (B, C * F, T)`` at the input
                pre-stage. Defaults to ``1``.
        """
        super().__init__()
        if num_input_channels < 1:
            raise ValueError(
                f"num_input_channels must be >= 1, got {num_input_channels}."
            )
        self.input_dim = input_dim
        self.num_input_channels = int(num_input_channels)
        self.flat_input_dim = self.input_dim * self.num_input_channels
        self.transform = transform
        self.max_sources = max_sources

    @staticmethod
    def _flatten_channel_features(
        tensor: torch.Tensor,
        expected_channels: int,
        stream_name: str = "features",
    ) -> torch.Tensor:
        """Flattens a 4D feature tensor ``(B, C, F, T)`` into ``(B, C * F, T)``.

        Args:
            tensor (torch.Tensor): Input feature tensor of shape ``(B, C, F, T)``
                or ``(B, F_flat, T)``.
            expected_channels (int): Expected channel count ``C`` when 4D input is
                provided.
            stream_name (str, optional): Stream label for error diagnostics.
                Defaults to ``"features"``.

        Returns:
            torch.Tensor: Flattened 3D feature tensor of shape ``(B, C * F, T)``.

        Raises:
            ValueError: If a 4D tensor's channel dimension does not match
                ``expected_channels``.
        """
        if tensor.dim() == 4:
            batch_size, num_channels, feat_dim, num_frames = tensor.shape
            if num_channels != expected_channels:
                raise ValueError(
                    f"{stream_name} has {num_channels} channels, but this fixed-width "
                    f"estimator expects {expected_channels} channels (check `num_mics`)."
                )
            return tensor.reshape(batch_size, num_channels * feat_dim, num_frames)
        return tensor

    def forward(
        self,
        features: torch.Tensor,
        cond_features: torch.Tensor | None = None,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Processes input features and returns estimated source activity logits.

        Args:
            features (torch.Tensor): Input features of shape ``(B, C, F, T)`` or
                ``(B, C * F, T)``.
            cond_features (torch.Tensor | None, optional): Optional conditioning
                feature tensor for multi-stream fusion estimators. Defaults to ``None``.
            valid_frames (torch.Tensor | None, optional): Optional valid frame counts
                of shape ``(B,)`` used to mask right-padded frames in variable-length
                batches. Defaults to ``None``.

        Returns:
            torch.Tensor: Source activity logits of shape ``(B, T, max_sources + 1)``.

        Raises:
            ValueError: If ``cond_features`` is passed to a single-stream estimator.
        """
        if cond_features is not None:
            raise ValueError(
                f"{self.__class__.__name__} is a single-stream estimator and does not "
                "accept `cond_features`. Use FusionTransformerEstimator for dual-stream fusion."
            )
        _ = valid_frames
        flat_features = self._flatten_channel_features(
            features, self.num_input_channels, stream_name="features"
        )
        return self.forward_tensor(flat_features)

    @abstractmethod
    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """Abstract method to process a flattened single-stream feature tensor.

        Args:
            features (torch.Tensor): Input features of shape ``(B, C * F, T)``.

        Returns:
            torch.Tensor: Source activity logits of shape ``(B, T, max_sources + 1)``.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration dictionary used to initialize this detector.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        pass

    def _verbose_parameters(self, indent: str = "") -> None:
        """
        Logs the parameters of the module in a structured, indented format.
        Child classes should extend this method to include their specific parameters.

        Args:
            indent (str, optional): A string to prepend to each line for indentation.
                                    Defaults to "".
        """
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Max Sources: {self.max_sources}")
