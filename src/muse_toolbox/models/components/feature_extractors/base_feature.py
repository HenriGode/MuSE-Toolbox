import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class BaseFeatureExtractor(nn.Module, ABC):
    """
    Abstract base class for all feature extractors in the COSAD framework.

    This class enforces a consistent interface where inputs and outputs are always
    dictionaries. It handles the dispatch between processing raw audio scenarios
    and passing through pre-computed features based on the 'input_type' key.
    """

    def __init__(self, transform: STFTtransform | None = None):
        """
        Initialize the base feature extractor.
        """
        super().__init__()
        self.transform = transform

    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        """
        Indicates whether this feature extractor contains learnable parameters.
        """
        pass

    @property
    def uses_stft(self) -> bool:
        """
        Indicates whether this feature extractor requires STFT input
        by determining whether self.transform is of type STFTtransform.
        """
        return isinstance(self.transform, STFTtransform)

    @property
    @abstractmethod
    def signature(self) -> str:
        """
        Provides a unique signature describing the features produced by this extractor.

        Returns:
            str: A str containing feature name and config parameters in a condensed form.
        """
        pass

    @property
    def full_signature(self) -> str:
        """
        Returns the complete, unhashed signature.
        Defaults to self.signature if not overridden.
        """
        return self.signature

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """
        Returns the dimension (J) of the feature vector at each time step.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration dictionary used to initialize this feature extractor.

        Returns:
            dict[str, Any]: A dictionary containing configuration parameters.
        """
        pass

    @property
    def precompute_type(self) -> str | None:
        """
        Returns the type of data produced by the precompute method.
        If precomputation is not supported, returns None.
        """
        if self.is_trainable:
            if self.uses_stft:
                return "stft"
            return None
        return "features"

    def precompute(
        self, batch: torch.Tensor, input_type: str = "raw_audio"
    ) -> dict[str, torch.Tensor]:
        """
        Method to precompute features for a given input tensor.
        Must be overridden if the feature extractor is trainable.
        """
        precomputedict = defaultdict(torch.Tensor)

        if (
            self.uses_stft
            and isinstance(self.transform, STFTtransform)
            and input_type == "raw_audio"
        ):
            precomputedict["stft"] = self.transform.encode(batch)
            batch = precomputedict["stft"]
            input_type = "stft"

        if not self.is_trainable:
            precomputedict["features"] = self.forward(batch, input_type=input_type)

        batch = None  # Free memory

        return precomputedict

    def forward(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        input_type: str = "raw_audio",
        valid_mics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Dispatch method to process input based on its type.
        """
        is_unbatched = False
        if isinstance(batch, torch.Tensor):
            if input_type == "raw_audio" and batch.dim() == 2:
                is_unbatched = True
            elif input_type == "stft" and batch.dim() == 3:
                is_unbatched = True
            elif input_type == "features" and batch.dim() == 2:
                is_unbatched = True

            if is_unbatched:
                batch = batch.unsqueeze(0)

        if input_type == "raw_audio" and isinstance(batch, torch.Tensor):
            out = self.forward_raw_audio(batch, valid_mics=valid_mics)
        elif (
            input_type == "stft" and self.uses_stft and isinstance(batch, torch.Tensor)
        ):
            # STFT natively has shape (..., F, M, T). Transpose to (..., M, F, T).
            batch = batch.transpose(-2, -3)
            out = self.forward_stft(batch, valid_mics=valid_mics)
        elif input_type == "features" and isinstance(batch, torch.Tensor):
            out = self.forward_precomputed_features(batch, valid_mics=valid_mics)
        elif input_type == "features" and isinstance(batch, dict):
            out = self.forward_precomputed_features_dict(batch, valid_mics=valid_mics)
        else:
            raise ValueError(
                f"Invalid input_type '{input_type}' for feature extractor"
                f" '{self.__class__.__name__}'."
            )
            
        if is_unbatched and isinstance(out, torch.Tensor):
            out = out.squeeze(0)
            
        return out

    def forward_raw_audio(self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None) -> torch.Tensor:
        """Compute features from raw audio tensor (B, M, N)."""
        if isinstance(batch, dict):
            raise NotImplementedError(
                "BaseFeatureExtractor does not support dictionary input for raw audio by default."
            )
        if isinstance(self.transform, STFTtransform) and self.uses_stft:
            stft_audio = self.transform.encode(batch)
            # STFT natively has shape (..., F, M, T). Transpose to (..., M, F, T).
            stft_audio = stft_audio.transpose(-2, -3)
            return self.forward_stft(stft_audio, valid_mics=valid_mics)
        else:
            raise NotImplementedError(
                "forward_raw_audio method is not implemented for feature extractors that do not use STFT."
            )
        pass

    def forward_stft(self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None) -> torch.Tensor:
        """Compute features from STFT tensor (B, M, F, T)."""
        if self.uses_stft:
            raise NotImplementedError(
                f"forward_stft method must be implemented for feature extractors that use STFT but is not for {self.__class__.__name__}."
            )
        raise NotImplementedError(
            "forward_stft method is not implemented for this feature extractor since it does not use STFT."
        )

    def forward_precomputed_features(self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None) -> torch.Tensor:
        """
        Pass-through features tensor (B, J, T).
        Can be overridden by trainable feature extractors to apply the trainable part.
        """
        if isinstance(batch, dict):
            raise NotImplementedError(
                "BaseFeatureExtractor does not support dictionary input for precomputed features by default."
            )
        return batch

    def forward_precomputed_features_dict(
        self, batch: dict[str, torch.Tensor], valid_mics: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError(
            "forward_precomputed_features_dict method must be implemented for feature extractors that accept dictionary input for precomputed features."
        )

    def get_valid_feature_mask(self, valid_mics_count: torch.Tensor | None, max_M: int) -> torch.Tensor | None:
        """
        Returns a boolean mask of shape (B, C_out) indicating which output features are valid.
        
        Args:
            valid_mics_count (torch.Tensor | None): Tensor of shape (B,) with the number of valid mics per batch element.
            max_M (int): The maximum number of microphones in the batch (i.e. the channel dimension).
            
        Returns:
            torch.Tensor | None: A boolean mask of shape (B, C_out) where True means the feature channel is valid.
        """
        if valid_mics_count is None:
            return None
            
        device = valid_mics_count.device
        # By default, output features map 1:1 to microphones
        mask = torch.arange(max_M, device=device).expand(
            valid_mics_count.shape[0], max_M
        ) < valid_mics_count.unsqueeze(1)
        return mask

    def _verbose_parameters(self, indent: str = "") -> None:
        """
        Logs the parameters of the feature extractor.
        """
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Trainable: {self.is_trainable}")
