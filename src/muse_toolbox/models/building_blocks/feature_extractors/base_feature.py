from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
from utilities.sigproc4torch import STFTtransform


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
    def get_config(self) -> dict:
        """
        Returns the configuration dictionary used to initialize this feature extractor.
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
    ) -> torch.Tensor:
        """
        Dispatch method to process input based on its type.
        """
        if input_type == "raw_audio" and isinstance(batch, torch.Tensor):
            return self.forward_raw_audio(batch)
        elif (
            input_type == "stft" and self.uses_stft and isinstance(batch, torch.Tensor)
        ):
            return self.forward_stft(batch)
        elif input_type == "features" and isinstance(batch, torch.Tensor):
            return self.forward_precomputed_features(batch)
        elif input_type == "features" and isinstance(batch, dict):
            return self.forward_precomputed_features_dict(batch)
        else:
            raise ValueError(
                f"Invalid input_type '{input_type}' for feature extractor"
                f" '{self.__class__.__name__}'."
            )

    def forward_raw_audio(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute features from raw audio tensor (B, M, N)."""
        if isinstance(batch, dict):
            raise NotImplementedError(
                "BaseFeatureExtractor does not support dictionary input for raw audio by default."
            )
        if isinstance(self.transform, STFTtransform):
            stft_audio = self.transform.encode(batch)
            return self.forward_stft(stft_audio)
        else:
            raise NotImplementedError(
                "forward_raw_audio method is not implemented for feature extractors that do not use STFT."
            )
        pass

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute features from STFT tensor (B, M, F, T)."""
        if self.uses_stft:
            raise NotImplementedError(
                f"forward_stft method must be implemented for feature extractors that use STFT but is not for {self.__class__.__name__}."
            )
        raise NotImplementedError(
            "forward_stft method is not implemented for this feature extractor since it does not use STFT."
        )

    def forward_precomputed_features(self, batch: torch.Tensor) -> torch.Tensor:
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
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError(
            "forward_precomputed_features_dict method must be implemented for feature extractors that accept dictionary input for precomputed features."
        )

    def _verbose_parameters(self, indent: str = "") -> None:
        print(f"{indent}{self.__class__.__name__} Parameters:")
        print(f"{indent}  Trainable: {self.is_trainable}")
