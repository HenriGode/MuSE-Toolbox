import hashlib
import logging
from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn as nn

from muse_toolbox.utils import STFTtransform

from .base_feature import BaseFeatureExtractor

log = logging.getLogger(__name__)


class StackedFeatureExtractor(BaseFeatureExtractor):
    """
    A meta-feature extractor that combines multiple feature extractors by stacking their outputs.

    It supports:
    1. Heterogeneous sub-extractors (e.g., one using STFT, one using raw audio).
    2. Mixed precomputation strategies (e.g., one fixed, one trainable).
    3. Efficient storage (combines precomputed outputs into a single dictionary).
    """

    def __init__(self, transform: STFTtransform, extractors: list[dict[str, Any]]) -> None:
        """
        Initializes the StackedFeatureExtractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            extractors (list[dict[str, Any]]): A list of dictionaries, where each 
                dictionary contains a single key (the class name of the feature extractor) 
                and its corresponding parameters as the value.
        """
        super().__init__(transform=transform)

        # Import here to avoid circular dependency issues at module level
        import muse_toolbox.models.components.feature_extractors as fe

        instantiated_extractors = []
        for item in extractors:
            # item is expected to be {ClassName: {params}}
            if isinstance(item, dict) and len(item) == 1:
                class_name = list(item.keys())[0]
                params = list(item.values())[0]
                if params is None:
                    params = {}

                if hasattr(fe, class_name):
                    cls = getattr(fe, class_name)
                    # Instantiate with transform and params
                    instantiated_extractors.append(cls(transform=transform, **params))
                else:
                    raise ValueError(f"Unknown feature extractor class: {class_name}")
            else:
                raise ValueError(
                    f"Invalid extractor config format: {item}. Expected single-key dict."
                )

        self.extractors = nn.ModuleList(instantiated_extractors)

    @property
    def _extractors(self) -> Iterable[BaseFeatureExtractor]:
        """Helper to satisfy type checker when iterating over ModuleList."""
        return cast(Iterable[BaseFeatureExtractor], self.extractors)

    @property
    def is_trainable(self) -> bool:
        return any(e.is_trainable for e in self._extractors)

    @property
    def uses_stft(self) -> bool:
        # If any extractor uses STFT, we might say this uses STFT,
        # but really it depends on what the sub-extractors need.
        # This property is mostly used to determine if 'stft' input is valid.
        return any(e.uses_stft for e in self._extractors)

    @property
    def full_signature(self) -> str:
        return "+".join([e.signature for e in self._extractors])

    @property
    def signature(self) -> str:
        full_sig = self.full_signature
        # Linux filename limit is 255 bytes. We use a safe margin.
        if len(full_sig) > 200:
            hash_object = hashlib.md5(full_sig.encode())
            return f"Stacked_{hash_object.hexdigest()}"
        return full_sig

    @property
    def feature_dim(self) -> int:
        return sum(e.feature_dim for e in self._extractors)

    def get_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "extractors": [e.get_config() for e in self._extractors],
        }

    @property
    def precompute_type(self) -> str | None:
        return "features"

    def precompute(
        self, batch: torch.Tensor, input_type: str = "raw_audio"
    ) -> dict[str, torch.Tensor]:
        """
        Precomputes features for all sub-extractors and merges them into a single dictionary.
        Keys are prefixed with the index of the extractor (e.g., "0_features", "1_stft").
        """
        combined_output = {}
        for i, extractor in enumerate(self._extractors):
            out = extractor.precompute(batch, input_type=input_type)
            for k, v in out.items():
                combined_output[f"{i}_{k}"] = v
        return combined_output

    def forward(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        input_type: str = "raw_audio",
    ) -> torch.Tensor:
        if input_type == "raw_audio" and isinstance(batch, torch.Tensor):
            return self.forward_raw_audio(batch)
        elif input_type == "stft" and isinstance(batch, torch.Tensor):
            return self.forward_stft(batch)
        elif input_type == "features" and isinstance(batch, dict):
            return self.forward_precomputed_features_dict(batch)
        elif input_type == "features" and isinstance(batch, torch.Tensor):
            raise ValueError(
                "When input_type is 'features', batch must be a dict of precomputed features."
            )
        else:
            raise ValueError(f"Invalid input_type: {input_type}")

    def forward_raw_audio(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = []
        for extractor in self._extractors:
            outputs.append(extractor.forward_raw_audio(batch))
        return torch.cat(outputs, dim=-2)

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = []
        for extractor in self._extractors:
            if extractor.uses_stft:
                outputs.append(extractor.forward_stft(batch))
            else:
                # Fallback: If we have STFT but extractor needs raw, we can't easily invert.
                # We assume that if input is STFT, all extractors must support it.
                raise NotImplementedError(
                    f"Extractor {extractor.__class__.__name__} does not support STFT input, "
                    "but StackedFeatureExtractor was called with input_type='stft'."
                )
        return torch.cat(outputs, dim=-2)

    def forward_precomputed_features_dict(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Processes precomputed features.
        'batch' is expected to be a dictionary where keys are like "0_features", "1_stft", etc.
        """
        outputs = []
        for i, extractor in enumerate(self._extractors):
            # Determine which key to use for this extractor
            feat_key = f"{i}_features"
            stft_key = f"{i}_stft"

            if feat_key in batch:
                # Fixed feature: pass through
                outputs.append(extractor.forward_precomputed_features(batch[feat_key]))
            elif stft_key in batch:
                # Trainable feature: run forward_stft on the saved STFT
                outputs.append(extractor.forward_stft(batch[stft_key]))
            else:
                # Try to find any key starting with i_?
                # This handles cases where a sub-extractor might return something else
                found = False
                for k, v in batch.items():
                    if k.startswith(f"{i}_"):
                        # Assuming if it's not 'features' or 'stft', it might be handled by forward_precomputed_features
                        # But strictly speaking, our protocol defines 'features' and 'stft'.
                        pass

                if not found:
                    raise ValueError(
                        f"No precomputed data found for extractor {i} in batch keys: {list(batch.keys())}"
                    )

        return torch.cat(outputs, dim=-2)
