from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from muse_toolbox.utils import STFTtransform


class BaseSourceCountEstimator(nn.Module, ABC):
    """
    Abstract base class for all Source Count Estimators (Detectors) in the COSAD framework.

    This class defines the common interface for all detector implementations.
    Its primary role is to enforce a consistent input/output structure.

    Child classes must implement the `forward` method, which is responsible for
    processing input features and returning a dictionary containing the estimated
    source activity tensor.
    """

    def __init__(self, input_dim: int, transform: STFTtransform, max_sources: int):
        """
        Initializes the base detector.

        Args:
            input_dim (int): The dimension of the input feature vector (J).
            transform (STFTtransform): An STFT transformation object.
            max_sources (int): The maximum number of sources to consider. This
                defines the number of output classes (C = max_sources + 1).
        """
        super().__init__()
        self.input_dim = input_dim
        self.transform = transform
        self.max_sources = max_sources

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes the input features and returns the estimated source activity logits.

        Args:
            features (torch.Tensor): Input features of shape (B, J, T), where
                B = batch size, J = feature dimension, T = time frames.
        Returns:
            torch.Tensor: Source activity logits of shape (B, T, C), where
                C = number of classes (max_sources + 1).
        """
        return self.forward_tensor(features)

    def forward_old(
        self, features: torch.Tensor | list[torch.Tensor]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Handles input dispatch (Tensor vs List) and standardizes output format.
        """
        if isinstance(features, torch.Tensor):
            # Case 1: Single Tensor (B, J, T)
            logits = self.forward_tensor(features)
            return {"estimated_source_activity": logits}

        elif isinstance(features, list):
            # Case 2: List of Tensors [ (1, J, T1), (1, J, T2), ... ]
            outputs = []
            for x in features:
                # Process each item individually
                # x is (1, J, Ti) -> forward_tensor -> (1, Ti, C)
                out = self.forward_tensor(x)
                outputs.append(out)

            # Try to stack if possible (same time dimension)
            try:
                # Check if all time dimensions are equal
                first_shape = outputs[0].shape
                if all(o.shape == first_shape for o in outputs):
                    stacked_logits = torch.cat(outputs, dim=0)  # (B, T, C)
                    return {"estimated_source_activity": stacked_logits}
            except (IndexError, RuntimeError):
                pass

            # Return list if stacking fails
            return {"estimated_source_activity": outputs}

        else:
            raise TypeError(f"Unsupported input type: {type(features)}")

    @abstractmethod
    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to process a single feature tensor.

        Args:
            features (torch.Tensor): Input features (B, J, T).

        Returns:
            torch.Tensor: Source activity logits (B, T, C).
        """
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """
        Returns the configuration dictionary used to initialize this detector.
        """
        pass

    def _verbose_parameters(self, indent: str = "") -> None:
        """
        Prints the parameters of the module in a structured, indented format.
        Child classes should extend this method to include their specific parameters.

        Args:
            indent (str, optional): A string to prepend to each line for indentation.
                                    Defaults to "".
        """
        print(f"{indent}{self.__class__.__name__} Parameters:")
        print(f"{indent}  Max Sources: {self.max_sources}")
