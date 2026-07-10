"""Base class for all losses in the MuSE Toolbox."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Union, List


class BaseLoss(nn.Module, ABC):
    """Abstract base class for all losses.

    This class provides a common interface for computing losses. It inherits
    from torch.nn.Module to allow for parameter registration and integration
    with standard PyTorch workflows.
    """

    def __init__(self, **kwargs: Any):
        """Initializes the base loss module."""
        super().__init__()

    def forward(
        self,
        prediction: Union[torch.Tensor, List[torch.Tensor]],
        target: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Computes the loss between predictions and targets.

        This method acts as the standard PyTorch forward pass, handling both
        single tensors and lists of tensors (e.g., for variable-length sequences
        or multiple outputs).

        Args:
            prediction (torch.Tensor | list[torch.Tensor]): The predicted values.
            target (torch.Tensor | list[torch.Tensor]): The ground truth values.

        Returns:
            torch.Tensor: The computed scalar loss.
            
        Raises:
            ValueError: If `prediction` and `target` types or lengths do not match.
        """
        if isinstance(prediction, list):
            if not isinstance(target, list):
                raise ValueError("When prediction is a list, target must also be a list.")
            if len(prediction) != len(target):
                raise ValueError("Prediction and target lists must have the same length.")

            frames = 0
            loss = torch.tensor(0.0, device=prediction[0].device)
            for out, tgt in zip(prediction, target):
                frames += tgt.numel()
                loss += self.compute_loss(out, tgt)

            return loss / frames
        else:
            if isinstance(target, list):
                raise ValueError("When prediction is a tensor, target must also be a tensor.")
            
            # Here we expect compute_loss to return a pre-reduced mean loss if appropriate,
            # or we normalize by the number of elements depending on the sub-class implementation.
            # Usually CrossEntropyLoss with reduction='mean' handles normalization automatically.
            return self.compute_loss(prediction, target)

    @abstractmethod
    def compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Computes the core loss function for a single tensor pair.

        Subclasses must implement this method to define the specific
        loss logic (e.g., CrossEntropy, MSE, etc.).

        Args:
            prediction (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        pass
