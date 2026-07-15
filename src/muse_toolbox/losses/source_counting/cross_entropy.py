"""Cross-entropy loss for frame-wise source counting."""

import torch
import torch.nn as nn
from typing import Any
from muse_toolbox.losses.base_loss import BaseLoss


class CrossEntropy(BaseLoss):
    """Standard cross-entropy loss extended for 3D tensors.

    Computes the cross-entropy loss between predicted probabilities (or logits)
    and target class indices. Automatically handles the necessary permutation
    of axes for 3D prediction tensors.
    """

    def __init__(
        self, weight: torch.Tensor | None = None, reduction: str = "mean", **kwargs: Any
    ):
        """Initializes the CrossEntropy loss.

        Args:
            weight (Optional[torch.Tensor]): A manual rescaling weight given to each class.
                If given, has to be a Tensor of size C.
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'.
            **kwargs: Additional keyword arguments passed to BaseLoss.
        """
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Computes the cross-entropy loss.

        Args:
            prediction (torch.Tensor): The predicted logits. Expected shapes:
                - (B, T, C) for batched temporal predictions.
                - (1, T, C) or (T, C) for single temporal predictions.
            target (torch.Tensor): The ground truth labels. Expected shape: (B, T) or (T,).

        Returns:
            torch.Tensor: The computed cross-entropy loss.
            
        Raises:
            ValueError: If the prediction tensor doesn't have 3 dimensions.
        """
        if prediction.dim() == 3:
            # CrossEntropyLoss expects (B, C, T) for sequence data.
            # Permute from (B, T, C) -> (B, C, T)
            return self.loss_fn(prediction.permute(0, 2, 1), target)
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}. Expected (B, T, C).")
