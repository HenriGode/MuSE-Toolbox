"""Callback for catching and zeroing NaN gradients during backpropagation."""

import logging
import torch
from lightning.pytorch import Callback, Trainer, LightningModule

log = logging.getLogger(__name__)


class NaNGradientCallback(Callback):
    """Zeroes out NaN gradients after the backward pass and logs a counter."""

    def __init__(self) -> None:
        """Initializes the callback."""
        super().__init__()
        self.nan_batch_counter = 0.0

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Handles NaN gradients by setting them to zero.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer.
            pl_module (LightningModule): The active PyTorch Lightning module.
        """
        increase_nan_batch_counter = False
        for param in pl_module.parameters():
            if param.grad is not None:
                nan_grads = torch.isnan(param.grad)
                if torch.any(nan_grads):
                    param.grad[nan_grads] = 0.0
                    increase_nan_batch_counter = True
        
        if increase_nan_batch_counter:
            self.nan_batch_counter += 1

        batch_size = getattr(pl_module, "batch_size", 1)
        pl_module.log(
            "ptl/nan_batch_counter",
            self.nan_batch_counter,
            batch_size=batch_size,
        )
