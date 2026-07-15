import torch
import wandb
import pandas as pd
from torchmetrics.classification import MulticlassConfusionMatrix
from muse_toolbox.metrics.base_metric import BaseMetric
import matplotlib.pyplot as plt


class ConfusionMatrix(BaseMetric):
    """Confusion Matrix metric class for source counting.
    
    Inherits from `BaseMetric` to generate and log a multi-class 
    confusion matrix tracking predicted vs actual active sources.
    """
    is_differentiable = False
    higher_is_better = None
    full_state_update = True
    requires_reference = True

    def __init__(self, max_sources: int, stage: str, *args, **kwargs):
        """Initializes the ConfusionMatrix metric.

        Args:
            max_sources (int): The maximum number of sources expected.
            stage (str): The execution stage (e.g., 'val', 'test') for logging.
            *args: Variable length arguments passed to BaseMetric.
            **kwargs: Arbitrary keyword arguments passed to BaseMetric.
        """
        super().__init__(*args, requires_numpy=False, **kwargs)
        self.num_classes = max_sources + 1
        self.conf_matrix_metric = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        )
        self.stage = stage

    def update(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        meta: dict,
        dataloader_idx: int,
    ):
        """Updates the ConfusionMatrix metric state with new batch data.

        Args:
            preds: List of prediction tensors.
            targets: List of ground truth target tensors.
            meta (dict): Dictionary with scenario metadata.
            dataloader_idx (int): Current dataloader index.
        """
        if self.requires_reference and targets is None:
            return

        pred_one_hot = torch.cat(preds).permute(0, 1)
        true_counts = torch.cat(targets)

        # The metric expects (N, C, ...) for preds and (N, ...) for target.
        # Your inputs are (B, T, C) and (B, T), which is incompatible, so we permute preds.
        self.conf_matrix_metric.update(pred_one_hot.cpu(), true_counts.cpu())

    def compute(self) -> dict:
        """
        Computes the confusion matrix and logs it as a W&B image.
        Returns an empty dictionary as the raw tensor is not suitable for scalar logging.
        """
        conf_matrix = self.conf_matrix_metric.compute()
        fig, ax = self.conf_matrix_metric.plot(val=conf_matrix)

        # Add titles and labels for clarity
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # plt.tight_layout()

        # Log plot to wandb
        if wandb.run:
            wandb.log({f"{self.stage}/ConfusionMatrix": wandb.Image(fig)})

        plt.close(fig)  # Close the figure to free memory

        # The raw matrix is not a scalar, so we don't return it for standard logging.
        # The visual plot is logged to W&B directly.
        return {}

    def get_dataframe(self) -> pd.DataFrame | None:
        """The confusion matrix is an aggregate metric and does not produce per-sample results.
        
        Returns:
            pd.DataFrame | None: Always returns None.
        """
        return None

    def reset(self) -> None:
        """Resets the internal state of the metric."""
        self.conf_matrix_metric.reset()
