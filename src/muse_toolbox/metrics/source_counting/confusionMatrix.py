import torch
import wandb
import pandas as pd
from torchmetrics.classification import MulticlassConfusionMatrix
from muse_toolbox.metrics.common.base_metric import BaseMetric
from typing import Optional, List
import matplotlib.pyplot as plt
import io
import PIL


class ConfusionMatrix(BaseMetric):
    is_differentiable = False
    higher_is_better = None
    full_state_update = True
    requires_reference = True

    def __init__(self, max_sources: int, stage: str, *args, **kwargs):
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

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        The confusion matrix is an aggregate metric and does not produce per-sample results.
        Therefore, it does not return a DataFrame.
        """
        return None

    def reset(self) -> None:
        """Resets the internal state of the metric."""
        self.conf_matrix_metric.reset()
