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
        self.enabled: bool = True
        self.has_updates: bool = False

    def _expand_num_classes(self, new_num_classes: int) -> None:
        """Dynamically expands the internal confusion matrix when higher source counts appear.

        Args:
            new_num_classes (int): Required number of classes (`max_observed + 1`).
        """
        if new_num_classes <= self.num_classes:
            return
        old_confmat = self.conf_matrix_metric.confmat.detach().clone()
        old_c = old_confmat.shape[0]
        new_metric = MulticlassConfusionMatrix(num_classes=new_num_classes).to(
            old_confmat.device
        )
        new_metric.confmat[:old_c, :old_c] = old_confmat
        self.conf_matrix_metric = new_metric
        self.num_classes = new_num_classes

    def update(
        self,
        preds: list[torch.Tensor] | torch.Tensor,
        targets: list[torch.Tensor] | torch.Tensor,
        meta: dict,
        dataloader_idx: int,
    ) -> None:
        """Updates the ConfusionMatrix metric state with new batch data.

        Args:
            preds: List of prediction tensors or a single prediction tensor.
            targets: List of ground truth target tensors or a single target tensor.
            meta (dict): Dictionary with scenario metadata.
            dataloader_idx (int): Current dataloader index.
        """
        if not self.enabled:
            return
        if self.requires_reference and targets is None:
            return

        if isinstance(preds, list):
            if not preds:
                return
            pred_list = [
                torch.argmax(p, dim=-1).reshape(-1) if p.ndim >= 2 else p.reshape(-1)
                for p in preds
            ]
            pred_counts = torch.cat(pred_list).long().cpu()
        else:
            pred_counts = (
                torch.argmax(preds, dim=-1) if preds.ndim >= 2 else preds
            ).reshape(-1).long().cpu()

        if isinstance(targets, list):
            if not targets:
                return
            true_counts = torch.cat([t.reshape(-1) for t in targets]).long().cpu()
        else:
            true_counts = targets.reshape(-1).long().cpu()

        if pred_counts.numel() == 0 or true_counts.numel() == 0:
            return

        max_observed = int(
            max(pred_counts.max().item(), true_counts.max().item())
        )
        if max_observed >= self.num_classes:
            self._expand_num_classes(max_observed + 1)

        self.conf_matrix_metric.update(pred_counts, true_counts)
        self.has_updates = True

    def compute(self) -> dict:
        """Computes the confusion matrix and logs it as a W&B image.

        Returns an empty dictionary as the raw tensor is not suitable for scalar logging.
        """
        if not self.enabled or not self.has_updates:
            return {}

        conf_matrix = self.conf_matrix_metric.compute()
        fig, ax = self.conf_matrix_metric.plot(val=conf_matrix)

        # Add titles and labels for clarity
        ax.set_title(f"Confusion Matrix ({self.stage})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Log plot to wandb
        if wandb.run:
            wandb.log({f"{self.stage}/ConfusionMatrix": wandb.Image(fig)})

        plt.close(fig)  # Close the figure to free memory

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
        self.has_updates = False
