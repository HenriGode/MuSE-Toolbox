import torch
import pandas as pd
from muse_toolbox.metrics.common.base_metric import BaseMetric
from typing import Optional, List


class Accuracy(BaseMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False  # Set to False for list states
    requires_reference = True

    correct_predictions: torch.Tensor
    total_samples: torch.Tensor
    per_sample_results: List[torch.Tensor]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, requires_numpy=False, **kwargs)

        # Use primitives for default values to ensure correct device placement
        self.add_state(
            "correct_predictions", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        # This state will be a list of tensors, which can be concatenated across devices
        self.add_state("per_sample_results", default=[], dist_reduce_fx="cat")

        # scenario_ids is not a metric state, just a regular attribute
        self.scenario_ids: List[str] = []

    def update(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        meta: dict,
        dataloader_idx: int,
    ):
        if self.requires_reference and targets is None:
            return

        pred_counts = [torch.argmax(p, dim=-1) for p in preds]
        true_counts = targets

        correct = [
            torch.sum((p.float() == t.float()).int())
            for p, t in zip(pred_counts, true_counts)
        ]
        num_frames = torch.tensor([t.numel() for t in true_counts])
        # For per-sample logging, calculate MAE for each item in the batch
        mean_correct = torch.tensor([ae / nf for ae, nf in zip(correct, num_frames)])
        self.correct_predictions += torch.sum(torch.stack(correct)).item()
        self.total_samples += torch.sum(num_frames).item()

        # Append the raw tensor to the list state
        self.per_sample_results.append(mean_correct)
        # Extend the regular python list with string IDs
        self.scenario_ids.extend(meta["scenario_id"])

    def compute(self) -> dict:
        """Computes the final accuracy over all batches for W&B logging."""
        if self.total_samples == 0:
            return {"Accuracy": 0.0}
        accuracy = self.correct_predictions.float() / self.total_samples
        return {"Accuracy": accuracy}

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Creates a DataFrame with per-sample Accuracy results."""
        if not self.scenario_ids:
            return None

        # After compute(), self.per_sample_results is a gathered list of tensors.
        # Concatenate them into a single tensor before moving to CPU.
        all_results = torch.cat(self.per_sample_results).cpu().numpy()

        df = pd.DataFrame({"Accuracy": all_results}, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df

    def reset(self) -> None:
        """Resets metric states and the scenario_ids list."""
        super().reset()
        self.scenario_ids = []
