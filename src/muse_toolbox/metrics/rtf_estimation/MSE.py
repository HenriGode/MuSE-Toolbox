import torch
import pandas as pd
from muse_toolbox.metrics.common.base_metric import BaseMetric
from typing import Optional, List


class MSE(BaseMetric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = True
    requires_reference = True

    total_squared_error: torch.Tensor
    total_samples: torch.Tensor
    per_sample_results: List[torch.Tensor]  # State will be a list of Tensors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, requires_numpy=False, **kwargs)

        # Use primitives for default values to ensure correct device placement
        self.add_state(
            "total_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
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

        squared_error = [
            torch.sum(torch.square(p.float() - t.float()))
            for p, t in zip(pred_counts, true_counts)
        ]
        num_frames = torch.tensor([t.numel() for t in true_counts])
        # For per-sample logging, calculate MSE for each item in the batch
        mean_squared_error = torch.tensor(
            [se / nf for se, nf in zip(squared_error, num_frames)]
        )
        self.total_squared_error += torch.sum(torch.stack(squared_error)).item()
        self.total_samples += torch.sum(num_frames).item()

        # Append the raw tensor to the list state
        self.per_sample_results.append(mean_squared_error)
        # Extend the regular python list with string IDs
        self.scenario_ids.extend(meta["scenario_id"])

    def compute(self) -> dict:
        """Computes the final MSE over all batches for W&B logging."""
        if self.total_samples == 0:
            return {"MSE": 0.0}
        mse = self.total_squared_error / self.total_samples
        return {"MSE": mse}

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Creates a DataFrame with per-sample MSE results."""
        if not self.scenario_ids:
            return None

        # After compute(), self.per_sample_results is a gathered list of tensors.
        # Concatenate them into a single tensor before moving to CPU.
        all_results = torch.cat(self.per_sample_results).cpu().numpy()

        df = pd.DataFrame({"MSE": all_results}, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df

    def reset(self) -> None:
        """Resets metric states and the scenario_ids list."""
        super().reset()
        self.scenario_ids = []
