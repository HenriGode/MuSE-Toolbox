from typing import Any
import pandas as pd
import torch
import torchmetrics


class BaseMetric(torchmetrics.Metric):
    """Base class for all metrics in the MuSE-Toolbox.

    Inherits from `torchmetrics.Metric` and establishes the foundational 
    interface for both source counting and RTF estimation metrics.

    Attributes:
        requires_reference (bool): Whether this metric requires a ground-truth reference signal.
        requires_numpy (bool): Whether this metric expects NumPy arrays instead of PyTorch tensors.
        dataframe (pd.DataFrame): DataFrame for storing per-scenario evaluation results.
    """

    def __init__(
        self,
        *args,
        requires_reference: bool = True,
        requires_numpy: bool = True,
        **kwargs,
    ):
        """Initializes the BaseMetric.

        Args:
            requires_reference (bool): If True, the metric needs reference targets. Defaults to True.
            requires_numpy (bool): If True, inputs are converted to NumPy. Defaults to True.
            *args: Variable length argument list passed to `torchmetrics.Metric`.
            **kwargs: Arbitrary keyword arguments passed to `torchmetrics.Metric`.
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.requires_reference = requires_reference
        self.requires_numpy = requires_numpy
        self.dataframe = pd.DataFrame()

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Core computation logic for the metric (must be implemented by subclasses).

        Args:
            preds (torch.Tensor): The predicted output from the model.
            target (torch.Tensor): The ground truth target.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError

    def update(
        self,
        preds: Any,
        targets: Any,
        meta: dict,
        dataloader_idx: int,
    ) -> None:
        """Updates the metric state with new batch data.

        Args:
            preds (Any): Predictions for the current batch.
            targets (Any): Ground truth targets for the current batch.
            meta (dict): Dictionary containing scenario and session metadata.
            dataloader_idx (int): Index of the current dataloader.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError

    def update_dataframe(self, meta: dict, results: dict) -> None:
        """Updates the internal pandas DataFrame with scenario results.

        Args:
            meta (dict): Dictionary containing metadata for the current batch.
            results (dict): Dictionary of computed metric results.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError

    def compute(self):
        """Aggregates the accumulated states and returns the final metric value.

        Returns:
            The computed metric(s).

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError
        # count nan and its percentage

        # move to cuda required due to lightning and torchmetrics quirk...
        # return {
        #     # "noisy": (self.noisy_total.float() / self.numel).to("cuda"),
        #     "enhanced": (self.enhanced_total.float() / self.numel).to("cuda"),
        # }
