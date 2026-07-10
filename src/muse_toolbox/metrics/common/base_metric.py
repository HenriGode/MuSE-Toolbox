import pandas as pd
import torch
import torchmetrics


class BaseMetric(torchmetrics.Metric):
    def __init__(
        self,
        *args,
        requires_reference: bool = True,
        requires_numpy: bool = True,
        # name: str = "",
        **kwargs,
    ):
        super().__init__(
            *args,
            # compute_on_cpu=True,
            **kwargs,
        )
        self.requires_reference = requires_reference
        self.requires_numpy = requires_numpy
        self.dataframe = pd.DataFrame()

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        meta: dict,
        dataloader_idx: int,
    ) -> None:
        raise NotImplementedError

    def update_dataframe(self, meta, results):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError
        # count nan and its percentage

        # move to cuda required due to lightning and torchmetrics quirk...
        # return {
        #     # "noisy": (self.noisy_total.float() / self.numel).to("cuda"),
        #     "enhanced": (self.enhanced_total.float() / self.numel).to("cuda"),
        # }
