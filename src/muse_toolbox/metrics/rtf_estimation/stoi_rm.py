import torch
from muse_toolbox.metrics.common.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform
from torchmetrics.audio import ShortTimeObjectiveIntelligibility


class STOI(RefMetric):
    def __init__(
        self,
        transform: STFTtransform,
        extended: bool,
        ref_channels: list[int],
        *args,
        **kwargs
    ):
        super().__init__(
            metric_name="STOI",
            transform=transform,
            ref_channels=ref_channels,
            *args,
            **kwargs
        )

        self.STOI_fun = ShortTimeObjectiveIntelligibility(
            fs=int(self.transform.sampling_frequency), extended=extended
        )

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        try:
            return self.STOI_fun(deg.squeeze(0), ref.squeeze(0))
        except Exception:
            return torch.tensor(float("nan"), device=ref.device)
