import torch
from muse_toolbox.metrics.common.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SISDR(RefMetric):
    def __init__(
        self,
        transform: STFTtransform,
        zero_mean: bool = False,
        ref_channels: list[int] = [0],
        *args,
        **kwargs,
    ):
        super().__init__(
            metric_name="SISDR",
            transform=transform,
            ref_channels=ref_channels,
            *args,
            **kwargs,
        )

        self.SISDR_fun = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        try:
            return self.SISDR_fun(deg, ref)
        except Exception:
            return torch.tensor(float("nan"), device=ref.device)
