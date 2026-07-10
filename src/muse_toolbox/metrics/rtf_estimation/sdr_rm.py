import torch
from typing import Optional
from muse_toolbox.metrics.common.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform
from torchmetrics.audio import SignalDistortionRatio


class SDR(RefMetric):
    def __init__(
        self,
        transform: STFTtransform,
        use_cg_iter: Optional[int] = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        load_diag: Optional[float] = None,
        ref_channels: list[int] = [0],
        *args,
        **kwargs,
    ):
        super().__init__(
            metric_name="SDR",
            transform=transform,
            ref_channels=ref_channels,
            *args,
            **kwargs,
        )

        self.SDR_fun = SignalDistortionRatio(
            use_cg_iter=use_cg_iter,
            filter_length=filter_length,
            zero_mean=zero_mean,
            load_diag=load_diag,
        )

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        try:
            return self.SDR_fun(deg, ref)
        except Exception:
            return torch.tensor(float("nan"), device=ref.device)
