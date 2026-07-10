import torch
from muse_toolbox.metrics.common.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform, fwssnr


class FWSSNR(RefMetric):
    def __init__(
        self,
        transform: STFTtransform,
        frameLen: float,
        overlap: float,
        ref_channels: list[int],
        *args,
        **kwargs
    ):
        super().__init__(
            metric_name="FWSSNR",
            transform=transform,
            ref_channels=ref_channels,
            *args,
            **kwargs
        )

        self.frameLen = frameLen
        self.overlap = overlap
        self.fs = int(self.transform.sampling_frequency)

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        try:
            return fwssnr(
                ref=ref.squeeze(0),
                sig=deg.squeeze(0),
                fs=self.fs,
                frameLen=self.frameLen,
                overlap=self.overlap,
            )
        except Exception:
            return torch.tensor([float("nan")], device=ref.device)
