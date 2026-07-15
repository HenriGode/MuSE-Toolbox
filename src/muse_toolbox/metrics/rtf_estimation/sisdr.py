import torch
from muse_toolbox.metrics.rtf_estimation.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SISDR(RefMetric):
    """Scale-Invariant Signal Distortion Ratio (SI-SDR) metric class.
    
    Inherits from `RefMetric` and evaluates audio quality using the 
    `torchmetrics.audio.ScaleInvariantSignalDistortionRatio` implementation.
    """
    def __init__(
        self,
        transform: STFTtransform,
        zero_mean: bool = False,
        ref_channels: list[int] = [0],
        *args,
        **kwargs,
    ):
        """Initializes the SI-SDR metric.

        Args:
            transform (STFTtransform): Transformer to convert STFT inputs back to time-domain.
            zero_mean (bool): If True, zero mean the signals before evaluation. Defaults to False.
            ref_channels (list[int]): Indices of channels to use as reference signals.
            *args: Variable length arguments passed to `RefMetric`.
            **kwargs: Arbitrary keyword arguments passed to `RefMetric`.
        """
        super().__init__(
            metric_name="SISDR",
            transform=transform,
            ref_channels=ref_channels,
            *args,
            **kwargs,
        )

        self.SISDR_fun = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Evaluates the SI-SDR score for a given degraded and reference signal.

        Args:
            deg (torch.Tensor): The degraded (or enhanced) audio signal.
            ref (torch.Tensor): The ground truth reference audio signal.

        Returns:
            torch.Tensor: The computed SI-SDR value. Returns NaN if evaluation fails.
        """
        try:
            return self.SISDR_fun(deg, ref)
        except Exception:
            return torch.tensor(float("nan"), device=ref.device)

