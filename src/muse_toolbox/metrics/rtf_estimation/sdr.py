import torch
from muse_toolbox.metrics.rtf_estimation.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform
from torchmetrics.audio import SignalDistortionRatio


class SDR(RefMetric):
    """Signal Distortion Ratio (SDR) evaluation metric.
    
    Inherits from `RefMetric` and wraps `torchmetrics.audio.SignalDistortionRatio`.
    """
    def __init__(
        self,
        transform: STFTtransform,
        use_cg_iter: int | None = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        load_diag: float | None = None,
        ref_channels: list[int] = [0],
        *args,
        **kwargs,
    ):
        """Initializes the SDR metric.

        Args:
            transform (STFTtransform): Transformer to convert STFT back to time-domain.
            use_cg_iter (int | None): Number of CG iterations for SDR calculation.
            filter_length (int): Length of the filter. Defaults to 512.
            zero_mean (bool): If True, zero mean the signals. Defaults to False.
            load_diag (float | None): Diagonal loading for numerical stability.
            ref_channels (list[int]): Indices of reference channels. Defaults to [0].
            *args: Variable length arguments passed to `RefMetric`.
            **kwargs: Arbitrary keyword arguments passed to `RefMetric`.
        """
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
        """Evaluates the SDR score for a given degraded and reference signal.

        Args:
            deg (torch.Tensor): The degraded (or enhanced) audio signal.
            ref (torch.Tensor): The ground truth reference audio signal.

        Returns:
            torch.Tensor: The computed SDR value. Returns NaN if evaluation fails.
        """
        try:
            return self.SDR_fun(deg, ref)
        except Exception:
            return torch.tensor(float("nan"), device=ref.device)

