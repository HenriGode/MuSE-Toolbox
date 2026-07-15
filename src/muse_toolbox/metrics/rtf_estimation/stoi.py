import torch
from muse_toolbox.metrics.rtf_estimation.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform
from torchmetrics.audio import ShortTimeObjectiveIntelligibility


class STOI(RefMetric):
    """Short-Time Objective Intelligibility (STOI) metric class.
    
    Inherits from `RefMetric` and evaluates speech intelligibility using the 
    `torchmetrics.audio.ShortTimeObjectiveIntelligibility` implementation.
    """
    def __init__(
        self,
        transform: STFTtransform,
        extended: bool,
        ref_channels: list[int],
        *args,
        **kwargs
    ):
        """Initializes the STOI metric.

        Args:
            transform (STFTtransform): Transformer to convert STFT inputs back to time-domain.
            extended (bool): If True, uses the Extended STOI (eSTOI) algorithm.
            ref_channels (list[int]): Indices of channels to use as reference signals.
            *args: Variable length arguments passed to `RefMetric`.
            **kwargs: Arbitrary keyword arguments passed to `RefMetric`.
        """
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
        """Evaluates the STOI score for a given degraded and reference signal.

        Args:
            deg (torch.Tensor): The degraded (or enhanced) audio signal.
            ref (torch.Tensor): The ground truth reference audio signal.

        Returns:
            torch.Tensor: The computed STOI value. Returns NaN if evaluation fails.
        """
        try:
            return self.STOI_fun(deg.squeeze(0), ref.squeeze(0))
        except Exception:
            return torch.tensor(float("nan"), device=ref.device)

