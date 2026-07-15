import torch
import logging

from muse_toolbox.metrics.rtf_estimation.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

log = logging.getLogger(__name__)


class PESQ(RefMetric):
    """Perceptual Evaluation of Speech Quality (PESQ) metric class.
    
    Inherits from `RefMetric` and evaluates audio quality using the 
    `torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality` implementation.
    Includes handling for very short signals via zero-padding.
    """
    def __init__(
        self,
        transform: STFTtransform,
        mode: str,
        n_processes: int = 1,
        ref_channels: list[int] = [0],
        *args,
        **kwargs,
    ):
        """Initializes the PESQ metric.

        Args:
            transform (STFTtransform): Transformer to convert STFT inputs back to time-domain.
            mode (str): Either 'wb' (wideband) or 'nb' (narrowband).
            n_processes (int): Number of parallel processes to use. Defaults to 1.
            ref_channels (list[int]): Indices of channels to use as reference signals.
            *args: Variable length arguments passed to `RefMetric`.
            **kwargs: Arbitrary keyword arguments passed to `RefMetric`.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        super().__init__(
            metric_name="PESQ",
            transform=transform,
            ref_channels=ref_channels,
            *args,
            **kwargs,
        )

        fs = int(self.transform.sampling_frequency)
        if fs not in [8000, 16000]:
            log.warning(
                f"Warning: Sampling frequency {fs} not supported for PESQ. Defaulting to 'wb' (16kHz). This might crash."
            )
            fs = 16000
        if mode not in ["wb", "nb"]:
            raise ValueError(f"Invalid mode {mode} for PESQ. Must be 'wb' or 'nb'.")

        self.PESQ_fun = PerceptualEvaluationSpeechQuality(
            fs=fs, mode=mode, n_processes=n_processes
        )

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Computes PESQ with robustness to short signals by zero-padding.
        
        Ensures signals are at least 0.25 seconds long by appending zeros.

        Args:
            deg (torch.Tensor): The degraded (or enhanced) audio signal.
            ref (torch.Tensor): The ground truth reference audio signal.

        Returns:
            torch.Tensor: The computed PESQ value. Returns NaN if evaluation fails.
        """
        fs = int(self.transform.sampling_frequency)
        min_samples = int(fs * 0.25)

        current_samples = ref.shape[-1]
        if current_samples < min_samples:
            pad_len = min_samples - current_samples
            # Pad last dimension
            ref = torch.nn.functional.pad(ref, (0, pad_len))
            deg = torch.nn.functional.pad(deg, (0, pad_len))

        try:
            return self.PESQ_fun(deg.squeeze(0), ref.squeeze(0))
        except Exception:
            # PESQ raises NoUtterancesError if it cannot detect speech.
            # We return NaN to indicate invalid measurement for this segment.
            return torch.tensor(float("nan"), device=ref.device)

