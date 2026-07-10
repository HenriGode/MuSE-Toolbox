import torch
from muse_toolbox.metrics.common.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


class PESQ(RefMetric):
    def __init__(
        self,
        transform: STFTtransform,
        mode: str,
        n_processes: int = 1,
        ref_channels: list[int] = [0],
        *args,
        **kwargs,
    ):
        super().__init__(
            metric_name="PESQ",
            transform=transform,
            ref_channels=ref_channels,
            *args,
            **kwargs,
        )

        fs = int(self.transform.sampling_frequency)
        if fs not in [8000, 16000]:
            print(
                f"Warning: Sampling frequency {fs} not supported for PESQ. Defaulting to 'wb' (16kHz). This might crash."
            )
            fs = 16000
        if mode not in ["wb", "nb"]:
            raise ValueError(f"Invalid mode {mode} for PESQ. Must be 'wb' or 'nb'.")

        self.PESQ_fun = PerceptualEvaluationSpeechQuality(
            fs=fs, mode=mode, n_processes=n_processes
        )

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Compute PESQ with robustness to short signals by zero-padding.
        Ensures signals are at least 0.25 seconds long.
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
