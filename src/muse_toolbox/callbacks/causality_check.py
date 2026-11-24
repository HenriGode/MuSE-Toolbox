"""Callback for validating algorithmic causality during inference."""

import logging
import torch
from lightning.pytorch import Callback, Trainer, LightningModule

log = logging.getLogger(__name__)


class CausalityCheckCallback(Callback):
    """Checks if the model satisfies the algorithmic latency requirements."""

    def __init__(self, allowed_latency_s: float = 0.010) -> None:
        """Initializes the causality check callback.

        Args:
            allowed_latency_s (float): The maximum allowed algorithmic latency in seconds.
        """
        super().__init__()
        self.allowed_latency_s = allowed_latency_s

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Triggers the causality check before testing starts.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): The active PyTorch Lightning module.
        """
        # Safely check if causality checking is disabled via model hyperparameters
        check_causality = getattr(pl_module, "check_causality", None)
        if check_causality is None and hasattr(pl_module, "hparams"):
            check_causality = pl_module.hparams.get("check_causality", True)
        if check_causality is None:
            check_causality = True

        if not check_causality:
            log.info("Causality check disabled for this model (check_causality=False).")
            return

        pl_module.eval()

        num_channels = int(getattr(pl_module, "num_mics", None) or getattr(pl_module, "num_channels", 1))
        fs = int(getattr(pl_module, "fs", 16000))

        allowed_latency = int(self.allowed_latency_s * fs)
        sig_len_range = [2.0, 8.0]
        num_repetitions = 10

        for _ in range(num_repetitions):
            length = (
                int(
                    torch.rand(1).item() * (sig_len_range[1] - sig_len_range[0])
                    + sig_len_range[0]
                )
                * fs
            )
            sig = torch.randn(length)
            sig = sig / (sig.abs().max() + 1e-8) * 0.9

            p = int(torch.randint(0, length, (1,)).item())
            sig[p:] = float("nan")

            # Dictionary batch expects input of shape (B, M, T_samples)
            sig_channel = sig.unsqueeze(0).unsqueeze(0).repeat(1, num_channels, 1)
            batch = {
                "input": sig_channel.to(pl_module.device),
                "mic_lengths": torch.tensor(
                    [num_channels], dtype=torch.long, device=pl_module.device
                ),
                "time_lengths": torch.tensor(
                    [length], dtype=torch.long, device=pl_module.device
                ),
                "meta": {},
            }

            with torch.no_grad():
                out = pl_module(batch)
                est_sig = (
                    out.get("input_proc", out.get("estimates", None))
                    if isinstance(out, dict)
                    else None
                )
                if est_sig is None:
                    log.warning("CausalityCheckCallback could not find output tensor to check.")
                    return

            if p - allowed_latency + 1 >= 1 and torch.any(
                torch.isnan(est_sig[0, 0, : p - allowed_latency + 1])
            ):
                log.error(
                    f"Your model does NOT satisfy the algorithmic latency requirement of {self.allowed_latency_s} s!"
                )
                return

        log.info(
            f"Your model satisfies the algorithmic latency requirement of {self.allowed_latency_s} s!"
        )
