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
        pl_module.eval()

        num_channels = getattr(pl_module, "num_channels", 1)
        binaural = getattr(pl_module, "binaural", False)
        model_name = getattr(pl_module, "model_name", "")
        fs = getattr(pl_module, "fs", 16000)

        if "Bilat" in model_name or model_name == "BiConvTasNet":
            binaural = True
        if binaural:
            num_channels *= 2

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
            
            p = torch.randint(0, length, (1,)).item()
            sig[p:] = float("nan")
            sig = sig.unsqueeze(0).unsqueeze(0)
            sig = sig.repeat(1, num_channels, 1)

            inp = {"input": sig.to(pl_module.device)}
            with torch.no_grad():
                out = pl_module(inp)
                est_sig = out.get("input_proc", out.get("estimates", None))
                
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
