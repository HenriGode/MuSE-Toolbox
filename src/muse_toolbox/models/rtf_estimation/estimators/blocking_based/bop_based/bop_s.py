from typing import Any

import torch

from muse_toolbox.utils import noise_subtraction

from .bop import BOP


class BOP_S(BOP):
    """
    RTF estimator using the BOP-S method.
    It uses random additional vectors and noise subtraction.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__(mode=mode)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "BOP_S",
            "mode": self.mode,
        }

    @staticmethod
    def noise_handling(
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        # Subtract noise covariance from noisy covariance
        R = noise_subtraction(Rn, Ry, ensure_PSD=False)
        return R, G, None
