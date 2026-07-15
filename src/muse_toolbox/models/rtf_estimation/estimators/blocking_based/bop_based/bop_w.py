from typing import Any

import torch

from muse_toolbox.utils import noise_whitening_4_BOP

from .bop import BOP


class BOP_W(BOP):
    """
    RTF estimator using the BOP-W method.
    It uses random additional vectors and noise whitening.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__(mode=mode)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "BOP_W",
            "mode": self.mode,
        }

    @staticmethod
    def noise_handling(
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # noise whitening
        L, R, G = noise_whitening_4_BOP(Rn, Ry, G)
        return R, G, L
