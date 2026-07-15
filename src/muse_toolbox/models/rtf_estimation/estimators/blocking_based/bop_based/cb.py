from typing import Any

import torch

from .bop import BOP


class CB(BOP):
    """
    RTF estimator using the CB method.
    It uses no additional vectors and no noise handling.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__(mode=mode)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "CB",
            "mode": self.mode,
        }

    @staticmethod
    def generate_add_vecs(G: torch.Tensor, R: torch.Tensor, Ka: int) -> torch.Tensor:
        # No additional vectors
        return torch.zeros_like(G)[..., 0:0]
