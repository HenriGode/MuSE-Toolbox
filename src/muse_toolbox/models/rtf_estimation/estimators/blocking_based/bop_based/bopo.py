import torch
from typing import Any
from .bop import BOP
from muse_toolbox.utils import characteristic_subspace_h


class BOPO(BOP):
    """
    RTF estimator using the BOPO method.
    It uses orthogonal additional vectors and no noise handling.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__(mode=mode)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "BOPO",
            "mode": self.mode,
        }

    @staticmethod
    def generate_add_vecs(G: torch.Tensor, R: torch.Tensor, Ka: int) -> torch.Tensor:
        return characteristic_subspace_h(R, order=range(-Ka, 0))
