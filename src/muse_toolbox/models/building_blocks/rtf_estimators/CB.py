import torch
from .BOP import BOP


class CB(BOP):
    """
    RTF estimator using the CB method.
    It uses no additional vectors and no noise handling.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form"):
        super().__init__(mode=mode)

    @staticmethod
    def generate_add_vecs(G: torch.Tensor, R: torch.Tensor, Ka: int) -> torch.Tensor:
        # No additional vectors
        return torch.zeros_like(G)[..., 0:0]
