import torch
from .BOP import BOP
from muse_toolbox.utils import noise_whitening_4_BOP


class BOP_W(BOP):
    """
    RTF estimator using the BOP-W method.
    It uses random additional vectors and noise whitening.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form"):
        super().__init__(mode=mode)

    @staticmethod
    def noise_handling(
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # noise whitening
        L, R, G = noise_whitening_4_BOP(Rn, Ry, G)
        return R, G, L
