import torch
from .BOP import BOP
from muse_toolbox.utils import noise_subtraction


class BOP_S(BOP):
    """
    RTF estimator using the BOP-S method.
    It uses random additional vectors and noise subtraction.

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
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        # Subtract noise covariance from noisy covariance
        R = noise_subtraction(Rn, Ry, ensure_PSD=False)
        return R, G, None
