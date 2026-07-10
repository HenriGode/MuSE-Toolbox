from .BOPO import BOPO
from .BOP_W import BOP_W


class BOPO_W(BOPO, BOP_W):
    """
    RTF estimator using the BOPO-W method.
    It uses orthogonal additional vectors and noise whitening.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form"):
        super().__init__(mode=mode)
