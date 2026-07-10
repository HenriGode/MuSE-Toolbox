from .BOPO import BOPO
from .BOP_S import BOP_S


class BOPO_S(BOPO, BOP_S):
    """
    RTF estimator using the BOPO-S method.
    It uses orthogonal additional vectors and noise subtraction.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form"):
        super().__init__(mode=mode)
