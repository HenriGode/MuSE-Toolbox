from .CB import CB
from .BOP_S import BOP_S


class CB_S(CB, BOP_S):
    """
    RTF estimator using the CB-S method.
    It uses no additional vectors and noise subtraction.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form"):
        super().__init__(mode=mode)
