from .CB import CB
from .BOP_W import BOP_W


class CB_W(CB, BOP_W):
    """
    RTF estimator using the CB-W method.
    It uses no additional vectors and noise whitening.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form"):
        super().__init__(mode=mode)
