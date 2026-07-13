from typing import Any
from .cb import CB
from .bop_w import BOP_W


class CB_W(CB, BOP_W):
    """
    RTF estimator using the CB-W method.
    It uses no additional vectors and noise whitening.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__(mode=mode)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "CB_W",
            "mode": self.mode,
        }
