from typing import Any

from .bop_s import BOP_S
from .cb import CB


class CB_S(CB, BOP_S):
    """
    RTF estimator using the CB-S method.
    It uses no additional vectors and noise subtraction.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__(mode=mode)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "CB_S",
            "mode": self.mode,
        }
