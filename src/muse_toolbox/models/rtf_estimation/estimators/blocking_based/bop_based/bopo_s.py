from typing import Any
from .bopo import BOPO
from .bop_s import BOP_S


class BOPO_S(BOPO, BOP_S):
    """
    RTF estimator using the BOPO-S method.
    It uses orthogonal additional vectors and noise subtraction.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__(mode=mode)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "BOPO_S",
            "mode": self.mode,
        }
