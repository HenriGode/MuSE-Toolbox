from typing import Any
from .bopo import BOPO
from .bop_w import BOP_W


class BOPO_W(BOPO, BOP_W):
    """
    RTF estimator using the BOPO-W method.
    It uses orthogonal additional vectors and noise whitening.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__(mode=mode)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "BOPO_W",
            "mode": self.mode,
        }
