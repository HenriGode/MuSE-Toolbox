import torch
import logging
from typing import Any

from muse_toolbox.models.rtf_estimation.estimators.base_rtf_estimator import BaseRTFestimator
from muse_toolbox.utils.util_classes import covblockwhiten

log = logging.getLogger(__name__)


class CBW(BaseRTFestimator):
    """RTF estimator using Covariance Blocking and Whitening."""

    def __init__(self) -> None:
        super().__init__()

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {"type": "CBW"}

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates the RTF using covariance blocking and whitening.
        """
        return covblockwhiten(Ry, Rn, G)
