import torch
import logging
from typing import Any

from muse_toolbox.models.rtf_estimation.estimators.base_rtf_estimator import BaseRTFestimator
from muse_toolbox.utils import peigvech

log = logging.getLogger(__name__)


class C(BaseRTFestimator):
    """RTF estimator using the principle eigenvector of the Covariance matrix."""

    def __init__(self) -> None:
        super().__init__()

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {"type": "C"}

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates the RTF using the principle eigenvector of the noisy covariance matrix.
        """
        return peigvech(Ry)
