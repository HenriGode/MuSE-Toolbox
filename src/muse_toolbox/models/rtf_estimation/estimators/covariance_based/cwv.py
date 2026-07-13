import torch
import logging
from typing import Any

from muse_toolbox.models.rtf_estimation.estimators.base_rtf_estimator import BaseRTFestimator
from muse_toolbox.utils import covarianceWhitening

log = logging.getLogger(__name__)


class CWv(BaseRTFestimator):
    """RTF estimator using Covariance Whitening (undesired)."""

    def __init__(self) -> None:
        super().__init__()

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {"type": "CWv"}

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates the RTF using covariance whitening with undesired covariance.
        """
        return covarianceWhitening(whiteningCovMat=Rv, covMat=Ry)
