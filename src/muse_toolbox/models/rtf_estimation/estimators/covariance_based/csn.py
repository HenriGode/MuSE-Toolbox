import logging
from typing import Any

import torch

from muse_toolbox.models.rtf_estimation.estimators.base_rtf_estimator import (
    BaseRTFestimator,
)
from muse_toolbox.utils import covarianceSubtraction

log = logging.getLogger(__name__)


class CSn(BaseRTFestimator):
    """RTF estimator using Covariance Subtraction (noise)."""

    def __init__(self) -> None:
        super().__init__()

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {"type": "CSn"}

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates the RTF using covariance subtraction.
        """
        return covarianceSubtraction(noiseCovMat=Rn, covMat=Ry)
