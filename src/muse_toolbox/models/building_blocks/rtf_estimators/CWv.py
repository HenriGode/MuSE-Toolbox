import torch
from .base_rtf_estimator import BaseRTFestimator
from muse_toolbox.utils import covarianceWhitening


class CWv(BaseRTFestimator):
    """RTF estimator using the C method."""

    def __init__(self):
        super().__init__()

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ):
        # Implementation of the CWv RTF estimation method goes here
        return covarianceWhitening(whiteningCovMat=Rv, covMat=Ry)
