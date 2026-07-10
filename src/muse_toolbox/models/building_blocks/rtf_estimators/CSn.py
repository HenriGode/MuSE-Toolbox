import torch
from .base_rtf_estimator import BaseRTFestimator
from muse_toolbox.utils import covarianceSubtraction


class CSn(BaseRTFestimator):
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
        # Implementation of the CSn RTF estimation method goes here
        return covarianceSubtraction(noiseCovMat=Rn, covMat=Ry)
