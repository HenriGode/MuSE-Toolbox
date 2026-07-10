import torch
from .base_rtf_estimator import BaseRTFestimator
from utilities.util_classes import covblockwhiten


class CBW(BaseRTFestimator):
    """RTF estimator using the principle eigenvector of the Covariance matrix."""

    def __init__(self):
        super().__init__()

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ):
        # Implementation of the C RTF estimation method goes here
        return covblockwhiten(Ry, Rn, G)
