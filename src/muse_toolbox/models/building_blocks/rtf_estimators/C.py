import torch
from .base_rtf_estimator import BaseRTFestimator
from muse_toolbox.utils import peigvech


class C(BaseRTFestimator):
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
        return peigvech(Ry)
