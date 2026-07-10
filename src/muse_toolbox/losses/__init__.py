"""Loss functions for the MuSE-Toolbox."""

from .common.base_loss import BaseLoss
from .source_counting.cross_entropy import CrossEntropy

__all__ = ["BaseLoss", "CrossEntropy"]
