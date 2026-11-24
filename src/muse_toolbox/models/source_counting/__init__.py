"""
Source Counting Module.

This package contains models and routines for estimating 
the number of active speakers.
"""

from .osc import OSCmodule
from .precomputed_sad import PrecomputedSAD

__all__ = [
    "OSCmodule",
    "PrecomputedSAD",
]
