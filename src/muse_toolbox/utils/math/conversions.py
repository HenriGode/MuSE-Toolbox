"""Conversions"""

import logging

import torch

log = logging.getLogger(__name__)


def db2amp(db: torch.Tensor) -> torch.Tensor:
    """
    Converts decibels (dB) to linear amplitude.

    Args:
        db (torch.Tensor): Tensor containing values in decibels.

    Returns:
        torch.Tensor: Tensor containing corresponding linear amplitude values.
    """
    return torch.pow(10.0, db / 20.0)


def db2pow(db: torch.Tensor) -> torch.Tensor:
    """
    Converts decibels (dB) to linear power.

    Args:
        db (torch.Tensor): Tensor containing values in decibels.

    Returns:
        torch.Tensor: Tensor containing corresponding linear power values.
    """
    return torch.pow(10.0, db / 10.0)


def amp2db(amp: torch.Tensor) -> torch.Tensor:
    """
    Converts linear amplitude to decibels (dB).

    Args:
        amp (torch.Tensor): Tensor containing linear amplitude values.

    Returns:
        torch.Tensor: Tensor containing corresponding decibel values.
    """
    return 20 * torch.log10(amp)


def pow2db(power: torch.Tensor) -> torch.Tensor:
    """
    Converts linear power to decibels (dB).

    Args:
        power (torch.Tensor): Tensor containing linear power values.

    Returns:
        torch.Tensor: Tensor containing corresponding decibel values.
    """
    return 10 * torch.log10(power)


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    """
    Converts angles from radians to degrees.

    Args:
        rad (torch.Tensor): Tensor containing angles in radians.

    Returns:
        torch.Tensor: Tensor containing corresponding angles in degrees.
    """
    return rad / torch.pi * 180


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    """
    Converts angles from degrees to radians.

    Args:
        deg (torch.Tensor): Tensor containing angles in degrees.

    Returns:
        torch.Tensor: Tensor containing corresponding angles in radians.
    """
    return deg / 180 * torch.pi