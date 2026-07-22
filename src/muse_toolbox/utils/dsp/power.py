import logging

import torch

from muse_toolbox.utils.math.conversions import pow2db
from muse_toolbox.utils.math.stats import wmean

log = logging.getLogger(__name__)


def compute_power(
    signal_components: torch.Tensor, vad: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes the power of a multichannel signal, optionally weighted by a Voice Activity Detector (VAD).

    Handles both per-channel VAD `[..., C, N]` and shared-channel VAD `[..., 1, N]`.

    Args:
        signal_components (torch.Tensor): The signal components of shape `(..., C, N)`.
        vad (torch.Tensor | None): The VAD mask. Defaults to None.

    Returns:
        torch.Tensor: The computed power.
    """
    log.debug("Computing signal power.")
    if vad is None:
        return torch.mean(
            (torch.abs(signal_components) ** 2), dim=[-2, -1], keepdim=True
        )
    else:
        # Check if the VAD is shared (channel dim is 1) and the signal is multichannel
        num_channels = signal_components.shape[-2]
        if vad.shape[-2] == 1 and num_channels > 1:
            # Explicitly repeat the VAD along the channel dimension to match the signal
            repeat_dims = [1] * (vad.dim() - 2) + [num_channels, 1]
            vad = vad.repeat(*repeat_dims)

        # Now that VAD has the correct shape, the original calculation is correct.
        return wmean(torch.abs(signal_components) ** 2, dims=(-2, -1), weights=vad)


def compute_rms(
    signal_components: torch.Tensor, vad: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes the Root Mean Square (RMS) of a signal, optionally weighted by a VAD.

    Args:
        signal_components (torch.Tensor): The signal components.
        vad (torch.Tensor | None): The VAD mask. Defaults to None.

    Returns:
        torch.Tensor: The computed RMS value.
    """
    log.debug("Computing signal RMS.")
    return torch.sqrt(compute_power(signal_components=signal_components, vad=vad))


def normalize_components(
    signal_components: torch.Tensor,
    vad: torch.Tensor | None = None,
    norm_power: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes the signal components to a target power level.

    Args:
        signal_components (torch.Tensor): The signal to normalize.
        vad (torch.Tensor | None): Optional VAD to only consider active regions. Defaults to None.
        norm_power (float): The target power level to normalize to. Defaults to 1e-2.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the normalized signal and the normalization factors.
    """
    log.debug(f"Normalizing signal components to power {norm_power}.")
    norm_factors = norm_power / compute_rms(signal_components, vad)
    return signal_components * norm_factors, norm_factors


def computeSNR(
    signal: torch.Tensor, noise: torch.Tensor, vad: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes the Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        signal (torch.Tensor): The target signal.
        noise (torch.Tensor): The noise signal.
        vad (torch.Tensor | None): Optional VAD mask to restrict computation to active regions. Defaults to None.

    Returns:
        torch.Tensor: The SNR in dB.
    """
    log.debug("Computing SNR.")
    return pow2db(
        compute_power(signal_components=signal, vad=vad)
        / compute_power(signal_components=noise, vad=vad)
    )