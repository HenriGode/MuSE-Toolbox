import logging

import numpy as np
import torch

from muse_toolbox.utils.math.matrix_ops import (
    makeHermitian,
    makeMatrixUnitNorm,
    makeVectorUnitNorm,
    regularize,
)

log = logging.getLogger(__name__)


def Beamformer(
    covMat2min: torch.Tensor,  # (..., F, T, M, M)
    RTFs4constraints: torch.Tensor,  # (..., F, M, K)
    gains: torch.Tensor,  # (..., K, 1)
    signal: torch.Tensor | None = None,  # (..., F, M, T)
) -> torch.Tensor:
    """
    Computes a beamformer using a constrained variance minimization approach.

    Args:
        covMat2min (torch.Tensor): Covariance matrix to minimize, of shape (..., F, T, M, M).
        RTFs4constraints (torch.Tensor): Relative Transfer Functions (RTFs) for constraints, of shape (..., F, M, K).
        gains (torch.Tensor): Target gains for the constraints, of shape (..., K, 1).
        signal (Optional[torch.Tensor]): Optional input signal to apply the beamformer to, of shape (..., F, M, T). Defaults to None.

    Returns:
        torch.Tensor: The beamformer weights if `signal` is None, otherwise the beamformed signal.
    """
    log.debug("Computing Beamformer weights.")
    RTFs4constraints = makeVectorUnitNorm(RTFs4constraints)
    RinvC = torch.linalg.solve(
        makeHermitian(regularize(makeMatrixUnitNorm(covMat2min), 1e-6)),
        RTFs4constraints,
    )
    covMat2min = torch.empty_like(covMat2min)
    beamformer = RinvC @ torch.linalg.solve(
        regularize(RTFs4constraints.mH @ RinvC, 1e-6),
        torch.diag_embed(gains[..., 0]) @ RTFs4constraints.mH,
    )
    
    # Free memory
    RTFs4constraints = torch.empty_like(RTFs4constraints)
    gains = torch.empty_like(gains)
    RinvC = None

    if signal is None:
        return beamformer
    else:
        if True:  # TODO: memory(beamformer) < 1024**3 check was bypassed here originally
            return beamformer.mH @ signal
        else:
            return torch.cat(
                [
                    beamformer.mH[..., [frame], :, :] @ signal[..., [frame], :, :]
                    for frame in range(beamformer.shape[-3])
                ],
                dim=-3,
            )


def calc_beam_pattern(
    W: np.ndarray, fs: int, mic_loc: np.ndarray, degrees: np.ndarray, c: float = 343.0
) -> np.ndarray:
    """
    Calculates the directivity pattern of a microphone array.

    Args:
        W (np.ndarray): Demixing matrices of shape (n_freq, n_out, n_chan).
        fs (int): Sampling frequency in Hz.
        mic_loc (np.ndarray): The locations of microphones of shape (n_chan, 3).
        degrees (np.ndarray): The degrees of beam patterns in degrees.
        c (float): The speed of sound. Defaults to 343.0.

    Returns:
        np.ndarray: Beam pattern of the microphone array in decibel, of shape (n_freq, n_out, n_deg).
    """
    log.debug("Calculating beam pattern.")
    n_deg = degrees.size
    n_freq, n_out, _ = W.shape
    n_fft = (n_freq - 1) * 2

    # Defines the origin of degree from x-axis as counter clock-wise
    rad = np.deg2rad(degrees)  # (n_deg,)
    unit_vec = np.array([np.cos(rad), np.sin(rad), np.zeros(n_deg)])  # (3, n_deg)
    delay = mic_loc @ unit_vec / c  # (n_chan, n_deg)

    omega = 2 * np.pi * np.arange(n_freq) * fs / n_fft  # (n_freq,)

    # (n_freq, n_chan, n_deg)
    phase = np.exp(1j * omega[:, None, None] * delay[None, :, :])

    # demixing matrix * amplitude (1) * phase (simulated)
    beam_pattern = W @ phase

    return 20 * np.log10(np.abs(beam_pattern))