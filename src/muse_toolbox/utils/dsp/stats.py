import logging
from typing import Optional

import torch

from muse_toolbox.utils.math.covariance import (
    covariance_SCM,
    make2covariance_matrix_rel_lower_bound,
)
from muse_toolbox.utils.math.matrix_ops import (
    makeHermitian,
    mytorch_eigvalsh,
    regularize,
    vec2diagMat,
)
from muse_toolbox.utils.math.windowing import (
    exp_windowing_conv,
    exp_windowing_recursive,
    windowing,
)

log = logging.getLogger(__name__)


def smoothCovarianceMatrix(
    stft_signal: torch.Tensor,
    smoothing_factor: float,
    init_cov: Optional[torch.Tensor] = None,
    init_smoothing_factor: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute the smoothed covariance matrix using recursive exponential windowing.

    Args:
        stft_signal (torch.Tensor): STFT signal of shape (..., F, M, T).
        smoothing_factor (float): Smoothing factor for exponential windowing.
        init_cov (Optional[torch.Tensor]): Initial covariance matrix for warm start. Shape (..., F, 1, M, M).
        init_smoothing_factor (Optional[float]): Smoothing factor used for initial covariance.

    Returns:
        torch.Tensor: Smoothed covariance matrix of shape (..., F, T, M, M).
    """
    log.debug("Computing smoothed covariance matrix.")
    instantaneous_cov_mat = covariance_SCM(stft_signal.transpose(-2, -1)[..., None])

    if init_cov is not None:
        if init_smoothing_factor is None:
            instantaneous_cov_mat = torch.cat([init_cov, instantaneous_cov_mat], dim=-3)
        else:
            gamma = init_smoothing_factor / smoothing_factor
            init_cov_weighted = (
                gamma * init_cov + (1 - gamma) * instantaneous_cov_mat[..., :1, :, :]
            )
            instantaneous_cov_mat = torch.cat(
                [init_cov_weighted, instantaneous_cov_mat], dim=-3
            )
    # else: # TODO: Regularization
    #     num_channels = instantaneous_cov_mat.shape[-2]
    #     instantaneous_cov_mat[..., :num_channels, :, :] = regularize(
    #         instantaneous_cov_mat[..., :num_channels, :, :], reg_factor=1e-1
    #     )

    smoothCov = exp_windowing_recursive(
        data=instantaneous_cov_mat,
        smoothing_factor=smoothing_factor,
        dim=-3,
    )

    if init_cov is not None:
        smoothCov = smoothCov[..., 1:, :, :]

    return make2covariance_matrix_rel_lower_bound(smoothCov, 1e-6)


def windowedCovarianceMatrix(
    stft_signal: torch.Tensor, window: torch.Tensor
) -> torch.Tensor:
    """
    Compute the windowed covariance matrix using a sliding window.

    Args:
        stft_signal (torch.Tensor): STFT signal.
        window (torch.Tensor): Window function to apply over time frames.

    Returns:
        torch.Tensor: The windowed covariance matrix.
    """
    log.debug("Computing windowed covariance matrix.")
    instantaneous_cov_mat = covariance_SCM(stft_signal.transpose(-2, -1)[..., None])
    num_channels = instantaneous_cov_mat.shape[-2]
    # instantaneous_cov_mat[..., :num_channels, :, :] = regularize(
    #     instantaneous_cov_mat[..., :num_channels, :, :], reg_factor=1e-1
    # )
    return make2covariance_matrix_rel_lower_bound(
        windowing(
            data=instantaneous_cov_mat,
            window=window.to(stft_signal.device),
            dim=-3,
        ),
        1e-6,
    )


def smoothCovarianceMatrix_conv(
    stft_signal: torch.Tensor, smoothing_factor: float
) -> torch.Tensor:
    """
    Compute the smoothed covariance matrix using 1D convolution (non-recursive).

    Args:
        stft_signal (torch.Tensor): STFT signal.
        smoothing_factor (float): Smoothing factor for the exponential window.

    Returns:
        torch.Tensor: The smoothed covariance matrix.
    """
    log.debug("Computing smoothed covariance matrix via convolution.")
    instantaneous_cov_mat = covariance_SCM(stft_signal.transpose(-2, -1)[..., None])
    num_channels = instantaneous_cov_mat.shape[-2]
    instantaneous_cov_mat[..., :num_channels, :, :] = regularize(
        instantaneous_cov_mat[..., :num_channels, :, :], reg_factor=1e-1
    )
    return makeHermitian(
        exp_windowing_conv(
            data=instantaneous_cov_mat,
            smoothing_factor=smoothing_factor,
            dim=-3,
        )
    )


def coherenceMatrix(covMat: torch.Tensor) -> torch.Tensor:
    """
    Computes the spatial coherence matrix from a covariance matrix.

    Args:
        covMat (torch.Tensor): The covariance matrix.

    Returns:
        torch.Tensor: The corresponding coherence matrix.
    """
    log.debug("Computing coherence matrix.")
    Dsqrtinv = vec2diagMat(1 / covMat.diagonal(dim1=-2, dim2=-1).sqrt()[..., None])
    return makeHermitian(Dsqrtinv @ covMat @ Dsqrtinv)


def gmsc(
    coherenceMat: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Generalized Magnitude Squared Coherence (GMSC) from a coherence matrix.

    Args:
        coherenceMat (torch.Tensor): The coherence matrix.

    Returns:
        torch.Tensor: The GMSC values.
    """
    log.debug("Computing GMSC.")
    eigvals = mytorch_eigvalsh(coherenceMat)
    return ((eigvals[..., [-1]][..., None] - 1) / (coherenceMat.shape[-1] - 1)) ** 2


def wdo(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes the W-disjoint orthogonality measure.

    Outputs a value between 1 and 0, where 1 is completely orthogonal and 0 is not orthogonal.
    Inputs are typically spectrograms (e.g. STFT data).

    Args:
        A (torch.Tensor): First signal representation.
        B (torch.Tensor): Second signal representation.

    Returns:
        torch.Tensor: The computed W-disjoint orthogonality measure.
    """
    log.debug("Computing W-disjoint orthogonality measure (WDO).")
    return 1 - (A * B).abs().sum() / torch.max(A.abs(), B.abs()).sum()