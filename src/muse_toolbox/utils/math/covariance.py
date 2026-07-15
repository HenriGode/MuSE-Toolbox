import logging

import torch

from .matrix_ops import evd2matrix_h, makeHermitian, mytorch_eigh, trace
from .stats import deviation, wmean

log = logging.getLogger(__name__)


def covariance_SCM(data: torch.Tensor) -> torch.Tensor:
    """
    Estimates the covariance matrix of data using the Sample Covariance Matrix (SCM) method.

    Args:
        data (torch.Tensor): Audio STFT data with dimensions (..., channels, frames).

    Returns:
        torch.Tensor: Covariance matrix for each batch, shape (..., channels, channels).
    """
    if False:  # data.shape[-2] >= data.shape[-1]:
        return make2covariance_matrix(crossCovariance_SCM(data, data))
    else:
        return crossCovariance_SCM(data, data)


def growing_average_SCM(data: torch.Tensor) -> torch.Tensor:
    """
    Computes the recursive/growing average Sample Covariance Matrix (SCM).

    For an input signal X of T frames, the output at index t is:
    R[t] = (1 / (t+1)) * Sum_{i=0}^{t} (x[i] @ x[i]^H)

    Args:
        data (torch.Tensor): Input audio data, typically STFT.
            Expected shape: (..., channels, frames) or (..., frames, channels) depending on dim.
            Standard usage here assumes (..., channels, frames).

    Returns:
        torch.Tensor: A tensor of shape (..., frames, channels, channels) containing
            the cumulative average covariance up to that frame.
    """

    # (..., T, M, M)
    R_inst = covariance_SCM(data.transpose(-1, -2)[..., None])

    # 3. Cumulative Sum along the Time dimension (which is now -3 due to M, M at end)
    # R_sum: (..., T, M, M)
    R_sum = torch.cumsum(R_inst, dim=-3)

    # 4. Normalize by count (1, 2, 3, ..., T)
    T = data.shape[-1]
    counts = torch.arange(1, T + 1, device=data.device, dtype=data.real.dtype)

    # Reshape counts for broadcasting: (T, 1, 1)
    counts = counts.view(-1, 1, 1)

    R_avg = R_sum / counts

    return R_avg


def weighted_SCM(data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Estimate the covariance matrix of data using
    using weighted / windowed Sample Covariance Matrix (SCM) method.

    Parameters:
        data: torch.Tensor
            Input tensor of shape (..., channels, frames).
        weights: torch.Tensor
            Weights tensor of shape (..., frames).
            Usually just (frames)
    Returns:
        torch.Tensor
            Covariance matrix for each batch, shape (..., channels, channels).
    """
    # Compute instantaneous SCM for each time frame with dimensions (..., frames, channels, channels,)
    instant_SCM = covariance_SCM(data.transpose(-1, -2)[..., None])
    return make2covariance_matrix_rel_lower_bound(
        wmean(instant_SCM, weights=weights[..., None, None], dims=-3).squeeze(-3), 1e-6
    )


def crossCovariance_SCM(
    data1: torch.Tensor,
    data2: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the cross-covariance matrix between two data tensors using the Sample Covariance Matrix (SCM) method.

    Args:
        data1 (torch.Tensor): First input tensor of shape (..., channels, frames).
        data2 (torch.Tensor): Second input tensor of shape (..., channels, frames).

    Returns:
        torch.Tensor: Cross-covariance matrix of shape (..., channels, channels).
    """
    num_samples = data1.shape[-1]
    return data1 @ data2.mH / num_samples


def covariance_Tyler(
    data: torch.Tensor, max_iters: int = 1000, tol: float = 1e-8
) -> torch.Tensor:
    """
    Tyler's Estimator for Cross-Covariance.

    Parameters:
        data: torch.Tensor
            First input tensor of shape (..., channels, frames).
        max_iters: int
            Maximum number of iterations for convergence.
        tol: float
            Convergence tolerance for Tyler's estimator.

    Returns:
        torch.Tensor
            Covariance matrix for each batch, shape (..., channels, channels).
    """

    # Initialize cross-covariance matrices for each batch
    cov = covariance_SCM(
        data
    )  # torch.eye(data.shape[-2], dtype=data.dtype, device=data.device)#covariance_SCM(data)
    power = trace(cov)[..., None, :, :]
    # cov = torch.eye(cov.shape[-2], dtype=data.dtype, device=data.device).expand(cov.shape)#covariance_SCM(data)

    # Get dimensionality d and number of samples n
    d, n = data.shape[-2:]
    a = max(0, d / n - 1)
    S = cov[..., None, :, :]
    x = data.mT[..., None]
    mask = torch.ones_like(S[..., 0, 0, 0], dtype=torch.bool, device=S.device)

    for iter in range(max_iters):

        old_S = S.clone()

        S[mask] = d / ((1 + a) * n) * torch.sum(
            (x[mask] @ x[mask].mH)
            / (x[mask].mH @ torch.linalg.solve(S[mask], x[mask], left=True)),
            dim=-3,
            keepdim=True,
        ) + a / (1 + a) * torch.eye(d, device=data.device, dtype=data.dtype)
        S[mask] /= trace(S[mask])

        criterion = deviation(old_S, S, relative=True)[..., 0, 0, 0]
        mask = criterion > tol
        log.info(
            f"Iteration: {iter}, Converged: {(~mask).sum().item()}/{mask.numel()}, max change: {criterion.max().item():.3e}"
        )
        if not mask.any():
            break

    return make2covariance_matrix(S / trace(S) * power)[..., 0, :, :]


def make2covariance_matrix(
    matrix: torch.Tensor, reg_factor: float = 1e-6
) -> torch.Tensor:
    """
    Forces a matrix to be a valid covariance matrix (Hermitian and positive semi-definite).
    Negative eigenvalues are replaced based on a regularization factor relative to the maximum eigenvalue.

    Args:
        matrix (torch.Tensor): Input square matrix.
        reg_factor (float, optional): Regularization factor. Defaults to 1e-6.

    Returns:
        torch.Tensor: Valid covariance matrix.
    """
    matrix = makeHermitian(matrix)

    eigvals, eigvecs = mytorch_eigh(matrix)

    # Find the maximum eigenvalue for each set of eigenvalues (last dimension)
    max_eigvals = torch.max(eigvals, dim=-1, keepdim=True)[0]

    # Create a mask where the eigenvalues are negative
    negative_mask = eigvals < 0

    # # find the minimal positive eigenvalue for each set of eigenvalues (last dimension)
    # min_positive_eigvals = torch.min(torch.where(eigvals > 0, eigvals, torch.tensor(float('inf'), device=eigvals.device)), dim=-1, keepdim=True)[0]

    # Replace the negative eigenvalues with reg_factor * max_eigval for each set
    # modified_eigvals = torch.where(negative_mask, torch.min(reg_factor * abs(max_eigvals), min_positive_eigvals), eigvals)
    modified_eigvals = torch.where(
        negative_mask, reg_factor * abs(max_eigvals), eigvals
    )

    return evd2matrix_h(eigvals=modified_eigvals, eigvecs=eigvecs)


def make2covariance_matrix_rel_lower_bound(
    matrix: torch.Tensor, reg_factor: float = 1e-6
) -> torch.Tensor:
    """
    Forces a matrix to be a valid covariance matrix by enforcing a relative lower bound
    on all eigenvalues (relative to the maximum eigenvalue).

    Args:
        matrix (torch.Tensor): Input square matrix.
        reg_factor (float, optional): Regularization factor. Defaults to 1e-6.

    Returns:
        torch.Tensor: Valid covariance matrix.
    """
    matrix = makeHermitian(matrix)

    eigvals, eigvecs = mytorch_eigh(matrix)

    # Find the maximum eigenvalue for each set of eigenvalues (last dimension)
    max_eigvals = torch.max(eigvals, dim=-1, keepdim=True)[0]

    # Calculate the dynamic minimum threshold per matrix
    min_threshold = reg_factor * abs(max_eigvals)

    # Replace any eigenvalue smaller than the threshold with the threshold
    modified_eigvals = torch.where(eigvals < min_threshold, min_threshold, eigvals)

    return evd2matrix_h(eigvals=modified_eigvals, eigvecs=eigvecs)
