# %% Coordinate Transformations

import logging
from typing import Union, Tuple, Optional, List

import torch

from .complex_angles import atan2, hermitian_angle
from ..tensor_ops import check_broadcastable

log = logging.getLogger(__name__)


def spherical2cartesian(spherical_coords: torch.Tensor) -> torch.Tensor:
    """
    Convert n-dimensional spherical coordinates to Cartesian coordinates.

    Args:
        spherical_coords (torch.Tensor): Tensor of shape [..., M, N] where M is the vector dimension
                                         (including the radial and angular coordinates) and N is the broadcast dimension.

    Returns:
        torch.Tensor: Cartesian coordinates tensor of shape [..., M, N].
    """
    # Split the spherical coordinates into radial and angular parts
    radius = spherical_coords[..., :1, :]  # Radial  coordinate  [...,   1, N]
    angles = spherical_coords[..., 1:, :]  # Angular coordinates [..., M-1, N]

    # Compute the remaining Cartesian coordinates
    sin_product = torch.nn.functional.pad(
        torch.cumprod(torch.sin(angles), dim=-2), (0, 0, 1, 0), "constant", 1
    )
    cos_terms = torch.nn.functional.pad(torch.cos(angles), (0, 0, 0, 1), "constant", 1)

    return radius * sin_product * cos_terms


def cartesian2spherical(
    cartesian_coords: torch.Tensor, shift_interval=(-torch.pi, torch.pi)
) -> torch.Tensor:
    """
    Convert n-dimensional Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coords (torch.Tensor): Tensor of shape [..., M, :] where M is the vector dimension

    Returns:
        torch.Tensor: Spherical coordinates tensor of shape [..., M, :].
    """

    # Calculate the radial distance (r)
    radius = (
        (cartesian_coords.flip(dims=(-2,)) ** 2).cumsum(dim=-2).flip(dims=(-2,)).sqrt()
    )
    radius[..., -1, :] = cartesian_coords[..., -1, :]

    # Compute the remaining angular coordinates
    angles = moduloshift(
        atan2(radius[..., 1:, :], cartesian_coords[..., :-1, :]),
        real_interval=shift_interval,
    )

    # Concatenate the radial distance and angular coordinates
    return torch.cat([radius[..., :1, :], angles], dim=-2)


# %% Interpolation


def slerp(
    vector_start: torch.Tensor,
    vector_end: torch.Tensor,
    interpolation_factor: torch.Tensor,
) -> torch.Tensor:
    """
    Perform spherical linear interpolation (SLERP) between two vectors, supporting multidimensional tensors.

    This function supports multi-dimensional tensors where all but the last two dimensions are broadcast dimensions.
    The second-to-last dimension represents the vectors, and the last dimension is a broadcast dimension.
    The interpolation factor `t` must have the same broadcastable dimensions, with the second-to-last dimension being 1.

    Args:
        vector_start (torch.Tensor): Starting vectors of shape (..., M, N), where M is the vector dimension and N is the broadcast dimension.
        vector_end (torch.Tensor): End vectors of shape (..., M, N), same shape as vector_start.
        t (torch.Tensor): Interpolation factor(s), shape should be broadcastable to (..., 1, N), where the second-to-last dimension is 1.

    Returns:
        torch.Tensor: The interpolated vectors of shape (..., M, N).
    """
    # Check if `interpolation_factor` is 1D
    if interpolation_factor.dim() == 1:
        # If it's 1D, the condition is satisfied
        second_to_last_dim_is_valid = True
    else:
        # Otherwise, check if the second-to-last dimension is 1
        second_to_last_dim_is_valid = interpolation_factor.shape[-2] == 1

    # Assert that the second-to-last dimension is valid
    assert (
        second_to_last_dim_is_valid
    ), "The second-to-last dimension of `interpolation_factor` must be 1, or it must be a 1D tensor."

    # Ensure broadcast compatibility between vector_start, vector_end, and interpolation_factor
    assert check_broadcastable(
        vector_start.shape, vector_end.shape, interpolation_factor.shape
    ), "The dimensions of `vector_start`, `vector_end`, and `interpolation_factor` must align for broadcasting."

    # Normalize the input vectors along the vector dimension (second-to-last dimension)
    vector_start = vector_start / torch.linalg.vector_norm(
        vector_start, dim=-2, keepdim=True
    )
    vector_end = vector_end / torch.linalg.vector_norm(vector_end, dim=-2, keepdim=True)

    # Compute the dot product between the vectors along the vector dimension
    dot_product = torch.sum(vector_start * vector_end, dim=-2, keepdim=True)

    # Compute the angle between the vectors
    theta = hermitian_angle(vector_start, vector_end)

    # Avoid division by zero by checking for small angles (theta close to 0)
    sin_theta = torch.sin(theta)

    abs_sin_theta = sin_theta.abs()

    # Create a mask for very small angles where sin(theta) is close to zero
    small_angle_mask = torch.isclose(
        abs_sin_theta,
        torch.tensor(0.0, device=theta.device, dtype=sin_theta.abs().dtype),
    )

    # Perform SLERP where sin(theta) is not close to zero
    v_interp = (
        torch.sin((1 - interpolation_factor) * theta) / sin_theta
    ) * vector_start + (
        torch.sin(interpolation_factor * theta) / sin_theta
    ) * vector_end

    # For small angles (sin(theta) close to zero), perform linear interpolation instead
    v_interp = torch.where(
        small_angle_mask,
        (1 - interpolation_factor) * vector_start + interpolation_factor * vector_end,
        v_interp,
    )

    return v_interp


# W.-K. Ma et al., A signal processing perspective on hyperspectral unmixing: Insights from remote sensing. IEEE Signal Process Mag 31(1), 67–81 (2014)
def successive_projections(X: torch.Tensor, num_selections: int) -> List[int]:
    """
    Performs the Successive Projections Algorithm (SPA) to select a set of
    representative features from the dataset.

    Args:
        X (torch.Tensor): Input data matrix of shape (m, n) where each column is a feature vector.
        num_selections (int): Number of indices to select.

    Returns:
        list: Indices of the selected columns.
    """
    # Check dimensions
    m, n = X.shape

    # Normalize columns of X for numerical stability
    X = X / (X.norm(dim=0) + 1e-10)

    # Initialize an empty list to store selected indices
    selected_indices = []

    # Initialize residual matrix (initially, it's the entire matrix X)
    residual = X.clone()

    for _ in range(num_selections):
        # Compute the norms of each column in the residual matrix
        norms = residual.norm(dim=0)

        # Find the index of the column with the maximum norm
        max_index = norms.argmax().item()
        selected_indices.append(max_index)

        # Extract the selected vector and normalize it
        selected_vector = X[:, max_index].unsqueeze(1)

        # Project all columns of X onto the orthogonal complement of the selected vector
        projection = (
            selected_vector
            @ (selected_vector.T @ residual)
            / (selected_vector.T @ selected_vector + 1e-10)
        )
        residual = residual - projection

    return selected_indices


def moduloshift(
    vals: Union[torch.Tensor, float, complex],
    real_interval: Optional[Tuple[float, float]] = None,
    imag_interval: Optional[Tuple[float, float]] = None,
) -> Union[torch.Tensor, float, complex]:
    """
    Shift values into specified intervals using modulo arithmetic.
    Supports both real and complex inputs.

    Args:
        vals (torch.Tensor or float or complex): Tensor of values to be shifted.
        real_interval (tuple): A tuple (min_val, max_val) specifying the target interval for the real part.
        imag_interval (tuple, optional): A tuple (min_val, max_val) specifying the target interval for the imaginary part.
                                          If None, no shift is performed (for real / imaginary independently).

    Returns:
        torch.Tensor or float or complex: Tensor of values shifted into the specified intervals.
    """

    def shift(vals, interval):
        min_val, max_val = interval
        range_val = max_val - min_val

        # Perform the shift
        shifted_vals = (vals - min_val) % range_val + min_val

        # Handle edge case where shifted_vals might equal max_val (should wrap to min_val)
        shifted_vals = torch.where(shifted_vals == max_val, min_val, shifted_vals)

        return shifted_vals

    if torch.is_complex(vals):

        if real_interval is None:
            shifted_real = torch.real(vals)
        else:
            shifted_real = shift(torch.real(vals), real_interval)

        if imag_interval is None:
            shifted_imag = torch.imag(vals)
        else:
            shifted_imag = shift(torch.imag(vals), imag_interval)

        # Recombine into a complex tensor
        shifted_vals = torch.complex(shifted_real, shifted_imag)
    else:
        # Apply the shift to real values directly
        shifted_vals = shift(vals, real_interval)

    return shifted_vals