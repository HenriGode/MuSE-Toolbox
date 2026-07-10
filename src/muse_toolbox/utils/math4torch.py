# %%
import os
import torch
from typing import Union, Optional, Callable
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# %% Trigonometry Functions


def atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Extended atan2 function for complex-valued inputs.

    Args:
        y (torch.Tensor): Complex-valued tensor representing the numerator.
        x (torch.Tensor): Complex-valued tensor representing the denominator.

    Returns:
        torch.Tensor: Complex-valued tensor representing the extended atan2 result.
    """
    return -1j * torch.log((x + 1j * y) / torch.sqrt(x**2 + y**2))


def atan3(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Extended atan2 function for complex-valued inputs.

    Args:
        y (torch.Tensor): Complex-valued tensor representing the numerator.
        x (torch.Tensor): Complex-valued tensor representing the denominator.

    Returns:
        torch.Tensor: Complex-valued tensor representing the extended atan2 result.
    """
    r = (
        torch.sqrt(abs(x) ** 2 + abs(y) ** 2)
        * torch.sqrt(x**2 + y**2)
        / abs(torch.sqrt(x**2 + y**2))
    )
    # return -1j * torch.log((x + 1j * y) / (torch.sqrt(abs(x)**2 + abs(y)**2) * torch.exp(1j * torch.angle(torch.sqrt(x**2 + y**2)))))
    return -1j * torch.log((x + 1j * y) / r)


def quadrant(z: torch.Tensor) -> torch.Tensor:
    """
    Determine the quadrant of a complex number.

    Args:
        z (torch.Tensor): A complex tensor of any shape.

    Returns:
        torch.Tensor: An integer tensor of the same shape as z, where each element
                      is the quadrant (1, 2, 3, or 4) of the corresponding element in z.
    """

    # Get the phase angle of the complex number
    # Map the angle to a quadrant number
    return (torch.angle(z) // (torch.pi / 2)).int() % 4 + 1


# %% Aggregation Operations


def wmean(
    tensor: torch.Tensor,
    dims: int | tuple[int, ...],
    weights: torch.Tensor | None = None,
    keepdim: bool = True,
) -> torch.Tensor:
    if weights == None:
        return (tensor).mean(dim=dims, keepdim=keepdim)
    else:
        return (tensor * weights).sum(dim=dims, keepdim=keepdim) / weights.sum(
            dim=dims, keepdim=keepdim
        )


def norm_by_sum(tensor: torch.Tensor, dims: int, keepdim: bool = True) -> torch.Tensor:
    denominator = tensor.sum(dim=dims, keepdim=keepdim)
    return tensor / denominator


def windowing(data: torch.Tensor, window: torch.Tensor, dim: int = -1) -> torch.Tensor:
    datatype = data.dtype
    data_tmp = data.swapaxes(-1, dim).to(
        torch.complex128 if data.is_complex() else torch.float64
    )
    data_len = data_tmp.shape[-1]
    index = (data.ndim - 1) * (None,) + (slice(None),)
    window_tmp = window.squeeze()[index]

    # Compute normalization
    normalization = torch.cat(
        [
            torch.cumsum(window_tmp[..., :data_len], dim=-1),
            torch.sum(window_tmp, dim=-1, keepdim=True)
            * torch.ones(
                max([data_len - window_tmp.shape[-1], 0]), device=window.device
            ),
        ],
        dim=-1,
    )

    # Perform FFT-based convolution
    windowed_Data_tmp = (
        torchaudio_functional_fftconvolve_complex(data_tmp, window_tmp, mode="full")[
            ..., :data_len
        ]
        / normalization
    )

    # Convert back to original dtype and shape
    windowedData = windowed_Data_tmp.to(dtype=datatype).swapaxes(-1, dim)

    return windowedData


def windowing_conv(
    data: torch.Tensor, window: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Performs windowing (running weighted average) using a standard 1D convolution.

    This function is an alternative to the FFT-based windowing and may be more
    numerically stable for certain inputs, though potentially slower.
    """
    datatype = data.dtype
    is_complex = data.is_complex()
    # Use float64 for precision during intermediate calculations
    compute_dtype = torch.float64

    data_tmp = data.swapaxes(-1, dim).to(
        torch.complex128 if is_complex else compute_dtype
    )
    data_len = data_tmp.shape[-1]

    # Prepare window for convolution
    index = (data.ndim - 1) * (None,) + (slice(None),)
    window_tmp = window.squeeze()[index].to(data_tmp.dtype)
    window_len = window_tmp.shape[-1]

    # Compute normalization factor for the running average
    normalization = torch.cat(
        [
            torch.cumsum(window_tmp[..., :data_len], dim=-1),
            torch.sum(window_tmp, dim=-1, keepdim=True)
            * torch.ones(
                max([data_len - window_tmp.shape[-1], 0]),
                device=window.device,
                dtype=window_tmp.dtype,
            ),
        ],
        dim=-1,
    )

    # Reshape for conv1d: (..., L) -> (B, 1, L) where B is the flattened batch size
    original_batch_shape = data_tmp.shape[:-1]
    flat_data = data_tmp.reshape(-1, 1, data_len)

    # To perform convolution, the kernel (window) must be flipped.
    # conv1d expects kernel shape (C_out, C_in, K)
    flipped_window = torch.flip(window_tmp, dims=[-1]).reshape(1, 1, window_len)

    # Padding for 'full' convolution mode
    padding = window_len - 1

    if is_complex:
        # Perform complex convolution using four real convolutions
        # (a+ib) * (c+id) = (ac-bd) + i(ad+bc)
        data_real = flat_data.real.to(compute_dtype)
        data_imag = flat_data.imag.to(compute_dtype)
        win_real = flipped_window.real.to(compute_dtype)
        win_imag = flipped_window.imag.to(compute_dtype)

        conv_ac = torch.nn.functional.conv1d(data_real, win_real, padding=padding)
        conv_bd = torch.nn.functional.conv1d(data_imag, win_imag, padding=padding)
        conv_ad = torch.nn.functional.conv1d(data_real, win_imag, padding=padding)
        conv_bc = torch.nn.functional.conv1d(data_imag, win_real, padding=padding)

        conv_res_real = conv_ac - conv_bd
        conv_res_imag = conv_ad + conv_bc
        conv_res = torch.complex(conv_res_real, conv_res_imag)
    else:
        conv_res = torch.nn.functional.conv1d(
            flat_data.to(compute_dtype),
            flipped_window.to(compute_dtype),
            padding=padding,
        )

    # Reshape back to original batch shape and truncate to original length
    windowed_Data_tmp = conv_res.reshape(*original_batch_shape, -1)[..., :data_len]

    # Apply normalization and convert back to original dtype and shape
    windowedData = (windowed_Data_tmp / normalization).to(datatype).swapaxes(-1, dim)

    return windowedData


def exp_windowing(
    data: torch.Tensor, smoothing_factor: float, dim: int = -1
) -> torch.Tensor:
    return windowing(
        data=data,
        window=torch.tensor(
            smoothing_factor, dtype=get_real_dtype(data), device=data.device
        )
        ** (
            torch.arange(
                0, data.shape[dim], dtype=get_real_dtype(data), device=data.device
            )
        ),
        dim=dim,
    )


def exp_windowing_conv(
    data: torch.Tensor, smoothing_factor: float, dim: int = -1
) -> torch.Tensor:
    return windowing_conv(
        data=data,
        window=torch.tensor(
            smoothing_factor, dtype=get_real_dtype(data), device=data.device
        )
        ** (
            torch.arange(
                0, data.shape[dim], dtype=get_real_dtype(data), device=data.device
            )
        ),
        dim=dim,
    )


def exp_windowing_recursive(
    data: torch.Tensor, smoothing_factor: float, dim: int = -1
) -> torch.Tensor:
    """Applies exponential windowing recursively along a specified dimension.

    Args:
        data (torch.Tensor): Input tensor to be windowed.
        smoothing_factor (float): Smoothing factor for the exponential window.
        dim (int, optional): Dimension along which to apply the windowing. Defaults to -1.

    Returns:
        torch.Tensor: Tensor after applying exponential windowing.
    """
    length = data.shape[dim]
    windowed_data = torch.zeros_like(data)
    for i in range(length):
        if i == 0:
            windowed_data.select(dim, i).copy_(data.select(dim, i))
        else:
            windowed_data.select(dim, i).copy_(
                smoothing_factor * windowed_data.select(dim, i - 1)
                + (1 - smoothing_factor) * data.select(dim, i)
            )

    return windowed_data


def exp_windowing_recursive_changing_factor(
    data: torch.Tensor, smoothing_factor: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Applies exponential windowing recursively along a specified dimension.

    Args:
        data (torch.Tensor): Input tensor to be windowed.
        smoothing_factor (torch.Tensor): Smoothing factors for the exponential window.
                                         Can be changing along the specified dimension.
                                         Must be 1D tensor with length equal to data.shape[dim].
        dim (int, optional): Dimension along which to apply the windowing. Defaults to -1.

    Returns:
        torch.Tensor: Tensor after applying exponential windowing.
    """
    assert smoothing_factor.dim() == 1, "smoothing_factor must be a 1D tensor"
    assert (
        smoothing_factor.shape[-1] == data.shape[dim]
    ), "smoothing_factor must be a 1D tensor with length equal to data.shape[dim]"
    length = data.shape[dim]
    windowed_data = torch.zeros_like(data)
    for i in range(length):
        if i == 0:
            windowed_data.select(dim, i).copy_(
                smoothing_factor[i] * torch.zeros_like(data.select(dim, i))
                + (1 - smoothing_factor[i]) * data.select(dim, i)
            )
        else:
            windowed_data.select(dim, i).copy_(
                smoothing_factor[i] * windowed_data.select(dim, i - 1)
                + (1 - smoothing_factor[i]) * data.select(dim, i)
            )

    return windowed_data


# %% Comparison


def deviation(
    original: torch.Tensor,
    comparator: torch.Tensor,
    dim=(-1, -2),
    relative: bool = True,
) -> torch.Tensor:
    if relative:
        return torch.linalg.norm(
            comparator - original, dim=dim, keepdim=True
        ) / torch.linalg.norm(original, dim=dim, keepdim=True)
    else:
        return torch.linalg.norm(comparator - original, dim=dim, keepdim=True)


# %% Conversions


def db2amp(db: torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, db / 20.0)


def db2pow(db: torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, db / 10.0)


def amp2db(amp: torch.Tensor) -> torch.Tensor:
    return 20 * torch.log10(amp)


def pow2db(power: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(power)


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    return rad / torch.pi * 180


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    return deg / 180 * torch.pi


# def timeConstant2smoothingFactor(time_constant: float, time_interval: float) -> float:
#     return np.exp(-time_interval / time_constant)


def moduloshift(vals, real_interval=None, imag_interval=None):
    """
    Shift values into specified intervals using modulo arithmetic.
    Supports both real and complex inputs.

    Args:
        vals (torch.Tensor or float or complex): Tensor of values to be shifted.
        real_interval (tuple): A tuple (min_val, max_val) specifying the target interval for the real part.
        imag_interval (tuple, optional): A tuple (min_val, max_val) specifying the target interval for the imaginary part.
                                          If None, the real_interval is used for both real and imaginary parts.

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


# %% Coordinate Transformations


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


# %% Generate random numbers


def randdir(
    *size, device: Union[torch.device, str] = "cuda:0", dtype=torch.complex128
) -> torch.Tensor:
    return makeVectorUnitNorm(torch.randn(*size, device=device, dtype=dtype))


def randdir_orthogonal2vec(vector: torch.Tensor, N: int) -> torch.Tensor:
    assert vector.shape[-1] == 1, "The input vector must have a shape of (..., 1)."
    return torch.cat(
        [orthogonal_complement(vector), vector], dim=-1
    ) @ torch.nn.functional.pad(
        randdir((vector.shape[-2] - 1, N), device=vector.device, dtype=vector.dtype),
        (0, 0, 0, 1),
        mode="constant",
        value=0,
    )


def sample_complex_multivariate(mean, covariance, relation, num_samples):
    """
    Generate complex-valued vectors from a multivariate normal distribution.

    Parameters:
        mean (torch.Tensor): Mean vector (complex-valued), shape (n,).
        covariance (torch.Tensor): Covariance matrix (Hermitian), shape (n, n).
        relation (torch.Tensor): Relation matrix (symmetric, complex-valued), shape (n, n).
        num_samples (int): Number of samples to draw.

    Returns:
        torch.Tensor: Complex-valued samples, shape (num_samples, n).
    """
    # Ensure inputs are tensors
    # mean = mean.to(dtype=torch.cfloat)
    # covariance = covariance.to(dtype=torch.cfloat)
    # relation = relation.to(dtype=torch.cfloat)

    # Dimension of the vector
    n = mean.shape[-2]

    # Decompose the covariance and relation matrices into real components
    K_xx = 0.5 * torch.real(covariance + relation)
    K_yy = 0.5 * torch.real(covariance - relation)
    K_yx = 0.5 * torch.imag(relation - covariance)
    K_xy = 0.5 * torch.imag(relation + covariance)

    # Form the real-valued block covariance matrix
    block_covariance = torch.cat(
        [torch.cat([K_xx, K_yx], dim=-1), torch.cat([K_xy, K_yy], dim=-1)], dim=-2
    )

    # General square root for the block covariance matrix
    eigvals, eigvecs = torch.linalg.eigh(block_covariance)  # Hermitian decomposition
    print(eigvals[None, :].mT)
    sqrt_block_covariance = (
        eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.H
    )
    # sqrt_block_covariance = eigvecs.to(dtype=torch.cdouble) @ torch.diag(torch.sqrt(eigvals.to(dtype=torch.cdouble))) @ eigvecs.H.to(dtype=torch.cdouble)

    # Generate real-valued standard normal samples
    real_samples = torch.randn(2 * n, num_samples, dtype=torch.float64)

    # Transform samples using the square root of the block covariance matrix
    transformed_samples = sqrt_block_covariance @ real_samples

    # Split back into real and imaginary parts
    samples_real = transformed_samples[..., :n, :]
    samples_imag = transformed_samples[..., n:, :]

    # Construct complex samples
    complex_samples = samples_real + 1j * samples_imag

    # Add the mean vector
    samples = complex_samples + mean

    return samples


# # Example Usage
# n = 4  # Dimension of the complex vector
# mean = torch.tensor([[1+1j], [2-1j], [1+2j], [3-3j]], dtype=torch.cdouble)  # Mean vector
# covariance = torch.tensor(
#     [[2, 1+0.5j, 0.3-0.2j, 0.1],
#      [1-0.5j, 2, 0.4+0.1j, 0.2],
#      [0.3+0.2j, 0.4-0.1j, 1.5, 0.1],
#      [0.1, 0.2, 0.1, 1]],
#     dtype=torch.cdouble
# )  # Hermitian covariance matrix

# relation = torch.tensor(
#     [[1, 0.2+0.1j, 0.1-0.2j, 0.3],
#      [0.2+0.1j, 1, 0.4+0.2j, 0.2],
#      [0.1-0.2j, 0.4+0.2j, 1, 0.1],
#      [0.3, 0.2, 0.1, 1]],
#     dtype=torch.cdouble
# )  # Symmetric relation matrix


# mean = torch.randn(n, 1, dtype=torch.cdouble)
# covariance = torch.randn(n, n, dtype=torch.cdouble)
# relation = torch.randn(n, n, dtype=torch.cdouble)
# covariance = make2covariance_matrix(covariance @ covariance.mH)
# relation = relation @ relation.mT
# relation = 1/10 * relation

# num_samples = 100000  # Number of samples to generate
# samples = sample_complex_multivariate(mean, covariance, relation, num_samples)

# print(samples[...,:5])  # Print the first 5 samples
# sample_mean = samples.mean(dim=-1, keepdim=True)
# print("Mean Difference:", torch.norm(sample_mean - mean) / torch.norm(mean))
# centered_samples = samples - mean
# estimated_covariance = (centered_samples @ centered_samples.mH) / num_samples
# print("Covariance Difference:", torch.norm(estimated_covariance - covariance) / torch.norm(covariance))
# estimated_relation = (centered_samples @ centered_samples.mT) / num_samples
# print("Relation Difference:", torch.norm(estimated_relation - relation) / torch.norm(relation))


# W-disjoint orthogonality measure (inputs are usually spectrograms such as STFT-data)
# Outputs a value between 1 and 0, where 1 is orthogonal and 0 is not orthogonal
def wdo(A: torch.Tensor, B: torch.Tensor):
    return 1 - (A * B).abs().sum() / torch.max(A.abs(), B.abs()).sum()


# %% Linaer Algebra (Matrices)


def cpu_gen_solve(A, B):
    return torch.linalg.lstsq(A.cpu(), B.cpu()).solution.to(A.device)


def covariance_SCM(data: torch.Tensor) -> torch.Tensor:
    """
    Estimate the covariance matrix of data using
    Sample Covariance Matrix (SCM) method.

    Args:
        data (torch.Tensor): audio stft data with dimensions (..., channels, frames)

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
        print(
            f"Iteration: {iter}, Coverged: {(~mask).sum().item()}/{mask.numel()}, max change: {criterion.max().item():.3e}"
        )
        if not mask.any():
            break

    return make2covariance_matrix(S / trace(S) * power)[..., 0, :, :]


def effective_rank(matrix: torch.Tensor, order: float | str = 1) -> torch.Tensor:
    p = makeVectorUnitNorm(torch.linalg.svdvals(matrix)[..., None], order)
    H = -torch.sum(torch.nan_to_num(p * torch.log(p), nan=0), dim=-2, keepdim=True)
    return torch.exp(H)


def is_hermitian(matrix: torch.Tensor) -> torch.Tensor:
    return torch.all(matrix - matrix.mH == 0, dim=-1, keepdim=True).all(
        dim=-2, keepdim=True
    )


def is_symmetric(matrix: torch.Tensor) -> torch.Tensor:
    return torch.all(matrix - matrix.mT == 0, dim=-1, keepdim=True).all(
        dim=-2, keepdim=True
    )


def is_positive_definite_h(matrix: torch.Tensor) -> torch.Tensor:
    eigvals = mytorch_eigvalsh(matrix)
    return torch.all(eigvals > 0, dim=-1, keepdim=True)[..., None] * is_hermitian(
        matrix
    )


def is_positive_semi_definite_h(matrix: torch.Tensor) -> torch.Tensor:
    eigvals = mytorch_eigvalsh(matrix)
    return torch.all(eigvals >= 0, dim=-1, keepdim=True)[..., None] * is_hermitian(
        matrix
    )


def hermitian_angle(
    vector_1: torch.Tensor, vector_2: torch.Tensor, dim: int = -2
) -> torch.Tensor:
    return torch.acos(
        torch.min(
            torch.abs(torch.sum(vector_1.conj() * vector_2, dim=dim, keepdim=True))
            / (
                vector_1.norm(dim=dim, keepdim=True)
                * vector_2.norm(dim=dim, keepdim=True)
            ),
            torch.tensor(1),
        )
    )


def complex_angle(
    vector_1: torch.Tensor, vector_2: torch.Tensor, dim: int = -2
) -> torch.Tensor:
    return torch.acos(
        torch.sum(vector_1.conj() * vector_2, dim=dim, keepdim=True)
        / (vector_1.norm(dim=dim, keepdim=True) * vector_2.norm(dim=dim, keepdim=True))
    )


def subspace_angles(U, V):
    """
    Compute the subspace angles between two subspaces U and V.

    Parameters:
        U (torch.Tensor): A tensor of shape (..., d, m), where d is the total considered space dimension,
                          and m is the subspace dimension for U.
        V (torch.Tensor): A tensor of shape (..., d, n), where d is the total considered space dimension,
                          and n is the subspace dimension for V.

    Returns:
        angles (torch.Tensor): A tensor of shape (...) containing the subspace angles in radians.
    """
    # Ensure U and V are tensors
    U = torch.tensor(U, dtype=torch.float32) if not isinstance(U, torch.Tensor) else U
    V = torch.tensor(V, dtype=torch.float32) if not isinstance(V, torch.Tensor) else V

    # QR decomposition for orthonormalization
    Q_U, _ = torch.linalg.qr(U)  # Shape: (..., d, m)
    Q_V, _ = torch.linalg.qr(V)  # Shape: (..., d, n)

    # Compute the inner product matrix between the orthonormal bases
    # Use transpose explicitly to avoid broadcasting mismatches
    P = Q_U.mH @ Q_V  # Shape: (..., m, n)

    # Compute singular values of the projection matrix
    sigma = torch.linalg.svdvals(P)[
        ..., None, :
    ]  # Singular values, shape: (..., min(m, n))

    # Compute subspace angles
    angles = torch.acos(
        torch.clamp(sigma, min=0.0, max=1.0)
    )  # Clamp to avoid numerical issues

    return angles


def trace(matrix: torch.Tensor) -> torch.Tensor:
    return torch.sum(matrix.diagonal(dim1=-2, dim2=-1), dim=-1)[..., None, None]


def makeHermitian(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.mH) / 2


def makeSymmetric(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.mT) / 2


def make_positive_definite_h(matrix: torch.Tensor) -> torch.Tensor:
    eigvals = mytorch_eigvalsh(matrix)
    return matrix + (
        eigvals.max(dim=-1)[0][..., None, None] * torch.finfo().eps
        - torch.min(eigvals.min(dim=-1)[0][..., None, None], torch.tensor(0))
    ) * torch.eye(matrix.shape[-1], device=matrix.device)


def make2covariance_matrix(
    matrix: torch.Tensor, reg_factor: float = 1e-6
) -> torch.Tensor:
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
    matrix = makeHermitian(matrix)

    eigvals, eigvecs = mytorch_eigh(matrix)

    # Find the maximum eigenvalue for each set of eigenvalues (last dimension)
    max_eigvals = torch.max(eigvals, dim=-1, keepdim=True)[0]

    # Calculate the dynamic minimum threshold per matrix
    min_threshold = reg_factor * abs(max_eigvals)

    # Replace any eigenvalue smaller than the threshold with the threshold
    modified_eigvals = torch.where(eigvals < min_threshold, min_threshold, eigvals)

    return evd2matrix_h(eigvals=modified_eigvals, eigvecs=eigvecs)


def evd2matrix_h(eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
    return makeHermitian(eigvecs @ (eigvals[..., None] * eigvecs.mH))


def makeMatrixUnitNorm(matrix: torch.Tensor, order: int | str = "fro") -> torch.Tensor:
    return matrix / torch.linalg.matrix_norm(
        matrix, ord=order, dim=(-2, -1), keepdim=True
    )


def makeMatricesMaxUnitNorm(
    matrix: torch.Tensor, dependent_dim: int = -3, order: int | str = "fro"
) -> torch.Tensor:
    return (
        matrix
        / torch.max(
            torch.linalg.matrix_norm(matrix, ord=order, dim=(-2, -1), keepdim=True),
            dim=dependent_dim,
            keepdim=True,
        )[0]
    )


def makeVectorUnitNorm(vector: torch.Tensor, order: float | str = 2) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, ord=order, dim=(-2), keepdim=True)


def makeVectorUnitNorm_inPlace(vector: torch.Tensor, order: float | str = 2):
    norm_factor = torch.linalg.vector_norm(vector, ord=order, dim=(-2), keepdim=True)
    vector.div_(norm_factor)


def peigvech(matrix: torch.Tensor) -> torch.Tensor:
    return characteristic_subspace_h(matrix)


def characteristic_subspace_h(matrix: torch.Tensor, order=[0]) -> torch.Tensor:
    return mytorch_eigh(matrix)[1].flip(dims=(-1,))[..., order]


def characteristic_subspace(matrix: torch.Tensor, order=[0], left=True) -> torch.Tensor:
    return (
        torch.linalg.svd(matrix.cpu())[0][..., order].to(device=matrix.device)
        if left
        else torch.linalg.svd(matrix.cpu())[2][..., order].to(device=matrix.device)
    )


def matrixsqrth(matrix: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = mytorch_eigh(matrix)
    if True:  # TODO Check this behavior and whether it is required
        eigvals = torch.clamp(eigvals, min=1e-7)
    return (eigvecs * eigvals[..., None, :].sqrt()) @ eigvecs.mH


def orthogonal_complement(
    matrix: torch.Tensor,
) -> (
    torch.Tensor
):  # TODO set to zero the to many vectors from the full column rank matrices
    M, N = matrix.shape[-2:]
    U, S, _ = torch.linalg.svd(matrix, full_matrices=True)
    rank = torch.linalg.matrix_rank(matrix)
    if check_all_elements_equal(rank):
        minrank = rank.min()
        return U[..., minrank:]
    else:
        raise NotImplementedError(
            "This function has not been implemented for a batch of matrices with different ranks yet. You can use a loop over your batch dimension(s) and you will receive output matrices of different dimensions!"
        )


def vec2diagMat(vector: torch.Tensor) -> torch.Tensor:
    M = vector.shape[-2]
    dev = vector.device
    dtype = vector.dtype
    return (vector @ torch.ones((1, M), device=dev, dtype=dtype)) * torch.eye(
        M, device=dev, dtype=dtype
    )


# Define projection operators
def parallel_projection(A: torch.Tensor, method: str = "fast") -> torch.Tensor:
    # Normalize input
    A_norm = makeVectorUnitNorm(A)
    A_norm[A_norm.isnan()] = 0

    # Store original dtype to cast back later
    orig_dtype = A.dtype

    match method:
        case "fast":
            # Fast method: A(A^H A)^-1 A^H
            # Squares condition number. Risky for rank determination.
            return makeHermitian(
                A_norm @ torch.linalg.solve(A_norm.mH @ A_norm, A_norm.mH)
            )

        case "exact":
            # SVD-based pseudo-inverse. Stable but limited by float32 precision.
            return makeHermitian(A_norm @ torch.linalg.pinv(A_norm))

        case "super_exact":
            # 1. Upcast to complex128 (double precision)
            # 2. Use SVD-based pinv
            # 3. Cast back
            A_high = A_norm.to(dtype=torch.complex128)
            P_high = A_high @ torch.linalg.pinv(A_high)
            return makeHermitian(P_high).to(dtype=orig_dtype)

        case _:
            raise ValueError(
                f"Unknown method '{method}' for parallel projection. Use 'fast', 'exact', or 'super_exact'."
            )


def orthogonal_projection(A: torch.Tensor, method: str = "fast") -> torch.Tensor:
    return makeHermitian(
        torch.eye(A.shape[-2], device=A.device, dtype=A.dtype)
        - parallel_projection(A, method)
    )


def oblique_projection(
    A: torch.Tensor, B: torch.Tensor, method: str = "fast"
) -> torch.Tensor:
    A = makeVectorUnitNorm(A)
    match method:
        case "fast":
            AHP_B = A.mH @ orthogonal_projection(B)
            return A @ torch.linalg.solve(AHP_B @ A, AHP_B)
        case "exact":
            return A @ torch.pinverse(orthogonal_projection(B) @ A)
        case "idk":
            C = generalized_cat([A, B], dim=-1)
            E = match_dims_to(
                torch.diag_embed(
                    torch.cat(
                        [
                            torch.ones(A.shape[-1], dtype=A.dtype, device=A.device),
                            torch.zeros(B.shape[-1], dtype=B.dtype, device=B.device),
                        ]
                    )
                ),
                C,
            )
            return C @ torch.linalg.lstsq(C.mT, E.mT).solution.mT
        case _:
            raise ValueError(
                f"Unknown method '{method}' for oblique projection. Use 'fast', 'exact', or 'idk'."
            )


def regularize(matrix: torch.Tensor, reg_factor: float = 0.0) -> torch.Tensor:
    return matrix + reg_factor * trace(matrix).abs() * torch.eye(
        matrix.shape[-1], device=matrix.device
    )


def zero2identity(matrix: torch.Tensor) -> torch.Tensor:
    """Replaces zero matrices in the batch with identity matrices.

    Args:
        matrix (torch.Tensor): Input batch of matrices of shape (..., M, M)

    Returns:
        torch.Tensor: Output batch with zero matrices replaced by identity matrices.
    """
    M = matrix.shape[-1]
    identity = torch.eye(M, device=matrix.device, dtype=matrix.dtype).expand(
        matrix.shape[:-2] + (M, M)
    )
    is_zero = (matrix.abs().sum(dim=(-1, -2)) == 0).unsqueeze(-1).unsqueeze(-1)
    return torch.where(is_zero, identity, matrix)


# %% Probaility Functions


def gaussian(x: torch.Tensor, mu: float = 0, sigma: float = 1) -> torch.Tensor:
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
        sigma * torch.sqrt(torch.tensor(2 * torch.pi))
    )


# %% Other


def get_real_dtype(input_tensor):
    """
    Returns the real dtype corresponding to the input tensor's dtype.
    If the input tensor is complex, it returns the corresponding real dtype.
    Otherwise, it returns the dtype of the input tensor.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor whose dtype is to be checked.

    Returns:
    - torch.dtype: The real dtype corresponding to the input tensor's dtype.
    """
    if input_tensor.is_complex():
        # If the input tensor is complex, return the corresponding real dtype
        if input_tensor.dtype == torch.complex128:
            return torch.float64
        elif input_tensor.dtype == torch.complex64:
            return torch.float32
    else:
        # If the input tensor is already real, return its dtype
        return input_tensor.dtype


# Automatic recursive divide and conquer if input broadcast dimenions are too large for cuda operations


# def fixcuda(fun: Callable, input: torch.Tensor) -> torch.Tensor:
#     try:
#         return fun(input)
#     except:
#         torch.cuda.empty_cache()
#         input_shape = None
#         if input.dim() > 3:
#             input_shape = input.shape
#             input = input.reshape(-1, input_shape[-2], input_shape[-1])
#         halfidx = input.shape[0] // 2
#         print("Halfing by fixcuda used!")
#         result = torch.cat(
#             [
#                 fixcuda(fun=fun, input=input[:halfidx]),
#                 fixcuda(fun=fun, input=input[halfidx:]),
#             ],
#             dim=0,
#         )
#         if input_shape is not None:
#             return result.reshape((*input_shape[:-2], *result.shape[1:]))
#         else:
#             return result


# def fixcuda4(fun: Callable, *args, **kwargs) -> torch.Tensor:
#     try:
#         return fun(*args, **kwargs)
#     except:
#         torch.cuda.empty_cache()
#         input = args[0]
#         input_shape = None
#         if input.dim() > 5:
#             input_shape = input.shape
#             input = input.reshape(
#                 -1, input_shape[-4], input_shape[-3], input_shape[-2], input_shape[-1]
#             )
#         halfidx = input.shape[0] // 2
#         print("Halfing by fixcuda used!")
#         args1 = tuple([input[:halfidx]] + list(args[1:]))
#         args2 = tuple([input[halfidx:]] + list(args[1:]))
#         result = torch.cat(
#             [fixcuda4(fun, *args1, **kwargs), fixcuda4(fun, *args2, **kwargs)], dim=0
#         )
#         if input_shape is not None:
#             return result.reshape((*input_shape[:-4], *result.shape[1:]))
#         else:
#             return result


# def fix_torch_on_cuda(fun: Callable, tensor: torch.Tensor):
#     """
#     A robust wrapper for torch.linalg.eigvalsh that handles potential CUDA errors
#     by attempting different strategies:
#     1. Tries to compute on the current device.
#     2. On failure, upcasts to complex128 for better precision.
#     3. On further failure, moves the computation to the CPU and then returns the
#        result to the original device.
#     4. As a last resort, switches the CUDA linear algebra backend to 'magma'.
#     """
#     try:
#         return fun(tensor)
#     except RuntimeError:
#         try:
#             dtype = tensor.dtype
#             # print(f"{fun.__name__} failed. Retrying with complex128.")
#             tensor_complex = tensor.to(dtype=torch.complex128)
#             output = fun(tensor_complex)
#             if isinstance(output, tuple):
#                 return tuple(out.to(dtype=dtype) for out in output)
#             return output.to(dtype=dtype)
#         except RuntimeError:
#             try:
#                 # print(f"{fun.__name__} failed again. Falling back to CPU.")
#                 original_device = tensor.device
#                 output_cpu = fun(tensor.cpu())
#                 if isinstance(output_cpu, tuple):
#                     return tuple(out.to(original_device) for out in output_cpu)
#                 return output_cpu.to(original_device)
#             except RuntimeError:
#                 # print(f"{fun.__name__} failed again. Retrying with 'magma' backend.")
#                 torch.backends.cuda.preferred_linalg_library(backend="magma")
#                 output = fun(tensor)
#                 torch.backends.cuda.preferred_linalg_library(backend="default")
#                 return output


def mytorch_eigvalsh(tensor: torch.Tensor) -> Any:
    return run_torch_function_with_settings(
        torch.linalg.eigvalsh, tensor, loop=True, broadcast_threshold=2**13
    )
    # return dac4torch_fun(torch.linalg.eigvalsh, tensor, 2**10)


def mytorch_eigh(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return run_torch_function_with_settings(
        torch.linalg.eigh, tensor, loop=True, broadcast_threshold=2**13
    )
    # return dac4torch_fun(torch.linalg.eigh, tensor, 2**10)


def dac4torch_fun(
    fun: Callable, tensor: torch.Tensor, broadcast_threshold, mode: str = "normal"
) -> Any:
    try:
        return run_torch_function_with_settings(
            fun,
            tensor,
            mode=mode,
            dac=True,
            broadcast_threshold=broadcast_threshold,
        )
    except RuntimeError:
        return dac4torch_fun(fun, tensor, broadcast_threshold // 2, mode=mode)


def run_torch_function_with_settings(
    fun: Callable,
    tensor: torch.Tensor,
    mode: str = "normal",
    dac: bool = False,
    loop: bool = False,
    **kwargs,
) -> Any:
    """
    A wrapper to run a torch function with specific settings.
    Parameters:
        fun (Callable): The torch function to be executed.
        mode (str): The mode of execution. Options are "normal", "magma", or "cpu".
    Returns:
        Any: The result of the torch function.
    """
    if dac:
        assert (
            "broadcast_threshold" in kwargs
        ), "Please provide 'broadcast_threshold' in kwargs for 'dac' mode."
        broadcast_threshold = kwargs["broadcast_threshold"]
        broadcast_shape = tensor.shape[:-2]
        num_matrices = tensor.numel() // (tensor.shape[-1] * tensor.shape[-2])
        if num_matrices > broadcast_threshold:
            flat_tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
            mid = flat_tensor.shape[0] // 2
            result1 = run_torch_function_with_settings(
                fun, flat_tensor[:mid], mode=mode, dac=dac, **kwargs
            )
            result2 = run_torch_function_with_settings(
                fun, flat_tensor[mid:], mode=mode, dac=dac, **kwargs
            )
            if isinstance(result1, tuple):
                flat_result = tuple(
                    torch.cat([r1, r2], dim=0) for r1, r2 in zip(result1, result2)
                )
                return tuple(
                    r.reshape(*broadcast_shape, *r.shape[1:]) for r in flat_result
                )
            else:
                flat_result = torch.cat([result1, result2], dim=0)
                return flat_result.reshape(*broadcast_shape, *flat_result.shape[1:])

    if loop:
        assert (
            "broadcast_threshold" in kwargs
        ), "Please provide 'broadcast_threshold' in kwargs for 'dac' mode."
        broadcast_threshold = kwargs["broadcast_threshold"]
        broadcast_shape = tensor.shape[:-2]
        num_matrices = tensor.numel() // (tensor.shape[-1] * tensor.shape[-2])
        if num_matrices > broadcast_threshold:
            flat_tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
            numblocks = flat_tensor.shape[0] // broadcast_threshold + 1

            results = []
            for i in range(numblocks):
                results.append(
                    run_torch_function_with_settings(
                        fun,
                        flat_tensor[
                            i * broadcast_threshold : (i + 1) * broadcast_threshold
                        ],
                        mode=mode,
                        loop=loop,
                        **kwargs,
                    )
                )
            if isinstance(results[0], tuple):
                flat_result = tuple(
                    torch.cat([res[i] for res in results], dim=0)
                    for i in range(len(results[0]))
                )
                return tuple(
                    r.reshape(*broadcast_shape, *r.shape[1:]) for r in flat_result
                )
            else:
                flat_result = torch.cat(results, dim=0)
                return flat_result.reshape(*broadcast_shape, *flat_result.shape[1:])

    if mode == "normal":
        return fun(tensor)
    elif mode == "cpu":
        original_device = tensor.device
        result = fun(tensor.cpu())
        if isinstance(result, tuple):
            return tuple(res.to(original_device) for res in result)
        return result.to(original_device)
    elif mode == "cuda":
        original_device = tensor.device
        result = fun(tensor.cuda())
        if isinstance(result, tuple):
            return tuple(res.to(original_device) for res in result)
        return result.to(original_device)
    elif mode == "magma":
        original_device = tensor.device
        torch.backends.cuda.preferred_linalg_library(backend="magma")
        result = fun(tensor.cuda())
        torch.backends.cuda.preferred_linalg_library(backend="default")
        if isinstance(result, tuple):
            return tuple(res.to(original_device) for res in result)
        return result.to(original_device)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'normal', 'cpu', 'cuda', or 'magma'."
        )


def memory(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def zeropad2fitdims(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    # Get the maximum size along each dimension
    max_sizes = [
        max(tensor.size(dim) for tensor in tensors) for dim in range(tensors[0].dim())
    ]

    padded_tensors = []

    for tensor in tensors:
        # Get the padding needed for each dimension
        padding = []
        for dim, max_size in enumerate(max_sizes):
            padding = [0, max_size - tensor.size(dim)] + padding

        # Apply padding to the tensor and append it to the list of padded tensors
        padded_tensors.append(torch.nn.functional.pad(tensor, padding))

    return padded_tensors


def check_all_elements_equal(tensor: torch.Tensor) -> bool:
    unique_elements = torch.unique(tensor)
    return len(unique_elements) == 1


def nanappend(tensor: torch.Tensor, dim: int, final_length) -> torch.Tensor:
    size = list(tensor.shape)
    size[dim] = int(final_length - size[dim])
    return torch.cat(
        [
            tensor,
            torch.full(size, float("nan"), device=tensor.device, dtype=tensor.dtype),
        ],
        dim=dim,
    )


def check_broadcastable(*shape_list: torch.Size) -> Union[tuple, bool]:
    """
    Check if a list of shapes are broadcastable and return the broadcasted shape.

    Args:
        shape_list (list[tuple]): List of shapes (tuples) representing the dimensions of tensors.

    Returns:
        tuple or bool: Returns the resulting broadcasted shape if broadcastable, else False.
    """
    # Initialize the result with an empty shape
    result_shape = []

    # Find the maximum number of dimensions in the shapes
    max_dims = max(len(shape) for shape in shape_list)

    # Iterate through each dimension from last to first (rightmost to leftmost)
    for dim in range(max_dims):
        current_dim = None  # Track the dimension being checked

        # Iterate through each shape
        for shape in shape_list:
            # Access the current dimension from the right (dim=-1 means last, dim=-2 means second last, etc.)
            shape_dim = shape[-(dim + 1)] if dim < len(shape) else 1

            # If this is the first dimension being checked, set it as the current dimension
            if current_dim is None:
                current_dim = shape_dim
            else:
                # Check if the current dimension is broadcastable
                if shape_dim != 1 and current_dim != 1 and shape_dim != current_dim:
                    return False  # Not broadcastable if neither is 1 and they are not equal

            # Update the current dimension to the larger of the two
            current_dim = max(current_dim, shape_dim)

        # Add the current dimension to the result shape (prepend since we're building the shape backwards)
        result_shape.insert(0, current_dim)

    return tuple(result_shape)


def inv_perm_indices(perm_indices: Union[list, tuple]) -> list:
    """
    Computes the inverse permutation indices such that
    tensor.permute(perm_indices).permute(inv_perm_indices(perm_indices))
    restores the original tensor.

    Parameters:
    - perm_indices (list or tuple): List of permutation indices.

    Returns:
    - inv_perm (list): Inverse permutation indices.
    """
    inv_perm = [0] * len(
        perm_indices
    )  # Initialize a list of zeros with the same length

    for i, p in enumerate(perm_indices):
        inv_perm[p] = i  # Assign the position to the inverse index

    return inv_perm


# %% Adjusted pytorch functions (e.g. for complex numbers)


def torchaudio_functional_fftconvolve_complex(
    x: torch.Tensor, y: torch.Tensor, mode: str = "full"
) -> torch.Tensor:
    r"""
    Convolves inputs along their last dimension using FFT. For inputs with large last dimensions, this function
    is generally much faster than :meth:`convolve`.
    Note that, in contrast to :meth:`torch.nn.functional.conv1d`, which actually applies the valid cross-correlation
    operator, this function applies the true `convolution`_ operator.
    Also note that this function can only output (c)float tensors (int tensor inputs will be cast to (c)float).

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        x (torch.Tensor): First convolution operand, with shape `(..., N)`.
        y (torch.Tensor): Second convolution operand, with shape `(..., M)`
            (leading dimensions must be broadcast-able with those of ``x``).
        mode (str, optional): Must be one of ("full", "valid", "same").

            * "full": Returns the full convolution result, with shape `(..., N + M - 1)`. (Default)
            * "valid": Returns the segment of the full convolution result corresponding to where
              the two inputs overlap completely, with shape `(..., max(N, M) - min(N, M) + 1)`.
            * "same": Returns the center segment of the full convolution result, with shape `(..., N)`.

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., L)`, where
        the leading dimensions match those of ``x`` and `L` is dictated by ``mode``.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """

    n = x.size(-1) + y.size(-1) - 1
    if x.is_complex() or y.is_complex():
        return torch.fft.ifft(torch.fft.fft(x, n=n) * torch.fft.fft(y, n=n), n=n)
    else:
        return torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(y, n=n), n=n)


# W.-K. Ma et al., A signal processing perspective on hyperspectral unmixing: Insights from remote sensing. IEEE Signal Process Mag 31(1), 67–81 (2014)
def successive_projections(X, num_selections):
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


def generalized_cat(tensor_list, dim):
    """
    Concatenates a list of tensors along the specified dimension,
    broadcasting the tensors to compatible shapes.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to concatenate.
        dim (int): The dimension along which to concatenate.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    if not tensor_list:
        raise ValueError("tensor_list must contain at least one tensor.")

    # Find the maximum number of dimensions
    max_dims = max(tensor.dim() for tensor in tensor_list)

    # Convert negative index to positive
    if dim < 0:
        dim += max_dims

    # Ensure all tensors have the same number of dimensions
    expanded_tensors = []
    shape_list = []
    for tensor in tensor_list:
        shape = list(tensor.shape)
        # Add leading singleton dimensions to match max_dims
        tensor = tensor.view(*([1] * (max_dims - len(shape))), *shape)
        expanded_tensors.append(tensor)
        shape_list.append(tensor.shape)

    # Determine the broadcast shape
    broadcast_shape = check_broadcastable(
        *[shape[:dim] + shape[dim + 1 :] for shape in shape_list]
    )
    assert broadcast_shape, "Shapes of tensors are not broadcastable!"

    # Concatenate along the specified dimension
    broadcast_shape = (
        broadcast_shape[:dim] + (-1,) + broadcast_shape[dim:]
        if not isinstance(broadcast_shape, bool)
        else None
    )
    # Expand all tensors to the broadcast shape
    broadcasted_tensors = [
        tensor.expand(broadcast_shape) for tensor in expanded_tensors
    ]

    # Concatenate along the specified dimension
    return torch.cat(broadcasted_tensors, dim=dim)


def match_dims_to(tensor, target_tensor):
    """
    Increase the dimensions of `tensor` to match the number of dimensions of `target_tensor`
    by adding dimensions of size 1 at the beginning.

    Parameters:
        tensor (torch.Tensor): The tensor to be expanded.
        target_tensor (torch.Tensor): The tensor whose dimensions we want to match.

    Returns:
        torch.Tensor: The expanded tensor.
    """
    # Get the number of dimensions
    tensor_dims = tensor.dim()
    target_dims = target_tensor.dim()

    # Calculate the number of dimensions to add
    dims_to_add = target_dims - tensor_dims

    # Add dimensions of size 1 at the beginning
    if dims_to_add > 0:
        tensor = tensor.view((1,) * dims_to_add + tensor.shape)

    return tensor
