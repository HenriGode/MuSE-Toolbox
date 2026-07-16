import logging

import torch

from ..tensor_ops import get_real_dtype

log = logging.getLogger(__name__)


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
    log.debug(f"Computing fftconvolve_complex with mode='{mode}'.")
    n = x.size(-1) + y.size(-1) - 1
    if x.is_complex() or y.is_complex():
        return torch.fft.ifft(torch.fft.fft(x, n=n) * torch.fft.fft(y, n=n), n=n)
    else:
        return torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(y, n=n), n=n)


def windowing(data: torch.Tensor, window: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Applies a windowing function to the data using FFT-based convolution.

    Args:
        data (torch.Tensor): Input tensor.
        window (torch.Tensor): Window tensor.
        dim (int, optional): Dimension along which to apply the windowing. Defaults to -1.

    Returns:
        torch.Tensor: Windowed tensor.
    """
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

    Args:
        data (torch.Tensor): Input tensor.
        window (torch.Tensor): Window tensor.
        dim (int, optional): Dimension along which to apply the windowing. Defaults to -1.

    Returns:
        torch.Tensor: Windowed tensor.
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
    """
    Applies exponential windowing to the data using FFT-based convolution.

    Args:
        data (torch.Tensor): Input tensor.
        smoothing_factor (float): Smoothing factor for the exponential window.
        dim (int, optional): Dimension along which to apply the windowing. Defaults to -1.

    Returns:
        torch.Tensor: Windowed tensor.
    """
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
    """
    Applies exponential windowing to the data using standard 1D convolution.

    Args:
        data (torch.Tensor): Input tensor.
        smoothing_factor (float): Smoothing factor for the exponential window.
        dim (int, optional): Dimension along which to apply the windowing. Defaults to -1.

    Returns:
        torch.Tensor: Windowed tensor.
    """
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