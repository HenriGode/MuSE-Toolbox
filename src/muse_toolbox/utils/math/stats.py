"""Generate random numbers"""

import logging

import torch

from .matrix_ops import makeVectorUnitNorm, orthogonal_complement

log = logging.getLogger(__name__)


def randdir(
    *size: int, device: torch.device | str = "cuda:0", dtype: torch.dtype = torch.complex128
) -> torch.Tensor:
    """
    Generates a random direction vector (unit norm).

    Args:
        size: Shape of the tensor to generate.
        device (Union[torch.device, str], optional): The device on which to create the tensor. Defaults to "cuda:0".
        dtype (torch.dtype, optional): The data type of the tensor. Defaults to torch.complex128.

    Returns:
        torch.Tensor: Random direction vector with unit norm.
    """
    return makeVectorUnitNorm(torch.randn(*size, device=device, dtype=dtype))


def randdir_orthogonal2vec(vector: torch.Tensor, N: int) -> torch.Tensor:
    """
    Generates a random direction vector orthogonal to the input vector.

    Args:
        vector (torch.Tensor): The input vector to which the generated vector should be orthogonal.
        N (int): The number of orthogonal vectors to generate.

    Returns:
        torch.Tensor: The orthogonal random direction vector.
    """
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
    log.debug(eigvals[None, :].mT)
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


# %% Probaility Functions


def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """
    Computes the Gaussian probability density function.

    Args:
        x (torch.Tensor): The input tensor.
        mu (float, optional): The mean of the distribution. Defaults to 0.0.
        sigma (float, optional): The standard deviation of the distribution. Defaults to 1.0.

    Returns:
        torch.Tensor: The evaluated Gaussian PDF.
    """
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
        sigma * torch.sqrt(torch.tensor(2 * torch.pi))
    )

# %% Aggregation Operations


def wmean(
    tensor: torch.Tensor,
    dims: int | tuple[int, ...],
    weights: torch.Tensor | None = None,
    keepdim: bool = True,
) -> torch.Tensor:
    """
    Computes the weighted mean of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        dims (Union[int, Tuple[int, ...]]): The dimensions to reduce.
        weights (Optional[torch.Tensor], optional): The weights tensor. Defaults to None.
        keepdim (bool, optional): Whether to retain reduced dimensions. Defaults to True.

    Returns:
        torch.Tensor: The weighted mean tensor.
    """
    if weights == None:
        return (tensor).mean(dim=dims, keepdim=keepdim)
    else:
        return (tensor * weights).sum(dim=dims, keepdim=keepdim) / weights.sum(
            dim=dims, keepdim=keepdim
        )


def norm_by_sum(tensor: torch.Tensor, dims: int, keepdim: bool = True) -> torch.Tensor:
    """
    Normalizes a tensor by its sum along a dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        dims (int): The dimension to sum over.
        keepdim (bool, optional): Whether to retain the reduced dimension. Defaults to True.

    Returns:
        torch.Tensor: The sum-normalized tensor.
    """
    denominator = tensor.sum(dim=dims, keepdim=keepdim)
    return tensor / denominator

# %% Comparison


def deviation(
    original: torch.Tensor,
    comparator: torch.Tensor,
    dim: int | tuple[int, ...] = (-1, -2),
    relative: bool = True,
) -> torch.Tensor:
    """
    Computes the deviation (norm difference) between two tensors.

    Args:
        original (torch.Tensor): The reference tensor.
        comparator (torch.Tensor): The tensor to compare.
        dim (Union[int, Tuple[int, ...]], optional): Dimensions to compute the norm over. Defaults to (-1, -2).
        relative (bool, optional): If True, computes relative deviation. Defaults to True.

    Returns:
        torch.Tensor: The deviation tensor.
    """
    if relative:
        return torch.linalg.norm(
            comparator - original, dim=dim, keepdim=True
        ) / torch.linalg.norm(original, dim=dim, keepdim=True)
    else:
        return torch.linalg.norm(comparator - original, dim=dim, keepdim=True)

def statistics(data: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and covariance matrix of a tensor.

    Args:
        data (torch.Tensor): The input tensor.
        dim (int, optional): The dimension along which to compute the statistics. Defaults to -1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and covariance matrix.
    """
    mean = torch.mean(data, dim=dim, keepdim=True)
    centered_data = data - mean
    if centered_data.is_complex():
        centered_data = torch.cat([centered_data, centered_data.conj()], dim=-2)
    covMat = covariance_Tyler(centered_data)
    return mean, covMat