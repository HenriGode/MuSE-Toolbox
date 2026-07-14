# %% Linear Algebra (Matrices)

import logging
from typing import Any, Union, Tuple, List, Optional

import torch

from ..tensor_ops import (
    check_all_elements_equal,
    generalized_cat,
    match_dims_to,
)
from ..system import run_torch_function_with_settings

log = logging.getLogger(__name__)


def cpu_gen_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Solves a linear system AX = B on the CPU, and moves the result back to the original device.

    Args:
        A (torch.Tensor): The left-hand side matrix.
        B (torch.Tensor): The right-hand side matrix/vector.

    Returns:
        torch.Tensor: The solution tensor X.
    """
    return torch.linalg.lstsq(A.cpu(), B.cpu()).solution.to(A.device)


def effective_rank(matrix: torch.Tensor, order: Union[float, str] = 1) -> torch.Tensor:
    """
    Computes the effective rank of a matrix based on the entropy of its singular values.

    Args:
        matrix (torch.Tensor): Input matrix.
        order (Union[float, str], optional): Norm order. Defaults to 1.

    Returns:
        torch.Tensor: The effective rank.
    """
    p = makeVectorUnitNorm(torch.linalg.svdvals(matrix)[..., None], order)
    H = -torch.sum(torch.nan_to_num(p * torch.log(p), nan=0.0), dim=-2, keepdim=True)
    return torch.exp(H)


def is_hermitian(matrix: torch.Tensor) -> torch.Tensor:
    """
    Checks if a matrix is Hermitian.

    Args:
        matrix (torch.Tensor): Input matrix.

    Returns:
        torch.Tensor: Boolean tensor indicating if the matrix is Hermitian.
    """
    return torch.all(matrix - matrix.mH == 0, dim=-1, keepdim=True).all(
        dim=-2, keepdim=True
    )


def is_symmetric(matrix: torch.Tensor) -> torch.Tensor:
    """
    Checks if a matrix is symmetric.

    Args:
        matrix (torch.Tensor): Input matrix.

    Returns:
        torch.Tensor: Boolean tensor indicating if the matrix is symmetric.
    """
    return torch.all(matrix - matrix.mT == 0, dim=-1, keepdim=True).all(
        dim=-2, keepdim=True
    )


def is_positive_definite_h(matrix: torch.Tensor) -> torch.Tensor:
    """
    Checks if a Hermitian matrix is positive definite.

    Args:
        matrix (torch.Tensor): Input Hermitian matrix.

    Returns:
        torch.Tensor: Boolean tensor indicating if the matrix is positive definite.
    """
    eigvals = mytorch_eigvalsh(matrix)
    return torch.all(eigvals > 0, dim=-1, keepdim=True)[..., None] * is_hermitian(
        matrix
    )


def is_positive_semi_definite_h(matrix: torch.Tensor) -> torch.Tensor:
    """
    Checks if a Hermitian matrix is positive semi-definite.

    Args:
        matrix (torch.Tensor): Input Hermitian matrix.

    Returns:
        torch.Tensor: Boolean tensor indicating if the matrix is positive semi-definite.
    """
    eigvals = mytorch_eigvalsh(matrix)
    return torch.all(eigvals >= 0, dim=-1, keepdim=True)[..., None] * is_hermitian(
        matrix
    )





def trace(matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes the trace of a matrix along the last two dimensions.

    Args:
        matrix (torch.Tensor): Input matrix.

    Returns:
        torch.Tensor: The trace.
    """
    return torch.sum(matrix.diagonal(dim1=-2, dim2=-1), dim=-1)[..., None, None]


def makeHermitian(matrix: torch.Tensor) -> torch.Tensor:
    """
    Forces a matrix to be Hermitian.

    Args:
        matrix (torch.Tensor): Input matrix.

    Returns:
        torch.Tensor: Hermitian matrix.
    """
    return (matrix + matrix.mH) / 2


def makeSymmetric(matrix: torch.Tensor) -> torch.Tensor:
    """
    Forces a matrix to be symmetric.

    Args:
        matrix (torch.Tensor): Input matrix.

    Returns:
        torch.Tensor: Symmetric matrix.
    """
    return (matrix + matrix.mT) / 2


def make_positive_definite_h(matrix: torch.Tensor) -> torch.Tensor:
    """
    Forces a Hermitian matrix to be positive definite by shifting its eigenvalues.

    Args:
        matrix (torch.Tensor): Input Hermitian matrix.

    Returns:
        torch.Tensor: Positive definite Hermitian matrix.
    """
    eigvals = mytorch_eigvalsh(matrix)
    return matrix + (
        eigvals.max(dim=-1)[0][..., None, None] * torch.finfo().eps
        - torch.min(eigvals.min(dim=-1)[0][..., None, None], torch.tensor(0.0, device=matrix.device))
    ) * torch.eye(matrix.shape[-1], device=matrix.device)







def evd2matrix_h(eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs a Hermitian matrix from its eigenvalues and eigenvectors.
    """
    return makeHermitian(eigvecs @ (eigvals[..., None] * eigvecs.mH))


def makeMatrixUnitNorm(matrix: torch.Tensor, order: Union[int, str] = "fro") -> torch.Tensor:
    """
    Normalizes a matrix to have unit norm.
    """
    return matrix / torch.linalg.matrix_norm(
        matrix, ord=order, dim=(-2, -1), keepdim=True
    )


def makeMatricesMaxUnitNorm(
    matrix: torch.Tensor, dependent_dim: int = -3, order: Union[int, str] = "fro"
) -> torch.Tensor:
    """
    Normalizes a batch of matrices such that the maximum norm across a dimension is 1.
    """
    return (
        matrix
        / torch.max(
            torch.linalg.matrix_norm(matrix, ord=order, dim=(-2, -1), keepdim=True),
            dim=dependent_dim,
            keepdim=True,
        )[0]
    )


def makeVectorUnitNorm(vector: torch.Tensor, order: Union[float, str] = 2) -> torch.Tensor:
    """
    Normalizes a vector to have unit norm.
    """
    return vector / torch.linalg.vector_norm(vector, ord=order, dim=(-2), keepdim=True)


def makeVectorUnitNorm_inPlace(vector: torch.Tensor, order: Union[float, str] = 2):
    """
    Normalizes a vector to have unit norm in-place.
    """
    norm_factor = torch.linalg.vector_norm(vector, ord=order, dim=(-2), keepdim=True)
    vector.div_(norm_factor)


def peigvech(matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes the principal eigenvector(s) of a Hermitian matrix.
    """
    return characteristic_subspace_h(matrix)


def characteristic_subspace_h(matrix: torch.Tensor, order: List[int] = [0]) -> torch.Tensor:
    """
    Returns the characteristic subspace (eigenvectors corresponding to specific eigenvalues)
    of a Hermitian matrix.
    """
    return mytorch_eigh(matrix)[1].flip(dims=(-1,))[..., order]


def characteristic_subspace(matrix: torch.Tensor, order: List[int] = [0], left: bool = True) -> torch.Tensor:
    """
    Returns the characteristic subspace of a matrix using SVD.
    """
    return (
        torch.linalg.svd(matrix.cpu())[0][..., order].to(device=matrix.device)
        if left
        else torch.linalg.svd(matrix.cpu())[2][..., order].to(device=matrix.device)
    )


def matrixsqrth(matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes the square root of a Hermitian positive semi-definite matrix.
    """
    eigvals, eigvecs = mytorch_eigh(matrix)
    if True:  # TODO Check this behavior and whether it is required
        eigvals = torch.clamp(eigvals, min=1e-7)
    return (eigvecs * eigvals[..., None, :].sqrt()) @ eigvecs.mH


def orthogonal_complement(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the orthogonal complement of the column space of a matrix.
    """
    # TODO set to zero the to many vectors from the full column rank matrices
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
    """
    Converts a vector into a diagonal matrix.
    """
    M = vector.shape[-2]
    dev = vector.device
    dtype = vector.dtype
    return (vector @ torch.ones((1, M), device=dev, dtype=dtype)) * torch.eye(
        M, device=dev, dtype=dtype
    )


# Define projection operators
def parallel_projection(A: torch.Tensor, method: str = "fast") -> torch.Tensor:
    """
    Computes the parallel projection matrix onto the subspace spanned by A.

    Args:
        A (torch.Tensor): Input matrix.
        method (str, optional): Projection method ("fast", "exact", or "super_exact"). Defaults to "fast".

    Returns:
        torch.Tensor: Parallel projection matrix.
    """
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
    """
    Computes the orthogonal projection matrix onto the orthogonal complement of the subspace spanned by A.

    Args:
        A (torch.Tensor): Input matrix.
        method (str, optional): Projection method. Defaults to "fast".

    Returns:
        torch.Tensor: Orthogonal projection matrix.
    """
    return makeHermitian(
        torch.eye(A.shape[-2], device=A.device, dtype=A.dtype)
        - parallel_projection(A, method)
    )


def oblique_projection(
    A: torch.Tensor, B: torch.Tensor, method: str = "fast"
) -> torch.Tensor:
    """
    Computes the oblique projection matrix onto the subspace spanned by A along the subspace spanned by B.

    Args:
        A (torch.Tensor): First input matrix.
        B (torch.Tensor): Second input matrix.
        method (str, optional): Projection method. Defaults to "fast".

    Returns:
        torch.Tensor: Oblique projection matrix.
    """
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
    """
    Regularizes a matrix by adding a scaled identity matrix.

    Args:
        matrix (torch.Tensor): Input matrix.
        reg_factor (float, optional): Regularization factor. Defaults to 0.0.

    Returns:
        torch.Tensor: Regularized matrix.
    """
    return matrix + reg_factor * trace(matrix).abs() * torch.eye(
        matrix.shape[-1], device=matrix.device
    )


def zero2identity(matrix: torch.Tensor) -> torch.Tensor:
    """
    Replaces zero matrices in the batch with identity matrices.

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


def mytorch_eigvalsh(tensor: torch.Tensor) -> Any:
    """
    Computes the eigenvalues of a Hermitian matrix using custom settings.

    Args:
        tensor (torch.Tensor): Input Hermitian matrix.

    Returns:
        Any: Eigenvalues of the matrix.
    """
    return run_torch_function_with_settings(
        torch.linalg.eigvalsh, tensor, loop=True, broadcast_threshold=2**13
    )
    # return dac4torch_fun(torch.linalg.eigvalsh, tensor, 2**10)


def mytorch_eigh(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the eigenvalues and eigenvectors of a Hermitian matrix using custom settings.

    Args:
        tensor (torch.Tensor): Input Hermitian matrix.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Eigenvalues and eigenvectors.
    """
    return run_torch_function_with_settings(
        torch.linalg.eigh, tensor, loop=True, broadcast_threshold=2**13
    )
    # return dac4torch_fun(torch.linalg.eigh, tensor, 2**10)