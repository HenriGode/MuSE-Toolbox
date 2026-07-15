import logging

import torch

log = logging.getLogger(__name__)


def hermitian_angle(
    vector_1: torch.Tensor, vector_2: torch.Tensor, dim: int = -2
) -> torch.Tensor:
    """
    Computes the Hermitian angle between two complex-valued tensors.

    Args:
        vector_1 (torch.Tensor): The first input tensor.
        vector_2 (torch.Tensor): The second input tensor.
        dim (int, optional): The dimension along which to compute the angle. Defaults to -2.

    Returns:
        torch.Tensor: A tensor containing the computed Hermitian angles.
    """
    return torch.acos(
        torch.min(
            torch.abs(torch.sum(vector_1.conj() * vector_2, dim=dim, keepdim=True))
            / (
                vector_1.norm(dim=dim, keepdim=True)
                * vector_2.norm(dim=dim, keepdim=True)
            ),
            torch.tensor(1.0, dtype=vector_1.dtype, device=vector_1.device),
        )
    )


def complex_angle(
    vector_1: torch.Tensor, vector_2: torch.Tensor, dim: int = -2
) -> torch.Tensor:
    """
    Computes the complex angle between two tensors.

    Args:
        vector_1 (torch.Tensor): The first input tensor.
        vector_2 (torch.Tensor): The second input tensor.
        dim (int, optional): The dimension along which to compute the angle. Defaults to -2.

    Returns:
        torch.Tensor: A tensor containing the computed complex angles.
    """
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