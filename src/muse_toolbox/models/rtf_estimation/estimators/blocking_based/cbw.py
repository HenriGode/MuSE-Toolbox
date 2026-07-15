import logging
from typing import Any

import torch

from muse_toolbox.models.rtf_estimation.estimators.base_rtf_estimator import (
    BaseRTFestimator,
)

log = logging.getLogger(__name__)


class CBW(BaseRTFestimator):
    """RTF estimator using Covariance Blocking and Whitening."""

    def __init__(self) -> None:
        super().__init__()

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {"type": "CBW"}

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates the RTF using covariance blocking and whitening.
        """
        return covblockwhiten(Ry, Rn, G)


def covblockwhiten(Ry: torch.Tensor, Rn: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Estimates the RTF vector of a newly activating source using the Covariance Blocking and Whitening (CBW) method.

    Implements the Covariance Blocking and Whitening (CBW) method for successive Relative Transfer Function (RTF)
    vector estimation in multi-speaker scenarios.

    Reference:
    Gode, H., & Doclo, S. (2023, October).
    Covariance Blocking and Whitening Method for Successive Relative Transfer Function Vector Estimation in Multi-Speaker Scenarios.
    In 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) (pp. 1-5). IEEE.

    Args:
        Ry (torch.Tensor): Covariance matrix of the noisy signal.
        Rn (torch.Tensor): Covariance matrix of the noise.
        G (torch.Tensor): RTF vector of the first speaker.

    Returns:
        torch.Tensor: Estimated RTF vector of the second speaker (unit norm).
    """

    # number of microphones
    M = G.shape[-2]
    # number of sources
    N = G.shape[-1] + 1

    # assert M >= 2*N-1, "Number of microphones must be at least 2 times the number of sources minus 1."
    if M < 2 * N - 1:
        print(
            "Warning: Number of microphones should be at least 2 times the number of sources minus 1."
        )

    # Step 1: Compute the dimension-reduced residual maker matrix P⊥_g
    P_G_perp_r = orthogonal_projection(G, "exact")[..., : M - N + 1]

    # Step 3: Block the first speaker and whiten the noise
    Rn_blocked = Rn @ P_G_perp_r
    Ry_blocked = Ry @ P_G_perp_r
    Rn_blocked_pinv = torch.linalg.pinv(Rn_blocked)
    Rw_y = Rn_blocked_pinv @ Ry_blocked - torch.eye(
        M - N + 1, device=Rn_blocked.device, dtype=Rn_blocked.dtype
    )

    # Step 4: Extract transformed RTF vectors using SVD
    U, _, Vh = torch.linalg.svd(Rw_y, full_matrices=False)
    qL = U[..., :, [0]]
    qR = Vh.mH[..., :, [0]]

    # Step 5: Solve for the weighting factor α
    B = torch.cat([Rn_blocked_pinv, P_G_perp_r.mH], dim=-2)
    P_B_perp = orthogonal_projection(B, "exact")
    P_B_L = P_B_perp[..., : (M - N + 1)]
    P_B_R = P_B_perp[..., (M - N + 1) :]
    alpha = -torch.linalg.pinv(P_B_R @ qR) @ (P_B_L @ qL)

    # Step 6: Estimate the scaled RTF vector and normalize
    h_tilde = torch.linalg.pinv(B) @ torch.cat([qL, qR * alpha], dim=-2)
    h_est = makeVectorUnitNorm(h_tilde)

    return h_est


# def covarianceBlockingWhitening(
#     noisyCovMat: torch.Tensor, noiseCovMat: torch.Tensor, oldRTFvecs: torch.Tensor
# ) -> torch.Tensor:
#     """
#     !!! Old code version !!! slow because of cpu_gen_solve
#     Implements the Covariance Blocking and Whitening (CBW) method for successive Relative Transfer Function (RTF)
#     vector estimation in multi-speaker scenarios.

#     Reference:
#     Gode, H., & Doclo, S. (2023, October).
#     Covariance Blocking and Whitening Method for Successive Relative Transfer Function Vector Estimation in Multi-Speaker Scenarios.
#     In 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) (pp. 1-5). IEEE.

#     Parameters:
#         noisyCovMat (torch.Tensor): The noisy covariance matrix.
#         noiseCovMat (torch.Tensor): The noise covariance matrix.
#         oldRTFvecs (torch.Tensor): Previously estimated RTF vectors.

#     Returns:
#         torch.Tensor: The updated RTF vector with unit norm.
#     """
#     # Determine the number of sources, microphones, and equations for the system
#     num_sources = (
#         oldRTFvecs.shape[-1] + 1
#     )  # Number of sources (adding one to account for the current source)
#     num_mics = oldRTFvecs.shape[-2]  # Number of microphones
#     num_equations = 2 * (
#         num_mics - num_sources + 1
#     )  # Required number of equations for the system

#     # Assert that the system is overdetermined for a valid solution
#     assert (
#         num_equations > num_mics
#     ), "The number of equations must be greater than the number of microphones."

#     # Compute the orthogonal projection matrix that blocks the previously estimated RTF vectors
#     Pgr = orthogonal_projection(oldRTFvecs)[..., : -(num_sources - 1)]

#     # Transform the noise covariance matrix with the projection matrix
#     RnPgr = noiseCovMat @ Pgr
#     RnPgr_pinv = torch.linalg.pinv(
#         RnPgr
#     )  # Compute the pseudo-inverse of the transformed noise covariance matrix

#     # Compute the whitened residual matrix
#     Rw = RnPgr_pinv @ noisyCovMat @ Pgr - torch.eye(
#         num_mics - num_sources + 1, device=noisyCovMat.device, dtype=noisyCovMat.dtype
#     )

#     # Perform Singular Value Decomposition (SVD) on the residual matrix
#     QL, _, QR = torch.linalg.svd(Rw)
#     qL = QL[..., [0]]  # Extract the principal left  singular vector
#     qR = QR.mH[
#         ..., [0]
#     ]  # Extract the principal right singular vector (Hermitian transpose)

#     # Construct the augmented matrix B for solving the linear system
#     B = torch.cat([RnPgr_pinv, Pgr.mH], dim=-2)

#     # Compute the orthogonal projection matrix PB for the augmented matrix B
#     PB = orthogonal_projection(B)
#     PBL = PB[
#         ..., : (num_mics - num_sources + 1)
#     ]  # Left  partition of the projection matrix
#     PBR = PB[
#         ..., (num_mics - num_sources + 1) :
#     ]  # Right partition of the projection matrix

#     # Solve for the scaling factor alpha using the partitions of PB and the singular vectors
#     alpha = -cpu_gen_solve(PBR @ qR, PBL @ qL)

#     # Solve for the updated RTF vector and normalize it to unit norm
#     updated_RTF = cpu_gen_solve(B, torch.cat([qL, qR * alpha], dim=-2))
#     return makeVectorUnitNorm(updated_RTF)


# def test_covarianceBlockingWhitening():
#     """
#     Tests the covarianceBlockingWhitening function with synthetic data.
#     """
#     device = "cuda:0"
#     M = 5  # Number of microphones
#     N = 100000  # Number of samples

#     # Generate random RTF vectors and noise
#     g1 = randdir(M, 1)
#     g2 = randdir(M, 1)
#     g3 = randdir(M, 1)
#     n = randdir(M, N)

#     # Generate microphone signals for various sources
#     x1 = g1 * torch.randn(1, N, device=device)
#     x2 = g2 * torch.randn(1, N, device=device)
#     x3 = g3 * torch.randn(1, N, device=device)
#     y1 = x1 + n
#     y2 = x1 + x2 + n
#     y3 = x1 + x2 + x3 + n

#     # Compute covariance matrices
#     Rn = covariance_SCM(n)
#     Rx1 = covariance_SCM(x1)
#     Rx2 = covariance_SCM(x2)
#     Rx3 = covariance_SCM(x3)
#     Ry_1 = Rn + Rx1
#     Ry_2 = Rn + Rx1 + Rx2
#     Ry_3 = Rn + Rx1 + Rx2 + Rx3
#     Ry1 = covariance_SCM(y1)
#     Ry2 = covariance_SCM(y2)
#     Ry3 = covariance_SCM(y3)

#     # Perform CBW and CW
#     h_1 = covarianceWhitening(whiteningCovMat=Rn, covMat=Ry_1)
#     h_2 = covarianceBlockingWhitening(noisyCovMat=Ry_2, noiseCovMat=Rn, oldRTFvecs=h_1)
#     h_3 = covarianceBlockingWhitening(
#         noisyCovMat=Ry_3, noiseCovMat=Rn, oldRTFvecs=torch.cat([h_1, h_2], dim=-1)
#     )
#     h1 = covarianceWhitening(whiteningCovMat=Rn, covMat=Ry1)
#     h2 = covarianceBlockingWhitening(noisyCovMat=Ry2, noiseCovMat=Rn, oldRTFvecs=h1)
#     h3 = covarianceBlockingWhitening(
#         noisyCovMat=Ry3, noiseCovMat=Rn, oldRTFvecs=torch.cat([h1, h2], dim=-1)
#     )

#     # Compare against known ground truth RTF vectors
#     HA_1 = hermitian_angle(g1, h_1)
#     HA_2 = hermitian_angle(g2, h_2)
#     HA_3 = hermitian_angle(g3, h_3)
#     HA1 = hermitian_angle(g1, h1)
#     HA2 = hermitian_angle(g2, h2)
#     HA3 = hermitian_angle(g3, h3)

#     # Print results
#     for angle in [HA_1, HA_2, HA_3, HA1, HA2, HA3]:
#         print(f"Hermitian angle: {angle.item() / torch.pi * 180:.2f} degrees")

#     # Assert expected behavior (e.g., angles close to zero for correct estimation)
#     assert HA_1.item() < 1e-2, "Angle for g1 mismatch"
#     assert HA_2.item() < 1e-2, "Angle for g2 mismatch"
#     assert HA_3.item() < 1e-2, "Angle for g3 mismatch"