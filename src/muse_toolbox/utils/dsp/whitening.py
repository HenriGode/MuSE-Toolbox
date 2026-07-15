import logging

import torch

from muse_toolbox.utils.math.matrix_ops import (
    makeHermitian,
    makeVectorUnitNorm,
    matrixsqrth,
    mytorch_eigvalsh,
    peigvech,
    regularize,
)

log = logging.getLogger(__name__)


def noise_subtraction(
    subtractingCovMat: torch.Tensor, covMat: torch.Tensor, ensure_PSD: bool = False
) -> torch.Tensor:
    """
    Subtracts a noise covariance matrix from a signal covariance matrix.

    Args:
        subtractingCovMat (torch.Tensor): The noise covariance matrix to subtract.
        covMat (torch.Tensor): The original signal covariance matrix.
        ensure_PSD (bool): Whether to scale the subtraction to ensure the result is Positive Semi-Definite. Defaults to False.

    Returns:
        torch.Tensor: The resulting subtracted covariance matrix.
    """
    log.debug("Performing noise subtraction.")
    subtractingCovMat = makeHermitian(subtractingCovMat)
    covMat = makeHermitian(covMat)
    subtractedCovMat = makeHermitian(covMat - subtractingCovMat)
    if ensure_PSD:
        loweigval = mytorch_eigvalsh(covMat)[..., 0]
        loweigvalsub = mytorch_eigvalsh(subtractedCovMat)[..., 0]
        factor = (loweigval / (loweigval - loweigvalsub)).clamp(max=1)[..., None, None]
        return makeHermitian(covMat - factor * subtractingCovMat)
        # return is_positive_definite_h(subtractedCovMat) * subtractedCovMat + (~is_positive_definite_h(subtractedCovMat)) * covMat
    else:
        return subtractedCovMat


def noise_whitening(
    whiteningCovMat: torch.Tensor,
    covMat: torch.Tensor,
    RTFvecs: torch.Tensor | None = None,
    subtract_identity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Whitens the covariance matrix using Cholesky decomposition.

    Args:
        whiteningCovMat (torch.Tensor): The covariance matrix used for whitening (e.g. noise covariance).
        covMat (torch.Tensor): The covariance matrix to be whitened.
        RTFvecs (Optional[torch.Tensor]): Optional Relative Transfer Functions to whiten. Defaults to None.
        subtract_identity (bool): Whether to subtract the identity matrix from the whitened covariance matrix. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: The whitening matrix L, the whitened covariance matrix, and the whitened RTF vectors.
    """
    log.debug("Performing noise whitening.")
    L = regularize(
        torch.linalg.cholesky(regularize(whiteningCovMat, reg_factor=1e-6)),
        reg_factor=1e-8,
    )
    whiteRTFVecs = (
        makeVectorUnitNorm(
            torch.linalg.solve_triangular(L, RTFvecs, upper=False, left=True)
        )
        if RTFvecs is not None
        else None
    )
    # whiteCovMat = L^-1 * covMat * L^-H - I
    whiteCovMat = torch.linalg.solve_triangular(
        L,
        torch.linalg.solve_triangular(L.mH, covMat, upper=True, left=False),
        upper=False,
        left=True,
    )
    if subtract_identity:
        whiteCovMat = makeHermitian(
            whiteCovMat
            - torch.eye(*covMat.shape[-2:], device=covMat.device, dtype=covMat.dtype)
        )
    else:
        whiteCovMat = makeHermitian(whiteCovMat)
    return L, whiteCovMat, whiteRTFVecs


def noise_whitening_robust(
    whiteningCovMat: torch.Tensor,
    covMat: torch.Tensor,
    RTFvecs: torch.Tensor | None = None,
    subtract_identity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Whitens the covariance matrix, falling back to matrix square root if Cholesky decomposition fails.

    Args:
        whiteningCovMat (torch.Tensor): The covariance matrix used for whitening.
        covMat (torch.Tensor): The covariance matrix to be whitened.
        RTFvecs (Optional[torch.Tensor]): Optional Relative Transfer Functions to whiten. Defaults to None.
        subtract_identity (bool): Whether to subtract the identity matrix. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: The whitening matrix L, the whitened covariance matrix, and the whitened RTF vectors.
    """
    log.debug("Performing robust noise whitening.")
    cholesky_failed = False
    try:
        L = regularize(
            torch.linalg.cholesky(whiteningCovMat),
            reg_factor=1e-6,
        )
    except Exception as e:
        log.warning(f"Cholesky decomposition failed, falling back to matrix square root. Error: {e}")
        cholesky_failed = True
        L = regularize(
            matrixsqrth(regularize(whiteningCovMat, reg_factor=1e-5)),
            reg_factor=1e-6,
        )

    if not cholesky_failed:
        whiteRTFVecs = (
            makeVectorUnitNorm(
                torch.linalg.solve_triangular(L, RTFvecs, upper=False, left=True)
            )
            if RTFvecs is not None
            else None
        )
        # whiteCovMat = L^-1 * covMat * L^-H - I
        whiteCovMat = torch.linalg.solve_triangular(
            L,
            torch.linalg.solve_triangular(L.mH, covMat, upper=True, left=False),
            upper=False,
            left=True,
        )

    else:
        whiteRTFVecs = (
            makeVectorUnitNorm(torch.linalg.solve(L, RTFvecs, left=True))
            if RTFvecs is not None
            else None
        )
        # whiteCovMat = L^-1 * covMat * L^-H - I
        whiteCovMat = torch.linalg.solve(
            L,
            torch.linalg.solve(L.mH, covMat, left=False),
            left=True,
        )

    if subtract_identity:
        whiteCovMat = makeHermitian(
            whiteCovMat
            - torch.eye(*covMat.shape[-2:], device=covMat.device, dtype=covMat.dtype)
        )
    else:
        whiteCovMat = makeHermitian(whiteCovMat)
    return L, whiteCovMat, whiteRTFVecs


def noise_whitening_4_BOP(
    whiteningCovMat: torch.Tensor,
    covMat: torch.Tensor,
    RTFvecs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust noise whitening intended for Block Orthogonal Projection (BOP) scenarios.

    Always subtracts the identity matrix from the whitened covariance matrix.

    Args:
        whiteningCovMat (torch.Tensor): The covariance matrix used for whitening.
        covMat (torch.Tensor): The covariance matrix to be whitened.
        RTFvecs (torch.Tensor): Relative Transfer Functions to whiten.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The whitening matrix L, the whitened covariance matrix, and the whitened RTF vectors.
    """
    log.debug("Performing noise whitening for BOP.")
    cholesky_failed = False
    try:
        L = regularize(
            torch.linalg.cholesky(whiteningCovMat),
            reg_factor=1e-6,
        )
    except Exception as e:
        log.warning(f"Cholesky decomposition failed, falling back to matrix square root. Error: {e}")
        cholesky_failed = True
        L = regularize(
            matrixsqrth(whiteningCovMat),
            reg_factor=1e-6,
        )

    if not cholesky_failed:
        whiteRTFVecs = makeVectorUnitNorm(
            torch.linalg.solve_triangular(L, RTFvecs, upper=False, left=True)
        )
        # whiteCovMat = L^-1 * covMat * L^-H - I
        whiteCovMat = torch.linalg.solve_triangular(
            L,
            torch.linalg.solve_triangular(L.mH, covMat, upper=True, left=False),
            upper=False,
            left=True,
        )

    else:
        whiteRTFVecs = makeVectorUnitNorm(torch.linalg.solve(L, RTFvecs, left=True))
        # whiteCovMat = L^-1 * covMat * L^-H - I
        whiteCovMat = torch.linalg.solve(
            L,
            torch.linalg.solve(L.mH, covMat, left=False),
            left=True,
        )

    whiteCovMat = makeHermitian(
        whiteCovMat
        - torch.eye(*covMat.shape[-2:], device=covMat.device, dtype=covMat.dtype)
    )
    return L, whiteCovMat, whiteRTFVecs


def noise_whitening_noncholesky(
    whiteningCovMat: torch.Tensor,
    covMat: torch.Tensor,
    RTFvecs: torch.Tensor | None = None,
    subtract_identity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Whitens the covariance matrix using matrix square root instead of Cholesky decomposition.

    Args:
        whiteningCovMat (torch.Tensor): The covariance matrix used for whitening.
        covMat (torch.Tensor): The covariance matrix to be whitened.
        RTFvecs (Optional[torch.Tensor]): Optional Relative Transfer Functions to whiten. Defaults to None.
        subtract_identity (bool): Whether to subtract the identity matrix. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: The whitening matrix L, the whitened covariance matrix, and the whitened RTF vectors.
    """
    log.debug("Performing non-Cholesky noise whitening.")
    L = regularize(matrixsqrth(whiteningCovMat))
    whiteRTFVecs = (
        makeVectorUnitNorm(torch.linalg.solve(L, RTFvecs, left=True))
        if RTFvecs is not None
        else None
    )
    whiteCovMat = torch.linalg.solve(
        L.mH, torch.linalg.solve(L, covMat, left=True), left=False
    )
    if subtract_identity:
        whiteCovMat = makeHermitian(
            whiteCovMat
            - torch.eye(*covMat.shape[-2:], device=covMat.device, dtype=covMat.dtype)
        )
    else:
        whiteCovMat = makeHermitian(whiteCovMat)
    return L, whiteCovMat, whiteRTFVecs


def covarianceWhitening(
    whiteningCovMat: torch.Tensor, covMat: torch.Tensor
) -> torch.Tensor:
    """
    Estimates the RTF vector by whitening the target covariance matrix with the noise covariance matrix.

    Args:
        whiteningCovMat (torch.Tensor): The noise covariance matrix.
        covMat (torch.Tensor): The signal covariance matrix.

    Returns:
        torch.Tensor: The unit-norm RTF vector corresponding to the principal eigenvector of the whitened covariance matrix.
    """
    log.debug("Performing covariance whitening for RTF estimation.")
    Rnsqrt, Rw, _ = noise_whitening(whiteningCovMat=whiteningCovMat, covMat=covMat)
    return makeVectorUnitNorm(Rnsqrt @ peigvech(Rw))


def covarianceSubtraction(
    noiseCovMat: torch.Tensor, covMat: torch.Tensor
) -> torch.Tensor:
    """
    Estimates the RTF vector by subtracting the noise covariance matrix from the signal covariance matrix.

    Args:
        noiseCovMat (torch.Tensor): The noise covariance matrix.
        covMat (torch.Tensor): The signal covariance matrix.

    Returns:
        torch.Tensor: The unit-norm RTF vector corresponding to the principal eigenvector of the noiseless covariance matrix.
    """
    log.debug("Performing covariance subtraction for RTF estimation.")
    noiselessCovMat = noise_subtraction(
        subtractingCovMat=noiseCovMat, covMat=covMat, ensure_PSD=False
    )
    return makeVectorUnitNorm(peigvech(noiselessCovMat))