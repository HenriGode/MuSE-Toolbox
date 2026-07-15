import logging
from typing import Any

import torch

from muse_toolbox.models.rtf_estimation.estimators.base_rtf_estimator import (
    BaseRTFestimator,
)
from muse_toolbox.utils import (
    characteristic_subspace,
    check_broadcastable,
    makeMatrixUnitNorm,
    makeVectorUnitNorm,
    orthogonal_projection,
    randdir,
)

log = logging.getLogger(__name__)


class BOP(BaseRTFestimator):
    """
    RTF estimator using the BOP method.
    It uses random additional vectors and no noise handling.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form") -> None:
        super().__init__()
        self.mode = mode

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary."""
        return {
            "type": "BOP",
            "mode": self.mode,
        }

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates the RTF vector from the input statistics.

        Args:
            Rn (torch.Tensor): Noise covariance matrix.
            Ry (torch.Tensor): Noisy covariance matrix.
            G (torch.Tensor): Previous RTF estimates.
            Rv (torch.Tensor): Previous noisy covariance matrix.

        Returns:
            torch.Tensor: Estimated RTF vector.
        """
        # Step 1: Noise handling
        R, G, Rnsqrt = self.noise_handling(Rn, Ry, G)
        # Step 1.1: Normalize R and G to avoid numerical issues
        R = makeMatrixUnitNorm(R)
        G = makeVectorUnitNorm(G)
        # Step 1.2: Ensure G has equal dimensions as R
        broadcast_dims = check_broadcastable(G.shape[:-2], R.shape[:-2])
        if isinstance(broadcast_dims, bool):
            raise ValueError(
                "G and R are not broadcastable in their leading dimensions."
            )
        G = G.expand(broadcast_dims + tuple(G.shape[-2:]))
        # Step 2: Generate additional vectors
        G = self.add_artificial_vectors(R, G)
        # Step 3: Estimate RTF
        h = self.estimate_rtf(R, G)
        # Step 4: Dewhitening (only for noise whitening)
        if Rnsqrt is not None:
            h = self.dewhitening(Rnsqrt, h)

        return h

    def estimate_rtf(self, R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Estimates the RTF using the chosen operational mode.

        Args:
            R (torch.Tensor): Handled covariance matrix.
            G (torch.Tensor): Handled previous RTF estimates.

        Returns:
            torch.Tensor: Estimated RTF vector.

        Raises:
            NotImplementedError: If the gradient mode is selected but not implemented.
            ValueError: If an unknown mode is provided.
        """
        if self.mode == "closed-form":
            return self.closed_form(R, G)
        elif self.mode == "gradient":
            raise NotImplementedError("Gradient mode not implemented yet.")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @staticmethod
    def noise_handling(
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Performs noise handling for the BOP estimator.
        Since BOP does not natively handle noise, it returns the noisy covariance unaltered.

        Args:
            Rn (torch.Tensor): Noise covariance matrix.
            Ry (torch.Tensor): Noisy covariance matrix.
            G (torch.Tensor): Previous RTF estimates.

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Handled covariance matrix, RTFs, and optionally the whitening matrix.
        """
        # BOP does not handle noise, so we simply return Ry and G
        return Ry, G, None

    @staticmethod
    def dewhitening(
        Rnsqrt: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dewhitens the estimated RTF vector.

        Args:
            Rnsqrt (torch.Tensor): Whitening matrix.
            h (torch.Tensor): Whitened RTF estimate.

        Returns:
            torch.Tensor: Dewhitened RTF estimate.
        """
        return makeVectorUnitNorm(Rnsqrt @ h)

    def add_artificial_vectors(self, R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Adds artificial vectors to the previous RTF estimates to match dimensions.

        Args:
            R (torch.Tensor): Covariance matrix.
            G (torch.Tensor): Previous RTF estimates.

        Returns:
            torch.Tensor: Concatenated RTF estimates and artificial vectors.
        """
        Ka = G.shape[-2] - G.shape[-1] - 1
        Ga = self.generate_add_vecs(G, R, Ka)
        return torch.cat([G, Ga], dim=-1)

    @staticmethod
    def generate_add_vecs(G: torch.Tensor, R: torch.Tensor, Ka: int) -> torch.Tensor:
        """
        Generates random additional vectors to append to the existing RTF set.

        Args:
            G (torch.Tensor): Previous RTF estimates.
            R (torch.Tensor): Covariance matrix.
            Ka (int): Number of additional vectors to generate.

        Returns:
            torch.Tensor: Generated artificial vectors.
        """
        return randdir(G.shape[:-1] + (Ka,), device=G.device, dtype=G.dtype)

    @staticmethod
    def closed_form(R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Computes the closed-form BOP solution.

        Args:
            R (torch.Tensor): Covariance matrix.
            G (torch.Tensor): Existing RTFs and artificial vectors.

        Returns:
            torch.Tensor: Estimated RTF vector.
        """
        M = R.shape[-1]
        Kold = G.shape[-1]
        if Kold == M - 1:
            return makeVectorUnitNorm(
                torch.mean(R @ orthogonal_projection(G), dim=-1, keepdim=True)
            )
        else:
            return characteristic_subspace(R @ orthogonal_projection(G), left=True)

    @staticmethod
    def cost_function(
        h: torch.Tensor,
        R: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the BOP cost function for gradient-based estimation.

        Args:
            h (torch.Tensor): Current RTF estimate.
            R (torch.Tensor): Covariance matrix.
            G (torch.Tensor): Existing RTFs.

        Returns:
            torch.Tensor: Computed cost.

        Raises:
            NotImplementedError: As it is not yet implemented.
        """
        # Implementation of the BOP cost function goes here
        raise NotImplementedError("Cost function not implemented yet.")
