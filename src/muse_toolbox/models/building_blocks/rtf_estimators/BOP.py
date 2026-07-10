import torch
from .base_rtf_estimator import BaseRTFestimator
from muse_toolbox.utils import (
    makeVectorUnitNorm,
    orthogonal_projection,
    randdir,
    characteristic_subspace,
    makeMatrixUnitNorm,
    check_broadcastable,
)
from typing import Optional


class BOP(BaseRTFestimator):
    """
    RTF estimator using the BOP method.
    It uses random additional vectors and no noise handling.

    Args:
        mode (str): Mode of operation, either "closed-form" or "gradient".
    """

    def __init__(self, mode: str = "closed-form"):
        super().__init__()
        self.mode = mode

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ) -> torch.Tensor:

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
        G = self.add_artifical_vectors(R, G)
        # Step 3: Estimate RTF
        h = self.esimtate_rtf(R, G)
        # Step 4: Dewhitening (only for noise whitening)
        if Rnsqrt is not None:
            h = self.dewhitening(Rnsqrt, h)

        return h

    def esimtate_rtf(self, R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
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
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # BOP does not handle noise, so we simply return Ry and G
        return Ry, G, None

    @staticmethod
    def dewhitening(
        Rnsqrt: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        return makeVectorUnitNorm(Rnsqrt @ h)

    def add_artifical_vectors(self, R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        Ka = G.shape[-2] - G.shape[-1] - 1
        Ga = self.generate_add_vecs(G, R, Ka)
        return torch.cat([G, Ga], dim=-1)

    @staticmethod
    def generate_add_vecs(G: torch.Tensor, R: torch.Tensor, Ka: int) -> torch.Tensor:
        return randdir(G.shape[:-1] + (Ka,), device=G.device, dtype=G.dtype)

    @staticmethod
    def closed_form(R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
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
    ):
        # Implementation of the BOP cost function goes here
        raise NotImplementedError("Cost function not implemented yet.")
