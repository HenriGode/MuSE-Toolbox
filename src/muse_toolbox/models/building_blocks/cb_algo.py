import torch
import torch.nn as nn
from utilities.math4torch import *
from utilities.sigproc4torch import *
from utilities.model_utils import *
from typing import List, Tuple, Union
import warnings


class CovarianceBlocking(nn.Module):
    def __init__(
        self,
        computation_method: str = "closed_form",  # 'closed-form' or 'gradient-descent'
        additional_vectors: Union[
            str, List[str]
        ] = "none",  # ['none', 'orthogonal', 'random']
        noise_handling: Union[
            str, List[str]
        ] = "whitening",  # ['whitening', 'subtraction', 'none']
    ):
        """
        Initialize the RTFEstimator module.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output RTF vector.
        """
        super().__init__()
        ## Choice of RTF estimation method(s)
        # Choice of computation method
        # 'closed-form' or 'gradient-descent'
        self.computation_method = computation_method
        # Choice of additional vector(s)
        # 'none', 'orthogonal' or 'random'
        self.additional_vectors = (
            additional_vectors
            if isinstance(additional_vectors, list)
            else [additional_vectors]
        )
        # Choice of noise handling method(s)
        # 'whitening', 'subtraction' or 'none'
        self.noise_handling = (
            noise_handling if isinstance(noise_handling, list) else [noise_handling]
        )

    def BOPcostfun(
        self, R: torch.Tensor, G: torch.Tensor, h: torch.Tensor, comp_grad: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        P_Gh = oblique_projection(G, h)
        cost = torch.real(trace(P_Gh @ R @ P_Gh.mH))
        if not comp_grad:
            return cost
        else:
            return cost, self.BOPgradient(R, G, h)

    def BOPgradient(
        self, R: torch.Tensor, G: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        Poblique_Gh = oblique_projection(G, h)
        summand_1 = -(
            Poblique_Gh.mH
            @ (
                Poblique_Gh
                @ (
                    R
                    @ (
                        (
                            torch.eye(
                                *Poblique_Gh.shape[-2:],
                                device=Poblique_Gh.device,
                                dtype=Poblique_Gh.dtype,
                            )
                            - Poblique_Gh.mH
                        )
                        @ (h / (torch.linalg.vector_norm(h, dim=-2, keepdim=True) ** 2))
                    )
                )
            )
        )
        if G.shape[-1] == G.shape[-2] - 1:
            return summand_1
        else:
            Porthogonal_GH = orthogonal_projection(generalized_cat([G, h], dim=-1))
            Ph = orthogonal_projection(h)
            summand_2 = -(
                Porthogonal_GH
                @ (
                    R
                    @ (
                        Poblique_Gh.mH
                        @ (
                            G
                            @ torch.linalg.solve(
                                G.mH @ (Ph @ G),
                                (
                                    G.mH
                                    @ (
                                        h
                                        / torch.linalg.vector_norm(
                                            h, dim=-2, keepdim=True
                                        )
                                        ** 2
                                    )
                                ),
                            )
                        )
                    )
                )
            )
            return summand_1 + summand_2

    def closedForm(self, R: torch.Tensor, Glist: List[torch.Tensor]) -> torch.Tensor:
        hlist = []
        for G in Glist:
            if (
                G.shape[-1] == G.shape[-2] - 1
            ):  # "G should be of dimension M x M-1, where M is the number of microphones!"
                hlist += [
                    makeVectorUnitNorm(
                        torch.mean(R @ orthogonal_projection(G), dim=-1, keepdim=True)
                    )
                ]
            else:
                hlist += [
                    characteristic_subspace(R @ orthogonal_projection(G), left=True)
                ]
        return torch.cat(hlist, dim=0)

    def gradientDescent(self):
        pass

    def forward(self, x: dict[str, torch.Tensor]):
        """
        Forward pass of the CovarianceBlocking module.

        Args:
            x (dict[str, torch.Tensor]):
            Input containing:
            - Ry: Covariance matrix of the mixture, shape (..., 1, NH, F, [T,1], M, M)
            - Rn: Covariance matrix of the noise, shape (..., 1, 1, F, [T,1], M, M)
            - Gy: RTF vectors of already known sources, shape (..., AV, NH, F, [T,1], M, N-1)
              where:
              - M is the number of microphones
              - N is the number of sources
              - F is the number of frequency bins
              - T is the number of time frames
              - AV is the number of additional vector methods
              - NH is the number of noise handling methods

        Returns:
            h (torch.Tensor): estimated RTF of the new source (..., AV, NH, F, [T,1], M, 1).
        """
        # Check input types and dimensions
        Ry, Rn, Gy = x["mixCovMat"], x["noiseCovMat"], x["oldSourceRTFs"]
        if not all(isinstance(t, torch.Tensor) for t in [Ry, Rn, Gy]):
            raise TypeError("Ry, Rn, and Gy must all be torch.Tensor objects.")
        # Check dimensions
        AV_Ry, NH_Ry, F_Ry, T_Ry, M1_Ry, M2_Ry = Ry.shape[-6:]
        AV_Rn, NH_Rn, F_Rn, T_Rn, M1_Rn, M2_Rn = Rn.shape[-6:]
        AV_Gy, NH_Gy, F_Gy, T_Gy, M_Gy, N_Gy_minus_1 = Gy.shape[-6:]
        if not (
            (
                AV_Ry == 1
                and AV_Rn == 1
                and (AV_Gy == 1 or AV_Gy == len(self.additional_vectors))
            )
            and (
                NH_Ry == 1
                and NH_Rn == 1
                and (NH_Gy == 1 or NH_Gy == len(self.noise_handling))
            )
            and (F_Ry == F_Rn == F_Gy)
            and (T_Rn in [T_Ry, 1] and T_Gy in [T_Ry, 1])
            and (M1_Ry == M1_Rn == M_Gy == M2_Ry == M2_Rn)
        ):
            raise ValueError(
                "Input tensors must have the correct dimensions: Ry: (..., 1, 1, F, T, M, M), "
                "Rn: (..., 1, 1, F, [T,1], M, M), Gy: (..., [AV,1], [NH,1], F, [T,1], M, N-1)."
            )
        if not check_broadcastable(Ry.shape[:-2], Rn.shape[:-2], Gy.shape[:-2]):
            raise ValueError(
                f"Input tensors of shapes {Ry.shape}, {Rn.shape}, {Gy.shape} are not broadcastable."
            )
        if not Gy.shape[-6] in [len(self.additional_vectors), 1]:
            raise ValueError(
                f"Expected dimension -6 of the RTF vectors of already known sources Gy corresponding to the additional vector methods {self.additional_vectors} to be {len(self.additional_vectors)} or 1, but got {Gy.shape[-6]}."
            )
        if not Gy.shape[-5] in [len(self.noise_handling), 1]:
            raise ValueError(
                f"Expected dimension -5 of the RTF vectors of already known sources Gy corresponding to the noise handling methods {self.noise_handling} to be {len(self.noise_handling)} or 1, but got {Gy.shape[-5]}."
            )
        # Ensure that the covariance matrices Ry and Rn are hermitian
        Ry, Rn = makeHermitian(Ry), makeHermitian(Rn)
        # Prepare the covariance matrices and RTF vectors according to the number of additional vectors and noise handling methods in terms of dimensions
        M, N = Gy.shape[-2:]
        AV, NH = len(self.additional_vectors), len(self.noise_handling)
        N += 1
        Na = M - N
        if Gy.shape[-6] != AV and Na > 0:
            Gy = Gy.repeat_interleave(AV, dim=-6)
        if Gy.shape[-5] != NH:
            Gy = Gy.repeat_interleave(NH, dim=-5)
        if NH != 1:
            Ry = Ry.repeat_interleave(NH, dim=-5)
        # Perform noise whitening if specified
        for idx, nh in enumerate(self.noise_handling):
            if nh == "subtraction":
                Ry[..., idx : idx + 1, :, :, :, :] = (
                    Ry[..., idx : idx + 1, :, :, :, :] - Rn
                )
            elif nh == "whitening":
                (
                    Rnsqrt,
                    Ry[..., idx : idx + 1, :, :, :, :],
                    Gy[..., idx : idx + 1, :, :, :, :],
                ) = noise_whitening(
                    Rn,
                    Ry[..., idx : idx + 1, :, :, :, :],
                    Gy[..., idx : idx + 1, :, :, :, :],
                )
        # Normalize Ry for numerical stability and ensure hermitian if not already
        Ry = makeMatrixUnitNorm(makeHermitian(Ry))
        # Add the additional vectors to the RTF vectors of already known sources
        Ga_list = []
        if Na > 0:
            for idx, av in enumerate(self.additional_vectors):
                gy = (
                    Gy[..., idx : idx + 1, :, :, :, :, :].repeat_interleave(
                        Ry.shape[-3], dim=-3
                    )
                    if (Gy.shape[-3] != Ry.shape[-3] and av in ["random", "orthogonal"])
                    else Gy[..., idx : idx + 1, :, :, :, :, :]
                )
                if av == "random":
                    Ga_list.append(
                        torch.cat(
                            [
                                gy,
                                randdir(
                                    gy.shape[:-1] + (Na,),
                                    device=gy.device,
                                    dtype=gy.dtype,
                                ),
                            ],
                            dim=-1,
                        )
                    )
                elif av == "orthogonal":
                    Ga_list.append(
                        torch.cat(
                            [
                                gy,
                                characteristic_subspace_h(Ry, order=range(-Na, 0)),
                            ],
                            dim=-1,
                        )
                    )
                elif av == "none":
                    Ga_list.append(gy)
        else:
            Ga_list.append(Gy)
            warnings.warn(
                "No additional vectors required for the RTF estimation. Outputs of all additional vector methods will be the same."
            )
        # Compute the RTF vectors using the specified computation method
        if self.computation_method == "closed_form":
            h = self.closedForm(Ry, Ga_list)
        else:
            raise NotImplementedError(
                f"Computation method {self.computation_method} not implemented."
            )
        # Perform noise dewhitening if specified
        for idx, nh in enumerate(self.noise_handling):
            if nh == "whitening":
                h[..., idx : idx + 1, :, :, :, :] = (
                    Rnsqrt @ h[..., idx : idx + 1, :, :, :, :]
                )
        # Make the RTF vectors unit norm
        h = makeVectorUnitNorm(h)
        # Repeat the RTF vectors of already known sources for all additional vector methods when additional vectors are not needed. So the methods make not difference and we saved computation time in only computing one of them and copying the result here.
        if Na == 0:
            if h.shape[-6] == 1:
                h = h.repeat_interleave(AV, dim=-6)
            else:
                raise ValueError(
                    f"Expected dimension -6 of the RTF vectors of already known sources Gy corresponding to the additional vector methods {self.additional_vectors} to be 1 if number of sources matches number of microphones, but got {h.shape[-6]}."
                )

        return h


class CBwrapper(nn.Module):

    def __init__(
        self,
        transform: STFTtransform,
        smoothing_time_constant: float,  # [s]
        only_noise_period: float,  # [s]
        fix_old_RTF_method: str,  # ['last', 'half']
        computation_method: str = "closed_form",  # 'closed-form' or 'gradient-descent'
        additional_vectors: Union[
            str, List[str]
        ] = "none",  # ['none', 'orthogonal', 'random']
        noise_handling: Union[
            str, List[str]
        ] = "whitening",  # ['whitening', 'subtraction', 'none']
    ):
        """
        Initialize the RTFEstimator module.

        Args:

        """
        super().__init__()
        ## Processing parameters
        self.transform = transform
        self.smoothing_time_constant = smoothing_time_constant
        self.smoothing_factor = self.transform.timeConstant2smoothingFactor(
            self.smoothing_time_constant
        )
        self.only_noise_period = only_noise_period
        self.fix_old_RTF_method = fix_old_RTF_method
        ## Choice of RTF estimation method(s)
        # Choice of computation method
        # 'closed-form' or 'gradient-descent'
        self.computation_method = computation_method
        # Choice of additional vector(s)
        # 'none', 'orthogonal' or 'random'
        self.additional_vectors = (
            additional_vectors
            if isinstance(additional_vectors, list)
            else [additional_vectors]
        )
        # Choice of noise handling method(s)
        # 'whitening', 'subtraction' or 'none'
        self.noise_handling = (
            noise_handling if isinstance(noise_handling, list) else [noise_handling]
        )
        self.CB = CovarianceBlocking(
            computation_method=self.computation_method,
            additional_vectors=self.additional_vectors,
            noise_handling=self.noise_handling,
        )

    def forward(self, x: dict[str, torch.Tensor]):
        mix = x["input"]
        activation_times = x["activation_times"]
        # Convert the activation times to segment borders on the frame level
        num_frames = mix.shape[-1]
        segment_borders = self.transform.times2frames(activation_times)
        segment_borders = torch.cat(
            [
                torch.zeros_like(segment_borders[..., :1]),
                segment_borders,
                torch.full_like(segment_borders[..., :1], num_frames - 1),
            ],
            dim=-1,
        )
        # Compute the frame-wise temporally smooth covariance matrix of the mixture
        smoothCovMat_mix = smoothCovarianceMatrix(
            mix, smoothing_factor=self.smoothing_factor
        )

        batch_smoothCovMat_mix_segmented = [
            [
                seg.transpose(-3, -1)
                for seg in slice2segments(
                    smoothCovMat_mix[seg_idx].transpose(-3, -1),
                    segment_borders=seg_borders,
                )
            ]
            for seg_idx, seg_borders in enumerate(segment_borders)
        ]

        # Index correction according to the frame length and shift and frame position 'center'
        idx_correction = int(
            self.transform.frame_length / self.transform.frame_shift / 2
        )
        # Estimating RTFs for each item in batch
        estimated_RTFs = []
        for smoothCovMat_mix_segmented in batch_smoothCovMat_mix_segmented:
            h = []
            for seg_idx, Ry in enumerate(smoothCovMat_mix_segmented):
                if seg_idx == 0:
                    if self.only_noise_period > activation_times.min():
                        raise ValueError(
                            f"Only noise period ({self.only_noise_period} s) is longer than the only noise segment in the beginning of at least one scenario ({activation_times.min()} s)."
                        )
                    Rn = Ry[
                        ...,
                        [
                            self.transform.times2frames(self.only_noise_period)
                            - idx_correction
                        ],
                        :,
                        :,
                    ]
                elif (
                    seg_idx == 1
                ):  # Try to generalize by not using covarianceWhitening for seg_idx == 1 but already the multi-source methods which should generalize to the single-source case.
                    h.append(covarianceWhitening(whiteningCovMat=Rn, covMat=Ry))
                    if self.fix_old_RTF_method == "last":
                        fix_idx = -1 - idx_correction
                    elif self.fix_old_RTF_method == "half":
                        fix_idx = h[-1].shape[-3] // 2
                    Gy = h[-1][..., [fix_idx], :, :]
                elif seg_idx > 1:
                    if seg_idx == 2:
                        Gy = Gy.repeat_interleave(
                            len(self.additional_vectors), dim=-6
                        ).repeat_interleave(len(self.noise_handling), dim=-5)
                    x = {
                        "mixCovMat": Ry,
                        "noiseCovMat": Rn,
                        "oldSourceRTFs": Gy,
                    }
                    h.append(self.CB(x))
                    Gy = torch.cat([Gy, h[-1][..., [fix_idx], :, :]], dim=-1)
            estimated_RTFs.append(h)

        return estimated_RTFs
