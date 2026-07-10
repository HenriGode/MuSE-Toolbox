import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pyroomacoustics as pra
import anf_generator as anf
from .math4torch import *
from typing import Union, Optional
from muse_toolbox.utils import CodeTimer


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# %% Transforms


class STFTtransform:
    """
    Short-Time Fourier Transform (STFT) Framework for encoding and decoding signals.

    Attributes:
        frame_length (float): Frame length in seconds.
        frame_shift (float): Frame shift in seconds.
        sampling_frequency (float): Sampling frequency in Hz.
        window_type (str): Type of window function used.
        nfft (int): Number of FFT points.
        hop_length (int): Number of samples between successive frames.
        window (Tensor): Window tensor generated from window_type.
    """

    def __init__(
        self,
        frame_length: float = 32e-3,
        frame_shift: float = 16e-3,
        sampling_frequency: float = 16e3,
        window_type: str = "torch.hann_window(x).sqrt()",
        remove_DC: bool = True,
        remove_Nyquist: bool = True,
    ) -> None:
        """Initialize STFT transform with given parameters."""
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.sampling_frequency = sampling_frequency
        self.window_type = window_type
        self.remove_DC = remove_DC
        self.remove_Nyquist = remove_Nyquist

        # Calculate hop length and FFT size
        self.nfft = int(self.frame_length * self.sampling_frequency)
        self.hop_length = int(self.frame_shift * self.sampling_frequency)

        self.num_freq_bins = (
            self.nfft // 2 + 1 - int(self.remove_DC) - int(self.remove_Nyquist)
        )

        # Generate window function
        if isinstance(window_type, str):
            if window_type == "hann":
                self.window = torch.hann_window(
                    self.nfft,
                    periodic=True,
                    dtype=torch.get_default_dtype(),
                )
            elif window_type == "sqrt-hann":
                self.window = torch.hann_window(
                    self.nfft,
                    periodic=True,
                    dtype=torch.get_default_dtype(),
                ).sqrt()
            elif window_type == "hamming":
                self.window = torch.hamming_window(
                    self.nfft,
                    periodic=True,
                    dtype=torch.get_default_dtype(),
                )
            elif window_type == "bartlett":
                self.window = torch.bartlett_window(
                    self.nfft,
                    periodic=True,
                    dtype=torch.get_default_dtype(),
                )
            elif window_type == "blackman":
                self.window = torch.blackman_window(
                    self.nfft,
                    periodic=True,
                    dtype=torch.get_default_dtype(),
                )
            else:
                try:
                    self.window = eval(
                        self.window_type,
                        {
                            "x": self.nfft,
                            "periodic": True,
                            "dtype": torch.get_default_dtype(),
                            "torch": torch,
                        },
                    )
                except:
                    raise ValueError("unknown window type!")
        elif callable(window_type):
            self.window = window_type(
                self.frame_length,
                periodic=True,
                dtype=torch.get_default_dtype(),
            )
        elif type(window_type) is torch.Tensor:
            self.window = window_type
        else:
            raise NotImplementedError()

    def _verbose_parameters(self, indent: str = ""):
        print(f"{indent}{self.__class__.__name__} Parameters:")
        print(f"{indent}  Frame Length: {self.frame_length} s")
        print(f"{indent}  Frame Shift: {self.frame_shift} s")
        print(f"{indent}  Sampling Frequency: {self.sampling_frequency} Hz")
        print(f"{indent}  Window Type: {self.window_type}")
        print(f"{indent}  NFFT: {self.nfft} samples")
        print(f"{indent}  Hop Length: {self.hop_length} samples")

    def get_config(self) -> dict:
        return {
            "frame_length": self.frame_length,
            "frame_shift": self.frame_shift,
            "sampling_frequency": self.sampling_frequency,
            "window_type": self.window_type,
        }

    @property
    def signature(self) -> str:
        return (
            f"STFT_fl{self.frame_length}_fs{self.frame_shift}_"
            f"sf{self.sampling_frequency}_wt{self.window_type}"
        )

    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply STFT to encode the input signal."""
        orig_shape = signal.shape  #  (..., M, N_samples)

        stft_signal = torch.stft(
            signal.reshape(-1, *orig_shape[-1:]),
            self.nfft,
            hop_length=self.hop_length,
            window=self.window.to(signal.device),
            center=True,
            onesided=True,
            return_complex=True,
        )

        # Remove DC and/or Nyquist bands if requested
        start_idx = 1 if self.remove_DC else 0
        end_idx = -1 if self.remove_Nyquist else None
        stft_signal = stft_signal[..., start_idx:end_idx, :]

        stft_signal_rs = stft_signal.reshape(
            *orig_shape[:-1], *stft_signal.shape[-2:]
        ).transpose(
            -2, -3
        )  # (..., F, M, T_frames)

        return stft_signal_rs

    def decode(
        self,
        stft_signal: torch.Tensor,
        DC_zero: bool = True,
        Nyquist_zero: bool = True,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply inverse STFT to decode the input signal."""
        stft_signal = stft_signal.transpose(-2, -3)
        orig_shape = list(stft_signal.shape)
        orig_shape[-2] = orig_shape[-2] + int(self.remove_DC) + int(self.remove_Nyquist)

        # Ensuring that the DC- & Nyquist-Band are zero
        pad_top = 1 if self.remove_DC else 0
        pad_bottom = 1 if self.remove_Nyquist else 0

        if pad_top > 0 or pad_bottom > 0:
            stft_signal = torch.nn.functional.pad(
                stft_signal, (0, 0, pad_top, pad_bottom)
            )

        if not self.remove_DC and DC_zero:
            stft_signal[..., 0, :] = 0

        if not self.remove_Nyquist and Nyquist_zero:
            stft_signal[..., -1, :] = 0

        # Apply inverse STFT
        if memory(stft_signal) < 1024**3:
            time_signal = torch.istft(
                stft_signal.reshape(-1, *orig_shape[-2:]),
                self.nfft,
                hop_length=self.hop_length,
                window=self.window.to(stft_signal.device),
                center=True,
                onesided=True,
                return_complex=False,
            )
        else:
            time_signal = torch.stack(
                [
                    torch.istft(
                        stft_sig,
                        self.nfft,
                        hop_length=self.hop_length,
                        window=self.window.to(stft_signal.device),
                        center=True,
                        onesided=True,
                        return_complex=False,
                    )
                    for stft_sig in stft_signal.reshape(-1, *orig_shape[-2:])
                ]
            )

        time_signal_reshape = time_signal.reshape(*orig_shape[:-2], -1)

        # Truncate or zero pad in the time dimension if num_samples is specified
        if num_samples is not None:
            if time_signal_reshape.shape[-1] < num_samples:
                time_signal_reshape = torch.nn.functional.pad(
                    time_signal_reshape,
                    (0, num_samples - time_signal_reshape.shape[-1]),
                    "constant",
                    0,
                )
            elif time_signal_reshape.shape[-1] > num_samples:
                time_signal_reshape = time_signal_reshape[..., :num_samples]

        return time_signal_reshape

    def times2frames(
        self, time: Union[torch.Tensor, float, int, list, tuple], method="center"
    ) -> torch.Tensor:
        """Convert time (in seconds) to frame indices."""
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float64)
        match method:
            case "center":
                return (time / self.frame_shift).round().to(torch.int64)
            case _:
                raise ValueError(
                    f"Unknown method '{method}' for converting time to frames."
                )

    def times2samples(
        self, time: Union[torch.Tensor, float, int, list, tuple]
    ) -> torch.Tensor:
        """Convert time (in seconds) to sample indices."""
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float64)
        return (time * self.sampling_frequency).round().to(torch.int64)

    def frames2times(self, frames, method="center") -> torch.Tensor:
        """Convert frame indices to time (in seconds)."""
        match method:
            case "center":
                return (frames - 1) * self.frame_shift
            case _:
                raise ValueError(
                    f"Unknown method '{method}' for converting frames to time."
                )

    def samples2frames(self, samples: Union[torch.Tensor, int]) -> torch.Tensor:
        """Convert sample indices to frame indices."""
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples, dtype=torch.float64)
        return (samples / self.hop_length).round().to(torch.int64)

    def frames2samples(self, frames: Union[torch.Tensor, int]) -> torch.Tensor:
        """Convert frame indices to sample indices."""
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float64)
        return (frames * self.hop_length).round().to(torch.int64)

    def samples2frames_quantity(
        self, quantity: torch.Tensor, dim: int = -1, mode: str = "center"
    ) -> torch.Tensor:
        """
        Convert a sample-based quantity tensor to a frame-based quantity tensor.

        Args:
            quantity (torch.Tensor): Input tensor with a sample dimension.
            dim (int): The dimension corresponding to samples.
            mode (str): Method to select the frame value. 'center' picks the center sample of the frame.

        Returns:
            torch.Tensor: Tensor with the sample dimension replaced by the frame dimension.
        """
        num_samples = quantity.shape[dim]
        # Calculate number of frames corresponding to centered STFT (approx)
        num_frames = num_samples // self.hop_length + 1

        # Generate indices for the center of each frame (since center=True in STFT)
        # frame t centers at sample t * hop_length
        frame_indices = torch.arange(num_frames, device=quantity.device)
        sample_indices = (frame_indices * self.hop_length).long()

        # Clamp to valid range
        sample_indices = torch.clamp(sample_indices, max=num_samples - 1)

        return quantity.index_select(dim, sample_indices)

    def frames2samples_quantity(
        self, quantity: torch.Tensor, num_samples: int | None = None, dim: int = -1
    ) -> torch.Tensor:
        """
        Convert a frame-based quantity tensor to a sample-based quantity tensor.

        Uses nearest-neighbor interpolation to upsample frames to samples.

        Args:
            quantity (torch.Tensor): Input tensor with a frame dimension.
            num_samples (int, optional): Target number of samples. If None, inferred.
            dim (int): The dimension corresponding to frames.

        Returns:
            torch.Tensor: Tensor with the frame dimension replaced by the sample dimension.
        """
        # Move target dim to last for processing
        quantity_t = quantity.transpose(dim, -1)
        orig_shape = quantity_t.shape
        orig_dtype = quantity.dtype

        # Convert Bool to Float/Byte for interpolation support
        if quantity.dtype == torch.bool:
            quantity_processing = quantity_t.to(torch.uint8)
        else:
            quantity_processing = quantity_t

        # Flatten all other dimensions into batch
        reshaped = quantity_processing.reshape(
            -1, quantity_processing.shape[-1]
        ).unsqueeze(
            1
        )  # (Batch, 1, Frames)

        if num_samples is None:
            num_samples = (quantity_t.shape[-1] - 1) * self.hop_length

        # 1D Interpolation expects (Batch, Channels, Length)
        mode = "nearest"

        upsampled = torch.nn.functional.interpolate(
            reshaped, size=num_samples, mode=mode
        )

        # Reshape back
        output = upsampled.squeeze(1).reshape(*orig_shape[:-1], num_samples)

        # Restore original type and dimension order
        if orig_dtype == torch.bool:
            output = output.to(torch.bool)

        return output.transpose(dim, -1)

    def samples2times(self, samples: Union[torch.Tensor, int]) -> torch.Tensor:
        """Convert sample indices to time (in seconds)."""
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples, dtype=torch.float64)
        return samples / self.sampling_frequency

    def frequencies(self) -> torch.Tensor:
        """Get frequency bins."""
        freqs = torch.linspace(0, self.sampling_frequency / 2, self.nfft // 2 + 1)
        start_idx = 1 if self.remove_DC else 0
        end_idx = -1 if self.remove_Nyquist else None
        return freqs[start_idx:end_idx]

    def freqs2bins(self, freqs) -> torch.Tensor:
        """Convert given frequencies to STFT bin indices."""
        return (freqs / self.sampling_frequency * self.nfft).round().to(torch.int64) - 1

    def bins2freq(self, bins) -> torch.Tensor:
        """Convert STFT bin indices to frequencies."""
        if not isinstance(bins, torch.Tensor):
            bins = torch.tensor(bins, dtype=torch.float64)
        return bins * self.sampling_frequency / self.nfft

    def timeConstant2smoothingFactor(self, time_constant: float) -> float:
        """Convert time constant (in seconds) to a smoothing factor."""
        return np.exp(-self.frame_shift / time_constant)

    def plot(self, signal_length):
        """Placeholder for plotting the STFT representation."""
        plt.figure(dpi=600)
        # Further plotting implementation is needed here.


# %% Signal Statistics


def statistics(data: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.mean(data, dim=dim, keepdim=True)
    centered_data = data - mean
    if centered_data.is_complex():
        centered_data = torch.cat([centered_data, centered_data.conj()], dim=-2)
    covMat = covariance_Tyler(centered_data)
    return mean, covMat


def smoothCovarianceMatrix(
    stft_signal: torch.Tensor,
    smoothing_factor: float,
    init_cov: Optional[torch.Tensor] = None,
    init_smoothing_factor: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute the smoothed covariance matrix.

    Args:
        stft_signal (torch.Tensor): STFT signal of shape (..., F, M, T).
        smoothing_factor (float): Smoothing factor for exponential windowing.
        init_cov (Optional[torch.Tensor]): Initial covariance matrix for warm start. Shape (..., F, 1, M, M).
        init_smoothing_factor (Optional[float]): Smoothing factor used for initial covariance.

    Returns:
        torch.Tensor: Smoothed covariance matrix of shape (..., F, T, M, M).
    """

    instantaneous_cov_mat = covariance_SCM(stft_signal.transpose(-2, -1)[..., None])

    if init_cov is not None:
        if init_smoothing_factor is None:
            instantaneous_cov_mat = torch.cat([init_cov, instantaneous_cov_mat], dim=-3)
        else:
            gamma = init_smoothing_factor / smoothing_factor
            init_cov_weighted = (
                gamma * init_cov + (1 - gamma) * instantaneous_cov_mat[..., :1, :, :]
            )
            instantaneous_cov_mat = torch.cat(
                [init_cov_weighted, instantaneous_cov_mat], dim=-3
            )
    # else: # TODO: Regularization
    #     num_channels = instantaneous_cov_mat.shape[-2]
    #     instantaneous_cov_mat[..., :num_channels, :, :] = regularize(
    #         instantaneous_cov_mat[..., :num_channels, :, :], reg_factor=1e-1
    #     )

    smoothCov = exp_windowing_recursive(
        data=instantaneous_cov_mat,
        smoothing_factor=smoothing_factor,
        dim=-3,
    )

    if init_cov is not None:
        smoothCov = smoothCov[..., 1:, :, :]

    return make2covariance_matrix_rel_lower_bound(smoothCov, 1e-6)


def windowedCovarianceMatrix(
    stft_signal: torch.Tensor, window: torch.Tensor
) -> torch.Tensor:
    """Compute the windowed covariance matrix."""
    instantaneous_cov_mat = covariance_SCM(stft_signal.transpose(-2, -1)[..., None])
    num_channels = instantaneous_cov_mat.shape[-2]
    # instantaneous_cov_mat[..., :num_channels, :, :] = regularize(
    #     instantaneous_cov_mat[..., :num_channels, :, :], reg_factor=1e-1
    # )
    return make2covariance_matrix_rel_lower_bound(
        windowing(
            data=instantaneous_cov_mat,
            window=window.to(stft_signal.device),
            dim=-3,
        ),
        1e-6,
    )


def smoothCovarianceMatrix_conv(
    stft_signal: torch.Tensor, smoothing_factor: float
) -> torch.Tensor:
    instantaneous_cov_mat = covariance_SCM(stft_signal.transpose(-2, -1)[..., None])
    num_channels = instantaneous_cov_mat.shape[-2]
    instantaneous_cov_mat[..., :num_channels, :, :] = regularize(
        instantaneous_cov_mat[..., :num_channels, :, :], reg_factor=1e-1
    )
    return makeHermitian(
        exp_windowing_conv(
            data=instantaneous_cov_mat,
            smoothing_factor=smoothing_factor,
            dim=-3,
        )
    )


def coherenceMatrix(covMat: torch.Tensor) -> torch.Tensor:
    Dsqrtinv = vec2diagMat(1 / covMat.diagonal(dim1=-2, dim2=-1).sqrt()[..., None])
    return makeHermitian(Dsqrtinv @ covMat @ Dsqrtinv)


# def gmsc_fixcuda(
#     coherenceMat: torch.Tensor,
# ) -> torch.Tensor:  # generalized magnitude squared coherence
#     return (
#         (fixcuda(mytorch_eigvalsh, coherenceMat)[..., [-1]][..., None] - 1)
#         / (coherenceMat.shape[-1] - 1)
#     ) ** 2


def gmsc(
    coherenceMat: torch.Tensor,
) -> torch.Tensor:  # generalized magnitude squared coherence
    eigvals = mytorch_eigvalsh(coherenceMat)
    return ((eigvals[..., [-1]][..., None] - 1) / (coherenceMat.shape[-1] - 1)) ** 2


def noise_subtraction(
    subtractingCovMat: torch.Tensor, covMat: torch.Tensor, ensure_PSD: bool = False
) -> torch.Tensor:
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
) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
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
    RTFvecs: Optional[torch.Tensor] = None,
    subtract_identity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
    cholesky_failed = False
    try:
        # L = regularize(
        #     torch.linalg.cholesky(regularize(whiteningCovMat, reg_factor=1e-5)),
        #     reg_factor=1e-6,
        # )
        L = regularize(
            torch.linalg.cholesky(whiteningCovMat),
            reg_factor=1e-6,
        )
    except Exception as e:
        print(
            f"Cholesky decomposition failed, falling back to matrix square root. Error: {e}"
        )
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
    cholesky_failed = False
    try:
        L = regularize(
            torch.linalg.cholesky(whiteningCovMat),
            reg_factor=1e-6,
        )
    except Exception as e:
        print(
            f"Cholesky decomposition failed, falling back to matrix square root. Error: {e}"
        )
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
) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
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
    Rnsqrt, Rw, _ = noise_whitening(whiteningCovMat=whiteningCovMat, covMat=covMat)
    return makeVectorUnitNorm(Rnsqrt @ peigvech(Rw))


def covarianceSubtraction(
    noiseCovMat: torch.Tensor, covMat: torch.Tensor
) -> torch.Tensor:
    noiselessCovMat = noise_subtraction(
        subtractingCovMat=noiseCovMat, covMat=covMat, ensure_PSD=False
    )
    return makeVectorUnitNorm(peigvech(noiselessCovMat))


def covarianceBlockingWhitening(
    noisyCovMat: torch.Tensor, noiseCovMat: torch.Tensor, oldRTFvecs: torch.Tensor
) -> torch.Tensor:
    """
    !!! Old code version !!! slow because of cpu_gen_solve
    Implements the Covariance Blocking and Whitening (CBW) method for successive Relative Transfer Function (RTF)
    vector estimation in multi-speaker scenarios.

    Reference:
    Gode, H., & Doclo, S. (2023, October).
    Covariance Blocking and Whitening Method for Successive Relative Transfer Function Vector Estimation in Multi-Speaker Scenarios.
    In 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) (pp. 1-5). IEEE.

    Parameters:
        noisyCovMat (torch.Tensor): The noisy covariance matrix.
        noiseCovMat (torch.Tensor): The noise covariance matrix.
        oldRTFvecs (torch.Tensor): Previously estimated RTF vectors.

    Returns:
        torch.Tensor: The updated RTF vector with unit norm.
    """
    # Determine the number of sources, microphones, and equations for the system
    num_sources = (
        oldRTFvecs.shape[-1] + 1
    )  # Number of sources (adding one to account for the current source)
    num_mics = oldRTFvecs.shape[-2]  # Number of microphones
    num_equations = 2 * (
        num_mics - num_sources + 1
    )  # Required number of equations for the system

    # Assert that the system is overdetermined for a valid solution
    assert (
        num_equations > num_mics
    ), "The number of equations must be greater than the number of microphones."

    # Compute the orthogonal projection matrix that blocks the previously estimated RTF vectors
    Pgr = orthogonal_projection(oldRTFvecs)[..., : -(num_sources - 1)]

    # Transform the noise covariance matrix with the projection matrix
    RnPgr = noiseCovMat @ Pgr
    RnPgr_pinv = torch.linalg.pinv(
        RnPgr
    )  # Compute the pseudo-inverse of the transformed noise covariance matrix

    # Compute the whitened residual matrix
    Rw = RnPgr_pinv @ noisyCovMat @ Pgr - torch.eye(
        num_mics - num_sources + 1, device=noisyCovMat.device, dtype=noisyCovMat.dtype
    )

    # Perform Singular Value Decomposition (SVD) on the residual matrix
    QL, _, QR = torch.linalg.svd(Rw)
    qL = QL[..., [0]]  # Extract the principal left  singular vector
    qR = QR.mH[
        ..., [0]
    ]  # Extract the principal right singular vector (Hermitian transpose)

    # Construct the augmented matrix B for solving the linear system
    B = torch.cat([RnPgr_pinv, Pgr.mH], dim=-2)

    # Compute the orthogonal projection matrix PB for the augmented matrix B
    PB = orthogonal_projection(B)
    PBL = PB[
        ..., : (num_mics - num_sources + 1)
    ]  # Left  partition of the projection matrix
    PBR = PB[
        ..., (num_mics - num_sources + 1) :
    ]  # Right partition of the projection matrix

    # Solve for the scaling factor alpha using the partitions of PB and the singular vectors
    alpha = -cpu_gen_solve(PBR @ qR, PBL @ qL)

    # Solve for the updated RTF vector and normalize it to unit norm
    updated_RTF = cpu_gen_solve(B, torch.cat([qL, qR * alpha], dim=-2))
    return makeVectorUnitNorm(updated_RTF)


def test_covarianceBlockingWhitening():
    """
    Tests the covarianceBlockingWhitening function with synthetic data.
    """
    device = "cuda:0"
    M = 5  # Number of microphones
    N = 100000  # Number of samples

    # Generate random RTF vectors and noise
    g1 = randdir(M, 1)
    g2 = randdir(M, 1)
    g3 = randdir(M, 1)
    n = randdir(M, N)

    # Generate microphone signals for various sources
    x1 = g1 * torch.randn(1, N, device=device)
    x2 = g2 * torch.randn(1, N, device=device)
    x3 = g3 * torch.randn(1, N, device=device)
    y1 = x1 + n
    y2 = x1 + x2 + n
    y3 = x1 + x2 + x3 + n

    # Compute covariance matrices
    Rn = covariance_SCM(n)
    Rx1 = covariance_SCM(x1)
    Rx2 = covariance_SCM(x2)
    Rx3 = covariance_SCM(x3)
    Ry_1 = Rn + Rx1
    Ry_2 = Rn + Rx1 + Rx2
    Ry_3 = Rn + Rx1 + Rx2 + Rx3
    Ry1 = covariance_SCM(y1)
    Ry2 = covariance_SCM(y2)
    Ry3 = covariance_SCM(y3)

    # Perform CBW and CW
    h_1 = covarianceWhitening(whiteningCovMat=Rn, covMat=Ry_1)
    h_2 = covarianceBlockingWhitening(noisyCovMat=Ry_2, noiseCovMat=Rn, oldRTFvecs=h_1)
    h_3 = covarianceBlockingWhitening(
        noisyCovMat=Ry_3, noiseCovMat=Rn, oldRTFvecs=torch.cat([h_1, h_2], dim=-1)
    )
    h1 = covarianceWhitening(whiteningCovMat=Rn, covMat=Ry1)
    h2 = covarianceBlockingWhitening(noisyCovMat=Ry2, noiseCovMat=Rn, oldRTFvecs=h1)
    h3 = covarianceBlockingWhitening(
        noisyCovMat=Ry3, noiseCovMat=Rn, oldRTFvecs=torch.cat([h1, h2], dim=-1)
    )

    # Compare against known ground truth RTF vectors
    HA_1 = hermitian_angle(g1, h_1)
    HA_2 = hermitian_angle(g2, h_2)
    HA_3 = hermitian_angle(g3, h_3)
    HA1 = hermitian_angle(g1, h1)
    HA2 = hermitian_angle(g2, h2)
    HA3 = hermitian_angle(g3, h3)

    # Print results
    for angle in [HA_1, HA_2, HA_3, HA1, HA2, HA3]:
        print(f"Hermitian angle: {angle.item() / torch.pi * 180:.2f} degrees")

    # Assert expected behavior (e.g., angles close to zero for correct estimation)
    assert HA_1.item() < 1e-2, "Angle for g1 mismatch"
    assert HA_2.item() < 1e-2, "Angle for g2 mismatch"
    assert HA_3.item() < 1e-2, "Angle for g3 mismatch"


# %% Signal Enhancement Algorithms


def Beamformer(
    covMat2min: torch.Tensor,  # (..., F, T, M, M)
    RTFs4constraints: torch.Tensor,  # (..., F, M, K)
    gains: torch.Tensor,  # (..., K, 1)
    signal: torch.Tensor | None = None,  # (..., F, M, T)
) -> torch.Tensor:
    RTFs4constraints = makeVectorUnitNorm(RTFs4constraints)
    RinvC = torch.linalg.solve(
        makeHermitian(regularize(makeMatrixUnitNorm(covMat2min), 1e-6)),
        RTFs4constraints,
    )
    covMat2min = torch.empty_like(covMat2min)
    beamformer = RinvC @ torch.linalg.solve(
        regularize(RTFs4constraints.mH @ RinvC, 1e-6),
        torch.diag_embed(gains[..., 0]) @ RTFs4constraints.mH,
    )
    RTFs4constraints = torch.empty_like(RTFs4constraints)
    gains = torch.empty_like(gains)
    RinvC = None
    if signal is None:
        return beamformer
    else:
        if True:  # memory(beamformer) < 1024**3:
            return beamformer.mH @ signal
        else:
            return torch.cat(
                [
                    beamformer.mH[..., [frame], :, :] @ signal[..., [frame], :, :]
                    for frame in range(beamformer.shape[-3])
                ],
                dim=-3,
            )


# %%


def convolve_clean2microphone(
    clean: torch.Tensor, rirdata: torch.Tensor
) -> torch.Tensor:
    return torchaudio_functional_fftconvolve_complex(clean, rirdata, mode="full")[
        ..., : clean.shape[-1]
    ]


def convolve_white2microphone(
    rirdata: torch.Tensor, samples: int = 80000
) -> torch.Tensor:
    white = torch.randn([1, int(samples)], device=rirdata.device, dtype=rirdata.dtype)
    return torchaudio_functional_fftconvolve_complex(white, rirdata, mode="full")[
        ..., : white.shape[-1]
    ]


def rir2rtf(
    rir: torch.Tensor,
    transform: STFTtransform,
    ref_mic: Optional[int] = None,
    signal_len=1600000,
) -> torch.Tensor:

    directwhite = convolve_white2microphone(rir, samples=1600000)

    # Transform to STFT domain
    stftsig = transform.encode(directwhite)

    # Calculate Sample Covariance Matrix
    # Shape: (F, 1, M, M)
    covMat = covariance_SCM(stftsig)[..., None, :, :]

    # Extract Principal Eigenvector (RTF)
    # Shape: (F, 1, M, 1)
    rtf = peigvech(covMat)

    if ref_mic is not None:
        rtf = rtf / rtf[..., [ref_mic], :]

    return rtf


def slice2frames(
    signal: torch.Tensor, samples: int
) -> (
    torch.Tensor
):  # add one broadcast dim for time frames in the second last dim and the last dim remains samples
    dims = list(signal.shape)
    return signal[..., : (dims[-1] // samples) * samples].reshape(
        dims[:-1] + [dims[-1] // samples, samples]
    )


# %% Signal Evaluation


def vad_opt(
    audio: torch.Tensor, fs: float = 16000.0, thr: float = -30, min_on: float = 50e-3
) -> torch.Tensor:
    """
    Voice Activity Detection with gap bridging.

    This implementation thresholds energy and then bridges silences shorter than 'min_on'.
    Unlike simple extension, this does not prolong the end of an utterance artificially.
    """
    # raise Deprecation Error
    raise DeprecationWarning(
        "This function is deprecated. Please use vad_opt_fast_gen for improved performance."
    )
    raise NotImplementedError("This function is deprecated and not implemented.")
    # audio is a tensor of shape [num_signals, num_channels, num_samples]
    # Normalisation
    audio_centered = audio - torch.mean(audio, dim=-1, keepdim=True)
    max_vals = torch.max(torch.abs(audio_centered), dim=-1, keepdim=True)[0]
    audio_norm = audio_centered / (max_vals + 1e-8)
    # Compute energy in dB
    # Mean over channels
    energy_db = 10 * torch.log10(
        torch.mean(audio_norm.pow(2), dim=-2, keepdim=True) + 1e-8
    )
    # Initial binary determination
    vad = torch.ge(energy_db, thr)  # Shape: [num_signals, 1, num_samples]

    # Work on boolean or int representation for safety
    vad_int = vad.int()

    min_gap_samples = round(min_on * fs)
    if min_gap_samples <= 0:
        return vad
    # --- GAP BRIDGING (MORPHOLOGICAL CLOSING) ---
    # We want to turn 1-0-1 into 1-1-1 if the 0-part is short.

    # Alternative Efficient Float-Based Approach for Filling Small Gaps:
    # We can use a moving max (dilation) followed by a moving min (erosion).
    # Convolution with ones is Sum, Moving Max is Dilation.

    # Let's use MaxPool1d for Dilation (fill gaps) then MinPool (Erosion).
    # Since we want to preserve the original length and causality isn't strictly required
    # (usually offline VAD), we use centered windows.

    # Ideally: Mathematical Morphology Closing = Erosion(Dilation(X))

    # 1. Dilation: Expands 1s, filling 0s.
    # Kernel size = min_gap_samples. Stride 1. Padding same.
    # PyTorch's MaxPool1d is great here.

    # Ensure shape is [Batch, Channel, Time]
    # vad_int is [num_signals, 1, num_samples]

    # Kernel size logic:
    # If we want to bridge a gap of size K, we need a kernel of size K+1 roughly?
    # Actually, pure closing is best implemented via:
    #   Dilate: x[t] = max(window around t)
    #   Erode:  y[t] = min(window around t of x)

    pad_size = min_gap_samples // 2

    # DILATION (Max Pool)
    # We need to cover the gap.
    dilated = torch.nn.functional.max_pool1d(
        vad_int.float(), kernel_size=min_gap_samples, stride=1, padding=pad_size
    )

    # EROSION (Min Pool = -Max(-X))
    # This shrinks the expanded regions back, but filled holes remain filled.
    closed = -torch.nn.functional.max_pool1d(
        -dilated, kernel_size=min_gap_samples, stride=1, padding=pad_size
    )

    # Fix potential size mismatch due to odd/even kernel/padding
    if closed.shape[-1] != vad.shape[-1]:
        closed = closed[..., : vad.shape[-1]]

    return closed > 0.5


def vad_opt_fast(
    audio: torch.Tensor, fs: float = 16000.0, thr: float = -30, min_on: float = 50e-3
) -> torch.Tensor:
    """
    Voice Activity Detection with gap bridging.
    Optimized: Uses linear energy comparison to avoid expensive log10() on tensors.
    """

    # raise Deprecation Error
    raise DeprecationWarning(
        "This function is deprecated. Please use vad_opt_fast_gen for improved performance."
    )
    raise NotImplementedError("This function is deprecated and not implemented.")

    # 1. Normalization & Energy Computation (Linear Domain)

    # DC removal
    audio_centered = audio - torch.mean(audio, dim=-1, keepdim=True)

    # Calculate Max per signal per channel for normalization
    # Shape: [num_signals, num_channels, 1]
    max_vals = torch.max(torch.abs(audio_centered), dim=-1, keepdim=True)[0]

    # Denominator for normalization (max^2)
    # Add small epsilon to prevent division by zero
    denom = max_vals.pow(2) + 1e-16

    # Normalized Energy per channel/sample: (x^2) / (max^2)
    # Then take mean over channels
    # Shape: [num_signals, 1, num_samples]
    energy_norm = torch.mean(audio_centered.pow(2) / denom, dim=-2, keepdim=True)

    # 2. Thresholding (Linear Domain)
    # Convert dB threshold to linear power ratio
    # Formula: 10 * log10(E) >= thr_db  <==>  E >= 10^(thr_db / 10)
    # Note: We subtract the epsilon used in the original log formulation (1e-8),
    # but it's usually negligible compared to typical thresholds (e.g., -30dB = 1e-3).
    thr_lin = 10.0 ** (thr / 10.0)

    vad = energy_norm >= thr_lin

    # 3. Gap Bridging (Morphological Closing)
    min_gap_samples = round(min_on * fs)
    if min_gap_samples <= 0:
        return vad

    pad_size = min_gap_samples // 2
    vad_float = vad.float()

    # Dilation (Max Pool): Bridges gaps (fills 0s with 1s)
    dilated = torch.nn.functional.max_pool1d(
        vad_float, kernel_size=min_gap_samples, stride=1, padding=pad_size
    )

    # Erosion (Min Pool): Restores boundaries (shrinks 1s back)
    # Implemented as -Max(-X)
    closed = -torch.nn.functional.max_pool1d(
        -dilated, kernel_size=min_gap_samples, stride=1, padding=pad_size
    )

    # Handle potential size mismatch due to padding logic
    if closed.shape[-1] != vad.shape[-1]:
        closed = closed[..., : vad.shape[-1]]

    return closed > 0.5


def vad_opt_fast_gen(
    audio: torch.Tensor,
    fs: float = 16000.0,
    thr: float = -30,
    min_on: float = 50e-3,
    mode: str = "highpass",
    cutoff_freq: float = 80.0,
) -> torch.Tensor:
    """
    Generalized Voice Activity Detection with gap bridging.
    Optimized: Uses linear energy comparison to avoid expensive log10() on tensors.

    Args:
        mode (str): 'normal', 'highpass', or 'zeromask'
        cutoff_freq (float): Cutoff frequency for highpass mode (default: 50.0)
    """
    # 1. Normalization & Energy Computation (Linear Domain)

    if mode == "normal":
        # Standard DC removal
        audio_centered = audio - torch.mean(audio, dim=-1, keepdim=True)

    elif mode == "highpass":
        # Highpass filter (Biquad) instead of simple DC subtraction
        # Ensure audio is float for filtering
        # Note: torchaudio.functional.highpass_biquad expects tensor
        # If input is double/half, we cast to float for stability in IIR
        audio_float = audio.float()
        audio_centered = torchaudio.functional.highpass_biquad(
            audio_float, int(fs), cutoff_freq
        )
        if audio.dtype != audio_centered.dtype:
            audio_centered = audio_centered.to(dtype=audio.dtype)

    elif mode == "zeromask":
        # Identify non-zeros (perfect zeros are padding)
        mask = audio != 0

        # Compute mean on non-zeros
        # Sum over last dim (time)
        sum_vals = torch.sum(audio, dim=-1, keepdim=True)
        count_vals = torch.sum(mask, dim=-1, keepdim=True).float()

        # Avoid div by zero
        count_vals = count_vals.clamp(min=1.0)
        mean_vals = sum_vals / count_vals

        # Subtract mean ONLY from non-zero values. Zeros remain 0.
        # If mask is False (zero), result is 0 (since mask sets it to audio, which is 0).
        # audio - mean where mask is true, else audio (0)
        audio_centered = torch.where(mask, audio - mean_vals, audio)

    else:
        raise ValueError(f"Unknown VAD mode: {mode}")

    # Calculate Max per signal per channel for normalization
    # Shape: [num_signals, num_channels, 1]
    # max_vals = torch.max(torch.abs(audio_centered), dim=-1, keepdim=True)[0]
    max_vals = torch.quantile(torch.abs(audio_centered), 0.999, dim=-1, keepdim=True)

    # Denominator for normalization (max^2)
    # Add small epsilon to prevent division by zero
    denom = max_vals.pow(2) + 1e-16

    # Normalized Energy per channel/sample: (x^2) / (max^2)
    # Then take mean over channels
    # Shape: [num_signals, 1, num_samples]
    energy_norm = torch.mean(audio_centered.pow(2) / denom, dim=-2, keepdim=True)

    # 2. Thresholding (Linear Domain)
    # Convert dB threshold to linear power ratio
    # Formula: 10 * log10(E) >= thr_db  <==>  E >= 10^(thr_db / 10)
    # Note: We subtract the epsilon used in the original log formulation (1e-8),
    # but it's usually negligible compared to typical thresholds (e.g., -30dB = 1e-3).
    thr_lin = 10.0 ** (thr / 10.0)

    vad = energy_norm >= thr_lin

    # 3. Gap Bridging (Morphological Closing)
    min_gap_samples = round(min_on * fs)
    if min_gap_samples <= 0:
        return vad

    pad_size = min_gap_samples // 2
    vad_float = vad.float()

    # Dilation (Max Pool): Bridges gaps (fills 0s with 1s)
    dilated = torch.nn.functional.max_pool1d(
        vad_float, kernel_size=min_gap_samples, stride=1, padding=pad_size
    )

    # Erosion (Min Pool): Restores boundaries (shrinks 1s back)
    # Implemented as -Max(-X)
    closed = -torch.nn.functional.max_pool1d(
        -dilated, kernel_size=min_gap_samples, stride=1, padding=pad_size
    )

    # Handle potential size mismatch due to padding logic
    if closed.shape[-1] != vad.shape[-1]:
        closed = closed[..., : vad.shape[-1]]

    return closed > 0.5


def vad_opt_bidirectional(
    audio: torch.Tensor, fs: float = 16000.0, thr: float = -30, min_on: float = 50e-3
) -> torch.Tensor:

    raise DeprecationWarning(
        "vad_opt_bidirectional is deprecated.\n"
        "It potentially detects activity in the center of silent gaps.\n"
        "Use vad_opt instead."
    )

    vad_mask = vad_opt_original(audio, fs, thr, min_on)
    vad_mask_rev = vad_opt_original(torch.flip(audio, dims=[-1]), fs, thr, min_on)
    vad_mask_rev = torch.flip(vad_mask_rev, dims=[-1])
    # combine both masks with and operation
    vad_combined = vad_mask & vad_mask_rev
    return vad_combined


def vad_opt_original(
    audio: torch.Tensor, fs: float = 16000.0, thr: float = -30, min_on: float = 50e-3
) -> torch.Tensor:
    # audio is a tensor of shape [num_signals, num_channels, num_samples]

    # Normalisation
    audio = audio - torch.mean(audio, dim=-1, keepdim=True)
    audio = audio / torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]

    # Compute energy in dB
    energy_db = 10 * torch.log10(
        torch.mean(audio.pow(2), dim=1, keepdim=True) + 1e-8
    )  # Add epsilon for stability

    # Detect voice activity using threshold
    vad = torch.ge(energy_db, thr)

    # --- REPLACEMENT START ---
    # Extend each active period efficiently using cumulative sum
    min_period = round(min_on * fs)
    if min_period <= 1:
        return vad

    vad_float = vad.to(audio.dtype)
    num_samples = vad_float.shape[-1]

    # Find the start of each active segment.
    # A segment starts if the current value is 1 and the previous was 0.
    padded_vad = torch.nn.functional.pad(vad_float, (1, 0), "constant", 0)
    is_start = (padded_vad[..., 1:] > padded_vad[..., :-1]).to(audio.dtype)

    # Create markers: +1 at the start of an extension, -1 at the end.
    markers = torch.zeros_like(vad_float)
    markers += is_start  # Add +1 at the start

    # Create the -1 markers for the end of the extension period
    end_markers = torch.nn.functional.pad(is_start, (min_period, 0))[..., :num_samples]
    markers -= end_markers

    # The cumulative sum will create blocks of '1's where VAD should be active.
    vad_extended = torch.cumsum(markers, dim=-1) > 0
    # --- REPLACEMENT END ---

    return vad_extended


def vad_opt_slow(
    audio: torch.Tensor, fs: float = 16000.0, thr: float = -30, min_on: float = 50e-3
) -> torch.Tensor:
    # audio is a tensor of shape [num_signals, num_channels, num_samples]

    # Normalisation
    audio = audio - torch.mean(audio, dim=-1, keepdim=True)
    audio = audio / torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]

    # Compute energy in dB
    energy_db = 10 * torch.log10(torch.mean(audio.pow(2), dim=1, keepdim=True))

    # Detect voice activity using threshold
    vad = torch.ge(energy_db, thr)

    # Extend each active period
    min_period = round(min_on * fs)
    extension = torch.ones((1, 1, min_period), device=audio.device, dtype=audio.dtype)
    vad_extended = (
        torch.nn.functional.conv1d(
            vad.to(audio.dtype), extension, padding=min_period - 1
        )
        > 0
    )

    return vad_extended[:, :, : audio.shape[-1]]


def normalize_components(
    signal_components: torch.Tensor,
    vad: Union[torch.Tensor, None] = None,
    norm_power: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vad is None:
        norm_factors = norm_power / compute_rms(signal_components)
    else:
        norm_factors = norm_power / compute_rms(signal_components, vad)
    return signal_components * norm_factors, norm_factors


def compute_rms(
    signal_components: torch.Tensor, vad: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    if vad is None:
        return torch.sqrt(compute_power(signal_components=signal_components))
    else:
        return torch.sqrt(compute_power(signal_components=signal_components, vad=vad))


def compute_power(
    signal_components: torch.Tensor, vad: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    Computes the power of a multichannel signal, optionally using a VAD.
    Handles both per-channel VAD [..., C, N] and shared-channel VAD [..., 1, N].
    """
    if vad is None:
        return torch.mean(
            (torch.abs(signal_components) ** 2), dim=[-2, -1], keepdim=True
        )
    else:
        # Check if the VAD is shared (channel dim is 1) and the signal is multichannel
        num_channels = signal_components.shape[-2]
        if vad.shape[-2] == 1 and num_channels > 1:
            # Explicitly repeat the VAD along the channel dimension to match the signal
            repeat_dims = [1] * (vad.dim() - 2) + [num_channels, 1]
            vad = vad.repeat(*repeat_dims)

        # Now that VAD has the correct shape, the original calculation is correct.
        return wmean(torch.abs(signal_components) ** 2, dims=(-2, -1), weights=vad)


def computeSNR(
    signal: torch.Tensor, noise: torch.Tensor, vad: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    return pow2db(
        compute_power(signal_components=signal, vad=vad)
        / compute_power(signal_components=noise, vad=vad)
    )


def gerkmannSPP_STFT(stft_sig: torch.Tensor, inti_frames=8) -> torch.Tensor:

    PH1mean = 0.5
    alphaPH1mean = 0.9
    alphaPSD = 0.8

    # Constants for a posteriori SPP
    q = 0.5
    priorFact = q / (1 - q)
    xiOptDb = 15
    xiOpt = 10 ** (xiOptDb / 10)
    logGLRFact = np.log(1 / (1 + xiOpt))
    GLRexp = xiOpt / (1 + xiOpt)

    noisyPer = stft_sig.abs() ** 2
    noisePow = noisyPer[..., :inti_frames].mean(dim=-1, keepdim=True)
    spp = torch.zeros(size=stft_sig.shape, device=noisyPer.device, dtype=noisyPer.dtype)

    for indFr in range(noisyPer.shape[-1]):
        noisyPerframe = noisyPer[..., [indFr]]
        snrPost1 = noisyPerframe / noisePow
        GLR = priorFact * torch.exp(
            torch.clamp(logGLRFact + GLRexp * snrPost1, max=200)
        )
        PH1 = GLR / (1 + GLR)
        # spp[..., indFr] = PH1[...,0]

        PH1mean = alphaPH1mean * PH1mean + (1 - alphaPH1mean) * PH1

        PH1[PH1mean > 0.99] = torch.clamp(PH1[PH1mean > 0.99], max=0.99)
        spp[..., indFr] = PH1[..., 0]

        estimate = PH1 * noisePow + (1 - PH1) * noisyPerframe
        noisePow = alphaPSD * noisePow + (1 - alphaPSD) * estimate

    return spp


# %% Frequency Weighting


class Frequency_Weighting:
    def __init__(self, name="LTASS", freqlist=None, weightlist=None) -> None:
        self.name = name
        self.freqs = (
            freqlist
            if self.name != "LTASS"
            else [
                0,
                63,
                80,
                100,
                125,
                160,
                200,
                250,
                315,
                400,
                500,
                630,
                800,
                1000,
                1250,
                1600,
                2000,
                2500,
                3150,
                4000,
                5000,
                6300,
                8000,
                10000,
                12500,
                16000,
            ]
        )
        self.power_dB = (
            weightlist
            if self.name != "LTASS"
            else [
                0,
                38.6,
                43.5,
                54.4,
                57.7,
                56.8,
                60.2,
                60.3,
                59.0,
                62.1,
                62.1,
                60.5,
                56.8,
                53.7,
                53.0,
                52.0,
                48.7,
                48.1,
                46.8,
                45.6,
                44.5,
                44.3,
                43.7,
                43.4,
                41.3,
                40.7,
            ]
        )

    def weights(
        self, transform=STFTtransform(), device: Union[torch.device, str] = "cpu"
    ):
        new_freqs = transform.frequencies()
        fun = interpolate.interp1d(self.freqs, self.power_dB, kind="cubic")
        return db2pow(torch.tensor(fun(new_freqs), device=device))

    def wmean(
        self, tensor: torch.Tensor, freq_dim: int = -1, transform=STFTtransform()
    ):
        return wmean(
            tensor.transpose(freq_dim, -1),
            dims=-1,
            weights=self.weights(transform=transform, device=tensor.device),
        ).transpose(freq_dim, -1)

    def wmean_dB(
        self, tensor: torch.Tensor, freq_dim: int = -1, transform=STFTtransform()
    ):
        return wmean(
            tensor.transpose(freq_dim, -1),
            dims=-1,
            weights=pow2db(self.weights(transform=transform, device=tensor.device)),
        ).transpose(freq_dim, -1)

    def plot(self, transform=STFTtransform()):
        new_freqs = transform.frequencies()
        fun = interpolate.interp1d(self.freqs, self.power_dB, kind="cubic")
        plt.figure(dpi=600)
        plt.semilogx(self.freqs, self.power_dB, "o", label="Original data")
        plt.semilogx(new_freqs, fun(new_freqs), "-", label="Interpolated data")
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Power / dB SPL")
        plt.grid()
        plt.savefig("Playground/Frequency_Weighting.png")


# %% generate RIRs


def circularPositions(center=[0, 0, 0], radius=1, num_items=1):
    """
    Compute positions of items (e.g., microphones or sources) in a circular arrangement.

    Parameters:
    - center (list): Center of the circle [x, y, z] (default: [0, 0, 0]).
    - radius (float): Radius of the circle (default: 1).
    - num_items (int): Number of items to position in the circular arrangement (default: 1).

    Returns:
    - positions (np.ndarray): Array of positions with shape (num_items, 3).
    """
    # Generate angles for equidistant points on the circle
    angles = np.linspace(0, 2 * np.pi, num_items, endpoint=False)

    # Calculate positions in 3D space (x, y, z)
    positions = np.array(
        [
            [
                center[0] + radius * np.cos(angle),  # x-coordinate
                center[1] + radius * np.sin(angle),  # y-coordinate
                center[2],
            ]  # z-coordinate remains the same
            for angle in angles
        ]
    )

    return positions


def simRIR_shoebox(
    room_dim,
    mic_positions,
    source_positions,
    noise_positions=None,
    noise_signal=None,
    rt60: float = 0.3,
    fs=16000,
):
    """
    Simulate room impulse responses (RIRs) and diffuse noise in a shoebox room using a microphone array and multiple sources.

    Parameters:
    - room_dim (list): Dimensions of the room [length, width, height] in meters.
    - mic_positions (np.ndarray): 2D array of microphone positions, shape (3, num_mics).
    - source_positions (np.ndarray): 2D array of source positions, shape (num_sources, 3).
    - noise_positions (np.ndarray): 2D array of noise positions, shape (num_sources, 3).
    - absorption (float): Wall absorption coefficient (default is 0.2).
    - fs (int): Sampling frequency in Hz (default is 16000).
    - max_order (int): Maximum order of reflections (default is 15).

    Returns:
    - rirs_tensor (torch.Tensor): RIRs as a tensor with shape (num_sources, num_mics, taps).
    """

    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the shoebox-shaped room with given dimensions
    room = pra.ShoeBox(
        room_dim, fs=fs, max_order=max_order, materials=pra.Material(e_absorption)
    )

    # Add the circular microphone array to the room
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))

    # Add sources to the room
    for source_position in source_positions:
        room.add_source(source_position)

    # Simulate room acoustics and compute room impulse responses (RIRs)
    room.image_source_model()
    room.compute_rir()

    rirs = torch.stack(
        zeropad2fitdims(
            [
                torch.stack(
                    zeropad2fitdims([torch.tensor(srcrir) for srcrir in micrir])
                )
                for micrir in room.rir  # type: ignore
            ]
        )
    ).swapaxes(0, 1)

    if noise_positions is not None and noise_signal is not None:

        # Create the shoebox-shaped room with given dimensions again for the diffuse noise
        noise_room = pra.ShoeBox(
            room_dim, fs=fs, max_order=max_order, materials=pra.Material(e_absorption)
        )

        # Add the circular microphone array to the room
        noise_room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))

        # Add noise sources to the noise room
        signal_len = np.floor(len(noise_signal) / len(noise_positions)).astype(int)
        for srcIdx, noise_pos in enumerate(noise_positions):
            noise_room.add_source(
                noise_pos,
                signal=noise_signal[srcIdx * signal_len : (srcIdx + 1) * signal_len],
            )

        # Simulate the noise room and get the microphone signals
        noise_room.image_source_model()
        noise_room.compute_rir()
        noise_room.simulate()

        noise = torch.tensor(
            np.stack(noise_room.mic_array.signals, axis=0)  # type: ignore
        )  # Shape: [num_mics, signal_length]

        return rirs, noise

    else:

        return rirs


def simRIR_shoebox_PRA(
    room_dim,
    mic_positions,
    source_positions,
    noise_positions=None,
    noise_signal=None,
    rt60: float = 0.3,
    fs=16000,
):
    """
    Simulate room impulse responses (RIRs) and diffuse noise in a shoebox room using a microphone array and multiple sources.

    Parameters:
    - room_dim (list): Dimensions of the room [length, width, height] in meters.
    - mic_positions (np.ndarray): 2D array of microphone positions, shape (3, num_mics).
    - source_positions (np.ndarray): 2D array of source positions, shape (num_sources, 3).
    - noise_positions (np.ndarray): 2D array of noise positions, shape (num_sources, 3).
    - absorption (float): Wall absorption coefficient (default is 0.2).
    - fs (int): Sampling frequency in Hz (default is 16000).
    - max_order (int): Maximum order of reflections (default is 15).

    Returns:
    - rirs_tensor (torch.Tensor): RIRs as a tensor with shape (num_sources, num_mics, taps).
    """

    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the shoebox-shaped room with given dimensions
    room = pra.ShoeBox(
        room_dim, fs=fs, max_order=max_order, materials=pra.Material(e_absorption)
    )

    # Add the circular microphone array to the room
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))

    # Add sources to the room
    for source_position in source_positions:
        room.add_source(source_position)

    # Simulate room acoustics and compute room impulse responses (RIRs)
    room.image_source_model()
    room.compute_rir()

    rirs = torch.stack(
        zeropad2fitdims(
            [
                torch.stack(
                    zeropad2fitdims([torch.tensor(srcrir) for srcrir in micrir])
                )
                for micrir in room.rir  # type: ignore
            ]
        )
    ).swapaxes(0, 1)

    return rirs


def calculate_t60(
    rirs: torch.Tensor, fs: int = 16000, taps_dim: int = -1
) -> torch.Tensor:
    """
    Calculate the T60 (reverberation time) for each RIR in a tensor.

    Parameters:
    - rirs (torch.Tensor): Input tensor containing RIRs. One of the dimensions is for the RIR taps.
    - taps_dim (int): The dimension of the RIR taps (default is the last dimension, i.e., -1).

    Returns:
    - t60_tensor (torch.Tensor): Tensor of T60 values with the same shape as `rirs`, except the `taps_dim` has size 1.
    """

    # Move the taps dimension to the last axis if it's not already
    perm_indices = None
    if taps_dim != -1:
        perm_indices = *[i for i in range(rirs.ndim) if i != taps_dim], taps_dim
        rirs = rirs.permute(perm_indices)

    # Convert to NumPy for use with pyroomacoustics' T60 function
    rirs_np = rirs.cpu().numpy()

    # Create an empty array to store the T60 values
    t60_values = np.zeros(
        rirs_np.shape[:-1]
    )  # Same shape but without the last dimension (taps)

    # Iterate over all RIRs and calculate the T60 for each
    it = np.nditer(t60_values, flags=["multi_index"])
    while not it.finished:
        # Access the RIR for the current index
        rir = rirs_np[it.multi_index]

        # Calculate the T60 using pyroomacoustics (Schroeder method)
        t60_estimate = pra.experimental.measure_rt60(rir, fs=16000)

        # Store the T60 value
        t60_values[it.multi_index] = t60_estimate
        it.iternext()

    # Convert the result back to a torch tensor and add the "taps" dimension with size 1
    t60_tensor = torch.tensor(
        t60_values, dtype=rirs.dtype, device=rirs.device
    ).unsqueeze(-1)

    # If taps_dim is not the last dimension, permute the result back to the original order
    # if taps_dim != -1:
    if perm_indices is not None:
        t60_tensor = t60_tensor.permute(inv_perm_indices(perm_indices))

    return t60_tensor


def save_rirNoise2wav(
    rirs: torch.Tensor,
    noise: torch.Tensor,
    roomname: str,
    arrayname: str,
    noisename: str,
    rirpath: str,
    noisepath: str,
    fs: int = 16000,
):
    """
    Save RIRs as WAV files from a tensor with shape [sources, mics, taps].

    All microphones are saved as separate channels in the same WAV file for each source.

    Naming convention: "RIR_roomname_Angle_arrayname.wav"

    Parameters:
    - rirs (torch.Tensor): Tensor containing the RIRs with shape [sources, mics, taps].
    - roomname (str): Name of the room (e.g., "sim300ms").
    - arrayname (str): Name of the array (e.g., "circ8center").
    - basepath (str): Path where the WAV files will be saved.
    - fs (int): Sampling frequency (default is 16000).

    Returns:
    None
    """

    # Ensure the base directory exists
    if not os.path.exists(rirpath):
        os.makedirs(rirpath)

    num_sources, num_mics, _ = rirs.shape
    angular_distance = (
        360 // num_sources
    )  # Compute the angular distance between sources

    # Calculate the angles for each source
    angles = [(i * angular_distance) for i in range(num_sources)]

    # Normalize angles to the range [-180, 180]
    angles = [(angle if angle <= 180 else angle - 360) for angle in angles]

    # Iterate over the sources
    for src_idx in range(num_sources):
        angle = angles[src_idx]
        angle_str = (
            f"A{angle:d}"  # Format angle with a leading "A" and always show the sign
        )

        # Get the RIRs for all microphones of the current source (as channels)
        rirs_for_source = rirs[src_idx].cpu()  # Shape: [num_mics, taps]

        # Create the filename
        filename = f"RIR_{roomname}_{angle_str}_{arrayname}.wav"
        filepath = os.path.join(rirpath, filename)

        # Save the RIR as a WAV file (all microphones as separate channels)
        torchaudio.save(filepath, rirs_for_source, fs)
        print(f"Saved: {filepath}")

    try:
        # Ensure the base directory exists
        if not os.path.exists(noisepath):
            os.makedirs(noisepath)

        # Create the filename
        filename = f"Noise_{roomname}_{arrayname}_{noisename}.wav"
        filepath = os.path.join(noisepath, filename)

        # Save the noise signal as a WAV file
        torchaudio.save(filepath, noise.cpu(), fs)
        print(f"Saved: {filepath}")

    except:
        print("No noise signal provided.")


def simDiffuseNoise(room: pra.Room, source_positions, signal, signal_length):
    """
    Simulates quasi-diffuse noise using room and multiple sources.

    Parameters:
    - room (pyroomacoustics.ShoeBox): The room object from Pyroomacoustics.
    - source_positions (numpy.ndarray): Array of shape (num_sources, 3) for source positions.
                                        If None, use the room's existing source positions.
    - signal (numpy.ndarray): The input signal to split between sources.
    - signal_length (int): The length of the signal to be played by each source.
    - fs (int): The sampling frequency of the signal.

    Returns:
    - mic_signals (numpy.ndarray): The microphone signals with shape (num_mics, signal_length).
    """

    # Ensure signal is longer than the signal length
    if signal.shape[0] < signal_length * (
        len(source_positions) if source_positions is not None else len(room.sources)
    ):
        raise ValueError(
            "The signal is too short to be divided among the sources with the given signal length."
        )

    # Step 1: Set up source positions and assign signal segments to each source
    if source_positions is not None:
        # Clear any existing sources
        room.sources = []

        # Divide the signal into equal segments for each source
        num_sources = len(source_positions)
        segment_length = len(signal) // num_sources

        # Place each source in the room with its corresponding signal segment
        for i, pos in enumerate(source_positions):
            segment_start = i * segment_length
            segment_end = segment_start + signal_length

            # Ensure the segment is longer than the desired signal length
            signal_segment = signal[segment_start:segment_end]

            # Add the source to the room
            room.add_source(pos, signal=signal_segment)
    else:
        # If source_positions is None, assume room already has sources
        num_sources = len(room.sources)
        segment_length = len(signal) // num_sources

        for i, source in enumerate(room.sources):
            segment_start = i * segment_length
            segment_end = segment_start + signal_length

            source.signal = signal[segment_start:segment_end]

    # Step 2: Simulate the room and get the microphone signals
    room.image_source_model()
    room.compute_rir()
    room.simulate()

    # Retrieve the microphone signals
    mic_signals = torch.tensor(
        np.stack(room.mic_array.signals, axis=0)[:, :signal_length]  # type: ignore
    )  # Shape: [num_mics, signal_length]

    return mic_signals


def simDiffuseNoiseANF(
    mic_positions: np.ndarray,
    input_signals: np.ndarray,
    fs: int,
    nfft: int = 1024,
    sc_type: str = "spherical",
    decomposition: str = "evd",
    processing: str = "balance+smooth",
) -> torch.Tensor:
    """
    Generates spatially coherent diffuse noise using the anf-generator library.

    This method creates multichannel noise signals that exhibit a predefined spatial
    coherence, as described in [1] and [2].

    [1] E.A.P. Habets, I. Cohen and S. Gannot, 'Generating nonstationary multisensor
        signals under a spatial coherence constraint,' JASA, 2008.
    [2] D. Mirabilii, S. J. Schlecht, E.A.P. Habets, 'Generating coherence-constrained
        multisensor signals using balanced mixing and spectrally smooth filters', JASA, 2021.

    Parameters:
        mic_positions (np.ndarray): Microphone positions as a NumPy array of shape (num_mics, 3).
        input_signals (np.ndarray): Uncorrelated input noise signals as a NumPy array
                                    of shape (num_mics, num_samples).
        fs (int): Sampling frequency.
        nfft (int): FFT size for coherence matrix calculation.
        sc_type (str): Spatial coherence model ('spherical', 'cylindrical', 'corcos').
        decomposition (str): Matrix decomposition method ('chd', 'evd').
        processing (str): Post-processing for the mixing matrix
                          ('standard', 'smooth', 'balanced', 'balanced+smooth').

    Returns:
        torch.Tensor: The generated multichannel noise signal as a tensor of shape (num_mics, num_samples).
    """
    # Define the parameters for the target spatial coherence matrix
    params = anf.CoherenceMatrix.Parameters(
        mic_positions=mic_positions,
        sc_type=sc_type,
        sample_frequency=fs,
        nfft=nfft,
    )

    # Generate the output signals with the desired spatial coherence
    output_signals, _, _ = anf.generate_signals(
        input_signals,
        params,
        decomposition=decomposition,
        processing=processing,
    )

    return torch.from_numpy(output_signals).float()


def calc_beam_pattern(W, fs, mic_loc, degrees, c=343):
    """
    Calculates a directivity pattern of a microphone array.

    Parameters
    ----------
    W : ndarray of shape (n_freq, n_out, n_chan)
        Demixing matrices
    fs : int
        Sampling frequency
    mic_loc : ndarray of shape (n_chan, 3)
        The locations of microphones
    degrees : ndarray of shape (n_deg,)
        The degrees of beam patterns in degree
    c: float, default=340
        The speed of sound

    Returns
    -------
    beam_pattern: ndarray of shape (n_freq, n_out, n_deg)
        Beam pattern of the microphone array in decibel.
    """
    n_deg = degrees.size
    n_freq, n_out, _ = W.shape
    n_fft = (n_freq - 1) * 2

    beam_pattern = np.zeros((n_freq, n_out, n_deg), dtype=W.dtype)

    # Defines the origin of degree from x-axis as counter clock-wise
    rad = np.deg2rad(degrees)  # (n_deg,)
    unit_vec = np.array([np.cos(rad), np.sin(rad), np.zeros(n_deg)])  # (3, n_deg)
    delay = mic_loc @ unit_vec / c  # (n_chan, n_deg)

    omega = 2 * np.pi * np.arange(n_freq) * fs / n_fft  # (n_freq,)

    # (n_freq, n_chan, n_deg)
    phase = np.exp(1j * omega[:, None, None] * delay[None, :, :])

    # demixing matrix * amplitude (1) * phase (simulated)
    beam_pattern = W @ phase

    return 20 * np.log10(np.abs(beam_pattern))


def plot_spectrogram(
    signal,
    transform,
    title="Spectrogram",
    clabel="Magnitude (dB)",
    clim=None,
    savename=None,
):
    """
    Plots the spectrogram of the input signal based on the provided transform configuration.

    Parameters:
    - signal: A torch tensor containing the STFT data.
    - transform: A transform object that should contain sampling_frequency, nfft, and frame_shift attributes.
    """

    # Convert the signal to a NumPy array (assuming 4D input with dimensions as per your original code)
    stft_data_np = signal.cpu().numpy()

    # Get the number of frequency bins and time frames
    num_frames = stft_data_np.shape[1]

    # Generate the frequency values for the y-axis (only positive frequencies)
    freqs = np.linspace(0, transform.sampling_frequency / 2, transform.nfft // 2 + 1)

    # Generate the time values for the x-axis
    times = np.arange(num_frames) * transform.frame_shift  # Time in seconds

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(
        20 * np.log10(np.abs(stft_data_np)),
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=(times[0], times[-1], freqs[0], freqs[-1]),
    )  # Adjust extent to actual time and frequency

    # Adding limits to the color bar
    if clim is not None:
        plt.clim(vmin=-clim[0], vmax=clim[1])

    # Adding labels and title
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    # Adding a color bar
    plt.colorbar(label=clabel)

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Save the plot
    if savename is not None:
        plt.savefig(savename)


def plot_phaseogram(
    signal: torch.Tensor,
    transform: STFTtransform,
    title="Phaseogram",
    clabel="Phase (rad)",
    savename=None,
):
    """
    Plots the phaseogram of the input signal based on the provided transform configuration.

    Parameters:
    - signal: A torch tensor containing the STFT data.
    - transform: A transform object that should contain sampling_frequency, nfft, and frame_shift attributes.
    """

    # Convert the signal to a NumPy array (assuming 4D input with dimensions as per your original code)
    stft_data_np = signal.cpu().numpy()

    # Get the number of frequency bins and time frames
    num_frames = stft_data_np.shape[1]

    # Generate the frequency values for the y-axis (only positive frequencies)
    # freqs = np.linspace(0, transform.sampling_frequency / 2, transform.nfft // 2 + 1)
    freqs = transform.frequencies().cpu().numpy()

    # Plot the phaseogram
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.angle(stft_data_np),
        aspect="auto",
        cmap="twilight",
        origin="lower",
        extent=(0.0, num_frames * transform.frame_shift, freqs[0], freqs[-1]),
    )  # Adjust extent to actual time and frequency

    # Adding labels and title
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    # Adding a color bar
    plt.colorbar(label=clabel)

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Save the plot
    if savename is not None:
        plt.savefig(savename)


def plot_coherence(
    STFT_signal: torch.Tensor,
    transform: STFTtransform,
    title: Union[str, None] = None,
    savename: Union[str, None] = None,
    mode: str = "batch",
):
    colormap = "inferno"
    match mode:
        case "batch":
            C = coherenceMatrix(covariance_SCM(STFT_signal))
            msc = abs(C) ** 2
            g_msc = gmsc(C)
        case "framewise":
            C = coherenceMatrix(
                smoothCovarianceMatrix(
                    STFT_signal,
                    smoothing_factor=transform.timeConstant2smoothingFactor(
                        STFT_signal.shape[-2] * transform.frame_shift,
                    ),
                )
            )
            msc = abs(C) ** 2
            g_msc = gmsc(C)
        case _:
            raise ValueError("Invalid mode. Use 'batch' or 'framewise'.")
    num_channels = msc.shape[-1]
    fig, axes = plt.subplots(
        num_channels - 1,
        num_channels - 1,
        figsize=(3 * (num_channels - 1), 3 * (num_channels - 1)),
        squeeze=False,
    )

    freqs = (
        transform.frequencies().cpu().numpy()
        if hasattr(transform, "frequencies")
        else np.arange(msc.shape[0])
    )

    for i in range(1, num_channels):
        for j in range(i):
            ax = axes[i - 1, j]
            if mode == "framewise":
                times = (
                    transform.frames2times(torch.arange(msc.shape[-3])).cpu().numpy()
                    if hasattr(transform, "frames2times")
                    else np.arange(msc.shape[1])
                )
                im = ax.imshow(
                    msc[:, :, i, j].real.cpu().numpy(),
                    aspect="auto",
                    extent=[times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000],
                    origin="lower",
                    cmap=colormap,
                )
                ax.set_title(f"MSC (Ch. {j} vs Ch. {i})")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Frequency [kHz]")
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Coherence")
            else:
                ax.plot(freqs, msc[:, i, j].real.cpu().numpy())
                ax.set_title(f"MSC (Ch. {j} vs Ch. {i})")
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("MSC")

    # Hide unused axes (upper triangle and diagonal)
    for i in range(num_channels - 1):
        for j in range(i + 1, num_channels - 1):
            axes[i, j].axis("off")

    if mode == "framewise":
        ax = axes[0, -1]
        ax.axis("on")
        times = (
            transform.frames2times(torch.arange(msc.shape[-3])).cpu().numpy()
            if hasattr(transform, "frames2times")
            else np.arange(msc.shape[1])
        )
        im = ax.imshow(
            g_msc[:, :, 0, 0].real.cpu().numpy(),
            aspect="auto",
            extent=[times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000],
            origin="lower",
            cmap=colormap,
        )
        ax.set_title("GMSC")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [kHz]")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Coherence")
    else:
        ax = axes[0, -1]
        ax.plot(freqs, g_msc[:, 0, 0].real.cpu().numpy())
        ax.set_title("GMSC")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("GMSC")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if savename:
        plt.savefig(savename)
    else:
        plt.show()
