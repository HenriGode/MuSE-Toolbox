import logging
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import interpolate

from muse_toolbox.utils.math.conversions import db2pow, pow2db
from muse_toolbox.utils.math.stats import wmean
from muse_toolbox.utils.system import memory

log = logging.getLogger(__name__)


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
        window_type: str | Callable | torch.Tensor = "torch.hann_window(x).sqrt()",
        remove_DC: bool = True,
        remove_Nyquist: bool = True,
    ) -> None:
        """
        Initialize STFT transform with given parameters.

        Args:
            frame_length (float): Window size in seconds. Defaults to 32e-3.
            frame_shift (float): Shift size in seconds. Defaults to 16e-3.
            sampling_frequency (float): Sampling frequency in Hz. Defaults to 16e3.
            window_type (Union[str, Callable, torch.Tensor]): The window function to apply. Defaults to "torch.hann_window(x).sqrt()".
            remove_DC (bool): Whether to remove the DC component. Defaults to True.
            remove_Nyquist (bool): Whether to remove the Nyquist frequency component. Defaults to True.
        """
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
                        window_type,
                        {
                            "x": self.nfft,
                            "periodic": True,
                            "dtype": torch.get_default_dtype(),
                            "torch": torch,
                        },
                    )
                except Exception as e:
                    log.error(f"Failed to evaluate window type string: {e}")
                    raise ValueError("unknown window type!") from e
        elif callable(window_type):
            self.window = window_type(
                self.frame_length,
                periodic=True,
                dtype=torch.get_default_dtype(),
            )
        elif type(window_type) is torch.Tensor:
            self.window = window_type
        else:
            raise NotImplementedError("Window type must be str, callable, or torch.Tensor.")
            
        log.debug(f"Initialized STFTtransform: {self.signature}")

    def _verbose_parameters(self, indent: str = "") -> None:
        """Log the parameters of the STFT transform."""
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Frame Length: {self.frame_length} s")
        log.info(f"{indent}  Frame Shift: {self.frame_shift} s")
        log.info(f"{indent}  Sampling Frequency: {self.sampling_frequency} Hz")
        log.info(f"{indent}  Window Type: {self.window_type}")
        log.info(f"{indent}  NFFT: {self.nfft} samples")
        log.info(f"{indent}  Hop Length: {self.hop_length} samples")

    def get_config(self) -> dict:
        """
        Get the STFT transform configuration.
        
        Returns:
            dict: Configuration dictionary.
        """
        return {
            "frame_length": self.frame_length,
            "frame_shift": self.frame_shift,
            "sampling_frequency": self.sampling_frequency,
            "window_type": self.window_type,
        }

    @property
    def signature(self) -> str:
        """Get a signature string uniquely identifying the STFT transform parameters."""
        return (
            f"STFT_fl{self.frame_length}_fs{self.frame_shift}_"
            f"sf{self.sampling_frequency}_wt{self.window_type}"
        )

    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply STFT to encode the input signal.
        
        Args:
            signal (torch.Tensor): Time-domain signal of shape `(..., M, N_samples)`.
            
        Returns:
            torch.Tensor: STFT signal of shape `(..., F, M, T_frames)`.
        """
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
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """
        Apply inverse STFT to decode the input signal back to time domain.
        
        Args:
            stft_signal (torch.Tensor): STFT signal.
            DC_zero (bool): If True and remove_DC was False, force DC band to zero.
            Nyquist_zero (bool): If True and remove_Nyquist was False, force Nyquist band to zero.
            num_samples (int | None): Target number of samples. If not None, truncates or zero-pads to this length.

        Returns:
            torch.Tensor: Time-domain signal.
        """
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
        self, time: torch.Tensor | float | int | list | tuple, method: str = "center"
    ) -> torch.Tensor:
        """
        Convert time (in seconds) to frame indices.
        
        Args:
            time: Time in seconds.
            method (str): Method to convert time to frame indices. Defaults to "center".
            
        Returns:
            torch.Tensor: Corresponding frame indices.
        """
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
        self, time: torch.Tensor | float | int | list | tuple
    ) -> torch.Tensor:
        """
        Convert time (in seconds) to sample indices.
        
        Args:
            time: Time in seconds.
            
        Returns:
            torch.Tensor: Corresponding sample indices.
        """
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float64)
        return (time * self.sampling_frequency).round().to(torch.int64)

    def frames2times(self, frames: torch.Tensor, method: str = "center") -> torch.Tensor:
        """
        Convert frame indices to time (in seconds).
        
        Args:
            frames: Frame indices.
            method (str): Method for conversion. Defaults to "center".
            
        Returns:
            torch.Tensor: Corresponding time in seconds.
        """
        match method:
            case "center":
                return (frames - 1) * self.frame_shift
            case _:
                raise ValueError(
                    f"Unknown method '{method}' for converting frames to time."
                )

    def samples2frames(self, samples: torch.Tensor | int) -> torch.Tensor:
        """
        Convert sample indices to frame indices.
        
        Args:
            samples: Sample indices.
            
        Returns:
            torch.Tensor: Corresponding frame indices.
        """
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples, dtype=torch.float64)
        return (samples / self.hop_length).round().to(torch.int64)

    def frames2samples(self, frames: torch.Tensor | int) -> torch.Tensor:
        """
        Convert frame indices to sample indices.
        
        Args:
            frames: Frame indices.
            
        Returns:
            torch.Tensor: Corresponding sample indices.
        """
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

    def samples2times(self, samples: torch.Tensor | int) -> torch.Tensor:
        """
        Convert sample indices to time (in seconds).
        
        Args:
            samples: Sample indices.
            
        Returns:
            torch.Tensor: Corresponding time in seconds.
        """
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples, dtype=torch.float64)
        return samples / self.sampling_frequency

    def frequencies(self) -> torch.Tensor:
        """
        Get frequency bins.
        
        Returns:
            torch.Tensor: Computed frequency bins.
        """
        freqs = torch.linspace(0, self.sampling_frequency / 2, self.nfft // 2 + 1)
        start_idx = 1 if self.remove_DC else 0
        end_idx = -1 if self.remove_Nyquist else None
        return freqs[start_idx:end_idx]

    def freqs2bins(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Convert given frequencies to STFT bin indices.
        
        Args:
            freqs (torch.Tensor): Frequencies.
            
        Returns:
            torch.Tensor: Corresponding STFT bin indices.
        """
        return (freqs / self.sampling_frequency * self.nfft).round().to(torch.int64) - 1

    def bins2freq(self, bins: torch.Tensor | int) -> torch.Tensor:
        """
        Convert STFT bin indices to frequencies.
        
        Args:
            bins: Bin indices.
            
        Returns:
            torch.Tensor: Corresponding frequencies.
        """
        if not isinstance(bins, torch.Tensor):
            bins = torch.tensor(bins, dtype=torch.float64)
        return bins * self.sampling_frequency / self.nfft

    def timeConstant2smoothingFactor(self, time_constant: float) -> float:
        """
        Convert time constant (in seconds) to a smoothing factor.
        
        Args:
            time_constant (float): Time constant.
            
        Returns:
            float: Smoothing factor.
        """
        return np.exp(-self.frame_shift / time_constant)

    def plot(self, signal_length: int) -> None:
        """Placeholder for plotting the STFT representation."""
        plt.figure(dpi=600)
        # Further plotting implementation is needed here.


def slice2frames(
    signal: torch.Tensor, samples: int
) -> torch.Tensor:
    """
    Slice a signal into frames of a given sample size.
    Adds one broadcast dim for time frames in the second last dim and the last dim remains samples.
    
    Args:
        signal (torch.Tensor): The input signal.
        samples (int): Number of samples per frame.
        
    Returns:
        torch.Tensor: The framed signal.
    """
    log.debug(f"Slicing signal into frames of {samples} samples.")
    dims = list(signal.shape)
    return signal[..., : (dims[-1] // samples) * samples].reshape(
        dims[:-1] + [dims[-1] // samples, samples]
    )


# %% Frequency Weighting


class Frequency_Weighting:
    """
    Provides frequency-dependent weighting (e.g., LTASS - Long-Term Average Speech Spectrum).
    """

    def __init__(self, name: str = "LTASS", freqlist: list | None = None, weightlist: list | None = None) -> None:
        """
        Initialize the Frequency Weighting.

        Args:
            name (str): The name of the weighting curve. Defaults to "LTASS".
            freqlist (list | None): List of frequencies. Overrides defaults if provided.
            weightlist (list | None): List of weights in dB. Overrides defaults if provided.
        """
        self.name = name
        self.freqs = (
            freqlist
            if self.name != "LTASS"
            else [
                0, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                10000, 12500, 16000,
            ]
        )
        self.power_dB = (
            weightlist
            if self.name != "LTASS"
            else [
                0, 38.6, 43.5, 54.4, 57.7, 56.8, 60.2, 60.3, 59.0, 62.1, 62.1,
                60.5, 56.8, 53.7, 53.0, 52.0, 48.7, 48.1, 46.8, 45.6, 44.5,
                44.3, 43.7, 43.4, 41.3, 40.7,
            ]
        )
        log.debug(f"Initialized Frequency_Weighting '{self.name}'.")

    def weights(
        self, transform: STFTtransform = STFTtransform(), device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """
        Compute interpolated weights in the linear power domain for the STFT frequencies.
        
        Args:
            transform (STFTtransform): The STFT transform defining the frequencies.
            device (torch.device | str): Device to place the returned tensor on.
            
        Returns:
            torch.Tensor: Frequency weights in power domain.
        """
        new_freqs = transform.frequencies()
        fun = interpolate.interp1d(self.freqs, self.power_dB, kind="cubic")
        return db2pow(torch.tensor(fun(new_freqs), device=device))

    def wmean(
        self, tensor: torch.Tensor, freq_dim: int = -1, transform: STFTtransform = STFTtransform()
    ) -> torch.Tensor:
        """
        Compute the weighted mean across the frequency dimension using linear power weights.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            freq_dim (int): Dimension index corresponding to frequencies. Defaults to -1.
            transform (STFTtransform): STFT transform context.
            
        Returns:
            torch.Tensor: Weighted mean tensor.
        """
        return wmean(
            tensor.transpose(freq_dim, -1),
            dims=-1,
            weights=self.weights(transform=transform, device=tensor.device),
        ).transpose(freq_dim, -1)

    def wmean_dB(
        self, tensor: torch.Tensor, freq_dim: int = -1, transform: STFTtransform = STFTtransform()
    ) -> torch.Tensor:
        """
        Compute the weighted mean across the frequency dimension using dB weights.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            freq_dim (int): Dimension index corresponding to frequencies. Defaults to -1.
            transform (STFTtransform): STFT transform context.
            
        Returns:
            torch.Tensor: Weighted mean tensor.
        """
        return wmean(
            tensor.transpose(freq_dim, -1),
            dims=-1,
            weights=pow2db(self.weights(transform=transform, device=tensor.device)),
        ).transpose(freq_dim, -1)

    def plot(self, transform: STFTtransform = STFTtransform()) -> None:
        """
        Plot the original and interpolated frequency weighting curve.
        
        Args:
            transform (STFTtransform): STFT transform context used to get evaluation frequencies.
        """
        new_freqs = transform.frequencies()
        fun = interpolate.interp1d(self.freqs, self.power_dB, kind="cubic")
        plt.figure(dpi=600)
        plt.semilogx(self.freqs, self.power_dB, "o", label="Original data")
        plt.semilogx(new_freqs, fun(new_freqs), "-", label="Interpolated data")
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Power / dB SPL")
        plt.grid()
        plt.savefig("Playground/Frequency_Weighting.png")
        log.info("Saved Frequency Weighting plot to Playground/Frequency_Weighting.png")