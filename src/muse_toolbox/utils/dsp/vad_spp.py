"""Broadband Voice Activity Detection (VAD) and Frequency-Dependent Speech Presence Probability (SPP)"""

import logging

import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt

log = logging.getLogger(__name__)

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
        audio (torch.Tensor): Input audio signal of shape `(..., num_channels, num_samples)`.
        fs (float): Sampling frequency in Hz. Defaults to 16000.0.
        thr (float): Threshold in dB. Defaults to -30.
        min_on (float): Minimum duration of silence to bridge in seconds. Defaults to 50e-3.
        mode (str): 'normal', 'highpass', or 'zeromask'. Defaults to 'highpass'.
        cutoff_freq (float): Cutoff frequency for highpass mode. Defaults to 80.0.
        
    Returns:
        torch.Tensor: Boolean VAD mask.
    """
    log.debug(f"Computing generalized VAD (mode: {mode}).")
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


def vad_opt_original(
    audio: torch.Tensor, fs: float = 16000.0, thr: float = -30, min_on: float = 50e-3
) -> torch.Tensor:
    """
    Original non-optimized Voice Activity Detection.

    Args:
        audio (torch.Tensor): audio is a tensor of shape [..., num_channels, num_samples].
        fs (float): Sampling frequency.
        thr (float): Threshold in dB.
        min_on (float): Minimum duration of silence to bridge in seconds.
        
    Returns:
        torch.Tensor: Boolean VAD mask.
    """
    log.debug("Computing original VAD.")
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
    """
    Slow Voice Activity Detection using conv1d for morphological closing.

    Args:
        audio (torch.Tensor): audio is a tensor of shape [..., num_channels, num_samples].
        fs (float): Sampling frequency.
        thr (float): Threshold in dB.
        min_on (float): Minimum duration of silence to bridge in seconds.
        
    Returns:
        torch.Tensor: Boolean VAD mask.
    """
    log.debug("Computing slow VAD.")
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


def gerkmannSPP_STFT(stft_sig: torch.Tensor, inti_frames: int = 8) -> torch.Tensor:
    """
    Calculates the frequency-dependent Speech Presence Probability (SPP) based on Gerkmann & Hendriks (2012).

    Args:
        stft_sig (torch.Tensor): The STFT of the input signal.
        inti_frames (int): Number of initial frames used to estimate the noise floor. Defaults to 8.

    Returns:
        torch.Tensor: The estimated Speech Presence Probability per time-frequency bin.
    """
    log.debug("Computing Gerkmann SPP.")
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


def plot_vad_debug(audio_signal: torch.Tensor, vad_mask: torch.Tensor, save_path: str):
    """
    Plots audio channels and the VAD mask to verify alignment.

    Args:
        audio_signal: [M, N] tensor (M channels, N samples)
        vad_mask: [N] tensor (N samples), binary (0 or 1) or boolean
        save_path: File path to save the plot (include extension like .png)
    """
    # 1. Data Preparation
    # Ensure CPU and Numpy
    if audio_signal.ndim == 1:
        audio_signal = audio_signal.unsqueeze(0)

    M, N = audio_signal.shape
    audio_np = audio_signal.detach().cpu().numpy()
    mask_np = vad_mask.detach().cpu().float().numpy()

    # Normalize audio to [-1, 1] for visualization scaling
    max_val = np.abs(audio_np).max()
    if max_val > 0:
        audio_np = audio_np / max_val

    time_axis = np.arange(N)

    # 2. Plotting
    plt.figure(figsize=(12, 6))

    # Plot Audio Channels
    # If many channels, plot first one bold, others faint
    if M > 1:
        for m in range(1, M):
            plt.plot(time_axis, audio_np[m], color="gray", alpha=0.3, linewidth=0.5)
        plt.plot(
            time_axis,
            audio_np[0],
            color="black",
            alpha=0.8,
            linewidth=1.0,
            label="Ch 0 Audio",
        )
    else:
        plt.plot(
            time_axis,
            audio_np[0],
            color="black",
            alpha=0.8,
            linewidth=1.0,
            label="Audio",
        )

    # Plot Mask Overlay (Green Shading)
    # where=mask_np>0.5 handles both boolean and float masks
    plt.fill_between(
        time_axis,
        -1,
        1,
        where=(mask_np > 0.5),  # type: ignore
        color="green",
        alpha=0.2,
        label="VAD Active (Region)",
    )

    # Plot Mask Line (Red Step)
    plt.plot(
        time_axis,
        mask_np * 0.9,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="VAD Mask (Signal)",
    )

    # 3. Styling
    plt.ylim(-1.1, 1.1)
    plt.title(f"Audio Signal (Normalized) vs VAD Mask\nChannels: {M}, Samples: {N}")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # 4. Save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Free memory
    print(f"VAD Plot saved to: {save_path}")
