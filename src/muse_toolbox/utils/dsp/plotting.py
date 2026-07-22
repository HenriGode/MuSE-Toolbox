import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from muse_toolbox.utils.dsp.signal_stats import (
    coherenceMatrix,
    gmsc,
    smoothCovarianceMatrix,
)
from muse_toolbox.utils.dsp.transforms import STFTtransform
from muse_toolbox.utils.math.covariance import covariance_SCM

log = logging.getLogger(__name__)


def plot_spectrogram(
    signal: torch.Tensor,
    transform: STFTtransform,
    title: str = "Spectrogram",
    clabel: str = "Magnitude (dB)",
    clim: tuple | None = None,
    savename: str | None = None,
) -> None:
    """
    Plots the spectrogram of the input signal based on the provided transform configuration.

    Args:
        signal (torch.Tensor): A torch tensor containing the STFT data.
        transform (STFTtransform): A transform object containing sampling_frequency, nfft, and frame_shift attributes.
        title (str): Title of the plot. Defaults to "Spectrogram".
        clabel (str): Color bar label. Defaults to "Magnitude (dB)".
        clim (tuple | None): Color limits for the plot `(vmin, vmax)`. Defaults to None.
        savename (str | None): Path to save the plot. If None, displays it instead. Defaults to None.
    """
    log.debug(f"Plotting spectrogram with title '{title}'.")
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
    
    # Save the plot
    if savename is not None:
        log.info(f"Saving spectrogram to {savename}")
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()


def plot_phaseogram(
    signal: torch.Tensor,
    transform: STFTtransform,
    title: str = "Phaseogram",
    clabel: str = "Phase (rad)",
    savename: str | None = None,
) -> None:
    """
    Plots the phaseogram of the input signal based on the provided transform configuration.

    Args:
        signal (torch.Tensor): A torch tensor containing the STFT data.
        transform (STFTtransform): A transform object containing sampling_frequency, nfft, and frame_shift attributes.
        title (str): Title of the plot. Defaults to "Phaseogram".
        clabel (str): Color bar label. Defaults to "Phase (rad)".
        savename (str | None): Path to save the plot. If None, displays it instead. Defaults to None.
    """
    log.debug(f"Plotting phaseogram with title '{title}'.")
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
    
    # Save the plot
    if savename is not None:
        log.info(f"Saving phaseogram to {savename}")
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()


def plot_coherence(
    STFT_signal: torch.Tensor,
    transform: STFTtransform,
    title: str | None = None,
    savename: str | None = None,
    mode: str = "batch",
) -> None:
    """
    Plots the spatial coherence and Generalized Magnitude Squared Coherence (GMSC) of the STFT signal.

    Args:
        STFT_signal (torch.Tensor): A torch tensor containing the STFT signal.
        transform (STFTtransform): The STFT transform configuration.
        title (str | None): Optional title for the figure. Defaults to None.
        savename (str | None): Optional path to save the figure. If None, it will be displayed. Defaults to None.
        mode (str): Plotting mode. Either "batch" or "framewise". Defaults to "batch".
    """
    log.debug(f"Plotting coherence in '{mode}' mode.")
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
        log.info(f"Saving coherence plot to {savename}")
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()