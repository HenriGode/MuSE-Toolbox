import logging
import os

import anf_generator as anf
import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio

from muse_toolbox.utils.dsp.transforms import STFTtransform
from muse_toolbox.utils.math.windowing import torchaudio_functional_fftconvolve_complex
from muse_toolbox.utils.math.covariance import covariance_SCM
from muse_toolbox.utils.math.matrix_ops import peigvech
from muse_toolbox.utils.tensor_ops import inv_perm_indices, zeropad2fitdims

log = logging.getLogger(__name__)


def convolve_clean2microphone(
    clean: torch.Tensor, rirdata: torch.Tensor
) -> torch.Tensor:
    """
    Convolves a clean audio signal with a Room Impulse Response (RIR).

    Args:
        clean (torch.Tensor): The clean input signal.
        rirdata (torch.Tensor): The Room Impulse Response data.

    Returns:
        torch.Tensor: The convolved signal, truncated to the original clean signal's length.
    """
    log.debug("Convolving clean signal with RIR.")
    return torchaudio_functional_fftconvolve_complex(clean, rirdata, mode="full")[
        ..., : clean.shape[-1]
    ]


def convolve_white2microphone(
    rirdata: torch.Tensor, samples: int = 80000
) -> torch.Tensor:
    """
    Convolves a generated white noise signal with a Room Impulse Response (RIR).

    Args:
        rirdata (torch.Tensor): The Room Impulse Response data.
        samples (int): Number of samples for the white noise signal. Defaults to 80000.

    Returns:
        torch.Tensor: The convolved signal, truncated to the generated white noise signal's length.
    """
    log.debug(f"Convolving generated white noise ({samples} samples) with RIR.")
    white = torch.randn([1, int(samples)], device=rirdata.device, dtype=rirdata.dtype)
    return torchaudio_functional_fftconvolve_complex(white, rirdata, mode="full")[
        ..., : white.shape[-1]
    ]


def rir2rtf(
    rir: torch.Tensor,
    transform: STFTtransform,
    ref_mic: int | None = None,
    signal_len: int = 1600000,
) -> torch.Tensor:
    """
    Computes the Relative Transfer Function (RTF) from a Room Impulse Response (RIR).

    Args:
        rir (torch.Tensor): The Room Impulse Response.
        transform (STFTtransform): The STFT transform configuration.
        ref_mic (Optional[int]): Index of the reference microphone. If provided, the RTF is normalized with respect to it. Defaults to None.
        signal_len (int): Length of the white noise signal to generate. Defaults to 1600000.

    Returns:
        torch.Tensor: The calculated RTF tensor.
    """
    log.debug("Computing RTF from RIR.")
    directwhite = convolve_white2microphone(rir, samples=signal_len)

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


# %% generate RIRs


def circularPositions(center: list = [0, 0, 0], radius: float = 1.0, num_items: int = 1) -> np.ndarray:
    """
    Compute positions of items (e.g., microphones or sources) in a circular arrangement.

    Args:
        center (list): Center of the circle `[x, y, z]`. Defaults to `[0, 0, 0]`.
        radius (float): Radius of the circle. Defaults to 1.
        num_items (int): Number of items to position in the circular arrangement. Defaults to 1.

    Returns:
        np.ndarray: Array of positions with shape `(num_items, 3)`.
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
    room_dim: list,
    mic_positions: np.ndarray,
    source_positions: np.ndarray,
    noise_positions: np.ndarray | None = None,
    noise_signal: np.ndarray | None = None,
    rt60: float = 0.3,
    fs: int = 16000,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate room impulse responses (RIRs) and optionally diffuse noise in a shoebox room.

    Args:
        room_dim (list): Dimensions of the room `[length, width, height]` in meters.
        mic_positions (np.ndarray): 2D array of microphone positions, shape `(3, num_mics)`.
        source_positions (np.ndarray): 2D array of source positions, shape `(num_sources, 3)`.
        noise_positions (Optional[np.ndarray]): 2D array of noise positions, shape `(num_sources, 3)`. Defaults to None.
        noise_signal (Optional[np.ndarray]): The noise signal to use. Defaults to None.
        rt60 (float): Target reverberation time (RT60) in seconds. Defaults to 0.3.
        fs (int): Sampling frequency in Hz. Defaults to 16000.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
            - If no noise is provided, returns RIRs as a tensor with shape `(num_sources, num_mics, taps)`.
            - If noise is provided, returns `(RIRs, noise_signals)`.
    """
    log.debug(f"Simulating RIRs in shoebox room (dim={room_dim}, rt60={rt60}).")
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
        log.debug("Simulating diffuse noise.")
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
    room_dim: list,
    mic_positions: np.ndarray,
    source_positions: np.ndarray,
    noise_positions: np.ndarray | None = None,
    noise_signal: np.ndarray | None = None,
    rt60: float = 0.3,
    fs: int = 16000,
) -> torch.Tensor:
    """
    Simulate room impulse responses (RIRs) and diffuse noise in a shoebox room using PyRoomAcoustics.

    Note: This version ignores the noise parameters and only returns the simulated RIRs. 
    Use `simRIR_shoebox` if noise simulation is needed.

    Args:
        room_dim (list): Dimensions of the room `[length, width, height]` in meters.
        mic_positions (np.ndarray): 2D array of microphone positions, shape `(3, num_mics)`.
        source_positions (np.ndarray): 2D array of source positions, shape `(num_sources, 3)`.
        noise_positions (Optional[np.ndarray]): (Ignored). Defaults to None.
        noise_signal (Optional[np.ndarray]): (Ignored). Defaults to None.
        rt60 (float): Wall absorption coefficient derived from RT60. Defaults to 0.3.
        fs (int): Sampling frequency in Hz. Defaults to 16000.

    Returns:
        torch.Tensor: RIRs as a tensor with shape `(num_sources, num_mics, taps)`.
    """
    log.debug(f"Simulating RIRs using PRA in shoebox room (dim={room_dim}, rt60={rt60}).")
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
    Calculate the T60 (reverberation time) for each RIR in a tensor using the Schroeder method.

    Args:
        rirs (torch.Tensor): Input tensor containing RIRs.
        fs (int): Sampling frequency in Hz. Defaults to 16000.
        taps_dim (int): The dimension of the RIR taps. Defaults to -1.

    Returns:
        torch.Tensor: Tensor of T60 values with the same shape as `rirs`, except the `taps_dim` has size 1.
    """
    log.debug("Calculating T60 for RIRs.")
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
        t60_estimate = pra.experimental.measure_rt60(rir, fs=fs)

        # Store the T60 value
        t60_values[it.multi_index] = t60_estimate
        it.iternext()

    # Convert the result back to a torch tensor and add the "taps" dimension with size 1
    t60_tensor = torch.tensor(
        t60_values, dtype=rirs.dtype, device=rirs.device
    ).unsqueeze(-1)

    # If taps_dim is not the last dimension, permute the result back to the original order
    if perm_indices is not None:
        t60_tensor = t60_tensor.permute(inv_perm_indices(perm_indices))

    return t60_tensor


def save_rirNoise2wav(
    rirs: torch.Tensor,
    noise: torch.Tensor | None,
    roomname: str,
    arrayname: str,
    noisename: str,
    rirpath: str,
    noisepath: str,
    fs: int = 16000,
) -> None:
    """
    Save RIRs and noise as WAV files from a tensor.

    All microphones are saved as separate channels in the same WAV file for each source.
    Naming convention: "RIR_roomname_Angle_arrayname.wav"

    Args:
        rirs (torch.Tensor): Tensor containing the RIRs with shape `[sources, mics, taps]`.
        noise (Optional[torch.Tensor]): The simulated noise signal.
        roomname (str): Name of the room (e.g., "sim300ms").
        arrayname (str): Name of the array (e.g., "circ8center").
        noisename (str): Name describing the noise.
        rirpath (str): Path where the RIR WAV files will be saved.
        noisepath (str): Path where the noise WAV files will be saved.
        fs (int): Sampling frequency. Defaults to 16000.
    """
    log.debug(f"Saving RIRs to {rirpath} and Noise to {noisepath}.")
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
        log.info(f"Saved RIR: {filepath}")

    if noise is not None:
        try:
            # Ensure the base directory exists
            if not os.path.exists(noisepath):
                os.makedirs(noisepath)

            # Create the filename
            filename = f"Noise_{roomname}_{arrayname}_{noisename}.wav"
            filepath = os.path.join(noisepath, filename)

            # Save the noise signal as a WAV file
            torchaudio.save(filepath, noise.cpu(), fs)
            log.info(f"Saved Noise: {filepath}")

        except Exception as e:
            log.error(f"Error saving noise signal: {e}")
    else:
        log.info("No noise signal provided to save.")


def simDiffuseNoise(
    room: pra.Room,
    source_positions: np.ndarray | None,
    signal: np.ndarray,
    signal_length: int,
) -> torch.Tensor:
    """
    Simulates quasi-diffuse noise using pyroomacoustics Room and multiple sources.

    Args:
        room (pra.Room): The room object from Pyroomacoustics.
        source_positions (Optional[np.ndarray]): Array of shape `(num_sources, 3)` for source positions.
                                                 If None, use the room's existing source positions.
        signal (np.ndarray): The input signal to split between sources.
        signal_length (int): The length of the signal to be played by each source.

    Returns:
        torch.Tensor: The microphone signals with shape `(num_mics, signal_length)`.
    """
    log.debug("Simulating diffuse noise in PRA room.")
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

    Args:
        mic_positions (np.ndarray): Microphone positions as a NumPy array of shape `(num_mics, 3)`.
        input_signals (np.ndarray): Uncorrelated input noise signals as a NumPy array
                                    of shape `(num_mics, num_samples)`.
        fs (int): Sampling frequency.
        nfft (int): FFT size for coherence matrix calculation. Defaults to 1024.
        sc_type (str): Spatial coherence model ('spherical', 'cylindrical', 'corcos'). Defaults to 'spherical'.
        decomposition (str): Matrix decomposition method ('chd', 'evd'). Defaults to 'evd'.
        processing (str): Post-processing for the mixing matrix
                          ('standard', 'smooth', 'balanced', 'balanced+smooth'). Defaults to 'balance+smooth'.

    Returns:
        torch.Tensor: The generated multichannel noise signal as a tensor of shape `(num_mics, num_samples)`.
    """
    log.debug("Simulating diffuse noise using ANF generator.")
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
