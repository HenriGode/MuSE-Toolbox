import os.path as path
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchcodec.decoders import AudioDecoder

from muse_toolbox.models.source_counting.estimators.base_estimator import (
    BaseSourceCountEstimator,
)
from losses import BaseLoss

from .dsp.acoustic_simulation import convolve_white2microphone
from .dsp.transforms import STFTtransform
from .math.complex_angles import hermitian_angle
from .math.covariance import covariance_SCM
from .math.matrix_ops import peigvech
from .tensor_ops import *
from .system import *
from pathlib import Path
from typing import Union, overload, TypeVar, cast, Any
import warnings
from scipy.spatial.distance import pdist
from shapely.geometry import Point
import trimesh
from components.feature_extractors.base_feature import BaseFeatureExtractor
from dataclasses import dataclass


class Clean_speech:
    """
    Represents and processes a single clean speech audio file.

    This class loads a clean speech signal, applies a linear onset ramp to prevent
    abrupt starts, and can prepend silence to simulate a delayed activation time.
    """

    def __init__(
        self,
        filename: str,
        signal_len: float,
        sampling_frequency: float = 16e3,
        activation_time: float | None = None,
        lin_onset_len: float = 50e-3,
    ) -> None:
        """
        Initializes the Clean_speech object.

        Args:
            filename (str): Path to the audio file. Expected format like 'f00061_0.wav'.
            signal_len (float): The desired length of the speech signal in seconds.
            sampling_frequency (float, optional): Target sampling frequency. Defaults to 16000.
            activation_time (float | None, optional): Time in seconds to prepend with silence. Defaults to None.
            lin_onset_len (float, optional): Length of the linear onset ramp in seconds. Defaults to 50e-3.
        """
        self.filename = filename
        self.signal_len = signal_len
        self.activation_time = activation_time
        self.lin_onset_len = lin_onset_len
        # Assumes filename format like 'f00061_0.wav' to extract source ID and sex
        self.source_id = path.basename(filename)[0:6]  # e.g., 'f00061'
        self.sex = path.basename(filename)[0]  # e.g., 'f' or 'm'
        self.data, self.sr = load_audio(
            filename,
            sampling_frequency=sampling_frequency,
            signal_len=self.signal_len,
        )
        # Squeeze the channel dimension to ensure the data is a 1D tensor (mono).
        if self.data.ndim > 1 and self.data.shape[0] == 1:
            self.data = self.data.squeeze(0)
        # Assert that the data is mono after potential squeezing.
        assert self.data.ndim < 2, f"Data is not mono. Shape is {self.data.shape}."

        # Apply a linear fade-in (onset ramp) to the beginning of the signal to avoid clicks.
        onset_samples = int(self.lin_onset_len * self.sr)
        ramp = torch.linspace(0, 1, onset_samples, device=self.data.device)
        self.data[:onset_samples] *= ramp

        # If an activation time is specified, prepend silence to the signal.
        if self.activation_time is not None:
            activation_samples = int(self.activation_time * self.sr)
            total_samples = int(self.signal_len * self.sr)
            speech_len_samples = total_samples - activation_samples

            # Create silence tensor
            silence = torch.zeros(
                activation_samples,
                device=self.data.device,
                dtype=self.data.dtype,
            )
            # Concatenate silence with the truncated speech signal
            self.data = torch.cat([silence, self.data[:speech_len_samples]])


class RIR:
    """
    Represents and processes a Room Impulse Response (RIR) file.

    This class loads an RIR and provides methods to extract its direct path
    component or compute an oracle Relative Transfer Function (RTF).
    """

    def __init__(
        self,
        filename: str,
        sampling_frequency: float = 16e3,
    ) -> None:
        """
        Initializes the RIR object.

        Args:
            filename (str): Path to the RIR file. Expected format like 'RIR_BX0310_A-150_BTE2x2center.wav'.
            sampling_frequency (float, optional): Target sampling frequency. Defaults to 16000.
        """
        self.filename = filename
        # Parse metadata from the filename based on an expected format.
        parts = path.basename(filename).split("_")
        self.room = parts[1]
        self.source_location = parts[2]
        self.array = parts[3][:-4]  # Remove '.wav' extension
        self.data, self.sr = load_audio(filename, sampling_frequency=sampling_frequency)

    def direct(
        self, time_after_max_power: float = 50e-3, provide_tail: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Separates the direct path and the reverberant tail of the RIR.

        The direct path is defined as the part of the RIR from the beginning up to
        a few milliseconds after the peak power.

        Args:
            time_after_max_power (float, optional): Time in seconds to include after the RIR's peak. Defaults to 50e-3.
            provide_tail (bool, optional): If True, also returns the reverberant tail. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None]: A tuple containing the direct path
            and optionally the reverberant tail.
        """
        # Find the sample index of the maximum power across all channels.
        max_power_sample = (
            torch.linalg.vector_norm(self.data, ord=2, dim=0).argmax().item()
        )
        end_of_direct_path = max_power_sample + int(time_after_max_power * self.sr)

        direct = self.data[:, :end_of_direct_path]
        tail = self.data[:, end_of_direct_path:] if provide_tail else None

        return direct, tail

    def oracleRTF(
        self, framework: STFTtransform, time_after_max_power: float = 50e-3
    ) -> torch.Tensor:
        """
        Computes the oracle Relative Transfer Function (RTF) from the RIR.

        This is done by convolving the RIR with white noise to get a broadband signal,
        transforming to the STFT domain, computing the spatial covariance matrix,
        and extracting the principal eigenvector.

        Args:
            framework (STFTtransform): The STFT transformation object.
            time_after_max_power (float, optional): Time window for direct path. Defaults to 50e-3.

        Returns:
            torch.Tensor: The estimated RTF.
        """
        # TODO: store these in signal object to not compute them multiple times !!!
        direct = self.data  # Using the full RIR for RTF estimation
        # Convolve with white noise to create a wideband signal for robust covariance estimation.
        directwhite = convolve_white2microphone(direct, samples=1600000)
        stftsig = framework.encode(directwhite)
        # Calculate the Sample Covariance Matrix (SCM).
        covMat = covariance_SCM(stftsig)[..., None, :, :]
        # Return the principal eigenvector, which corresponds to the RTF.
        return peigvech(covMat)


class Noise:
    """
    Represents a noise audio file, loaded and truncated to a specific length.
    """

    def __init__(
        self,
        filename: str,
        sampling_frequency: float = 16e3,
        signal_len: float = 10.0,
    ) -> None:
        """
        Initializes the Noise object.

        Args:
            filename (str): Path to the noise file. Expected format like 'Noise_BX0310_BTE2x2center_babble.wav'.
            sampling_frequency (float, optional): Target sampling frequency. Defaults to 16000.
            signal_len (float, optional): Desired length of the noise signal in seconds. Defaults to 10.0.
        """
        self.filename = filename
        # Parse metadata from the filename based on an expected format.
        parts = path.basename(filename).split("_")
        self.room = parts[1]
        self.array = parts[2]
        self.noise_type = parts[3][:-4]  # Remove '.wav' extension
        self.data, self.sr = load_audio(filename, sampling_frequency=sampling_frequency)
        # Truncate the noise signal to the desired length.
        self.data = self.data[:, : int(signal_len * self.sr)]


# Utility Functions


def load_audio(
    filename: str | Path,
    signal_len: float | None = None,
    sampling_frequency: float = 16000.0,
) -> tuple[torch.Tensor, float]:
    """
    Loads an audio file, resamples, normalizes, and optionally truncates it.

    Args:
        filename (str): The path to the audio file.
        signal_len (float | None, optional): The desired length of the audio in seconds.
                                             If None, the full audio is loaded. Defaults to None.
        sampling_frequency (float, optional): The target sampling frequency. Defaults to 16000.0.

    Returns:
        tuple[torch.Tensor, float]: A tuple containing:
            - The processed audio data as a tensor (DC offset removed).
            - The sampling frequency of the audio data.
    """
    # Use torchcodec's AudioDecoder to load and resample the audio.
    decoder = AudioDecoder(source=filename, sample_rate=int(sampling_frequency))

    # Get samples up to signal_len if specified, otherwise get all samples.
    # The get_samples_played_in_range method handles truncation.
    samples = decoder.get_samples_played_in_range(stop_seconds=signal_len)
    audio_tensor = samples.data
    actual_sampling_rate = samples.sample_rate

    # Normalize the audio by removing the DC offset (subtracting the mean).
    # AudioDecoder normalizes to [-1, 1], but doesn't guarantee zero mean.
    mean_normalized_audio = audio_tensor - audio_tensor.mean(dim=-1, keepdim=True)

    return mean_normalized_audio, actual_sampling_rate


def generate_activation_pattern(
    source_ids: list[str],
    max_sources: int = 4,
    total_duration: float = 60.0,  # [s]
    initial_noise_only_duration: Union[float, list[float], tuple[float, float]] = (
        2.0,
        5.0,
    ),  # [s]
    time_between_events: Union[float, list[float], tuple[float, float]] = (
        2.0,
        5.0,
    ),  # [s]
    activations_only: bool = False,
):
    """
    Generates a sequence of source activation and deactivation events.

    This function simulates a scenario where sources turn on and off over time,
    following a simple state machine logic.

    Args:
        source_ids (List[str]): A list of unique string identifiers for all potential sources.
        max_sources (int): Maximum number of simultaneous sources.
        time_between_events (Union[float, List[float], Tuple[float, float]]): Time between events.
            Can be a single float for a fixed duration, or a list/tuple of two
            floats for a random duration within that interval.
        total_duration (float): Total duration of the scenario in seconds.
        initial_noise_only_duration (Union[float, List[float], Tuple[float, float]]): Initial noise period.
            Can be a single float for a fixed duration, or a list/tuple of two
            floats for a random duration within that interval.
        activations_only (bool): If True, only activation events are generated (sources never turn off).

    Returns:
        list[dict]: A list of event dictionaries. Each dictionary contains:
                    - 'time' (float): The time of the event in seconds.
                    - 'type' (int): The event type (+1 for activation, -1 for deactivation).
                    - 'num_sources' (int): The number of active sources after the event.
                    - 'source_id' (str): The ID of the source involved in this event.
                    - 'active_sources' (list[str]): A list of IDs of all active sources after this event.
    """
    event_log = []
    available_sources = list(source_ids)
    active_sources = []

    # Ensure max_sources does not exceed the total number of available sources
    if max_sources > len(source_ids):

        warnings.warn(
            f"max_sources ({max_sources}) is greater than the number of available sources ({len(source_ids)}). "
            f"It will be capped at {len(source_ids)}."
        )
        max_sources = len(source_ids)

    # Set initial time, allowing for a random duration from an interval.
    current_time = sample_parameter(initial_noise_only_duration)

    # Loop until the scenario duration is reached.
    while current_time < total_duration:
        possible_actions = []

        # Determine possible actions based on the current number of sources.
        if len(active_sources) < max_sources and available_sources:
            possible_actions.append("activation")

        if active_sources and not activations_only:
            possible_actions.append("deactivation")

        # Stop if no more actions can be taken.
        if not possible_actions:
            break

        # Randomly choose the next action (activation or deactivation).
        action = random.choice(possible_actions)
        source_id_for_event = None

        # Update the number of sources and log the event.
        if action == "activation":
            event_type = 1
            source_id_for_event = random.choice(available_sources)
            active_sources.append(source_id_for_event)
            available_sources.remove(source_id_for_event)
        else:  # deactivation
            event_type = -1
            source_id_for_event = random.choice(active_sources)
            available_sources.append(source_id_for_event)
            active_sources.remove(source_id_for_event)

        event_log.append(
            {
                "time": current_time,
                "type": event_type,
                "num_sources": len(active_sources),
                "source_id": source_id_for_event,
                "active_sources": active_sources.copy(),
            }
        )

        # Determine the time for the next event, allowing for a random duration from an interval.
        current_time += sample_parameter(time_between_events)

    return event_log


def convert_pattern_to_time_series(
    activation_pattern: list[dict],
    total_duration: float,
    sampling_frequency: int,
) -> torch.Tensor:
    """
    Converts an event-based activation pattern to a sample-wise time series.

    Args:
        activation_pattern (List[dict]): A list of event dictionaries from
                                          generate_activation_pattern.
        total_duration (float): The total duration of the scenario in seconds.
        sampling_frequency (int): The sampling frequency in Hz.

    Returns:
        torch.Tensor: A 1D integer tensor where each element represents the
                      number of active sources at that time sample.
    """
    total_samples = int(total_duration * sampling_frequency)
    time_series = torch.zeros(total_samples, dtype=torch.int)

    last_sample_idx = 0
    num_sources_before_event = 0

    # Iterate through each activation or deactivation event
    for event in activation_pattern:
        event_sample_idx = int(event["time"] * sampling_frequency)

        # Fill the tensor from the last event up to the current one
        if event_sample_idx > last_sample_idx:
            time_series[last_sample_idx:event_sample_idx] = num_sources_before_event

        # Update the number of sources for the next segment
        num_sources_before_event = event["num_sources"]
        last_sample_idx = event_sample_idx

    # Fill the remainder of the tensor after the last event
    time_series[last_sample_idx:] = num_sources_before_event

    return time_series


# Define a TypeVar for the types we can sample
intfloatstr = TypeVar("intfloatstr", int, float, str)


@overload
def sample_parameter(param: intfloatstr, num: None = None) -> intfloatstr: ...
@overload
def sample_parameter(param: intfloatstr, num: int) -> list[intfloatstr]: ...
@overload
def sample_parameter(param: list[Any], num: None = None) -> Any: ...
@overload
def sample_parameter(param: list[Any], num: int) -> list[Any]: ...
@overload
def sample_parameter(
    param: tuple[intfloatstr, ...], num: None = None
) -> intfloatstr: ...


@overload
def sample_parameter(param: tuple[intfloatstr, ...], num: int) -> list[intfloatstr]: ...


@overload
def sample_parameter(param: dict, num: None = None) -> dict: ...


@overload
def sample_parameter(param: dict, num: int) -> dict: ...
def sample_parameter(
    param: Union[intfloatstr, list[Any], tuple[intfloatstr, ...], dict],
    num: int | None = None,
) -> Any:
    """
    Samples a value or a list of values from a parameter configuration.

    - If param is a single value (int, float, str), it returns that value.
    - If param is a list or a tuple of choices, it returns a random choice.
    - If param is a tuple representing a numeric range, it samples from that range.
    - If param is a dictionary, it samples key-value pairs.

    If the 'num' argument is provided, it returns a list/dict of 'num' unique samples.

    Args:
        param: The parameter configuration (single value, list, tuple, or dict).
        num (int, optional): The number of samples to return. If None, returns a
                                single value. Defaults to None.

    Returns:
        A single sampled value or a list/dict of sampled values.
    """

    def _get_single_sample() -> Any:
        if isinstance(param, (int, float, str)):
            return param
        elif isinstance(param, list):
            return random.choice(param)
        elif isinstance(param, tuple):
            if (
                len(param) == 2
                and isinstance(param[0], (int, float))
                and isinstance(param[1], (int, float))
            ):
                low, high = param
                if isinstance(low, int) and isinstance(high, int):
                    return cast(intfloatstr, random.randint(low, high))
                else:
                    return cast(intfloatstr, random.uniform(float(low), float(high)))
            else:  # Assumes a tuple of choices
                return random.choice(param)
        # Handle any dictionary by sampling a single key-value pair
        elif isinstance(param, dict):
            key = random.choice(list(param.keys()))
            return {key: param[key]}
        else:
            raise TypeError(f"Unsupported type for parameter: {param}")

    if num is None:
        return _get_single_sample()
    else:
        # Handle multi-sampling
        if isinstance(param, (list, tuple)):
            # If it's a numeric range, draw 'num' independent samples
            if (
                isinstance(param, tuple)
                and len(param) == 2
                and isinstance(param[0], (int, float))
                and isinstance(param[1], (int, float))
            ):
                return [_get_single_sample() for _ in range(num)]
            # Otherwise, perform unique sampling (without replacement)
            else:
                if num > len(param):
                    raise ValueError(
                        f"Cannot sample {num} unique items from a list of size {len(param)}."
                    )
                return random.sample(param, k=num)
        # Handle any dictionary by sampling 'num' key-value pairs
        elif isinstance(param, dict):
            if num > len(param):
                raise ValueError(
                    f"Cannot sample {num} unique items from a dictionary of size {len(param)}."
                )
            keys = random.sample(list(param.keys()), k=num)
            return {key: param[key] for key in keys}
        # If param is a single value, return a list of that value repeated num times
        elif isinstance(param, (int, float, str)):
            if num > 1:
                raise ValueError(
                    f"Cannot sample {num} unique items from a single value."
                )
            return [param] * num
        else:
            raise TypeError(f"Unsupported type for multi-sampling: {param}")


def plot_activity_pattern(
    source_activity: dict[str, torch.Tensor],
    sampling_frequency: int,
    output_path: str,
    detected_times: torch.Tensor | None = None,
):
    """
    Plots the source activity pattern as horizontal bars and saves it to a file.
    Optionally, it can also plot vertical lines for detected event times.

    Args:
        source_activity (dict[str, torch.Tensor]): A dictionary mapping source IDs
            to boolean time-series tensors indicating activity.
        sampling_frequency (int): The sampling frequency of the signals.
        output_path (str): The path to save the output PNG image.
        detected_times (torch.Tensor | None, optional): A 1D tensor of times (in seconds)
            to mark with vertical lines. Defaults to None.
    """
    source_ids = [sid for sid in source_activity.keys()]
    if not source_ids:
        print("No sources to plot.")
        return

    fig, ax = plt.subplots(figsize=(15, len(source_ids) * 0.5 + 1))

    for i, source_id in enumerate(source_ids):
        activity_ts = source_activity[source_id].numpy().squeeze()

        # Find where activity starts and ends
        diff = np.diff(np.concatenate(([0], activity_ts, [0])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Convert sample indices to time in seconds
        starts_sec = starts / sampling_frequency
        ends_sec = ends / sampling_frequency

        # Create (start, duration) tuples for plotting
        intervals = [(start, end - start) for start, end in zip(starts_sec, ends_sec)]

        if intervals:
            ax.broken_barh(intervals, (i - 0.4, 0.8), facecolors="darkgreen")

    # Add vertical lines for detected times if provided
    if detected_times is not None:
        for time_sec in detected_times:
            ax.axvline(
                x=time_sec.item(), color="darkred", linestyle="--", linewidth=1.5
            )

    ax.set_yticks(range(len(source_ids)))
    ax.set_yticklabels(source_ids)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Source ID")
    ax.set_title("Source Activity Pattern")
    ax.invert_yaxis()  # Puts the first source at the top
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)


class MicrophoneArray:
    """
    Generates microphone array coordinates based on abstract geometric rules.

    This class creates a blueprint for an array, centered at the origin,
    defined by its geometry, distribution type, number of mics, and a
    single sizing parameter: the maximum distance between any two mics.
    """

    def __init__(
        self,
        num_mics: int,
        geometry: str,
        distribution: str,
        max_distance: float | None = None,  # Made optional
        min_distance: float = 0.0,
        **kwargs,  # Added to capture 'radius'
    ):
        """
        Constructs the array blueprint and generates the local coordinates.

        Args:
            num_mics: The number of microphones.
            geometry: The shape to distribute mics in. One of:
                      ['line', 'circle', 'disk', 'square_area',
                       'sphere_surface', 'sphere_volume', 'cube_volume'].
            distribution: How to place mics. One of: ['regular', 'random'].
            max_distance: The maximum distance between any two mics in the array.
            min_distance: The minimum required distance between any two mics.
        """
        self.num_mics = sample_parameter(num_mics)
        self.geometry = geometry
        self.distribution = distribution
        # Handle 'radius' for double_tetraeder or custom configs
        self.radius = None
        if "radius" in kwargs:
            self.radius = sample_parameter(kwargs["radius"])
        self.max_distance = (
            max_distance if max_distance is not None else self._infer_max_distance()
        )
        self.min_distance = min_distance

        # Determine if the geometry is fundamentally 2D or 3D
        self._2d_geometries = ["line", "circle", "disk", "square_area"]
        self.dimensionality = 2 if self.geometry in self._2d_geometries else 3

        # This holds the final coordinates, shape (3, num_mics), centered at origin.
        self.local_locations = self._generate_locations()
        self.distance_matrix = self._compute_distance_matrix()

    def _infer_max_distance(self) -> float:
        """
        Infers a reasonable max_distance based on the geometry and number of mics.
        This is a fallback if max_distance is not provided.

        Returns:
            A float representing the inferred maximum distance between any two mics.
        """
        if self.geometry == "double_tetraeder":
            return self.radius * 2 if self.radius is not None else 1.0
        raise NotImplementedError(
            "Automatic max_distance inference is not implemented yet. "
            "Please provide max_distance explicitly when creating a MicrophoneArray instance."
        )
        if self.geometry == "line":
            return self.num_mics - 1  # Assuming unit spacing for regular line
        elif self.geometry == "circle":
            return self.num_mics / np.pi  # Approximate circumference for regular circle
        elif self.geometry == "disk":
            return np.sqrt(self.num_mics)  # Approximate diameter for regular disk
        elif self.geometry == "square_area":
            return np.sqrt(
                2 * self.num_mics
            )  # Diagonal of a square with num_mics points
        elif self.geometry in ["sphere_surface", "sphere_volume"]:
            return (
                self.num_mics ** (1 / 3)
            ) * 2  # Approximate diameter for regular sphere
        elif self.geometry == "cube_volume":
            return (
                self.num_mics ** (1 / 3)
            ) * 2  # Approximate diagonal for regular cube
        elif self.geometry == "double_tetraeder":
            return self.radius * 2 if self.radius is not None else 1.0
        else:
            raise NotImplementedError(f"Geometry '{self.geometry}' is not recognized.")

    @classmethod
    def from_locations(cls, local_locations: np.ndarray):
        """
        Alternative constructor to create a MicrophoneArray from pre-defined locations.
        This method will automatically center the provided locations and determine
        if the geometry is 2D (co-planar) or 3D.

        Args:
            local_locations (np.ndarray): A numpy array of shape (3, num_mics)
                                          containing the microphone coordinates.
        """
        # Validate input shape
        if (
            not isinstance(local_locations, np.ndarray)
            or local_locations.ndim != 2
            or local_locations.shape[0] != 3
        ):
            raise ValueError(
                f"local_locations must be a numpy array of shape (3, N), but got shape {local_locations.shape}"
            )

        num_mics = local_locations.shape[1]

        # 1. Center the locations by subtracting their center of mass.
        if num_mics > 0:
            center_of_mass = np.mean(local_locations, axis=1, keepdims=True)
            centered_locations = local_locations - center_of_mass
        else:
            centered_locations = local_locations

        # 2. Determine dimensionality by checking for co-planarity.
        # If 3 or fewer points, they are always co-planar.
        if num_mics <= 3:
            dimensionality = 2
        else:
            # Use SVD to find the variance along principal axes.
            # If the smallest singular value is near zero, the points are co-planar.
            _u, s, _vh = np.linalg.svd(centered_locations.T)
            is_planar = np.isclose(s[-1], 0)
            dimensionality = 2 if is_planar else 3

        # 3. Calculate max and min distances from the centered points
        if num_mics > 1:
            from scipy.spatial.distance import pdist

            pairwise_distances = pdist(centered_locations.T)
            max_dist = float(np.max(pairwise_distances))
            min_dist = float(np.min(pairwise_distances))
        else:
            max_dist = 0.0
            min_dist = 0.0

        # 4. Instantiate the class with inferred/placeholder params
        instance = cls(
            num_mics=num_mics,
            geometry="custom",
            distribution="custom",
            max_distance=max_dist,
            min_distance=min_dist,
        )

        # 5. Override locations and dimensionality
        instance.local_locations = centered_locations
        instance.dimensionality = dimensionality

        return instance

    def _generate_locations(self) -> np.ndarray:
        """
        Dispatcher method that calls the correct helper based on geometry and distribution.
        """
        # If geometry is 'custom', it means locations are provided externally.
        if self.geometry == "custom":
            return np.zeros((3, self.num_mics))  # Return a placeholder

        # A mapping from (geometry, distribution) to the appropriate helper function.
        generation_methods = {
            ("line", "regular"): self._generate_regular_line,
            ("line", "random"): self._generate_random_line,
            ("circle", "regular"): self._generate_regular_circle,
            ("circle", "random"): self._generate_random_circle,
            ("disk", "regular"): self._generate_regular_disk,
            ("disk", "random"): self._generate_random_disk,
            ("sphere_surface", "regular"): self._generate_regular_sphere_surface,
            ("sphere_surface", "random"): self._generate_random_sphere_surface,
            ("sphere_volume", "regular"): self._generate_regular_sphere_volume,
            ("sphere_volume", "random"): self._generate_random_sphere_volume,
            ("cube_volume", "random"): self._generate_random_cube_volume,
            ("square_area", "random"): self._generate_random_square_area,
            ("gen2D", "random"): self._generate_random_2D_array_old,
            ("gen3D", "random"): self._generate_random_3D_array_old,
            ("double_tetraeder", "fixed"): self._generate_regular_double_tetraeder,
        }

        method_key = (self.geometry, self.distribution)
        if method_key not in generation_methods:
            raise NotImplementedError(
                f"The combination of geometry='{self.geometry}' and "
                f"distribution='{self.distribution}' is not supported."
            )

        # Call the selected helper function.
        return generation_methods[method_key]()

    def _compute_distance_matrix(self) -> np.ndarray:
        """Computes the pairwise distance matrix between microphones."""
        if self.num_mics == 0:
            return np.zeros((0, 0))

        # Using broadcasting to compute pairwise distances efficiently.
        diff = self.local_locations[:, :, None] - self.local_locations[:, None, :]
        dist_matrix = np.linalg.norm(diff, axis=0)

        return dist_matrix

    # --- Helper Methods ---

    def _generate_regular_line(self) -> np.ndarray:
        """Generates regularly spaced points on a line of length `max_distance`."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        # Use np.linspace to create evenly spaced points from -half to +half length.
        # This ensures the total distance between the first and last mic is `max_distance`.
        half_length = self.max_distance / 2.0
        x_coords = np.linspace(-half_length, half_length, num=self.num_mics)

        # Create the final 3D coordinate array, placing points along the x-axis.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords

        return locations

    def _generate_random_line(self) -> np.ndarray:
        """Generates uniformly random points on a line of length `max_distance`."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        half_length = self.max_distance / 2.0

        # Sample `num_mics` points from a uniform distribution
        # within the bounds [-half_length, half_length].
        x_coords = np.random.uniform(-half_length, half_length, size=self.num_mics)

        # Create the final 3D coordinate array, placing points along the x-axis.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords

        return locations

    def _generate_regular_circle(self) -> np.ndarray:
        """Generates regularly spaced points on a circle's circumference."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Generate evenly spaced angles from 0 to 2*pi.
        # endpoint=False is important to avoid duplicating the first point at 2*pi.
        angles = np.linspace(0, 2 * np.pi, num=self.num_mics, endpoint=False)

        # Convert polar coordinates (radius, angle) to Cartesian (x, y).
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)

        # Create the final 3D coordinate array in the xy-plane.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords
        locations[1, :] = y_coords

        return locations

    def _generate_random_circle(self) -> np.ndarray:
        """Generates uniformly random points on a circle's circumference."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Generate random angles from a uniform distribution between 0 and 2*pi.
        angles = np.random.uniform(0, 2 * np.pi, size=self.num_mics)

        # Convert polar coordinates (radius, angle) to Cartesian (x, y).
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)

        # Create the final 3D coordinate array in the xy-plane.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords
        locations[1, :] = y_coords

        return locations

    def _generate_regular_disk(self) -> np.ndarray:
        """
        Generates regularly spaced points within a 2D disk using a Fermat's spiral pattern.
        This ensures a quasi-uniform distribution for any number of microphones.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Golden angle for spiral distribution
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        # Create an index for each point (0 to N-1)
        indices = np.arange(self.num_mics)

        # Calculate the radius for each point. The sqrt ensures uniform area distribution.
        # The radius scales up to the maximum radius for the last point.
        radii = radius * np.sqrt(indices / (self.num_mics - 1))

        # Calculate the angle for each point using the golden angle
        angles = golden_angle * indices

        # Convert polar coordinates to Cartesian
        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)

        # Create the final 3D coordinate array in the xy-plane.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords
        locations[1, :] = y_coords

        return locations

    def _generate_random_disk(self) -> np.ndarray:
        """Generates uniformly random points within a 2D disk."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Generate random angles uniformly
        angles = np.random.uniform(0, 2 * np.pi, size=self.num_mics)

        # Generate random radii. Taking the sqrt of a uniform variable
        # ensures that the points are uniformly distributed by area.
        sqrt_radii = radius * np.sqrt(np.random.uniform(0, 1, size=self.num_mics))

        # Convert polar coordinates to Cartesian
        x_coords = sqrt_radii * np.cos(angles)
        y_coords = sqrt_radii * np.sin(angles)

        # Create the final 3D coordinate array in the xy-plane.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords
        locations[1, :] = y_coords

        return locations

    def _generate_regular_sphere_surface(self) -> np.ndarray:
        """
        Generates quasi-regularly spaced points on a sphere's surface using a Fibonacci lattice.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        # For a single mic on a surface, place it at an arbitrary point on the surface, e.g., along the x-axis.
        if self.num_mics == 1:
            locations = np.zeros((3, 1))
            locations[0, 0] = self.max_distance / 2.0
            return locations

        radius = self.max_distance / 2.0

        # Golden angle for spiral distribution on a sphere
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        # Create an index for each point (0 to N-1)
        indices = np.arange(self.num_mics)

        # Calculate the y-coordinate (latitude). This distributes points evenly along the y-axis.
        y = 1 - (2 * indices) / (self.num_mics - 1)

        # Calculate the radius of the circle at that y-height
        radius_at_y = np.sqrt(1 - y**2)

        # Calculate the angle (longitude) for each point using the golden angle
        theta = golden_angle * indices

        # Convert to Cartesian coordinates (unit sphere)
        x = radius_at_y * np.cos(theta)
        z = radius_at_y * np.sin(theta)

        # Scale by the desired radius and combine into the final array
        locations = np.vstack([x, y, z]) * radius

        return locations

    def _generate_random_sphere_surface(self) -> np.ndarray:
        """Generates uniformly random points on a sphere's surface."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        # For a single mic on a surface, place it at an arbitrary point on the surface, e.g., along the x-axis.
        if self.num_mics == 1:
            locations = np.zeros((3, 1))
            locations[0, 0] = self.max_distance / 2.0
            return locations

        radius = self.max_distance / 2.0

        # Generate points from a 3D Gaussian distribution.
        # When normalized, these points are uniformly distributed on a sphere's surface.
        locations = np.random.randn(3, self.num_mics)

        # Normalize each column (each point) to have a length of 1.
        norms = np.linalg.norm(locations, axis=0)
        locations /= norms

        # Scale the points to the desired radius.
        locations *= radius

        return locations

    def _generate_regular_sphere_volume(self) -> np.ndarray:
        """
        Generates quasi-regularly spaced points within a 3D sphere using a 3D Fibonacci lattice.
        """
        print(
            "WARNING: This approximate method for regular distribution in a sphere volume has still a major error."
        )
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Create an index for each point (0 to N-1)
        indices = np.arange(self.num_mics) - self.num_mics / 2.0
        # Shift by 0.5 for better distribution
        # if num_mics is even then add 0.5 to center points better
        if self.num_mics % 2 == 0:
            indices += 0.5
        else:
            indices += 1.0

        indices_alt = []
        for ind in indices:
            if ind > 0:
                indices_alt.append(ind)
                indices_alt.append(-ind)
            elif ind == 0:
                indices_alt.append(ind)
            else:
                continue
        indices_alt = np.array(indices_alt)  # [: self.num_mics])

        # --- 1. Calculate the radial distance for each point ---
        # The cubic root ensures uniform volume distribution.
        # We use (indices + 0.5) to avoid a point at the exact center (radius=0)
        # and to distribute points more evenly.
        radii = radius * np.cbrt((indices_alt) / self.num_mics)

        # --- 2. Calculate the angular components using the golden angle ---
        # This distributes points evenly on a sphere's surface.
        golden_angle = np.pi * (1 + np.sqrt(5))  # Use the other golden ratio variant

        # Azimuthal angle (phi)
        phi = golden_angle * indices

        # Polar angle (theta)
        theta = np.pi * np.cos(2 * indices / self.num_mics)
        # cos_theta = 1 - (2 * indices + 1) / self.num_mics
        # sin_theta = np.sqrt(1 - cos_theta**2)

        # --- 3. Convert spherical to Cartesian coordinates ---
        # Each point's direction is determined by the angles, and its distance
        # from the origin is determined by its corresponding radius.
        x = radii * np.sin(theta) * np.cos(phi)
        y = radii * np.sin(theta) * np.sin(phi)
        z = radii * np.cos(theta)

        locations = np.vstack([x, y, z])

        return locations

    def _generate_random_sphere_volume(self) -> np.ndarray:
        """Generates uniformly random points within a 3D sphere."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # 1. Generate random directions uniformly on a sphere surface.
        # This is done by sampling from a 3D Gaussian and normalizing.
        locations = np.random.randn(3, self.num_mics)
        norms = np.linalg.norm(locations, axis=0)
        # Avoid division by zero for the unlikely case of a zero vector
        norms[norms == 0] = 1
        locations /= norms

        # 2. Generate random radii. To ensure uniform volume distribution,
        # the radii must be sampled from a distribution whose PDF is
        # proportional to r^2. This is achieved by taking the cube root
        # of a uniform random variable.
        random_radii = radius * np.cbrt(np.random.uniform(0, 1, size=self.num_mics))

        # 3. Scale the unit direction vectors by the random radii.
        locations *= random_radii

        return locations

    def _generate_random_square_area(self) -> np.ndarray:
        """
        Generates uniformly random points within a 2D square, respecting a minimum distance.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        # --- 1. Pre-check for physical impossibility (2D version) ---
        # Based on the densest packing of equal circles in a plane (hexagonal packing),
        # which can fill about 90.69% of the total area.
        if self.min_distance > 0:
            # The area of the exclusion circle around each microphone.
            area_of_exclusion_circle = np.pi * (self.min_distance / 2) ** 2
            total_exclusion_area = self.num_mics * area_of_exclusion_circle

            # The total area of the square in which mics are placed.
            square_area = self.max_distance**2

            # Maximum theoretical packing density for circles in 2D.
            max_packing_density = np.pi / (2 * np.sqrt(3))  # Approx. 0.9069
            max_fillable_area = square_area * max_packing_density

            assert total_exclusion_area <= max_fillable_area, (
                f"Configuration is physically impossible due to circle packing limits. "
                f"Required exclusion area for {self.num_mics} mics ({total_exclusion_area:.4f} m^2) "
                f"exceeds the maximum fillable area of the square ({max_fillable_area:.4f} m^2), "
                f"which is ~90.7% of the total square area ({square_area:.4f} m^2)."
            )

        # --- 2. Iterative placement ("dart throwing") ---
        half_length = self.max_distance / 2.0
        locations = np.zeros((3, self.num_mics))
        max_attempts_per_mic = 1000  # Failsafe to prevent infinite loops

        for i in range(self.num_mics):
            for attempt in range(max_attempts_per_mic):
                # Generate a random candidate point in the XY plane
                candidate_xy = np.random.uniform(-half_length, half_length, size=2)
                candidate = np.array([candidate_xy[0], candidate_xy[1], 0.0]).reshape(
                    3, 1
                )

                # If it's the first point or min_distance is zero, accept it immediately
                if i == 0 or self.min_distance == 0:
                    locations[:, i] = candidate.flatten()
                    break

                # Check distance to all previously placed points
                distances = np.linalg.norm(locations[:, :i] - candidate, axis=0)
                if np.all(distances >= self.min_distance):
                    locations[:, i] = candidate.flatten()
                    break  # Valid point found, move to the next mic
            else:
                # This 'else' belongs to the 'for attempt' loop.
                raise RuntimeError(
                    f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                    f"attempts. The configuration with num_mics={self.num_mics}, "
                    f"max_distance={self.max_distance}, and min_distance={self.min_distance} "
                    f"is likely too dense to solve."
                )

        return locations

    def _generate_random_cube_volume(self) -> np.ndarray:
        """
        Generates uniformly random points within a 3D cube, respecting a minimum distance.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        # --- 1. Pre-check for physical impossibility ---
        # Based on the Kepler conjecture, the densest possible packing of equal spheres
        # in 3D space can only fill about 74% of the total volume. We use this to
        # provide a much stricter and more realistic check for dense configurations.
        if self.min_distance > 0:
            # The volume of the exclusion sphere around each microphone.
            # The radius of this sphere is half the minimum distance.
            volume_of_exclusion_sphere = (4 / 3) * np.pi * (self.min_distance / 2) ** 3
            total_exclusion_volume = self.num_mics * volume_of_exclusion_sphere

            # The total volume of the cube in which mics are placed.
            cube_volume = self.max_distance**3

            # Maximum theoretical packing density for spheres in 3D.
            max_packing_density = np.pi / (3 * np.sqrt(2))  # Approx. 0.74048
            max_fillable_volume = cube_volume * max_packing_density

            assert total_exclusion_volume <= max_fillable_volume, (
                f"Configuration is physically impossible due to sphere packing limits. "
                f"Required exclusion volume for {self.num_mics} mics ({total_exclusion_volume:.4f} m^3) "
                f"exceeds the maximum fillable volume of the cube ({max_fillable_volume:.4f} m^3), "
                f"which is ~74% of the total cube volume ({cube_volume:.4f} m^3)."
            )

        # --- 2. Iterative placement ("dart throwing") ---
        half_length = self.max_distance / 2.0
        locations = np.zeros((3, self.num_mics))
        max_attempts_per_mic = 1000  # Failsafe to prevent infinite loops

        for i in range(self.num_mics):
            for attempt in range(max_attempts_per_mic):
                # Generate a random candidate point
                candidate = np.random.uniform(
                    -half_length, half_length, size=3
                ).reshape(3, 1)

                # If it's the first point or min_distance is zero, accept it immediately
                if i == 0 or self.min_distance == 0:
                    locations[:, i] = candidate.flatten()
                    break

                # Check distance to all previously placed points
                # `locations[:, :i]` slices the already placed microphones
                distances = np.linalg.norm(locations[:, :i] - candidate, axis=0)
                if np.all(distances >= self.min_distance):
                    locations[:, i] = candidate.flatten()
                    break  # Valid point found, move to the next mic
            else:
                # This 'else' belongs to the 'for attempt' loop.
                # It runs only if the loop completes without a 'break'.
                raise RuntimeError(
                    f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                    f"attempts. The configuration with num_mics={self.num_mics}, "
                    f"max_distance={self.max_distance}, and min_distance={self.min_distance} "
                    f"is likely too dense to solve, even if theoretically possible."
                )

        return locations

    def _generate_random_2D_array_old(self) -> np.ndarray:
        """
        Generates uniformly random points in a 2D plane, constrained only by min/max
        inter-microphone distances.

        The process is as follows:
        1. Iteratively place microphones ("dart throwing") ensuring only the `min_distance`
           is respected. The initial placement area is unbounded.
        2. After all points are placed, find the maximum pairwise distance in the generated set.
        3. Scale the entire array down so that this maximum distance equals `self.max_distance`.
        4. Center the final array by subtracting its center of mass.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            return np.zeros((3, 1))

        locations = [np.array([0.0, 0.0, 0.0])]
        max_attempts_per_mic = 100

        # 1. Iteratively place points respecting min_distance
        for i in range(1, self.num_mics):
            for attempt in range(max_attempts_per_mic):

                # # Pick a random existing point to place the new point near
                # anchor_point = locations[:, np.random.randint(i)]

                # Calculate center of mass of current points
                current_mics = np.column_stack(locations)
                center_of_mass = np.mean(current_mics, axis=1, keepdims=True)

                # Find the index of the mic closest to the center of mass
                distances_to_com = np.linalg.norm(current_mics - center_of_mass, axis=0)
                closest_mic_index = np.argmin(distances_to_com)

                # Set the anchor point to this closest mic
                anchor_point = current_mics[:, closest_mic_index]

                # Generate a candidate point in a random direction, at least min_distance away
                r = np.sqrt(
                    np.random.uniform(self.min_distance**2, self.max_distance**2)
                )
                angle = np.random.uniform(0, 2 * np.pi)
                offset = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
                candidate = anchor_point.reshape(3, 1) + offset.reshape(3, 1)

                # Check distance to all previously placed points
                distances = np.linalg.norm(current_mics[:, :i] - candidate, axis=0)
                if np.all(
                    (self.max_distance >= distances) & (distances >= self.min_distance)
                ):
                    print(
                        f"Placed mic #{i+1} at {candidate.flatten()} after {attempt+1} attempts."
                    )
                    locations.append(candidate.flatten())
                    break  # Valid point found
            else:
                # raise RuntimeError(
                #     f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                #     f"attempts. The min_distance={self.min_distance} might be too large."
                # )
                print(
                    f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                    f"attempts. The min_distance={self.min_distance} might be too large."
                )
                break

        # # 2. Scale the array to enforce max_distance
        # if self.num_mics > 1:
        #     from scipy.spatial.distance import pdist

        #     pairwise_distances = pdist(locations.T)
        #     current_max_dist = np.max(pairwise_distances)

        #     if current_max_dist > 0:
        #         scale_factor = self.max_distance / current_max_dist
        #         locations *= scale_factor

        # 3. Center the array
        center_of_mass = np.mean(np.column_stack(locations), axis=1, keepdims=True)
        centered_locations = np.column_stack(locations) - center_of_mass

        return centered_locations

    def _generate_random_3D_array_old(self) -> np.ndarray:
        """
        Generates uniformly random points in 3D space using a "place-then-scale" method.

        The process is as follows:
        1. Iteratively place microphones ("dart throwing") ensuring each new point
           is within [min_distance, max_distance] of all existing points.
        2. After all points are placed, find the maximum pairwise distance in the set.
        3. Scale the entire array down so that this maximum distance equals `self.max_distance`.
        4. Center the final array by subtracting its center of mass.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            return np.zeros((3, 1))

        locations = [np.array([0.0, 0.0, 0.0])]
        max_attempts_per_mic = 100

        # 1. Iteratively place points
        for i in range(1, self.num_mics):
            for attempt in range(max_attempts_per_mic):

                # # Pick a random existing point to place the new point near
                # anchor_point = locations[:, np.random.randint(i)]

                # Calculate center of mass of current points
                current_mics = np.column_stack(locations)
                center_of_mass = np.mean(current_mics, axis=1, keepdims=True)

                # Find the index of the mic closest to the center of mass
                distances_to_com = np.linalg.norm(current_mics - center_of_mass, axis=0)
                closest_mic_index = np.argmin(distances_to_com)

                # Set the anchor point to this closest mic
                anchor_point = current_mics[:, closest_mic_index]

                # Generate a candidate point in a random 3D direction
                r = np.cbrt(
                    np.random.uniform(self.min_distance**3, self.max_distance**3)
                )
                # Generate a random 3D unit vector for the direction
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                offset = r * direction
                candidate = anchor_point.reshape(3, 1) + offset.reshape(3, 1)

                # Check distance to all previously placed points
                distances = np.linalg.norm(current_mics - candidate, axis=0)
                if np.all(
                    (self.max_distance >= distances) & (distances >= self.min_distance)
                ):
                    print(
                        f"Placed mic #{i+1} at {candidate.flatten()} after {attempt+1} attempts."
                    )
                    locations.append(candidate.flatten())
                    break  # Valid point found
            else:
                # raise RuntimeError(
                #     f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                #     f"attempts. The constraints might be too tight."
                # )
                print(
                    f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                    f"attempts. The constraints might be too tight."
                )
                break

        # # 2. Scale the array to enforce the global max_distance
        # if self.num_mics > 1:
        #     from scipy.spatial.distance import pdist

        #     pairwise_distances = pdist(locations.T)
        #     current_max_dist = np.max(pairwise_distances)

        #     if current_max_dist > 0:
        #         scale_factor = self.max_distance / current_max_dist
        #         locations *= scale_factor

        # 3. Center the array
        center_of_mass = np.mean(np.column_stack(locations), axis=1, keepdims=True)
        centered_locations = np.column_stack(locations) - center_of_mass

        return centered_locations

    def _generate_random_2D_array(self) -> np.ndarray:
        """
        Generates random points in a 2D plane where each new point is constrained
        to be within a min/max distance from ALL existing points.

        The process uses the 'shapely' library:
        1. Start with one microphone at the origin.
        2. For each subsequent microphone:
           a. Calculate the valid placement area. This is the geometric intersection
              of annuli (rings) defined by the min/max distance from every
              existing microphone.
           b. Sample a random point from within this valid area.
        3. After all points are placed, center the final array by subtracting its
           center of mass so the array's centroid is at the origin.
        """

        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            return np.zeros((3, 1))

        # Start with the first mic at the origin
        mic_locations = [np.array([0.0, 0.0, 0.0])]
        center_point = Point(mic_locations[0][0], mic_locations[0][1])
        max_circle = center_point.buffer(self.max_distance)
        min_circle = center_point.buffer(self.min_distance)
        valid_area = max_circle.difference(min_circle)

        max_sampling_attempts = 10000  # Per point

        # Iteratively place the remaining microphones
        for i in range(1, self.num_mics):

            # --- 2. Check if a valid placement is possible ---
            if valid_area.is_empty:
                # raise RuntimeError(
                #     f"Failed to place microphone #{i+1}: No valid placement area exists. "
                #     f"The constraints (min_dist={self.min_distance}, max_dist={self.max_distance}) "
                #     f"are likely too tight for {self.num_mics} microphones."
                # )
                print(
                    f"Failed to place microphone #{i+1}: No valid placement area exists. "
                    f"The constraints (min_dist={self.min_distance}, max_dist={self.max_distance}) "
                    f"are likely too tight for {self.num_mics} microphones."
                )
                break

            # --- 3. Sample a point from within the valid area ---
            min_x, min_y, max_x, max_y = valid_area.bounds
            new_mic_point = None
            for j in range(max_sampling_attempts):
                candidate = Point(
                    np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)
                )
                if valid_area.contains(candidate):
                    print(f"Placed mic #{i+1} at {candidate} after {j+1} attempts.")
                    new_mic_point = candidate
                    break

            if new_mic_point is None:
                # raise RuntimeError(
                #     f"Failed to sample a point for microphone #{i+1} after "
                #     f"{max_sampling_attempts} attempts. The valid area might be "
                #     f"too small or fragmented."
                # )
                print(
                    f"Failed to sample a point for microphone #{i+1} after "
                    f"{max_sampling_attempts} attempts. The valid area might be "
                    f"too small or fragmented."
                )
                break

            # --- 4. Add the new microphone to the list ---
            new_mic_location = np.array([new_mic_point.x, new_mic_point.y, 0.0])
            mic_locations.append(new_mic_location)

            newPoint = Point(new_mic_location[0], new_mic_location[1])
            max_circle = newPoint.buffer(self.max_distance)
            min_circle = newPoint.buffer(self.min_distance)
            annulus = max_circle.difference(min_circle)
            valid_area = valid_area.intersection(annulus)

        # --- 5. Finalize the array ---
        # Convert list of arrays to a single (N, 3) numpy array
        final_locations = np.array(mic_locations)

        # Center the array by subtracting its center of mass
        center_of_mass = np.mean(final_locations, axis=0, keepdims=True)
        centered_locations = final_locations - center_of_mass

        # Return in the required (3, N) shape
        return centered_locations.T

    def _generate_random_3D_array(self) -> np.ndarray:
        """
        Generates random points in 3D space where each new point is constrained
        to be within a min/max distance from ALL existing points.

        The process uses the 'trimesh' library:
        1. Start with one microphone at the origin.
        2. For each subsequent microphone:
           a. Calculate the valid placement volume. This is the geometric
              intersection of spherical shells defined by the min/max distance
              from every existing microphone.
           b. Sample a random point from within this valid volume.
        3. After all points are placed, center the final array by subtracting its
           center of mass so the array's centroid is at the origin.
        """

        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            return np.zeros((3, 1))

        # Start with the first mic at the origin
        mic_locations = [np.array([0.0, 0.0, 0.0])]
        max_sampling_attempts = 10000  # Per point

        max_sphere = trimesh.primitives.Sphere(
            radius=self.max_distance, center=mic_locations[0]
        )
        min_sphere = trimesh.primitives.Sphere(
            radius=self.min_distance, center=mic_locations[0]
        )
        valid_volume = max_sphere.difference(min_sphere)

        # Iteratively place the remaining microphones
        for i in range(1, self.num_mics):

            # --- 2. Check if a valid placement is possible ---
            if valid_volume.is_empty:
                # raise RuntimeError(
                #     f"Failed to place microphone #{i+1}: No valid placement volume exists. "
                #     f"The constraints (min_dist={self.min_distance}, max_dist={self.max_distance}) "
                #     f"are likely too tight for {self.num_mics} microphones."
                # )
                print(
                    f"Failed to place microphone #{i+1}: No valid placement volume exists. "
                    f"The constraints (min_dist={self.min_distance}, max_dist={self.max_distance}) "
                    f"are likely too tight for {self.num_mics} microphones."
                )
                break

            # --- 3. Sample a point from within the valid volume ---
            min_b, max_b = valid_volume.bounds
            new_mic_point = None
            for j in range(max_sampling_attempts):
                candidate = np.random.uniform(low=min_b, high=max_b)
                if valid_volume.contains([candidate]):
                    print(f"Placed mic #{i+1} at {candidate} after {j+1} attempts.")
                    new_mic_point = candidate
                    break

            if new_mic_point is None:
                # raise RuntimeError(
                #     f"Failed to sample a point for microphone #{i+1} after "
                #     f"{max_sampling_attempts} attempts. The valid volume might be "
                #     f"too small or fragmented."
                # )
                print(
                    f"Failed to sample a point for microphone #{i+1} after "
                    f"{max_sampling_attempts} attempts. The valid volume might be "
                    f"too small or fragmented."
                )
                break

            # --- 4. Add the new microphone to the list ---
            mic_locations.append(new_mic_point)
            max_sphere = trimesh.primitives.Sphere(
                radius=self.max_distance, center=new_mic_point
            )
            min_sphere = trimesh.primitives.Sphere(
                radius=self.min_distance, center=new_mic_point
            )
            shell = max_sphere.difference(min_sphere)
            valid_volume = valid_volume.intersection(shell)

        # --- 5. Finalize the array ---
        # Convert list of arrays to a single (N, 3) numpy array
        final_locations = np.array(mic_locations)

        # Center the array by subtracting its center of mass
        center_of_mass = np.mean(final_locations, axis=0, keepdims=True)
        centered_locations = final_locations - center_of_mass

        # Return in the required (3, N) shape
        return centered_locations.T

    def _generate_regular_double_tetraeder(self) -> np.ndarray:
        """
        Generates a fixed double tetrahedron (Stella Octangula configuration).
        Consists of a Normal Tetrahedron (Tele) and an Inverted Tetrahedron (T2).
        Each has its center of mass at the origin.
        """
        # Ensure we have a valid radius
        if self.radius is None:
            self.radius = self.max_distance / 2.0

        R = self.radius

        # --- Tetra 1 (Normal) ---
        # Upright: One face pointing down (-Z), opposite corner pointing up (+Z).
        # Top vertex: (0, 0, R)
        # Base plane: z = -R/3
        # Radius of base circle: r_base = sqrt(R^2 - (R/3)^2) = 2*sqrt(2)/3 * R
        r_base = (2 * np.sqrt(2) / 3) * R
        h_base = -R / 3.0

        # Orientation:
        # "One edge parallel to first dimension (X)"
        # "A corner opposite of pointing towards positive second dimension" -> Pointing to -Y.

        # Base Vertex 1 (Pointing -Y):
        t1_v1 = [0, r_base, h_base]
        # Base Vertex 2 (210 deg):
        t1_v2 = [
            r_base * np.cos(np.deg2rad(210)),
            r_base * np.sin(np.deg2rad(210)),
            h_base,
        ]
        # Base Vertex 3 (330 deg):
        t1_v3 = [
            r_base * np.cos(np.deg2rad(330)),
            r_base * np.sin(np.deg2rad(330)),
            h_base,
        ]
        # Top Vertex:
        t1_top = [0, 0, R]

        # --- Tetra 2 (Inverted) ---
        # "Corners above the centerpoints of the faces of the normal one".
        # This describes the dual tetrahedron, which is strictly T2 = -T1.
        # Alternatively, defined by reflection or rotation logic provided:
        # Bottom vertex: (0, 0, -R)
        # Base plane: z = +R/3 (Since it's inverted)

        # Inverted coordinates are just negative of Normal coordinates
        t2_v1 = [-x for x in t1_v1]  # Points to +Y
        t2_v2 = [-x for x in t1_v2]
        t2_v3 = [-x for x in t1_v3]
        t2_bottom = [-x for x in t1_top]  # (0, 0, -R)

        # Collect all points. Order: [T1_Top, T1_Base1, T1_Base2, T1_Base3, T2_Bottom, T2_Base1, T2_Base2, T2_Base3]
        # You can adjust order if specific channel mapping is needed.
        locations = np.array(
            [t1_top, t1_v1, t1_v2, t1_v3, t2_bottom, t2_v1, t2_v2, t2_v3]
        ).T

        # Check num_mics
        if self.num_mics != 8:
            print(
                f"Warning: 'double_tetraeder' geometry generates 8 mics, but {self.num_mics} were requested. Using 8."
            )

        return locations

    def place(
        self,
        room_dims: list[float],
        min_dist_from_walls: float | None,  # CHANGE: Allow None
        plot_filepath: str | Path | None = None,
        restrict_rot_2_xy_plane: bool = False,
        fix_height: float | None = None,
        fixed_position: list[float] | None = None,  # CHANGE: New argument
        fixed_rotation: bool = False,  # CHANGE: New argument
    ) -> np.ndarray:
        """
        Places the array in a room with a random center and orientation.

        If the array is 2D and `restrict_rot_2_xy_plane` was set to True, the
        orientation is restricted to the XY-plane. Otherwise, a full 3D
        rotation is applied.
        """

        # 0. Warn if 2D rotation is requested for a 3D geometry and override.
        if restrict_rot_2_xy_plane and self.dimensionality == 3:
            warnings.warn(
                f"Restricted rotation to 2D for a 3D geometry ('{self.geometry}'). "
                f"Is this really desired?"
            )

        # 1. Determine Center
        if fixed_position is not None:
            center = np.array(fixed_position)
            # Optional: Check if center is in room
            if np.any(center < 0) or np.any(center > np.array(room_dims)):
                raise ValueError(
                    f"Fixed position {fixed_position} is outside room {room_dims}"
                )
        else:
            # Random placement logic
            safe_dist_walls = (
                min_dist_from_walls if min_dist_from_walls is not None else 0.0
            )
            effective_radius = self.max_distance / 2.0
            min_dist = safe_dist_walls + effective_radius

            center_x = random.uniform(min_dist, room_dims[0] - min_dist)
            center_y = random.uniform(min_dist, room_dims[1] - min_dist)
            if fix_height is not None:
                center_z = fix_height
            else:
                center_z = random.uniform(min_dist, room_dims[2] - min_dist)
            center = np.array([center_x, center_y, center_z])

        # 2. Apply a random rotation based on the flag
        if fixed_rotation:
            rotation_matrix = np.eye(3)
        elif restrict_rot_2_xy_plane:
            # Apply a 2D rotation around the Z-axis
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
            )
        else:
            # Apply a full random 3D rotation
            from scipy.spatial.transform import Rotation as R

            rotation_matrix = R.random().as_matrix()

        rotated_locs = rotation_matrix @ self.local_locations

        # 3. Translate to the final center position in the room
        global_locs = rotated_locs + center[:, np.newaxis]

        # 4. Plot the global locations if a filepath is provided
        if plot_filepath:
            self.plot_global_locations(global_locs, room_dims, plot_filepath)

        return global_locs

    # --- Plotting Methods ---

    def plot_local_locations(self, filepath: str | Path = "Playground/MA_local.png"):
        """
        Generates and saves a 2x2 plot visualizing the local (blueprint) coordinates.
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle("Microphone Array Blueprint (Local Coordinates)", fontsize=16)
        plot_margin = self.max_distance * 0.6

        # 3D Plot (Top-Left)
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax_3d.scatter(
            xs=self.local_locations[0, :],
            ys=self.local_locations[1, :],
            zs=self.local_locations[2, :],  # type: ignore
            c="r",
            marker="o",
        )
        ax_3d.set_title("3D View")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_xlim([-plot_margin, plot_margin])
        ax_3d.set_ylim([-plot_margin, plot_margin])
        ax_3d.set_zlim([-plot_margin, plot_margin])
        ax_3d.set_box_aspect([1, 1, 1])
        ax_3d.grid(True)

        # XY Projection (Top-Right)
        ax_xy = axs[0, 1]
        ax_xy.scatter(
            self.local_locations[0, :], self.local_locations[1, :], c="r", marker="o"
        )
        ax_xy.set_title("XY Projection (Top-Down View)")
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_xlim([-plot_margin, plot_margin])
        ax_xy.set_ylim([-plot_margin, plot_margin])
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(True)

        # XZ Projection (Bottom-Left)
        ax_xz = axs[1, 0]
        ax_xz.scatter(
            self.local_locations[0, :], self.local_locations[2, :], c="r", marker="o"
        )
        ax_xz.set_title("XZ Projection (Front View)")
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_xlim([-plot_margin, plot_margin])
        ax_xz.set_ylim([-plot_margin, plot_margin])
        ax_xz.set_aspect("equal", adjustable="box")
        ax_xz.grid(True)

        # YZ Projection (Bottom-Right)
        ax_yz = axs[1, 1]
        ax_yz.scatter(
            self.local_locations[1, :], self.local_locations[2, :], c="r", marker="o"
        )
        ax_yz.set_title("YZ Projection (Side View)")
        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_xlim([-plot_margin, plot_margin])
        ax_yz.set_ylim([-plot_margin, plot_margin])
        ax_yz.set_aspect("equal", adjustable="box")
        ax_yz.grid(True)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Saved local locations plot to: {filepath}")

    @staticmethod
    def plot_global_locations(
        global_locs: np.ndarray,
        room_dims: list[float],
        filepath: str | Path = "Playground/MA_in_room.png",
    ):
        """
        Generates and saves a 2x2 plot visualizing the global (placed) coordinates.
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(
            "Microphone Array Placed in Room (Global Coordinates)", fontsize=16
        )

        # 3D Plot (Top-Left)
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax_3d.scatter(
            xs=global_locs[0, :],
            ys=global_locs[1, :],
            zs=global_locs[2, :],  # type: ignore
            c="b",
            marker="^",
        )
        ax_3d.set_title("3D View")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_xlim([0, room_dims[0]])
        ax_3d.set_ylim([0, room_dims[1]])
        ax_3d.set_zlim([0, room_dims[2]])
        ax_3d.set_box_aspect(
            (
                np.ptp(ax_3d.get_xlim()),
                np.ptp(ax_3d.get_ylim()),
                np.ptp(ax_3d.get_zlim()),
            )
        )
        ax_3d.grid(True)

        # XY Projection (Top-Right)
        ax_xy = axs[0, 1]
        ax_xy.scatter(global_locs[0, :], global_locs[1, :], c="b", marker="^")
        ax_xy.set_title("XY Projection")
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_xlim([0, room_dims[0]])
        ax_xy.set_ylim([0, room_dims[1]])
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(True)

        # XZ Projection (Bottom-Left)
        ax_xz = axs[1, 0]
        ax_xz.scatter(global_locs[0, :], global_locs[2, :], c="b", marker="^")
        ax_xz.set_title("XZ Projection")
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_xlim([0, room_dims[0]])
        ax_xz.set_ylim([0, room_dims[2]])
        ax_xz.set_aspect("equal", adjustable="box")
        ax_xz.grid(True)

        # YZ Projection (Bottom-Right)
        ax_yz = axs[1, 1]
        ax_yz.scatter(global_locs[1, :], global_locs[2, :], c="b", marker="^")
        ax_yz.set_title("YZ Projection")
        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_xlim([0, room_dims[1]])
        ax_yz.set_ylim([0, room_dims[2]])
        ax_yz.set_aspect("equal", adjustable="box")
        ax_yz.grid(True)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Saved global locations plot to: {filepath}")


@dataclass
class Segment:
    start: int
    end: int
    num_sources: int
    event_type: str  # 'activation', 'deactivation', 'init', 'constant'


def identify_segments(source_activity: torch.Tensor) -> list[Segment]:
    T = len(source_activity)
    if T == 0:
        return []

    # 1. Find indices where activity changes
    # This returns indices `i` where activity[i] != activity[i+1]
    # We assume discrete integer counts.
    change_points = torch.nonzero(
        source_activity[:-1] != source_activity[1:], as_tuple=True
    )[0]

    # 2. Add the end of the last "change" (which is the index itself + 1)
    # Because if change is at index 0, it means segment 0 ends at 1.
    change_indices = change_points + 1

    # 3. Add Start (0) and End (T) boundaries
    # We want boundaries: [0, t1, t2, ..., T]
    boundaries = [0] + change_indices.tolist() + [T]

    segments = []

    # 4. Iterate over boundaries to create Segment objects
    # This loop runs O(N_segments) times, which is typically very small (< 100) compared to T (> 1000s).
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Get the count from the first frame of this segment
        # (Assuming constant count within segment)
        count = int(source_activity[start].item())

        # Determine event type based on previous count
        # Use 'segments' list to check previous state, or track prev_count variable
        event_type = determine_event_type(count, segments)

        segments.append(Segment(start, end, count, event_type))

    return segments


def determine_event_type(current_count: int, segments: list[Segment]) -> str:
    if len(segments) == 0:
        return "init"
    prev_count = segments[-1].num_sources
    if current_count > prev_count:
        return "activation"
    elif current_count < prev_count:
        return "deactivation"
    return "constant"


class HeterogeneousBatch:
    """
    A smart container for mixed-type, mixed-length audio/feature data.
    It handles device transfer and orchestrates feature extraction + padding.
    """

    def __init__(
        self,
        raw_audio: list[torch.Tensor] = [],
        stft_audio: list[torch.Tensor] = [],
        features: list[torch.Tensor | dict[str, torch.Tensor]] = [],
        meta: dict[str, list[Any]] = {},
        stft_info: Optional[dict[str, Any]] = None,
        feature_info: Optional[dict[str, Any]] = None,
        stft_data_needed: bool = False,
    ):
        self.raw_audio = raw_audio
        self.stft_audio = stft_audio
        self.features = features
        self.meta = meta
        self.stft_info = stft_info
        self.feature_info = feature_info
        self.stft_data_needed = stft_data_needed
        self.device = torch.device("cpu")
        self.batch_size = len(next(iter(meta.values()))) if meta else 0

        # Pipeline State
        self.status = "input"  # 'input' -> 'features' -> 'estimates'
        self.mask: Optional[torch.Tensor] = None  # (B, T_max)
        self.padded_features: Optional[torch.Tensor] = None  # (B, J, T_max)
        self.padded_estimates: Optional[torch.Tensor] = None  # (B, J, T_max)
        self.estimates: Optional[list[torch.Tensor] | torch.Tensor] = (
            None  # Output of estimator
        )

        if "gt_rtf_stream" not in self.meta:
            self.meta["gt_rtf_stream"] = []
            self.meta["gt_ids_stream"] = []
            self.meta["id_map"] = []
        for i in range(self.batch_size):
            sad_frames = self.meta["sad_frames"][i]  # Dict of SAD tensors (T,)
            rtfs = self.meta["rtfs"][i]  # Dict of RTF tensors
            segments = self.meta["segments"][i]  # List[Segment]

            # Compute ground-truth RTF stream for this item
            gt_rtf_stream, gt_ids_stream, id_map = ground_truth_rtf_stream(
                sad_frames, rtfs, segments
            )
            self.meta["gt_rtf_stream"].append(gt_rtf_stream)
            self.meta["gt_ids_stream"].append(gt_ids_stream)
            self.meta["id_map"].append(id_map)

    def to(self, device: torch.device):
        """
        Moves all internal tensors to the specified device.
        Called automatically by Lightning if transfer_batch_to_device is overridden.
        """
        self.device = device

        # Move raw audio
        self.raw_audio = [t.to(device) for t in self.raw_audio]

        # Move STFT audio
        self.stft_audio = [t.to(device) for t in self.stft_audio]

        # Move features
        new_features = []
        for t in self.features:
            if isinstance(t, torch.Tensor):
                new_features.append(t.to(device))
            elif isinstance(t, dict):
                new_features.append({k: v.to(device) for k, v in t.items()})
            else:
                raise TypeError(f"Unexpected type in self.features: {type(t)}")
        self.features = new_features

        # Move tensor targets in meta (e.g., source_count)
        if "source_count" in self.meta:
            sc = self.meta["source_count"]
            # source_count is likely a list of 1D tensors (variable length)
            self.meta["source_count"] = [t.to(device) for t in sc]

        # Move RTFs in meta
        if "rtfs" in self.meta:
            new_rtfs_batch = []
            for item in self.meta["rtfs"]:
                if isinstance(item, dict):
                    new_item = {k: v.to(device) for k, v in item.items()}
                    new_rtfs_batch.append(new_item)
                elif isinstance(item, list):
                    new_item = [t.to(device) for t in item]
                    new_rtfs_batch.append(new_item)
                else:
                    new_rtfs_batch.append(item)
            self.meta["rtfs"] = new_rtfs_batch

        if "gt_rtf_stream" in self.meta:
            new_gt_rtf_batch = []
            for item in self.meta["gt_rtf_stream"]:
                if isinstance(item, dict):
                    new_item = {k: v.to(device) for k, v in item.items()}
                    new_gt_rtf_batch.append(new_item)
                elif isinstance(item, list):
                    new_item = [t.to(device) for t in item]
                    new_gt_rtf_batch.append(new_item)
                else:
                    new_gt_rtf_batch.append(item)
            self.meta["gt_rtf_stream"] = new_gt_rtf_batch

        if "sad_frames" in self.meta:
            new_sad_batch = []
            for item in self.meta["sad_frames"]:
                if isinstance(item, dict):
                    new_item = {k: v.to(device) for k, v in item.items()}
                    new_sad_batch.append(new_item)
                elif isinstance(item, list):
                    new_item = [t.to(device) for t in item]
                    new_sad_batch.append(new_item)
                else:
                    new_sad_batch.append(item)
            self.meta["sad_frames"] = new_sad_batch

        return self

    def apply_feature_extractor(
        self, extractor: BaseFeatureExtractor
    ) -> "HeterogeneousBatch":
        """
        Runs the feature extractor on all available input types.
        Unifies results into self.features and clears raw inputs.
        """
        if self.status != "input":
            # If already processed, do nothing or raise error
            return self

        self.processed_features = []

        # 1. Process Raw Audio
        for item in self.raw_audio:
            # item: (M, N) -> Unsqueeze to (1, M, N) for batch processing
            feat = extractor.forward_raw_audio(item.unsqueeze(0))
            self.processed_features.append(feat.squeeze(0))  # Store as (J, T)

        # 2. Process STFT Audio
        for item in self.stft_audio:
            # item: (F, M, T) or similar.
            # We assume item has correct dimensions for the extractor except batch.
            feat = extractor.forward_stft(item.unsqueeze(0))
            self.processed_features.append(feat.squeeze(0))

        # 3. Process Precomputed Features
        for item in self.features:
            if isinstance(item, torch.Tensor):
                # item: (J, T) -> Unsqueeze to (1, J, T)
                feat = extractor.forward_precomputed_features(item.unsqueeze(0))
                self.processed_features.append(feat.squeeze(0))
            elif isinstance(item, dict):
                # item is a dict of tensors. We need to unsqueeze each tensor.
                unsqueezed_item = {k: v.unsqueeze(0) for k, v in item.items()}
                feat = extractor.forward_precomputed_features_dict(unsqueezed_item)
                self.processed_features.append(feat.squeeze(0))
            else:
                raise TypeError(f"Unexpected type in self.features: {type(item)}")

        # Update State
        self.raw_audio = []  # Clear to save memory
        self.stft_audio = []
        self.status = "features"

        return self

    def apply_source_count_estimator(
        self, estimator: BaseSourceCountEstimator
    ) -> "HeterogeneousBatch":
        """
        Pads features, creates mask, and runs the estimator.
        """
        if self.status != "features":
            raise ValueError(
                f"Cannot apply estimator. Current status: {self.status}. Expected: 'features'"
            )

        if not self.features:
            raise ValueError("Batch is empty, no features to process.")

        # 1. Determine Dimensions
        # features list: [ (J, T1), (J, T2), ... ]
        max_len = max([x.shape[-1] for x in self.processed_features])
        J = self.processed_features[0].shape[0]
        B = len(self.processed_features)

        # 2. Allocate Padded Tensor and Mask
        self.padded_features = torch.zeros(
            (B, J, max_len), device=self.device, dtype=self.processed_features[0].dtype
        )
        self.mask = torch.zeros((B, max_len), device=self.device, dtype=torch.bool)

        # 3. Fill Tensor and Mask
        for i, feat in enumerate(self.processed_features):
            length = feat.shape[-1]
            self.padded_features[i, :, :length] = feat
            self.mask[i, :length] = True

        # 4. Run Estimator
        # Estimator expects (B, J, T). Returns dict with (B, T, C)
        self.padded_estimates = estimator.forward(self.padded_features)

        # 5. Unpad Estimates into List Form
        self.estimates = [
            pe[self.mask[i], :] for i, pe in enumerate(self.padded_estimates)
        ]

        self.status = "estimates"
        return self

    def compute_loss(self, loss_fn: BaseLoss) -> dict[str, torch.Tensor]:
        """
        Computes loss using the internal estimates, targets, and mask.
        Handles the masking logic so the Loss function doesn't have to.
        """
        if self.status == "loss":
            return {"loss": self.loss}

        if self.status != "estimates" or self.padded_estimates is None:
            raise ValueError("Cannot compute loss. Estimates not available.")

        if "source_count" not in self.meta:
            raise ValueError(
                "Cannot compute loss. 'source_count' target missing in meta."
            )

        # 1. Get Predictions
        preds = self.padded_estimates  # (B, T_max, C)

        # 2. Prepare Targets (Pad to match T_max)
        targets_list = self.meta["source_count"]  # List of (Ti,)
        B, T_max = preds.shape[0], preds.shape[1]

        padded_targets = torch.zeros((B, T_max), device=self.device, dtype=torch.long)

        # We use the same mask we generated during feature padding
        # But we must ensure targets align with that mask
        for i, target in enumerate(targets_list):
            length = target.shape[0]
            # Check whether target length fits to feature length
            if self.mask is not None and not self.mask[i].sum() == length:
                raise ValueError(
                    f"Length mismatch for sample {i}: "
                    f"feature length={self.mask[i].sum().item()}, "
                    f"target length={length}."
                )
            valid_len = min(length, T_max)
            padded_targets[i, :valid_len] = target[:valid_len]

        # 3. Apply Masking (Flattening)
        # We select only the valid time steps for loss computation
        if self.mask is not None:
            preds_flat = preds.reshape(-1, preds.shape[-1])  # (B*T, C)
            targets_flat = padded_targets.reshape(-1)  # (B*T)
            mask_flat = self.mask.reshape(-1)  # (B*T)

            valid_preds = preds_flat[mask_flat].unsqueeze(0)  # (1, N_valid, C)
            valid_targets = targets_flat[mask_flat].unsqueeze(0)  # (1, N_valid)

            self.loss = loss_fn.compute_loss(valid_preds, valid_targets)
        else:
            self.loss = loss_fn.compute_loss(preds, padded_targets)

        self.status = "loss"
        return {"loss": self.loss}

    def print_summary(self):
        """
        Prints a summary of the batch contents and current status.
        """
        print("HeterogeneousBatch Summary:")
        print(f"  - Device: {self.device}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Status: {self.status}")
        print(f"  - Raw Audio Samples: {len(self.raw_audio)}")
        print(f"  - STFT Audio Samples: {len(self.stft_audio)}")
        print(f"  - Feature Samples: {len(self.features)}")
        print(f"  - Meta Keys: {list(self.meta.keys())}")
        if self.mask is not None:
            print(f"  - Mask Shape: {self.mask.shape}")
        if self.padded_features is not None:
            print(f"  - Padded Features Shape: {self.padded_features.shape}")
        if self.padded_estimates is not None:
            print(f"  - Padded Estimates Shape: {self.padded_estimates.shape}")
        if self.estimates is not None:
            print(f"  - Number of Estimate Samples: {len(self.estimates)}")
        print(50 * "-")
        if self.estimates is not None:
            HAs = self.compute_rtf_error()
            for bidx in range(self.batch_size):
                print(f"Sample {bidx}:")
                [
                    print(
                        f"RTFs: {list(est_rtf.shape)}\t SIDs: {est_sid[-1].tolist()}\t GT-SIDs: {gt_sid.tolist()}\t GT-RTFs: {list(gt_rtf.shape)}\t HA (mean, median, max): {ha}"
                    )
                    for est_rtf, est_sid, gt_rtf, gt_sid, ha in zip(
                        self.estimates[bidx][0],
                        self.estimates[bidx][1],
                        self.meta["gt_rtf_stream"][bidx],
                        self.meta["gt_ids_stream"][bidx],
                        HAs[bidx],
                    )
                ]
        else:
            for bidx in range(self.batch_size):
                [
                    print(f"GT-SIDs: {gt_sid.tolist()}\t GT-RTFs: {list(gt_rtf.shape)}")
                    for gt_rtf, gt_sid in zip(
                        self.meta["gt_rtf_stream"][bidx],
                        self.meta["gt_ids_stream"][bidx],
                    )
                ]

    def compute_rtf_error(self) -> list[list[tuple[float, float, float]]]:
        """
        Computes the mean, median, and max Hermitian angle between estimated and
        ground truth RTFs for each segment, accounting for source IDs.

        Returns:
            list[list[tuple[float, float, float]]]: A list (over samples) of lists (over segments)
            of tuples (mean_angle, median_angle, max_angle).
        """
        results = []
        if self.estimates is None:
            return results

        # Iterate over each sample in the batch
        for bidx in range(self.batch_size):
            sample_results = []

            # Check if we have data for this sample
            if (
                bidx >= len(self.estimates)
                or "gt_rtf_stream" not in self.meta
                or bidx >= len(self.meta["gt_rtf_stream"])
            ):
                results.append([])
                continue

            est_rtf_stream = self.estimates[bidx][0]
            est_sid_stream = self.estimates[bidx][1]
            gt_rtf_stream = self.meta["gt_rtf_stream"][bidx]
            gt_ids_stream = self.meta["gt_ids_stream"][bidx]

            # Iterate over segments within the sample
            for est_rtf, est_sid, gt_rtf, gt_sid in zip(
                est_rtf_stream, est_sid_stream, gt_rtf_stream, gt_ids_stream
            ):
                # Get IDs
                # est_sid is [T, K_est], we take the last time step to determine active IDs
                if isinstance(est_sid, torch.Tensor) and est_sid.ndim >= 2:
                    current_est_ids = est_sid[-1].tolist()
                else:
                    raise ValueError(
                        f"Unexpected shape for est_sid: {est_sid.shape if isinstance(est_sid, torch.Tensor) else 'N/A'}"
                    )
                # elif isinstance(est_sid, torch.Tensor):
                #     current_est_ids = est_sid.tolist()
                # elif isinstance(est_sid, (list, tuple)):
                #     current_est_ids = est_sid
                # else:
                #     current_est_ids = []

                if isinstance(gt_sid, torch.Tensor):
                    current_gt_ids = gt_sid.tolist()
                else:
                    raise ValueError(f"Unexpected type for gt_sid: {type(gt_sid)}")
                #
                # elif isinstance(gt_sid, (list, tuple)):
                #     current_gt_ids = gt_sid
                # else:
                #     current_gt_ids = []

                common_ids = set(current_est_ids) & set(current_gt_ids)

                segment_angles = []

                for uid in common_ids:
                    # Find indices
                    idx_est = current_est_ids.index(uid)
                    idx_gt = current_gt_ids.index(uid)

                    # Extract RTFs: [F, T, M]
                    vec_est = est_rtf[..., idx_est]
                    vec_gt = gt_rtf[..., idx_gt]

                    # Compute Hermitian angles
                    # hermitian_angle computes angle between vectors.
                    # Inputs are [F, T, M]. We want angle along dim M (last dim).
                    angles = hermitian_angle(vec_est, vec_gt, dim=-1)
                    segment_angles.append(angles.flatten())

                if segment_angles:
                    all_angles = torch.cat(segment_angles)

                    mean_val = all_angles.mean().item()
                    median_val = all_angles.median().item()
                    max_val = all_angles.max().item()

                    sample_results.append((mean_val, median_val, max_val))
                else:
                    sample_results.append((float("nan"), float("nan"), float("nan")))

            results.append(sample_results)

        return results


def ground_truth_rtf_stream(
    sad_frames: dict, rtfs: dict, segments: list[Segment]
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[int, str]]:
    """
    Computes the ground truth RTF stream and ID stream based on SAD frames.

    Args:
        sad_frames: A dictionary mapping source IDs to their SAD frame tensors.
        rtfs: A dictionary mapping source IDs to their RTF tensors.
        segments: A list of Segment objects defining the boundaries.

    Returns:
        gt_rtf_stream: A list of tensors (F, T_seg, M, K_seg) per segment.
        gt_ids_stream: A list of tensors (K_seg,) containing integer source IDs.
        id_map: A dictionary mapping integer IDs (0, 1...) to string source IDs.
    """
    gt_rtf_stream = []
    gt_ids_stream = []

    # Mappings for global ID tracking (Source String <-> Integer ID)
    str_to_int_id = {}
    next_global_id = 0

    # Helper to get device/dtype reference
    ref_tensor = next(iter(rtfs.values()))

    for seg in segments:
        start, end = seg.start, seg.end
        T_seg = end - start

        # Identify active sources in this segment
        active_source_ids = []

        # Use sorted keys so that the stacking order is deterministic.
        # This also ensures that if multiple sources appear for the first time
        # in the same segment, they are assigned IDs alphabetically.
        for source_id in sorted(sad_frames.keys()):
            if source_id == "noise":
                continue

            # Check if active in this window
            if sad_frames[source_id][start:end].sum() > 0:
                active_source_ids.append(source_id)

        K_seg = len(active_source_ids)

        # Determine Integer IDs for this segment.
        # Assign new IDs to first-time appearances.
        segment_int_ids = []
        for sid in active_source_ids:
            if sid not in str_to_int_id:
                str_to_int_id[sid] = next_global_id
                next_global_id += 1
            segment_int_ids.append(str_to_int_id[sid])

        if K_seg == 0:
            # No active sources: Shape (F, T_seg, M, 0)
            F = ref_tensor.shape[0]
            M = ref_tensor.shape[-2]  # Assuming (F, 1, M, 1)

            seg_rtfs = torch.zeros(
                (F, T_seg, M, 0), dtype=ref_tensor.dtype, device=ref_tensor.device
            )
            seg_ids = torch.tensor([], dtype=torch.long, device=ref_tensor.device)
        else:
            # Stack active RTFs
            stacked_rtfs_list = []
            for sid in active_source_ids:
                r = rtfs[sid]  # (F, 1, M, 1)

                # Tile over T_seg: (F, 1, M, 1) -> (F, T_seg, M, 1)
                r_expanded = r.expand(-1, T_seg, -1, -1)
                stacked_rtfs_list.append(r_expanded)

            # Concatenate along last dim -> (F, T_seg, M, K_seg)
            seg_rtfs = torch.cat(stacked_rtfs_list, dim=-1)

            # Create ID tensor -> (K_seg,)
            seg_ids = torch.tensor(
                segment_int_ids, dtype=torch.long, device=ref_tensor.device
            )

        gt_rtf_stream.append(seg_rtfs)
        gt_ids_stream.append(seg_ids)

    # Create output dictionary: Top-level Int -> Source String
    id_map = {v: k for k, v in str_to_int_id.items()}

    return gt_rtf_stream, gt_ids_stream, id_map


def gen_target_id_stream(
    source_ids: list[int], activity_tensor: torch.Tensor
) -> torch.Tensor:
    """Generates target ID stream for the entire utterance.
    The goal is always to extract the latest activated source.

    Args:
        source_ids (list[int]): List of source IDs.
        activity_tensor (torch.Tensor): Activity tensor of shape (K, T).

    Returns:
        torch.Tensor: Target ID stream of shape (T,).
    """
    target_id_stream = []
    priority_list = []  # List of source IDs in the order of activation
    diff = activity_tensor.float().mT.diff(dim=0)
    times, inds = torch.where(diff != 0)
    current_time = 0
    target_id_stream = -3 * torch.ones(activity_tensor.shape[-1], dtype=torch.long)
    for t, ind in zip(times, inds):
        if priority_list:
            target_id_stream[current_time : t + 1] = priority_list[-1]
        if diff[t, ind] == 1:
            # Source activated
            priority_list.append(source_ids[int(ind.item())])
        elif diff[t, ind] == -1:
            # Source deactivated
            priority_list.remove(source_ids[int(ind.item())])
        else:
            raise ValueError("Unexpected value in activity difference tensor.")
        current_time = t + 1
    if priority_list:
        target_id_stream[current_time:] = priority_list[-1]
    return target_id_stream


def activity_dict2tensor(
    activity_dict: dict[str, torch.Tensor], id_map: dict[int, str]
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts activity dictionary to tensor.

    Args:
        activity_dict (dict[str, torch.Tensor]): Activity dictionary.
        id_map (dict[int, str]): ID mapping dictionary.

    Returns:
        tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]: Source IDs and activity tensor.
    """
    sidm_str2int = {v: k for k, v in id_map.items()}
    sidm_str2int["noise"] = -2  # noise source id
    sids_and_act = [(sidm_str2int[k], v) for k, v in activity_dict.items()]
    source_ids = [src[0] for src in sids_and_act]
    activity_tensor = torch.stack([src[1] for src in sids_and_act], dim=0)  # (K, T)
    seg_borders = torch.cat(
        [
            torch.tensor([0], device=activity_tensor.device),
            torch.where(activity_tensor.diff().mT)[0] + 1,
            torch.tensor([activity_tensor.shape[-1]], device=activity_tensor.device),
        ]
    )
    return (
        source_ids,
        activity_tensor,
        gen_target_id_stream(source_ids, activity_tensor),
        seg_borders,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("--- Testing MicrophoneArray Class ---")

    # --- Test Case Configuration ---
    GEOMETRY = "gen2D"
    DISTRIBUTION = "random"
    NUM_MICS = 4
    MAX_DISTANCE = 0.1  # 10 cm diameter
    MIN_DISTANCE = 0.01  # 1 cm minimum distance between mics
    RESTRICT_ROT_2_XY_PLANE = True  # <-- Set this to True to test the new feature
    ROOM_DIMS = [5.0, 4.0, 3.0]
    MIN_DIST_WALLS = 0.5

    print(f"\n1. Testing: geometry='{GEOMETRY}', distribution='{DISTRIBUTION}'")
    print(
        f"   Params: num_mics={NUM_MICS}, max_dist={MAX_DISTANCE}, min_dist={MIN_DISTANCE}"
    )
    print(f"   Rotation restricted to 2D: {RESTRICT_ROT_2_XY_PLANE}")
    try:
        # 1. Create the array blueprint
        mic_array = MicrophoneArray(
            num_mics=NUM_MICS,
            geometry=GEOMETRY,
            distribution=DISTRIBUTION,
            max_distance=MAX_DISTANCE,
            min_distance=MIN_DISTANCE,
        )
        print(
            f"Successfully created blueprint. Shape: {mic_array.local_locations.shape}"
        )
        # print(f"  - Distance Matrix:\n{mic_array.distance_matrix}\n")
        print(f"  - Max Distance: {np.max(pdist(mic_array.local_locations.T)):.3f}")
        print(f"  - Min Distance: {np.min(pdist(mic_array.local_locations.T)):.3f}")

        # 2. Plot the local blueprint
        mic_array.plot_local_locations()

        # 3. Place the array in the room and plot the result
        placed_locations = mic_array.place(
            room_dims=ROOM_DIMS,
            min_dist_from_walls=MIN_DIST_WALLS,
            plot_filepath="Playground/MA_in_room.png",  # This triggers the global plot
            restrict_rot_2_xy_plane=RESTRICT_ROT_2_XY_PLANE,
        )
        # print(
        #     f"\nPlaced array in a {ROOM_DIMS} room at global locations: {placed_locations}."
        # )
        # print("Plots have been saved to the 'Playground' directory.")

    except Exception as e:
        import traceback

        print(f"An error occurred: {e}")
        traceback.print_exc()

    # # --- Test Case 2: Initialization from Custom Locations ---
    # print("\n2. Testing: Initialization from custom locations")
    # try:
    #     # Define a custom array, e.g., a T-shape
    #     custom_locs = np.array(
    #         [
    #             [-0.2, 0.0, 1.0],  # Bar of the T
    #             [-0.1, 0.0, 0.0],
    #             [0.0, 0.0, 0.0],
    #             [0.1, 0.0, 0.0],
    #             [0.2, 0.0, 0.0],
    #             [0.0, 0.1, 0.0],  # Stem of the T
    #             [0.0, 0.2, 0.0],
    #         ]
    #     ).T  # Transpose to get shape (3, 7)

    #     mic_array_custom = MicrophoneArray.from_locations(custom_locs)

    #     print(f"Successfully created array from custom locations.")
    #     print(f"  - Inferred num_mics: {mic_array_custom.num_mics}")
    #     print(f"  - Inferred max_distance: {mic_array_custom.max_distance:.3f}")
    #     print(f"  - Inferred min_distance: {mic_array_custom.min_distance:.3f}")
    #     print(f"  - Inferred dimensionality: {mic_array_custom.dimensionality}")

    #     mic_array_custom.plot_local_locations(filepath="Playground/MA_custom.png")

    #     # Place the custom array in the room
    #     placed_locations = mic_array_custom.place(
    #         room_dims=ROOM_DIMS,
    #         min_dist_from_walls=MIN_DIST_WALLS,
    #         plot_filepath="Playground/MA_custom_in_room.png",
    #         restrict_rot_2_xy_plane=True,
    #     )
    #     print("Successfully placed and plotted the custom array.")

    # except Exception as e:
    #     import traceback

    #     print(f"An error occurred during custom initialization: {e}")
    #     traceback.print_exc()

    # # --- Test Case 3: Unsupported Combination ---
    # print("\n3. Testing: Unsupported combination")
    # try:
    #     invalid_array = MicrophoneArray(
    #         num_mics=5,
    #         geometry="triangle",
    #         distribution="regular",
    #         max_distance=0.1,
    #     )
    # except NotImplementedError as e:
    #     print(f"Successfully caught expected error: {e}")
