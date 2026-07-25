from muse_toolbox.utils import sample_parameter
import torch
import random
import warnings
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_activation_pattern(
    source_ids: list[str],
    max_sources: int = 4,
    total_duration: float = 60.0,  # [s]
    initial_noise_only_duration: float | list[float] | tuple[float, float] = (
        2.0,
        5.0,
    ),  # [s]
    time_between_events: float | list[float] | tuple[float, float] = (
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



