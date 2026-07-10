import lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
import shutil
from collections.abc import Sized
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from building_blocks.feature_extractors.base_feature import BaseFeatureExtractor
from muse_toolbox.utils import (
    STFTtransform,
    convolve_clean2microphone,
    vad_opt_fast_gen,
    normalize_components,
    db2amp,
    sample_parameter,
    load_audio,
    HeterogeneousBatch,
    convolve_white2microphone,
    covariance_SCM,
    peigvech,
    Segment,
    identify_segments,
    rir2rtf,
    ground_truth_rtf_stream,
    move2device,
    print_gpu_tensors,
)


@dataclass
class ScenarioGenerationConfig:
    """Configuration for the generic scenario generation process."""

    max_sources: int | list[int] | tuple[int, int]
    signal_lengths: float | list[float] | tuple[float, float]
    initial_noise_only_duration: float | list[float] | tuple[float, float]
    snrs: float | list[float] | tuple[float, float]
    source_power_range: float
    time_between_events: float | list[float] | tuple[float, float]
    fix_seglen: bool
    activations_only: bool
    remove_silence: bool
    neglect_silence4oracle_sa: float
    bridge_clean_speech_gaps: float
    vad_threshold2define_oracle: float
    vad_threshold2select_clean_speech: float


class BaseDB(ABC):
    """
    The fundamental abstract base class for all database managers.

    It enforces that every database manager must have a root directory and
    a `prepare_data` method, which handles one-time setup like downloading
    or pre-processing.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    @abstractmethod
    def prepare_data(self):
        """
        Performs all one-time setup for the database. This can include
        downloading, unzipping, pre-processing, or creating cache files.
        This method should be idempotent (safe to run multiple times).
        """
        pass


class BaseSourceDB(BaseDB):
    """
    Abstract base class for databases that provide clean source signals (e.g., speech).
    """

    def __init__(
        self,
        root_dir: str | Path,
        signal_lengths: float,
    ):
        super().__init__(root_dir)
        self.signal_lengths = signal_lengths

    @abstractmethod
    def get_speaker_clips(self, split: str) -> dict[str, list]:
        """
        Returns a dictionary of speaker clips for a given split ('train', 'val', 'test').

        Returns:
            dict[str, list]: A dictionary where keys are unique speaker IDs and
                             values are lists of clips. A 'clip' can be a list
                             of file paths or any other identifier.
        """
        pass


class BaseRIRsDB(BaseDB):
    """
    Abstract base class for databases that provide Room Impulse Responses (RIRs).
    """

    @property
    @abstractmethod
    def reverb_conditions(self) -> list[str] | list[float] | tuple[float, ...]:
        """
        Should return a list of all available reverb condition identifiers (e.g., ['low', 'medium', 'high']).
        """
        pass

    @abstractmethod
    def get_rir(self, **kwargs: Any) -> torch.Tensor:
        """
        Returns the RIR tensor data based on given criteria.

        The specific keyword arguments will depend on the concrete implementation
        (e.g., reverb_condition, doa, microphone_array).
        """
        pass


class BaseNoiseDB(BaseDB):
    """
    Abstract base class for databases that provide noise signals.
    """

    @abstractmethod
    def get_noise(self, **kwargs: Any) -> torch.Tensor:
        """
        Returns the noise tensor data based on given criteria.

        The specific keyword arguments will depend on the concrete implementation
        (e.g., reverb_condition, noise_type, microphone_array).
        """
        pass


class BaseScenarioGenerator(Dataset, ABC):
    """
    A generic base class for creating dynamic, multi-source audio scenarios.

    This class encapsulates the project's core methodology for generating scenarios:
    1. A dynamic activation pattern where sources turn on and off.
    2. Rendering reverberant sources and placing them on a timeline.
    3. Mixing sources with noise at a specified SNR.
    4. Generating a frame-wise speaker count as the ground truth.

    It uses a template method pattern, requiring subclasses to implement a few
    hooks to provide dataset-specific ingredients.
    """

    def __init__(
        self,
        id: str,
        num_scenarios: int,
        transform: STFTtransform,
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        source_dbs: dict[str, BaseSourceDB],
        rir_db: BaseRIRsDB | None,
        noise_db: BaseNoiseDB | None,
        seed: int | None,
        acc_device: torch.device,
    ):
        # Store all generic parameters
        self.id = id
        self.num_scenarios = num_scenarios
        self.transform = transform
        self.sampling_frequency = sampling_frequency
        self.config = generation_config
        self.source_dbs = source_dbs
        self.rir_db = rir_db
        self.noise_db = noise_db
        self.seed = seed
        self.acc_device = acc_device
        self.max_possible_sources = (
            max(self.config.max_sources)
            if isinstance(self.config.max_sources, (list, tuple))
            else self.config.max_sources
        )
        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)

        # Get all clips from all source DBs
        split_name = self._get_split_name()
        self.all_clips = defaultdict(list)
        for db_name, db_manager in self.source_dbs.items():
            clips = db_manager.get_speaker_clips(split_name)
            for speaker_id, speaker_clips in clips.items():
                self.all_clips[f"{db_name}_{speaker_id}"].extend(speaker_clips)

        # The main construction call
        self.scenarios = self._construct_dataset()

    # --- Abstract Hooks for Subclasses ---

    @abstractmethod
    def _sample_scenario_parameters(self) -> dict:
        """
        Sample all dataset-specific parameters for a single scenario.
        Must return a dict with at least 'num_sources' and 'signal_length'.
        e.g., {'num_sources': 3, 'signal_length': 60.0, 'snr': 10, 'reverb_condition': 'low', ...}
        """
        pass

    @abstractmethod
    def _get_sources_for_scenario(self, scenario_params: dict) -> list[dict]:
        """
        Select clean speech clips and assign source properties.
        Must return a list of source dictionaries. Each dict must have an 'id' key.
        e.g., [{'id': 'spk1_doa30', 'doa': 30, 'clean_speech_paths': [...]}, ...]
        """
        pass

    def _get_split_name(self) -> str:
        """Helper to determine the current split ('train', 'val', 'test') from the ID."""
        if "train" in self.id:
            return "train"
        elif "val" in self.id:
            return "val"
        elif "test" in self.id:
            return "test"
        else:
            raise ValueError(f"Unknown split in id: {self.id}")

    def _construct_dataset(self) -> list[dict]:
        """Constructs the list of scenario configurations."""
        scenarios = []
        for _ in tqdm(
            range(self.num_scenarios),
            desc=f"Constructing {self.num_scenarios} scenarios for {self.id}",
        ):
            # 1. Sample dataset-specific parameters (e.g., SNR, reverb type)
            scenario_params = self._sample_scenario_parameters()
            num_sources = scenario_params["num_sources"]

            # 2. Get the specific sources for this scenario (e.g., speech files, DOAs)
            sources = self._get_sources_for_scenario(scenario_params)
            if len(sources) < num_sources:
                # Handle cases where not enough unique sources could be found
                num_sources = len(sources)
                scenario_params["num_sources"] = num_sources

            # 3. Generate the dynamic activation pattern (generic logic)
            activity_pattern, signal_length = self._generate_activation_pattern(
                sources=sources,
                max_sources=num_sources,
                scenario_params=scenario_params,
            )
            if self.config.activations_only:
                scenario_params["signal_length"] = signal_length
            scenario_params["activity_pattern"] = activity_pattern
            scenario_params["sources"] = sources
            scenario_params["transform"] = self.transform
            scenario_params["generator_id"] = self.id
            scenarios.append(scenario_params)
        return scenarios

    def _generate_activation_pattern(
        self, sources: list[dict], max_sources: int, scenario_params: dict
    ) -> tuple[list[dict], float]:
        # Logic from old _generate_activation_pattern
        event_log = []
        available_sources = list(sources)
        active_sources = []

        if max_sources > len(sources):
            max_sources = len(sources)

        current_time = sample_parameter(self.config.initial_noise_only_duration)

        max_event_time = (
            scenario_params["signal_length"]
            - torch.tensor(self.config.time_between_events).min().item()
        )

        while current_time < max_event_time:
            possible_actions = []
            if len(active_sources) < max_sources and available_sources:
                possible_actions.append("activation")
            if active_sources and not self.config.activations_only:
                possible_actions.append("deactivation")
            if not possible_actions:
                break

            action = random.choice(possible_actions)

            if action == "activation":
                event_type = 1  # activation
                source_for_event = random.choice(available_sources)
                active_sources.append(source_for_event)
                available_sources.remove(source_for_event)
            else:  # deactivation
                event_type = -1  # deactivation
                source_for_event = random.choice(active_sources)
                available_sources.append(source_for_event)
                active_sources.remove(source_for_event)

            event_log.append(
                {
                    "time": current_time,
                    "type": event_type,
                    "source": source_for_event,
                    "source_id": source_for_event["id"],
                    "num_sources": len(active_sources),
                    "active_sources": active_sources.copy(),
                }
            )
            if self.config.fix_seglen:
                current_time += scenario_params["fixed_time_between_events"]
            else:
                current_time += sample_parameter(self.config.time_between_events)
        return event_log, current_time

    def _convert_activity_pattern2samples(
        self, activity_pattern: list[dict], signal_length: float
    ) -> dict[str, list[torch.Tensor]]:
        # Logic from old _convert_activity_pattern2samples
        event_samples_by_speaker = defaultdict(list)
        for event in activity_pattern:
            source_id = event["source_id"]
            time = event["time"]
            sample_idx = self.transform.times2samples(time)
            event_samples_by_speaker[source_id].append(sample_idx)

        for source_id in event_samples_by_speaker:
            event_samples_by_speaker[source_id].sort()
            if len(event_samples_by_speaker[source_id]) % 2 != 0:
                end_sample = self.transform.times2samples(signal_length)
                event_samples_by_speaker[source_id].append(end_sample)
        return dict(event_samples_by_speaker)

    def _calculate_samples_per_source(
        self, event_samples_by_speaker: dict[str, list[torch.Tensor]]
    ) -> dict[str, int]:
        # Logic from old _calculate_samples_per_source
        samples_per_source = {}
        for source_id, event_samples in event_samples_by_speaker.items():
            total_active_samples = 0
            for i in range(0, len(event_samples), 2):
                if i + 1 < len(event_samples):
                    activation_sample = event_samples[i]
                    deactivation_sample = event_samples[i + 1]
                    duration = deactivation_sample - activation_sample
                    total_active_samples += duration.item()
            samples_per_source[source_id] = int(total_active_samples)
        return samples_per_source

    def _get_frame_wise_source_count(
        self, source_activity: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Logic from old _get_frame_wise_source_count
        if not source_activity:
            return torch.tensor([], dtype=torch.long)
        total_samples = next(iter(source_activity.values())).shape[0]
        source_vads = [vad for key, vad in source_activity.items() if key != "noise"]

        if not source_vads:
            num_frames = int(self.transform.samples2frames(total_samples).item())
            return torch.zeros(num_frames, dtype=torch.long)

        sample_wise_count = torch.stack(source_vads).long().sum(dim=0)
        framed_counts = sample_wise_count.unfold(
            dimension=0, size=self.transform.nfft, step=self.transform.hop_length
        )

        pad_size = self.transform.nfft // (2 * self.transform.hop_length)
        framed_counts = torch.nn.functional.pad(
            framed_counts.T, (pad_size, pad_size), "replicate"
        ).T

        frame_wise_count = torch.mode(framed_counts, dim=1).values
        return frame_wise_count

    def _load_source_materials(self, scenario_params: dict):
        # This method is already mostly correct from the plan. We just need to fill in the noise trimming.
        activity_samples = self._convert_activity_pattern2samples(
            scenario_params["activity_pattern"], scenario_params["signal_length"]
        )
        required_samples = self._calculate_samples_per_source(activity_samples)

        clean_speeches = {}
        for source_info in scenario_params["sources"]:
            source_id = source_info["id"]
            if source_id in required_samples and required_samples[source_id] > 0:
                clean_speeches[source_id] = self._load_clean_speech_for_source(
                    source_info, required_samples[source_id]
                )

        rirs = {}
        if self.rir_db:
            for source_info in scenario_params["sources"]:
                # Pass both scenario-wide and source-specific params to get_rir
                rirs[source_info["id"]] = self.rir_db.get_rir(
                    **scenario_params, **source_info
                ).to(self.acc_device)

        noise = None
        if self.noise_db:
            raw_noise = self.noise_db.get_noise(**scenario_params).to(self.acc_device)
            # Logic from old _load_noise
            target_samples = int(
                scenario_params["signal_length"] * self.sampling_frequency
            )
            if raw_noise.shape[1] >= target_samples:
                start_sample = random.randint(0, raw_noise.shape[1] - target_samples)
                noise = raw_noise[:, start_sample : start_sample + target_samples]
            else:
                repeats = (target_samples // raw_noise.shape[1]) + 1
                noise = raw_noise.repeat(1, repeats)[:, :target_samples]
            del raw_noise  # free memory

        torch.cuda.empty_cache()

        return clean_speeches, rirs, noise, activity_samples

    def _render_and_place_sources(
        self,
        # --- UPDATED: Change signature to accept long speeches ---
        clean_speeches_long: dict,
        # ---
        rirs: dict,
        activity_samples: dict,
        scenario_params: dict,
    ) -> dict[str, torch.Tensor]:
        total_samples = int(scenario_params["signal_length"] * self.sampling_frequency)
        if not rirs:
            return {}
        num_channels = next(iter(rirs.values())).shape[0]
        full_source_signals = {}

        sliced_speeches = defaultdict(list)
        for source_id, full_speech_signal in clean_speeches_long.items():
            if source_id not in activity_samples:
                continue
            event_samples = activity_samples[source_id]
            samples_processed = 0
            for i in range(0, len(event_samples), 2):
                if i + 1 < len(event_samples):
                    segment_length = (event_samples[i + 1] - event_samples[i]).item()
                    segment = full_speech_signal[
                        samples_processed : samples_processed + segment_length
                    ]
                    sliced_speeches[source_id].append(segment)
                    samples_processed += segment_length
        sliced_speeches = dict(sliced_speeches)

        for source_id, speech_segments in sliced_speeches.items():
            if source_id not in rirs:
                continue

            convolved_segments = [
                convolve_clean2microphone(cs_seg.unsqueeze(0), rirs[source_id])
                for cs_seg in speech_segments
            ]

            full_source_signal = torch.zeros(
                num_channels, total_samples, device=self.acc_device
            )
            event_samples = activity_samples[source_id]
            for i, segment in enumerate(convolved_segments):
                start_sample = event_samples[i * 2].item()
                segment_length = segment.shape[1]
                end_sample = min(start_sample + segment_length, total_samples)
                segment_length_to_copy = end_sample - start_sample
                full_source_signal[:, start_sample:end_sample] = segment[
                    :, :segment_length_to_copy
                ]

            full_source_signals[source_id] = full_source_signal
        full_speech_signal = None  # free memory
        sliced_speeches = None  # free memory
        convolved_segments = None  # free memory
        segments = None  # free memory
        full_source_signal = None  # free memory

        return full_source_signals

    def _mix_and_finalize_scenario(
        self,
        full_source_signals: dict,
        noise: torch.Tensor | None,
        scenario_params: dict,
    ):
        # Logic from old _mix_and_finalize_scenario
        signal_components = {}
        source_activity_vad = {}
        total_samples = int(scenario_params["signal_length"] * self.sampling_frequency)

        source_idx = 0
        for source_id, signal in full_source_signals.items():
            vad = vad_opt_fast_gen(
                signal,
                self.sampling_frequency,
                thr=self.config.vad_threshold2define_oracle,
                min_on=self.config.neglect_silence4oracle_sa,
                mode="highpass",
                cutoff_freq=80.0,
            ).squeeze()
            source_activity_vad[source_id] = vad
            norm_source = normalize_components(signal, vad=vad.unsqueeze(0))[0]
            signal_components[source_id] = (
                db2amp(torch.tensor(scenario_params["sirs"][source_idx])) * norm_source
            )
            source_idx += 1

        if noise is not None:
            noise_vad = torch.ones(total_samples, dtype=torch.bool, device=noise.device)
            source_activity_vad["noise"] = noise_vad
            norm_noise = normalize_components(noise, vad=noise_vad.unsqueeze(0))[0]
            signal_components["noise"] = (
                db2amp(torch.tensor(-scenario_params["snr"])) * norm_noise
            )

        if signal_components:
            stacked_sources = torch.stack(
                [s for s_id, s in signal_components.items() if s_id != "noise"]
            )
            noisy_signal = torch.sum(stacked_sources, dim=0) + signal_components.get(
                "noise", 0
            )
        elif noise is not None:
            noisy_signal = signal_components.get("noise")
        else:
            # This is the corrected fallback block
            # It now correctly determines the number of channels without referencing 'rirs'.
            # The number of channels is a property of the microphone array used for the scenario.
            # We can infer it from the noise tensor if it exists, or from the source signals.
            # If neither exists, we must make a safe assumption.
            print(
                f"Warning: Scenario {scenario_params.get('id', '')} has no active sources and no noise. Generating 2-channel (stereo) silence."
            )
            raise RuntimeError(
                f"Scenario {scenario_params.get('id', '')} has no active sources and no noise. Cannot generate a silent signal."
            )

        # # TODO: For J1 i change the oracle source activity definition to:
        # source_activity_vad = {
        #     k: (v != 0).all(dim=0) for k, v in signal_components.items()
        # }

        source_count = self._get_frame_wise_source_count(source_activity_vad)
        return noisy_signal, source_count, source_activity_vad, signal_components

    def _compute_oracle_rtfs(
        self, rirs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Computes oracle RTFs for all sources in the provided RIR dictionary.

        Args:
            rirs: Dictionary mapping source IDs to RIR tensors.

        Returns:
            Dictionary mapping source IDs to their corresponding RTF tensors.
        """
        rtfs = {}
        for source_id, rir in rirs.items():
            # # Convolve with white noise to create a wideband signal
            # # Use a sufficient length (e.g., 100s) to get a good estimate
            # directwhite = convolve_white2microphone(rir, samples=1600000)

            # # Transform to STFT domain
            # stftsig = self.transform.encode(directwhite)

            # # Calculate Sample Covariance Matrix
            # # Shape: (F, C, C)
            # covMat = covariance_SCM(stftsig)[..., None, :, :]

            # # Extract Principal Eigenvector (RTF)
            # # Shape: (F, C)
            # test = peigvech(covMat)

            rtfs[source_id] = rir2rtf(rir.to(self.acc_device), self.transform).cpu()

        return rtfs

    def _load_clean_speech_for_source(
        self, source_info: dict, required_samples: int
    ) -> torch.Tensor:
        """Loads and concatenates audio files for a single source."""
        concatenated_audio = []
        current_length = 0
        all_paths = source_info["clean_speech_paths"]

        resampler = torchaudio.transforms.Resample(
            orig_freq=self.sampling_frequency, new_freq=self.sampling_frequency
        )

        while current_length < required_samples:
            for file_path in all_paths:
                if current_length >= required_samples:
                    break
                audio_tensor, sr = load_audio(file_path)
                audio_tensor = audio_tensor.to(self.acc_device)
                sr = int(sr)
                if sr != self.sampling_frequency:
                    resampler.__init__(orig_freq=sr, new_freq=self.sampling_frequency)
                    audio_tensor = resampler(audio_tensor)

                if self.config.remove_silence and audio_tensor.numel() > 0:
                    vad_mask = vad_opt_fast_gen(
                        audio_tensor.unsqueeze(0),
                        self.sampling_frequency,
                        thr=self.config.vad_threshold2select_clean_speech,
                        min_on=self.config.bridge_clean_speech_gaps,
                        mode="highpass",
                        cutoff_freq=80.0,
                    ).squeeze()
                    audio_tensor = audio_tensor[:, vad_mask]

                if audio_tensor.numel() > 0:
                    concatenated_audio.append(audio_tensor.squeeze(0))
                    # try to remove audio_tensor from gpu memory. Dont know if this helps
                    current_length += audio_tensor.shape[-1]
                    audio_tensor = None

        if not concatenated_audio:
            # Handle case where all clips were silent
            return torch.zeros(required_samples)

        full_speech_signal = torch.cat(concatenated_audio)
        concatenated_audio = None  # free memory
        return full_speech_signal[:required_samples]

    # --- Standard Dataset Methods ---

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> dict:
        # This method is now a cleaner, high-level orchestrator.
        scenario_params = self.scenarios[idx]

        clean_speeches_long, rirs, noise, activity_samples = (
            self._load_source_materials(scenario_params)
        )

        full_source_signals = self._render_and_place_sources(
            clean_speeches_long, rirs, activity_samples, scenario_params
        )

        noisy_signal, source_count, sad_samples, signal_components = (
            self._mix_and_finalize_scenario(full_source_signals, noise, scenario_params)
        )

        rtfs = self._compute_oracle_rtfs(rirs)

        sad_frames = {
            k: self.transform.samples2frames_quantity(v) for k, v in sad_samples.items()
        }

        segments = identify_segments(source_count)

        # gt_rtf_stream, gt_ids_stream, id_map = ground_truth_rtf_stream(
        #     sad_frames, rtfs, segments
        # )

        meta_data = {
            "scenario_params": scenario_params,
            "source_count": source_count,
            "scenario_id": f"{self.id}_{idx}",
            "references": signal_components,
            "rtfs": rtfs,
            "sad_samples": sad_samples,
            "sad_frames": sad_frames,
            "segments": segments,
            # "gt_rtf_stream": gt_rtf_stream,
            # "gt_ids_stream": gt_ids_stream,
            # "id_map": id_map,
        }
        item = move2device(
            {"input": noisy_signal, "meta": meta_data}, torch.device("cpu")
        )
        del noisy_signal  # free memory
        # free memory:
        del rirs
        del clean_speeches_long
        del activity_samples
        del noise
        del full_source_signals
        del signal_components
        del sad_frames
        del source_count
        del rtfs
        del sad_samples
        del segments
        del meta_data
        del scenario_params

        torch.cuda.empty_cache()

        return item

    def collate_fn(self, batch: list[dict]) -> dict:
        """
        Collates items from the generator into a batch for saving.
        Since batch_size is 1 during generation, this primarily ensures
        the output format is consistent.
        """
        if not batch:
            return {}

        # 'input' is collected into a list.
        inputs = [item["input"] for item in batch]

        # 'meta' is collected into a dictionary of lists.
        collated_meta = {}
        if "meta" in batch[0]:
            for key in batch[0]["meta"].keys():
                collated_meta[key] = [d["meta"][key] for d in batch]

        return {"input": inputs, "meta": collated_meta}


class PrecomputedDataset(Dataset):
    """
    A generic, universal dataset for loading pre-computed .pt files from a
    standardized directory structure. It is designed to handle heterogeneous
    data by collating inputs into a list.
    """

    def __init__(
        self,
        precomputed_dir: str | Path | list[str | Path],
        preload_to_ram: bool = False,
    ):
        # Handle both single path and list of paths
        if isinstance(precomputed_dir, (str, Path)):
            self.precomputed_dirs = [Path(precomputed_dir)]
        elif isinstance(precomputed_dir, list):
            self.precomputed_dirs = [Path(p) for p in precomputed_dir]
        else:
            raise TypeError(
                "precomputed_dir must be a string, Path, or list of strings/Paths."
            )

        self.files = []
        for d in self.precomputed_dirs:
            if not d.exists():
                raise FileNotFoundError(
                    f"Precomputed directory not found: {d}. "
                    "Run the DataModule's prepare_data step first."
                )
            self.files.extend(d.glob("scenario_*.pt"))

        # Sorting is crucial! It ensures that index 'i' always maps to the same file,
        # regardless of the OS or file system order. This is required for reproducibility.
        self.files = sorted(self.files)

        # # TODO: Remove this since it is only for debugging DEBUG:
        # # alternate files form directories. The sorted structure is one directory after another. Now it sould be that neighbouring files are not from the same directory
        # if len(self.precomputed_dirs) > 1:
        #     interleaved_files = []
        #     num_dirs = len(self.precomputed_dirs)
        #     files_per_dir = len(self.files) // num_dirs
        #     for i in range(files_per_dir):
        #         for d in range(num_dirs):
        #             interleaved_files.append(self.files[d * files_per_dir + i])
        #     # Append any remaining files (if total number of files is not perfectly divisible)
        #     remaining_files = len(self.files) % num_dirs
        #     for r in range(remaining_files):
        #         interleaved_files.append(self.files[files_per_dir * num_dirs + r])
        #     self.files = interleaved_files

        if not self.files:
            raise FileNotFoundError(
                f"No 'scenario_*.pt' files found in {self.precomputed_dirs}."
            )

        print(
            f"Initialized dataset with {len(self.files)} files from {len(self.precomputed_dirs)} directories."
        )

        self.data_in_ram = []
        if preload_to_ram:
            print(f"Pre-loading {len(self.files)} files into RAM...")
            # Since self.files is sorted, data_in_ram will be in the exact same order.
            for f in tqdm(self.files, desc="Pre-loading data"):
                self.data_in_ram.append(self._load_scenario(f))
            print("...pre-loading complete.")

    def _load_scenario(self, file_path: Path) -> dict:
        scenario = torch.load(file_path, weights_only=False)
        if scenario["input_type"] == "raw_audio":
            scenario["raw_audio"] = torch.stack(
                [ref for _, ref in scenario["meta"]["references"].items()]
            ).sum(dim=0)
        elif scenario["input_type"] in ["stft", "features"]:
            meta_filename = (
                file_path.parent.parent.parent.parent
                / file_path.parent.name
                / file_path.name
            )
            scenario["meta"] = torch.load(meta_filename, weights_only=False)["meta"]
        return scenario

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        # Since both self.files and self.data_in_ram (if populated) are sorted
        # identically, idx always refers to the same scenario.
        if self.data_in_ram:
            return self.data_in_ram[idx]
        else:
            return self._load_scenario(self.files[idx])

    def collate_fn(self, batch: list[dict]) -> HeterogeneousBatch:
        """
        Collates items into a HeterogeneousBatch object.
        Crucially, it sorts items so that 'raw_audio' comes first, then 'features'.
        This aligns with the processing order in HeterogeneousBatch.unpack_and_process.
        """
        if not batch:
            # Return empty batch or raise error
            return HeterogeneousBatch([], [], [], {})

        # 1. Separate items by type
        raw_items = [x for x in batch if x.get("input_type") == "raw_audio"]
        stft_items = [x for x in batch if x.get("input_type") == "stft"]
        feat_items = [x for x in batch if x.get("input_type") == "features"]

        # 2. Create the sorted list for metadata extraction
        sorted_batch = raw_items + stft_items + feat_items

        # 3. Extract Data
        raw_audio_list = [x["raw_audio"] for x in raw_items]
        stft_audio_list = [x["stft"] for x in stft_items]
        features_list = [x["features"] for x in feat_items]

        # 4a. Extract STFT Info (from the first STFT item, or the first feature item if any)
        stft_info = None
        if stft_items:
            stft_info = stft_items[0].get("stft_info")
        # Optional: Validate consistency here if desired
        if stft_info is not None:
            for idx, item in enumerate(stft_items[1:], start=1):
                current_info = item.get("stft_info")
                if current_info != stft_info:
                    raise ValueError(
                        f"Inconsistent stft_info in batch. "
                        f"Item 0 has {stft_info}, but item {idx} has {current_info}."
                    )

        # 4b. Extract Feature Info (from the first feature item, if any)
        feature_info = None
        if feat_items:
            feature_info = feat_items[0].get("feature_info")
            # Optional: Validate consistency here if desired
            for idx, item in enumerate(feat_items[1:], start=1):
                current_info = item.get("feature_info")
                if current_info != feature_info:
                    raise ValueError(
                        f"Inconsistent feature_info in batch. "
                        f"Item 0 has {feature_info}, but item {idx} has {current_info}."
                    )

        # 5. Collate Metadata (using the sorted order)
        collated_meta = defaultdict(list)
        if sorted_batch:
            # Use keys from the first item
            keys = sorted_batch[0]["meta"].keys()
            for item in sorted_batch:
                for k in keys:
                    collated_meta[k].append(item["meta"][k])

        return HeterogeneousBatch(
            raw_audio=raw_audio_list,
            stft_audio=stft_audio_list,
            features=features_list,
            meta=dict(collated_meta),
            stft_info=stft_info,
            feature_info=feature_info,
        )


class BaseDataModule(pl.LightningDataModule, ABC):
    """
    Abstract base class for all data modules in this project.

    It defines the common structure and provides generic dataloader methods
    to reduce boilerplate code in specific implementations. Subclasses are
    expected to define `self.train_ds`, `self.val_ds`, and `self.test_ds`
    in their `setup()` method.
    """

    def __init__(
        self,
        project_root: str | Path,
        id: str,
        transform: STFTtransform,
        batch_size: int,
        num_workers: int,
        num_scenarios: list[int],
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        seed: int | None = None,
        reset: bool = False,
        acc_device: torch.device = torch.device("cpu"),
        feature_extractor: BaseFeatureExtractor | None = None,
        force_load_stft: bool = False,
    ):
        """
        Initializes the BaseDataModule.
        Args:
            project_root (str | Path): The root directory of the project.
            batch_size (int): The batch size for the dataloaders.
            num_workers (int): The number of worker processes for data loading.
            feature_extractor (BaseFeatureExtractor | None): Optional feature extractor.
            force_load_stft (bool): If True, forces loading of STFT (or raw) data even if features are precomputed.
        """
        super().__init__()
        self.project_root = Path(project_root)
        self.id = id
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_scenarios = num_scenarios
        self.sampling_frequency = sampling_frequency
        self.generation_config = generation_config
        self.seed = seed
        self.reset = reset
        self.feature_extractor = feature_extractor
        self.force_load_stft = force_load_stft
        self.acc_device = acc_device
        self.data_is_prepared = False

    # def _get_data_root(self) -> Path:
    #     """
    #     Implements the abstract method from BaseDataModule.
    #     Constructs and returns the specific root path for the data based
    #     on the current configuration.
    #     """
    #     # Define the generic base path for all precomputed data
    #     base_path = self.project_root / "databases" / "precomputed" / self.id
    #     feature_path = base_path / "features" / self.feature_config_id

    #     # Check if feature directory exists
    #     if self.feature_config_id and feature_path.exists():
    #         return feature_path

    #     # Fallback to base path (scenarios)
    #     if base_path.exists():
    #         return base_path

    #     raise FileNotFoundError(
    #         f"Precomputed directory not found. Checked:\n"
    #         f"  - Features: {feature_path}\n"
    #         f" - Scenarios: {base_path}\n"
    #         "Please run the prepare_data step first."
    #     )

    # --- Abstract methods for subclasses to implement ---
    @abstractmethod
    def _get_database_managers(self) -> list:
        """Provides all database manager objects (e.g., for speech, RIRs, noise)."""
        pass

    @abstractmethod
    def _get_scenario_generator(self, split: str) -> Dataset | None:
        """Provides the configured scenario generator for a given split."""
        pass

    # --- Main Orchestration Method ---
    def prepare_data(self):
        """
        Generic data preparation pipeline. Orchestrates all steps.
        """
        if self.data_is_prepared:
            print("\nData is already prepared. Skipping preparation step.")
        else:
            print("\nStarting data preparation...")
            self._handle_reset()
            self._prepare_source_databases()
            with torch.no_grad():
                self._generate_scenarios()
            # self._generate_features()
            print("\nData preparation finished.")
            self.data_is_prepared = True

    # --- Helper Functions for the Pipeline ---

    def _handle_reset(self):
        """Checks and executes the reset logic, including user confirmation."""
        if not (hasattr(self, "reset") and self.reset):
            return

        precomputed_base_path = self.project_root / "databases" / "precomputed"
        predictions_base_path = self.project_root / "predictions"
        checkpoints_base_path = self.project_root / "checkpoints"
        dir_to_delete = precomputed_base_path / self.id
        dependent_dirs_to_delete = [
            sd for sd in predictions_base_path.glob(f"{self.id}*")
        ] + [sd for sd in checkpoints_base_path.glob(f"{self.id}*")]

        if dir_to_delete.exists():
            print("\n" + "=" * 50)
            print("!! WARNING: RESET FLAG IS ENABLED !!")
            print("=" * 50)
            print("You have set the 'reset' flag to True. This will permanently delete")
            print("the following directory and all its pre-computed contents:")
            print(f"  -> {dir_to_delete}")
            print("-" * 50)

            confirmation = input(
                "Are you absolutely sure you want to proceed? Type 'yes' to confirm: "
            )

            if confirmation.lower() == "yes":
                print(f"\nUser confirmed. Deleting data at: {dir_to_delete}")
                try:
                    shutil.rmtree(dir_to_delete)
                    print("Deletion successful.")
                except OSError as e:
                    print(f"Error deleting directory {dir_to_delete}: {e}")
            else:
                print("\nDeletion aborted by user. The 'reset' flag will be ignored.")
                self.reset = False
        else:
            print(
                f"\nReset flag is True, but no existing directory found at: {dir_to_delete}"
            )

        if dependent_dirs_to_delete:
            print("\n" + "=" * 50)
            print(
                "The following dependent prediction directories should also be deleted:"
            )
            print("=" * 50)
            for d in dependent_dirs_to_delete:
                print(f"  -> {d}")
            print("-" * 50)
            confirmation = input("Type 'all' to delete all listed directories: ")
            if confirmation.lower() == "all":
                for d in dependent_dirs_to_delete:
                    try:
                        shutil.rmtree(d)
                        print(f"Deleted: {d}")
                    except OSError as e:
                        print(f"Error deleting directory {d}: {e}")
            else:
                print("\n Individually asking to delete each dependent directory.")
                for d in dependent_dirs_to_delete:
                    confirmation = input(
                        f"Type 'yes' to delete the directory:\n -> {d},\nor anything else to skip: "
                    )
                    if confirmation.lower() == "yes":
                        try:
                            shutil.rmtree(d)
                            print(f"Deleted: {d}")
                        except OSError as e:
                            print(f"Error deleting directory {d}: {e}")
                    else:
                        print(f"Skipped deletion of: {d}")

    def _prepare_source_databases(self):
        """Calls the prepare_data method on all registered database managers."""
        print("\n--- Preparing all source databases ---")
        for db_manager in self._get_database_managers():
            if hasattr(db_manager, "prepare_data"):
                db_manager.prepare_data()

    def _generate_scenarios(self):
        """
        Generates and saves scenario.pt files for all splits.
        Also pre-computes features and STFTs if applicable.
        """
        print("\n--- Starting scenario pre-computation ---")

        # 1. Setup Base Paths
        precomputed_base_path = self.project_root / "databases" / "precomputed"
        database_path = precomputed_base_path / self.id
        database_path.mkdir(parents=True, exist_ok=True)
        print(f"Scenarios will be saved to: {database_path}")

        # 2. Analyze Feature Extractor Capabilities
        feat_extr = self.feature_extractor
        feature_base_dir = None
        stft_base_dir = None

        # Capture original training state to restore later
        feat_extr_was_training = True

        if isinstance(feat_extr, BaseFeatureExtractor):
            feat_extr_was_training = feat_extr.training
            feat_extr.eval()

            # Condition A: Precompute Features (if supported)
            if feat_extr.precompute_type == "features":
                feature_base_dir = database_path / "features" / feat_extr.signature
                feature_base_dir.mkdir(parents=True, exist_ok=True)

                # Save full signature for reference (in case of hashing)
                with open(feature_base_dir / "signature.txt", "w") as f:
                    f.write(feat_extr.full_signature)

                print(f"Pre-computing features for: {feat_extr.full_signature}")

            # Condition B: Precompute STFT (if used by extractor)
            if feat_extr.uses_stft and isinstance(feat_extr.transform, STFTtransform):
                stft_signature = feat_extr.transform.signature
                stft_base_dir = database_path / "stft" / stft_signature
                stft_base_dir.mkdir(parents=True, exist_ok=True)
                print(f"Pre-computing STFTs for: {stft_signature}")

        # Set random global seed for reproducibility
        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)

        # 3. Iterate over Splits
        for split in ["train", "val", "test"]:
            generator = self._get_scenario_generator(split)
            if not generator or not isinstance(generator, Sized) or len(generator) == 0:
                print(f"No scenarios to generate for '{split}' split. Skipping.")
                continue

            # Prepare directories for this split
            dirs = {"raw": database_path / split}
            dirs["raw"].mkdir(parents=True, exist_ok=True)

            if feature_base_dir:
                dirs["feats"] = feature_base_dir / split
                dirs["feats"].mkdir(parents=True, exist_ok=True)
            if stft_base_dir:
                dirs["stft"] = stft_base_dir / split
                dirs["stft"].mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                # We use the generator directly.
                for i in tqdm(
                    range(len(generator)), desc=f"Pre-computing {split} scenarios"
                ):

                    # Define paths
                    paths = {k: d / f"scenario_{i}.pt" for k, d in dirs.items()}

                    # Check what is missing
                    missing = {k for k, p in paths.items() if not p.exists()}
                    if not missing:
                        continue

                    # Load or Generate Raw Audio
                    if "raw" in missing:
                        # print(50 * "=")
                        # print(50 * "=")
                        # print_gpu_tensors()
                        # data = generator[i]
                        data = generator.__getitem__(i)  # In case of custom logic
                        # torch.cuda.empty_cache()
                        # print_gpu_tensors()
                        # print(50 * "=")
                        input_tensor = data["input"]
                        meta = data["meta"]
                        torch.save(
                            {
                                "meta": meta,
                                "input_type": "raw_audio",
                            },
                            paths["raw"],
                        )
                    else:
                        # Only load if we need to compute derived data
                        if missing:
                            loaded = torch.load(paths["raw"], weights_only=False)
                            meta = loaded["meta"]
                            input_tensor = torch.stack(
                                [v for _, v in meta["references"].items()]
                            ).sum(dim=0)
                        else:
                            continue  # Nothing to do

                    # Compute Derived Data
                    if missing and isinstance(feat_extr, BaseFeatureExtractor):
                        # print(50 * "=")
                        # print_gpu_tensors()
                        derived = move2device(
                            feat_extr.precompute(
                                input_tensor.to(
                                    "cuda:" + str(self.trainer.device_ids[0])
                                )
                            ),  # self.acc_device)),
                            torch.device("cpu"),
                        )
                        # torch.cuda.empty_cache()
                        # print_gpu_tensors()
                        # print(50 * "=")

                        if (
                            "stft" in missing
                            and "stft" in derived
                            and isinstance(feat_extr.transform, STFTtransform)
                        ):
                            torch.save(
                                {
                                    "stft": derived["stft"],
                                    "input_type": "stft",
                                    "stft_info": feat_extr.transform.signature,
                                },
                                paths["stft"],
                            )

                        # Check for standard "features" key or stacked keys (e.g. "0_features", "1_stft")
                        has_features = "features" in derived
                        has_stacked = any(
                            k[0].isdigit() and "_features" in k for k in derived.keys()
                        )

                        if "feats" in missing and (has_features or has_stacked):
                            # If standard, save the tensor. If stacked, save the full dict.
                            payload = (
                                derived["features"]
                                if has_features
                                else {
                                    k: v
                                    for k, v in derived.items()
                                    if k[0].isdigit() and "_features" in k
                                }
                            )
                            torch.save(
                                {
                                    "features": payload,
                                    "input_type": "features",
                                    "feature_info": feat_extr.full_signature,
                                },
                                paths["feats"],
                            )

        # Restore original training state
        if isinstance(feat_extr, BaseFeatureExtractor):
            feat_extr.train(feat_extr_was_training)

    def setup(self, stage: str | None = None):
        """
        Smart setup method that assigns train/val/test datasets.
        It decides whether to load raw audio or precomputed features based on
        the trainability of the feature extractor.
        """
        # 1. Get the base root (usually .../precomputed/<dataset_id>)
        base_root = self.project_root / "databases" / "precomputed" / self.id

        if not base_root.exists():
            raise FileNotFoundError(
                f"Base data directory not found: {base_root}. "
                "Please run the prepare_data step first."
            )

        # 2. Determine the correct data source
        feature_extractor = self.feature_extractor

        if isinstance(feature_extractor, BaseFeatureExtractor):
            precompute_type = feature_extractor.precompute_type

            # Logic to determine what to load
            load_features = (precompute_type == "features") and (
                not self.force_load_stft
            )
            load_stft = (precompute_type == "stft") or self.force_load_stft

            if load_features:
                print("Loading precomputed features.")
                data_root = base_root / "features" / feature_extractor.signature
            elif load_stft and isinstance(feature_extractor.transform, STFTtransform):
                # Check if STFTs exist, otherwise fall back to raw (base_root)
                stft_dir = base_root / "stft" / feature_extractor.transform.signature
                if stft_dir.exists():
                    print("Loading precomputed STFTs.")
                    data_root = stft_dir
                else:
                    print("STFT directory not found. Loading raw audio.")
                    data_root = base_root
            else:
                print("Unknown precompute_type. Loading raw audio.")
                data_root = base_root
        else:
            print("No feature extractor defined. Loading raw audio.")
            data_root = base_root

        data_roots = [data_root]

        # # For Debug purposes TODO: remove
        # if True:
        #     data_roots = [
        #         data_root,
        #         base_root,
        #         base_root / "stft" / feature_extractor.transform.signature,
        #     ]
        #     print(f"Data roots are: {[dr for dr in data_roots]} \n")

        # 3. Assign Datasets
        if stage == "fit" or stage is None:
            self.train_ds = PrecomputedDataset(
                precomputed_dir=[dr / "train" for dr in data_roots],
                preload_to_ram=False,
            )
            self.val_ds = PrecomputedDataset(
                precomputed_dir=[dr / "val" for dr in data_roots],
                preload_to_ram=False,
            )
        if stage in ["test"] or stage is None:
            self.test_ds = PrecomputedDataset(
                precomputed_dir=[dr / "test" for dr in data_roots],
                preload_to_ram=False,
            )

    def train_dataloader(self):
        """Creates the DataLoader for the training set."""
        if not hasattr(self, "train_ds") or self.train_ds is None:
            raise NotImplementedError(
                "self.train_ds must be set in the setup() method."
            )

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.train_ds.collate_fn,
        )

    def val_dataloader(self):
        """Creates the DataLoader for the validation set."""
        if not hasattr(self, "val_ds") or self.val_ds is None:
            raise NotImplementedError("self.val_ds must be set in the setup() method.")

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.val_ds.collate_fn,
        )

    def test_dataloader(self):
        """Creates the DataLoader for the test set."""
        if not hasattr(self, "test_ds") or self.test_ds is None:
            raise NotImplementedError("self.test_ds must be set in the setup() method.")

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.test_ds.collate_fn,
        )
