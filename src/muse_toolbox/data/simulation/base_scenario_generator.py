import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import lightning as pl
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from muse_toolbox.data.databases.base_DBs import BaseNoiseDB, BaseRIRsDB, BaseSourceDB
from muse_toolbox.utils import calculate_t60
from .scenario_generation import identify_segments
from muse_toolbox.utils import (
    STFTtransform,
    convolve_clean2microphone,
    db2amp,
    load_audio,
    move2device,
    normalize_components,
    sample_parameter,
    rir2rtf,
    vad_opt_fast_gen,
)


log = logging.getLogger(__name__)


@dataclass
class ScenarioGenerationConfig:
    """Configuration for the generic scenario generation process."""

    max_sources: int | list[int] | tuple[int, int]
    signal_lengths: float | list[float] | tuple[float, float]
    initial_noise_only_duration: float | list[float] | tuple[float, float] | Any
    snrs: float | list[float] | tuple[float, float] | Any
    source_power_range: float
    time_between_events: float | list[float] | tuple[float, float] | Any
    fix_seglen: bool
    activations_only: bool
    remove_silence: bool
    neglect_silence4oracle_sa: float
    bridge_clean_speech_gaps: float
    vad_threshold2define_oracle: float
    vad_threshold2select_clean_speech: float
    save_signal_components: bool = True



class BaseScenarioGenerator(Dataset, ABC):
    """A generic base class for creating dynamic, multi-source audio scenarios.

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
    ) -> None:
        """Initializes the BaseScenarioGenerator.

        Args:
            id (str): Unique identifier for the generator (usually indicates split).
            num_scenarios (int): Number of scenarios to generate.
            transform (STFTtransform): Transform configuration.
            sampling_frequency (int): Target sampling frequency.
            generation_config (ScenarioGenerationConfig): Configuration parameters for scenario mixing.
            source_dbs (Dict[str, BaseSourceDB]): Dictionary of source database managers.
            rir_db (Optional[BaseRIRsDB]): Database manager for RIRs.
            noise_db (Optional[BaseNoiseDB]): Database manager for noise.
            seed (Optional[int]): Random seed for reproducibility.
            acc_device (torch.device): Torch device to use for computations.
        """
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

        split_name = self._get_split_name()
        self.all_clips = defaultdict(list)
        for db_name, db_manager in self.source_dbs.items():
            clips = db_manager.get_speaker_clips(split_name)
            for speaker_id, speaker_clips in clips.items():
                self.all_clips[f"{db_name}_{speaker_id}"].extend(speaker_clips)

        self.scenarios = self._construct_dataset()

    # --- Abstract Hooks for Subclasses ---

    @abstractmethod
    def _sample_scenario_parameters(self) -> dict[str, Any]:
        """Samples all dataset-specific parameters for a single scenario.

        Returns:
            Dict[str, Any]: Must contain at least 'num_sources' and 'signal_length'.
                e.g., {'num_sources': 3, 'signal_length': 60.0, 'snr': 10, ...}
        """
        pass

    @abstractmethod
    def _get_sources_for_scenario(self, scenario_params: dict[str, Any]) -> list[dict[str, Any]]:
        """Selects clean speech clips and assigns source properties.

        Args:
            scenario_params (Dict[str, Any]): The sampled parameters for this scenario.

        Returns:
            List[Dict[str, Any]]: A list of source dictionaries. Each dict must have an 'id' key.
                e.g., [{'id': 'spk1_doa30', 'doa': 30, 'clean_speech_paths': [...]}, ...]
        """
        pass

    def _get_split_name(self) -> str:
        """Determines the current split from the ID.

        Returns:
            str: The split name ('train', 'val', or 'test').

        Raises:
            ValueError: If the split cannot be determined from the ID.
        """
        if "train" in self.id:
            return "train"
        elif "val" in self.id:
            return "val"
        elif "test" in self.id:
            return "test"
        else:
            raise ValueError(f"Unknown split in id: {self.id}")

    def _construct_dataset(self) -> list[dict[str, Any]]:
        """Constructs the list of scenario configurations.

        Returns:
            List[Dict[str, Any]]: List of scenario configuration dictionaries.
        """
        scenarios = []
        for _ in tqdm(
            range(self.num_scenarios),
            desc=f"Constructing {self.num_scenarios} scenarios for {self.id}",
        ):
            scenario_params = self._sample_scenario_parameters()
            num_sources = scenario_params["num_sources"]

            sources = self._get_sources_for_scenario(scenario_params)
            if len(sources) < num_sources:
                num_sources = len(sources)
                scenario_params["num_sources"] = num_sources

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
        self, sources: list[dict[str, Any]], max_sources: int, scenario_params: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], float]:
        """Generates the dynamic activation pattern for a scenario.

        Args:
            sources (List[Dict[str, Any]]): Available sources.
            max_sources (int): Maximum active sources.
            scenario_params (Dict[str, Any]): Parameters for the scenario.

        Returns:
            Tuple[List[Dict[str, Any]], float]: Event log and final current time.
        """
        event_log = []
        available_sources = list(sources)
        active_sources = []

        if max_sources > len(sources):
            max_sources = len(sources)

        current_time = sample_parameter(self.config.initial_noise_only_duration)
        tbe = self.config.time_between_events
        if hasattr(tbe, "keys") and "min" in tbe: # type: ignore
            min_tbe = tbe["min"] # type: ignore
        elif hasattr(tbe, "__iter__") and not isinstance(tbe, str):
            min_tbe = min(tbe) # type: ignore
        else:
            min_tbe = float(tbe) # type: ignore
            
        max_event_time = scenario_params["signal_length"] - min_tbe

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
                event_type = 1
                source_for_event = random.choice(available_sources)
                active_sources.append(source_for_event)
                available_sources.remove(source_for_event)
            else:
                event_type = -1
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
        self, activity_pattern: list[dict[str, Any]], signal_length: float
    ) -> dict[str, list[torch.Tensor]]:
        """Converts activity pattern times to sample indices.

        Args:
            activity_pattern (List[Dict[str, Any]]): Log of activation/deactivation events.
            signal_length (float): Length of the signal in seconds.

        Returns:
            Dict[str, List[torch.Tensor]]: Sample indices grouped by source ID.
        """
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
        """Calculates total active samples required per source.

        Args:
            event_samples_by_speaker (Dict[str, List[torch.Tensor]]): Map of source ID to event samples.

        Returns:
            Dict[str, int]: The required active sample count per source ID.
        """
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
        """Calculates frame-wise source count from sample-wise VAD masks.

        Args:
            source_activity (Dict[str, torch.Tensor]): Dictionary of VAD tensors per source.

        Returns:
            torch.Tensor: The estimated count of active sources per frame.
        """
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

    def _load_source_materials(self, scenario_params: dict[str, Any]) -> tuple[dict, dict, torch.Tensor | None, dict]:
        """Loads clean speech, RIRs, and noise for the scenario.

        Args:
            scenario_params (Dict[str, Any]): Dictionary of scenario parameters.

        Returns:
            Tuple[Dict, Dict, Optional[torch.Tensor], Dict]: 
                Clean speeches, RIRs, noise tensor, and activity samples.
        """
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
                rirs[source_info["id"]] = self.rir_db.get_rir(
                    **scenario_params, **source_info
                ).to(self.acc_device)

        noise = None
        if self.noise_db:
            raw_noise = self.noise_db.get_noise(**scenario_params).to(self.acc_device)
            target_samples = int(scenario_params["signal_length"] * self.sampling_frequency)
            if raw_noise.shape[1] >= target_samples:
                start_sample = random.randint(0, raw_noise.shape[1] - target_samples)
                noise = raw_noise[:, start_sample : start_sample + target_samples]
            else:
                repeats = (target_samples // raw_noise.shape[1]) + 1
                noise = raw_noise.repeat(1, repeats)[:, :target_samples]
            del raw_noise

        torch.cuda.empty_cache()

        return clean_speeches, rirs, noise, activity_samples

    def _render_and_place_sources(
        self,
        clean_speeches_long: dict[str, torch.Tensor],
        rirs: dict[str, torch.Tensor],
        activity_samples: dict[str, list[torch.Tensor]],
        scenario_params: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Renders reverberant sources and places them into the timeline.

        Args:
            clean_speeches_long (Dict[str, torch.Tensor]): Dict of clean speech tensors.
            rirs (Dict[str, torch.Tensor]): Dict of RIR tensors.
            activity_samples (Dict[str, List[torch.Tensor]]): Dict of active sample intervals.
            scenario_params (Dict[str, Any]): The scenario parameters.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping source ID to the rendered source signal.
        """
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

        return full_source_signals

    def _mix_and_finalize_scenario(
        self,
        full_source_signals: dict[str, torch.Tensor],
        noise: torch.Tensor | None,
        scenario_params: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Mixes all signals and generates the final noisy scenario.

        Args:
            full_source_signals (Dict[str, torch.Tensor]): Rendered source signals.
            noise (Optional[torch.Tensor]): Rendered noise signal.
            scenario_params (Dict[str, Any]): Scenario configuration parameters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
                The noisy signal, frame-wise source count, activity VADs, and raw signal components.
                
        Raises:
            RuntimeError: If scenario has no active sources and no noise.
        """
        signal_components: dict[str, torch.Tensor] = {}
        source_activity_vad: dict[str, torch.Tensor] = {}
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
            noisy_signal = signal_components["noise"]
        else:
            log.warning(
                f"Scenario {scenario_params.get('id', '')} has no active sources and no noise. "
                "Generating 2-channel (stereo) silence."
            )
            raise RuntimeError(
                f"Scenario {scenario_params.get('id', '')} has no active sources and no noise. "
                "Cannot generate a silent signal."
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
        """Computes oracle RTFs for all sources in the provided RIR dictionary.

        Args:
            rirs (Dict[str, torch.Tensor]): Dictionary mapping source IDs to RIR tensors.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping source IDs to their corresponding RTF tensors.
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
        self, source_info: dict[str, Any], required_samples: int
    ) -> torch.Tensor:
        """Loads and concatenates audio files for a single source.

        Args:
            source_info (Dict[str, Any]): Source metadata containing file paths.
            required_samples (int): Required number of audio samples.

        Returns:
            torch.Tensor: The loaded and concatenated speech tensor.
        """
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
                    current_length += audio_tensor.shape[-1]
                    audio_tensor = None

        if not concatenated_audio:
            return torch.zeros(required_samples)

        full_speech_signal = torch.cat(concatenated_audio)
        return full_speech_signal[:required_samples]

    # --- Standard Dataset Methods ---

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Retrieves and constructs a single scenario.

        Args:
            index (int): Scenario index.

        Returns:
            Dict[str, Any]: Dictionary containing 'input' and 'meta' data.
        """
        scenario_params = self.scenarios[index]

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
        rT60s = torch.stack([calculate_t60(rir, int(self.transform.sampling_frequency)) for _, rir in rirs.items()])

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
            "scenario_id": f"{self.id}_{index}",
            "rtfs": rtfs,
            "sad_samples": sad_samples,
            "sad_frames": sad_frames,
            "segments": segments,
            "rirs": rirs,
            "rT60s": rT60s,
            # "gt_rtf_stream": gt_rtf_stream,
            # "gt_ids_stream": gt_ids_stream,
            # "id_map": id_map,
        }

        if self.config.save_signal_components:
            meta_data["references"] = signal_components

        item = move2device(
            {"input": noisy_signal, "meta": meta_data}, torch.device("cpu")
        )
        
        # Free memory
        del noisy_signal
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

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collates items from the generator into a batch for saving.

        Args:
            batch (List[Dict[str, Any]]): List of generated scenario items.

        Returns:
            Dict[str, Any]: The collated batch.
        """
        if not batch:
            return {}

        inputs = [item["input"] for item in batch]

        collated_meta = {}
        if "meta" in batch[0]:
            for key in batch[0]["meta"].keys():
                collated_meta[key] = [d["meta"][key] for d in batch]

        return {"input": inputs, "meta": collated_meta}
