from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from muse_toolbox.utils import STFTtransform, sample_parameter
import logging

from muse_toolbox.data.datamodules.base_datamodule import BaseDataModule
from muse_toolbox.data.databases.base_DBs import BaseSourceDB, BaseRIRsDB, BaseNoiseDB
from muse_toolbox.data.simulation.base_scenario_generator import BaseScenarioGenerator, ScenarioGenerationConfig
from muse_toolbox.models.components.feature_extractors.base_feature import BaseFeatureExtractor
import muse_toolbox.data.databases as all_dbs

log = logging.getLogger(__name__)


from muse_toolbox.data.databases.brudex_dbs import BrudexRIRsDB, BrudexNoiseDB

@dataclass
class BrudexConfig:
    """Configuration specific to the BRUDEX dataset parameters."""

    reverb_conditions: list[str]
    microphone_arrays: list[str]
    noise_types: tuple[list[str], list[str], list[str]]
    doas: tuple[list[int], list[int], list[int]]


from typing import Any

class BrudexScenarioGeneratorDataset(BaseScenarioGenerator):
    """
    Dataset class responsible for generating BRUDEX-specific acoustic scenarios.

    It extends the `BaseScenarioGenerator` to sample scenario parameters (e.g.,
    noise types, DOAs, reverb conditions) according to the splits defined in the 
    `BrudexConfig`.
    """
    def __init__(
        self,
        id: str,
        num_scenarios: int,
        transform: STFTtransform,
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        brudex_config: BrudexConfig,
        rir_db: BrudexRIRsDB,
        noise_db: BrudexNoiseDB,
        source_dbs: dict[str, BaseSourceDB],
        seed: int | None,
        acc_device: torch.device,
    ):
        """
        Initializes the BRUDEX scenario generator dataset.

        Args:
            id (str): Unique identifier for this dataset split (e.g., 'train_generator').
            num_scenarios (int): Number of scenarios to generate.
            transform (STFTtransform): The STFT transformation config.
            sampling_frequency (int): Sampling frequency of the audio.
            generation_config (ScenarioGenerationConfig): Generic scenario parameters.
            brudex_config (BrudexConfig): BRUDEX-specific scenario parameters.
            rir_db (BrudexRIRsDB): The Room Impulse Response database manager.
            noise_db (BrudexNoiseDB): The Noise database manager.
            source_dbs (dict[str, BaseSourceDB]): Dictionary of clean speech databases.
            seed (int | None): Random seed for reproducibility.
            acc_device (torch.device): Device on which to run accelerated data generation.
        """
        self.brudex_config = brudex_config
        self.train_noise_types, self.val_noise_types, self.test_noise_types = (
            brudex_config.noise_types
        )
        self.train_doas, self.val_doas, self.test_doas = brudex_config.doas

        super().__init__(
            id=id,
            num_scenarios=num_scenarios,
            transform=transform,
            sampling_frequency=sampling_frequency,
            generation_config=generation_config,
            source_dbs=source_dbs,
            rir_db=rir_db,
            noise_db=noise_db,
            seed=seed,
            acc_device=acc_device,
        )

    # --- Implementation of Abstract Hooks ---

    def _sample_scenario_parameters(self) -> dict[str, Any]:
        """
        Samples BRUDEX-specific acoustic parameters for a single scenario.

        Returns:
            dict[str, Any]: A dictionary containing sampled parameters such as 
                the number of sources, DOAs, noise types, and reverb conditions.
        """
        if self.rir_db is None:
            raise RuntimeError("RIR database must be initialized before sampling parameters.")

        # Determine split to select correct DOAs/noise types
        if "train" in self.id:
            split_name = "train"
        elif "val" in self.id:
            split_name = "val"
        elif "test" in self.id:
            split_name = "test"
        else:
            raise ValueError(f"Unknown split in id: {self.id}")

        doas = (
            self.train_doas
            if split_name == "train"
            else (self.val_doas if split_name == "val" else self.test_doas)
        )
        noise_types = (
            self.train_noise_types
            if split_name == "train"
            else (
                self.val_noise_types if split_name == "val" else self.test_noise_types
            )
        )

        num_sources = sample_parameter(
            self.config.max_sources
        )  # Use self.config from parent
        num_sources = min(num_sources, len(self.all_clips), len(doas))

        sirs = sample_parameter(
            (
                -self.config.source_power_range / 2,
                self.config.source_power_range / 2,
            ),
            num=num_sources,
        )

        if self.config.fix_seglen:
            fixed_time_between_events = sample_parameter(
                self.config.time_between_events
            )
        else:
            fixed_time_between_events = 0.0

        return {
            "num_sources": num_sources,
            "signal_length": sample_parameter(
                self.config.signal_lengths
            ),  # Use self.config
            "reverb_condition": sample_parameter(
                self.brudex_config.reverb_conditions
            ),  # Use self.brudex_config
            "microphone_array": sample_parameter(
                self.brudex_config.microphone_arrays
            ),  # Use self.brudex_config
            "noise_type": sample_parameter(noise_types),
            "snr": sample_parameter(self.config.snrs),  # Use self.config
            "sirs": sirs,
            "fixed_time_between_events": fixed_time_between_events,
        }

    def _get_sources_for_scenario(self, scenario_params: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Selects Librispeech clips and assigns DOAs for the sources in a scenario.

        Args:
            scenario_params (dict[str, Any]): The sampled scenario parameters.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each describing a source 
                with its ID, DOA, and file paths.
        """
        num_sources = scenario_params["num_sources"]

        # We need to decide which split to use ('train', 'val', 'test')
        # We can infer this from the generator's ID
        split_name = "train"
        if "val" in self.id:
            split_name = "val"
        elif "test" in self.id:
            split_name = "test"

        doas = (
            self.train_doas
            if split_name == "train"
            else (self.val_doas if split_name == "val" else self.test_doas)
        )

        # Sample unique speakers, then one clip per speaker
        speaker_ids = sample_parameter(list(self.all_clips.keys()), num_sources)
        speech_clips = [
            sample_parameter(self.all_clips[spk_id]) for spk_id in speaker_ids
        ]
        doas = sample_parameter(doas, num_sources)

        sources = []
        for clip_paths, doa in zip(speech_clips, doas):
            # Use the first file to get speaker and chapter ID.
            first_path = clip_paths[0]
            last_path = clip_paths[-1]
            parts = first_path.stem.split("-")
            speaker_id_from_file = parts[0]
            chapter_id = parts[1]
            first_utterance_id = parts[2]
            last_utterance_id = last_path.stem.split("-")[2]

            # Construct the unique source_id
            source_id = f"{speaker_id_from_file}_{chapter_id}_{first_utterance_id}-{last_utterance_id}_{doa}"

            sources.append(
                {"id": source_id, "doa": doa, "clean_speech_paths": clip_paths}
            )
        return sources


class BrudexDataModule(BaseDataModule):
    """
    Lightning DataModule for the BRUDEX dataset.

    This module handles the lifecycle of the BRUDEX data: dynamically initializing 
    the necessary Clean Speech, RIR, and Noise databases, and providing the 
    corresponding scenario generators for the train, validation, and test splits.
    """
    def __init__(
        self,
        id: str,
        data_dir: str | Path,
        transform: STFTtransform,
        batch_size: int,
        num_workers: int,
        num_scenarios: list[int],
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        brudex_config: BrudexConfig,
        clean_speech_databases: dict[str, dict[str, Any]],
        seed: int | None,
        reset: bool,
        feature_extractor: BaseFeatureExtractor | None,
        acc_device: torch.device,
        force_load_stft: bool,
    ):
        """
        Initializes the BrudexDataModule.

        Args:
            id (str): Unique identifier for this data module instance.
            transform (STFTtransform): The STFT transformation config.
            batch_size (int): Global batch size.
            num_workers (int): Number of dataloader workers.
            num_scenarios (list[int]): List containing the number of scenarios for [train, val, test].
            sampling_frequency (int): Audio sampling rate.
            generation_config (ScenarioGenerationConfig): Generic config for dataset generation.
            brudex_config (BrudexConfig): Config specific to the BRUDEX dataset splits.
            clean_speech_databases (dict[str, dict[str, Any]]): Configurations of the clean speech databases to load.
            seed (int | None): Random seed for reproducibility.
            reset (bool): Whether to force a reset of cached data.
            acc_device (torch.device): Device used for accelerated generation.
            feature_extractor (BaseFeatureExtractor | None): Optional pre-configured feature extractor.
            force_load_stft (bool): Whether to force STFT loading instead of waveform.
        """
        # --- Step 1: Call parent constructor and perform simple assignments ---
        super().__init__(
            data_dir=data_dir,
            id=id,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            num_scenarios=num_scenarios,
            sampling_frequency=sampling_frequency,
            generation_config=generation_config,
            seed=seed,
            reset=reset,
            acc_device=acc_device,
            feature_extractor=feature_extractor,
            force_load_stft=force_load_stft,
        )
        self.brudex_config = brudex_config
        self.clean_speech_databases_configs = clean_speech_databases

        # --- Step 2: Delegate complex initialization to helper methods ---
        self._init_clean_speech_dbs()
        self._init_brudex_dbs()

        # --- Step 3: Final logging call ---
        self._verbose_parameters()

    def _init_clean_speech_dbs(self) -> None:
        """Dynamically finds and instantiates clean speech database managers."""
        self.clean_speech_dbs = {}
        database_root = self.data_dir / "databases"

        for db_name, db_config in self.clean_speech_databases_configs.items():
            db_class_name = db_config.pop("class", None)
            if not db_class_name:
                raise ValueError(
                    f"Config for speech DB '{db_name}' is missing 'class' key."
                )

            db_class = getattr(all_dbs, db_class_name, None)

            if db_class and issubclass(db_class, BaseSourceDB):
                log.info(f"Found implementation for clean speech database: {db_class_name}")
                self.clean_speech_dbs[db_name] = db_class(
                    root_dir=database_root,
                    signal_lengths=np.max(self.generation_config.signal_lengths),
                    **db_config,
                )
            else:
                raise ImportError(
                    f"Could not find a valid BaseSourceDB class for '{db_name}' in databases."
                )

    def _init_brudex_dbs(self) -> None:
        """Initializes the specific BRUDEX RIR and Noise databases."""
        all_doas = sorted(
            list(
                set(self.brudex_config.doas[0])
                | set(self.brudex_config.doas[1])
                | set(self.brudex_config.doas[2])
            )
        )

        all_noise_types = sorted(
            list(
                set(self.brudex_config.noise_types[0])
                | set(self.brudex_config.noise_types[1])
                | set(self.brudex_config.noise_types[2])
            )
        )

        brudex_dir = self.data_dir / "databases" / "brudex"

        self.rir_db = BrudexRIRsDB(
            root_dir=brudex_dir,
            sampling_frequency=self.sampling_frequency,
            reverb_conditions=self.brudex_config.reverb_conditions,
            doas=all_doas,
            microphone_arrays=self.brudex_config.microphone_arrays,
        )
        self.noise_db = BrudexNoiseDB(
            root_dir=brudex_dir,
            sampling_frequency=self.sampling_frequency,
            reverb_conditions=self.brudex_config.reverb_conditions,
            noise_types=all_noise_types,
            microphone_arrays=self.brudex_config.microphone_arrays,
        )

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs the BRUDEX parameters to the standard logger."""
        log.info(f"{indent}Brudex Data Module Parameters:")
        log.info(f"{indent}  Batch Size: {self.batch_size}")
        log.info(f"{indent}  Num Workers: {self.num_workers}")
        log.info(f"{indent}  Num Scenarios: {self.num_scenarios}")
        self.transform._verbose_parameters(indent=indent + "  ")
        log.info(f"{indent}  Sampling Frequency: {self.sampling_frequency} [Hz]")
        log.info(f"{indent}  Max Sources: {self.generation_config.max_sources}")
        log.info(f"{indent}  Signal Lengths: {self.generation_config.signal_lengths} [s]")
        log.info(f"{indent}  Initial Noise Only Duration: {self.generation_config.initial_noise_only_duration} [s]")
        log.info(f"{indent}  Reverb Times: {self.brudex_config.reverb_conditions} [ms]")
        log.info(f"{indent}  Microphone Arrays: {self.brudex_config.microphone_arrays}")
        log.info(f"{indent}  Train Noise Types: {self.brudex_config.noise_types[0]}")
        log.info(f"{indent}  Val Noise Types: {self.brudex_config.noise_types[1]}")
        log.info(f"{indent}  Test Noise Types: {self.brudex_config.noise_types[2]}")
        log.info(f"{indent}  SNRs: {self.generation_config.snrs} [dB]")
        log.info(f"{indent}  Clean speech databases: {list(self.clean_speech_databases_configs.keys())}")
        log.info(f"{indent}  Train DOAs: {self.brudex_config.doas[0]} [°]")
        log.info(f"{indent}  Val DOAs: {self.brudex_config.doas[1]} [°]")
        log.info(f"{indent}  Test DOAs: {self.brudex_config.doas[2]} [°]")
        log.info(f"{indent}  Time Between Events: {self.generation_config.time_between_events} [s]")
        log.info(f"{indent}  Activations Only: {self.generation_config.activations_only}")
        log.info(f"{indent}  Seed: {self.seed}")

    def _get_database_managers(self) -> list[Any]:
        """Provides the specific database manager objects required for BRUDEX."""
        managers = list(self.clean_speech_dbs.values())
        managers.append(self.rir_db)
        managers.append(self.noise_db)
        return managers

    def _get_scenario_generator(self, split: str) -> Dataset | None:
        """Provides the configured BrudexScenarioGeneratorDataset for a given split."""

        if split == "train":
            num_scenarios = self.num_scenarios[0]
            seed_offset = 0
        elif split == "val":
            num_scenarios = self.num_scenarios[1]
            seed_offset = 1
        elif split == "test":
            num_scenarios = self.num_scenarios[2]
            seed_offset = 2
        else:
            raise ValueError(f"Unknown split: {split}")
        if num_scenarios == 0:
            return None

        return BrudexScenarioGeneratorDataset(
            id=f"{self.id}_{split}_generator",
            num_scenarios=num_scenarios,
            transform=self.transform,
            sampling_frequency=self.sampling_frequency,
            generation_config=self.generation_config,
            brudex_config=self.brudex_config,
            rir_db=self.rir_db,
            noise_db=self.noise_db,
            source_dbs=self.clean_speech_dbs,
            acc_device=self.acc_device,
            seed=self.seed + seed_offset if self.seed is not None else None,
        )
        