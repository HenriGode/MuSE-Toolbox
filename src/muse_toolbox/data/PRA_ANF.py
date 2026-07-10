import torch
import numpy as np
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Unpack, Any
from building_blocks.feature_extractors.base_feature import BaseFeatureExtractor
from .base_dataset import (
    BaseDataModule,
    BaseDB,
    BaseRIRsDB,
    BaseNoiseDB,
    BaseSourceDB,
    BaseScenarioGenerator,
    ScenarioGenerationConfig,
)
from muse_toolbox.data.sourceDBs import DemandDatabase
from muse_toolbox.utils import (
    STFTtransform,
    simRIR_shoebox_PRA,
    simDiffuseNoiseANF,
    MicrophoneArray,
    sample_parameter,
    load_audio,
)


# --- Step 1A: Define Type-Safe Dictionaries for RIR/Noise Generation ---


class PraRirKwargs(TypedDict):
    """Defines the required keyword arguments for generating a PRA RIR."""

    room_dims: list[float]
    source_pos: list[float]
    mic_pos: list[float]
    rt60: float
    mic_array: str  # To know which microphone positions to use


class AnfNoiseKwargs(TypedDict):
    """Defines the required keyword arguments for spatializing ANF noise."""

    room_dims: list[float]
    mic_pos: list[float]
    mic_array: str
    signal_length: int  # Added for clarity
    split: str  # Added to select the correct noise file
    noise_file_path: str  # Path to the noise file to be used


@dataclass
class PraAnfConfig:
    """Configuration specific to the Pyroomacoustics + ANF DataModule."""

    # Room dimensions (e.g., [[3, 8], [3, 8], [2.5, 4]])
    room_dims: list[list[float]] = field(
        default_factory=lambda: [[4, 8], [4, 8], [2.5, 4]]
    )

    # RT60 values in seconds
    rt60s: list[float] | float = 0.4

    microphone_arrays: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "num_mics": 4,
                "geometry": "cube_volume",
                "distribution": "random",
                "max_distance": 0.10,
                "min_distance": 0.01,
                "radius": 0.10,
                "rotation": "fixed",
                "restrict_rot_2_xy_plane": False,
                "fix_height": None,
            }
        ]
    )

    # Constraints for positioning the array center
    mic_array_min_distance: float = 0.5  # [m] from walls
    mic_array_position: list[list[float]] = field(
        default_factory=lambda: [[3.2], [3.2], [1.6]]
    )  # Default position if not randomly placed
    # Source position constraints relative walls the mic array and each other
    sources_min_distance: float = 0.5  # [m]
    min_doa_between_sources: float = (
        30.0  # [°], minimum angular separation between sources as seen from the mic array center
    )

    fix_source_heights: float | None = None  # [m], None to not fix heights


class PraRIRsDB(BaseRIRsDB):
    """
    Generates RIRs on-the-fly using Pyroomacoustics.
    This class does not pre-process or load files from disk.
    """

    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir=root_dir)
        # No file scanning needed as RIRs are generated dynamically.
        print("Initialized PraRIRsDB for on-the-fly RIR generation.")

    def prepare_data(self):
        """No data preparation is needed for generated RIRs."""
        pass

    @property
    def reverb_conditions(self) -> list[float]:
        """Returns an empty list as RT60 is sampled dynamically."""
        return []

    def get_rir(self, **kwargs: Unpack[PraRirKwargs]) -> torch.Tensor:
        """Generates a single RIR tensor based on the provided parameters."""
        # pyroomacoustics expects a list of source positions
        rir = simRIR_shoebox_PRA(
            room_dim=kwargs["room_dims"],
            mic_positions=kwargs["mic_pos"],
            source_positions=[kwargs["source_pos"]],
            rt60=kwargs["rt60"],
        )
        # Squeeze the source dimension as we generate one source at a time
        return rir.squeeze(0)


class AnfNoiseDB(BaseNoiseDB):
    """
    Generates diffuse noise on-the-fly using ANF-Generator. It uses a
    MyNoiseDB instance to dynamically load single-channel noise files.
    """

    def __init__(
        self,
        root_dir: str | Path,
        noise_source_db: DemandDatabase,
        sampling_frequency: int,
    ):
        super().__init__(root_dir=root_dir)
        self.noise_source_db: DemandDatabase = noise_source_db
        self.fs = sampling_frequency
        print(
            "Initialized AnfNoiseDB to use MyNoiseDB for on-the-fly diffuse noise generation."
        )

    def prepare_data(self):
        """No data preparation is needed here; it's handled by the source DB."""
        pass

    def get_noise(self, **kwargs: Unpack[AnfNoiseKwargs]) -> torch.Tensor:
        """Generates a diffuse noise tensor based on the room parameters."""
        num_mics = np.array(kwargs["mic_pos"]).shape[1]
        signal_len = kwargs["signal_length"]
        signal_len_samples = int(signal_len * self.fs)
        required_len = signal_len_samples * num_mics

        # --- Step 1: Get a noise file from the source DB ---
        noise_file_path = kwargs["noise_file_path"]

        # --- Step 2: Load the audio file using torchcodec ---
        noise_signal, _ = load_audio(noise_file_path, sampling_frequency=self.fs)

        # --- Step 3: Prepare uncorrelated signals ---
        # Ensure the loaded noise signal is long enough
        if noise_signal.shape[1] < required_len:
            # If not long enough, repeat the signal
            repeats = -(-required_len // noise_signal.shape[1])  # Ceiling division
            noise_signal = noise_signal.repeat(1, repeats)

        # Create a set of uncorrelated input signals by taking a random chunk
        start_idx = random.randint(0, noise_signal.shape[1] - required_len)
        uncorrelated_signals = (
            noise_signal[0, start_idx : start_idx + required_len]
            .reshape(num_mics, signal_len_samples)
            .numpy()
        )

        # --- Step 4: Generate the spatially coherent noise ---
        diffuse_noise = simDiffuseNoiseANF(
            mic_positions=np.array(kwargs["mic_pos"]).T,  # Expects (M, 3)
            input_signals=uncorrelated_signals,
            fs=self.fs,
        )

        return diffuse_noise

    # --- Step 4: Implement the Scenario Generator ---


class PraAnfScenarioGeneratorDataset(BaseScenarioGenerator):
    def __init__(
        self,
        id: str,
        num_scenarios: int,
        transform: STFTtransform,
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        pra_anf_config: PraAnfConfig,
        rir_db: PraRIRsDB,
        noise_db: AnfNoiseDB,
        source_dbs: dict[str, BaseSourceDB],
        seed: int | None,
        acc_device: torch.device,
    ):
        self.pra_anf_config = pra_anf_config
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
        self.noise_db: AnfNoiseDB = noise_db

    def _sample_scenario_parameters(self) -> dict:
        """Samples all dynamic parameters for one scenario."""

        split_name = self._get_split_name()

        # Sample room dimensions and RT60
        room_dims = [random.uniform(d[0], d[1]) for d in self.pra_anf_config.room_dims]
        rt60 = sample_parameter(self.pra_anf_config.rt60s)

        # Sample a microphone array configuration and place it in the room
        mic_array_config = random.choice(self.pra_anf_config.microphone_arrays)

        rotation = mic_array_config.get("rotation", "random")
        restrict_rot_2_xy = mic_array_config.pop("restrict_rot_2_xy_plane", False)
        fix_height = mic_array_config.pop("fix_height", None)
        mic_array = MicrophoneArray(**mic_array_config)

        # if the fixed position has ranges instead of fixed values, sample from the ranges
        if self.pra_anf_config.mic_array_position is not None:
            mic_array_position = []
            for dim in self.pra_anf_config.mic_array_position:
                if isinstance(dim, list) and len(dim) == 2:
                    mic_array_position.append(sample_parameter((dim[0], dim[1])))
                else:
                    mic_array_position.append(dim[0])  # Use the fixed value
        else:
            mic_array_position = None

        mic_pos = mic_array.place(
            room_dims,
            self.pra_anf_config.mic_array_min_distance,
            restrict_rot_2_xy_plane=restrict_rot_2_xy,
            fix_height=fix_height,
            fixed_position=mic_array_position,
            fixed_rotation=(rotation == "fixed"),
        )

        # Sample other generic parameters
        num_sources = sample_parameter(self.config.max_sources)
        # Ensure we don't request more sources than available speakers
        num_sources = min(num_sources, len(self.all_clips))

        min_dist = self.pra_anf_config.sources_min_distance
        min_doa_deg = self.pra_anf_config.min_doa_between_sources

        mic_array_center = np.mean(mic_pos, axis=1)
        source_positions = []
        max_attempts = 100  # Safeguard against infinite loops

        for _ in range(num_sources):
            for attempt in range(max_attempts):
                # Sample height: Handle both fixed float or range/list
                if self.pra_anf_config.fix_source_heights is not None:
                    z = sample_parameter(self.pra_anf_config.fix_source_heights)
                else:
                    z = random.uniform(min_dist, room_dims[2] - min_dist)

                # Sample position
                pos = np.array(
                    [
                        random.uniform(min_dist, room_dims[0] - min_dist),
                        random.uniform(min_dist, room_dims[1] - min_dist),
                        z,
                    ]
                )

                # Check distance to mic array center
                if np.linalg.norm(pos - mic_array_center) < min_dist:
                    continue

                # Check distance to other sources
                dist_valid = all(
                    np.linalg.norm(pos - np.array(p)) >= min_dist
                    for p in source_positions
                )
                if not dist_valid:
                    continue

                # Check DOA separation if required
                doa_valid = True
                if min_doa_deg > 0 and len(source_positions) > 0:
                    new_vec = pos - mic_array_center
                    new_norm = np.linalg.norm(new_vec)
                    # Safe guard for zero norms
                    if new_norm < 1e-6:
                        doa_valid = False
                    else:
                        for existing_pos in source_positions:
                            existing_vec = np.array(existing_pos) - mic_array_center
                            existing_norm = np.linalg.norm(existing_vec)
                            if existing_norm < 1e-6:
                                continue

                            cos_angle = np.dot(new_vec, existing_vec) / (
                                new_norm * existing_norm
                            )
                            # Clip to valid range for arccos due to numerical noise
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle_deg = np.rad2deg(np.arccos(cos_angle))

                            if angle_deg < min_doa_deg:
                                doa_valid = False
                                break

                if dist_valid and doa_valid:
                    source_positions.append(list(pos))
                    break
            else:
                # This else belongs to the for loop, executed if the loop finishes without a break
                raise RuntimeError(
                    f"Could not place source after {max_attempts} attempts. "
                    f"Consider reducing sources_min_distance ({min_dist}m), "
                    f"min_doa_between_sources ({min_doa_deg} deg), or the number of sources."
                )

        # for _ in range(num_sources):
        #     for attempt in range(max_attempts):
        #         # Sample a position with minimum distance from walls
        #         pos = np.array(
        #             [
        #                 random.uniform(min_dist, room_dims[0] - min_dist),
        #                 random.uniform(min_dist, room_dims[1] - min_dist),
        #                 (
        #                     random.uniform(min_dist, room_dims[2] - min_dist)
        #                     if self.pra_anf_config.fix_source_heights is None
        #                     else self.pra_anf_config.fix_source_heights
        #                 ),
        #             ]
        #         )

        #         # Check distance to mic array center
        #         if np.linalg.norm(pos - mic_array_center) < min_dist:
        #             continue

        #         # Check distance to other sources
        #         is_valid = all(
        #             np.linalg.norm(pos - np.array(p)) >= min_dist
        #             for p in source_positions
        #         )

        #         if is_valid:
        #             source_positions.append(list(pos))
        #             break
        #     else:
        #         # This else belongs to the for loop, executed if the loop finishes without a break
        #         raise RuntimeError(
        #             f"Could not place source after {max_attempts} attempts. "
        #             f"Consider reducing sources_min_distance ({min_dist}m) or the number of sources."
        #         )

        sirs = sample_parameter(
            (
                -self.config.source_power_range / 2,
                self.config.source_power_range / 2,
            ),
            num=num_sources,
        )

        noise_file = sample_parameter(
            self.noise_db.noise_source_db.noise_files_per_split[split_name]
        )

        if self.config.fix_seglen:
            fixed_time_between_events = sample_parameter(
                self.config.time_between_events
            )
        else:
            fixed_time_between_events = 0.0

        return {
            "num_sources": num_sources,
            "signal_length": sample_parameter(self.config.signal_lengths),
            "snr": sample_parameter(self.config.snrs),
            "sirs": sirs,
            "room_dims": room_dims,
            "rt60": rt60,
            "mic_array": mic_array_config.get("geometry", "UnnamedArray"),
            "mic_pos": mic_pos,
            "source_positions": source_positions,
            "noise_file_path": noise_file,
            "fixed_time_between_events": fixed_time_between_events,
        }

    def _get_sources_for_scenario(self, scenario_params: dict) -> list[dict]:
        """
        Selects clean speech clips
        """
        num_sources = scenario_params["num_sources"]

        # Ensure we don't try to sample more unique speakers than available
        num_speakers = len(self.all_clips)
        speaker_ids = sample_parameter(
            list(self.all_clips.keys()), min(num_sources, num_speakers)
        )
        # If num_sources > num_speakers, reuse speakers
        while len(speaker_ids) < num_sources:
            speaker_ids.append(random.choice(list(self.all_clips.keys())))

        speech_clips = [random.choice(self.all_clips[spk_id]) for spk_id in speaker_ids]

        sources = []
        for i, clip_paths in enumerate(speech_clips):
            source_id = f"{Path(clip_paths[0]).stem}_pos{i}"
            sources.append(
                {
                    "id": source_id,
                    "clean_speech_paths": clip_paths,
                    "source_pos": scenario_params["source_positions"][i],
                }
            )
        return sources


class PraAnfDataModule(BaseDataModule):
    def __init__(
        self,
        id: str,
        batch_size: int,
        num_workers: int,
        num_scenarios: list[int],
        transform: STFTtransform,
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        pra_anf_config: PraAnfConfig,
        clean_speech_databases: dict[str, dict],
        noise_databases: dict[str, dict],
        feature_extractor: BaseFeatureExtractor | None,
        seed: int | None,
        reset: bool,
        acc_device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            project_root=Path(__file__).parent.parent.parent,
            id=id,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            num_scenarios=num_scenarios,
            sampling_frequency=sampling_frequency,
            generation_config=generation_config,
            seed=seed,
            reset=reset,
            feature_extractor=feature_extractor,
            acc_device=acc_device,
        )
        self.pra_anf_config = pra_anf_config
        self.clean_speech_databases_configs = clean_speech_databases
        self.noise_databases_configs = noise_databases

        self._init_clean_speech_dbs()
        self._init_noise_source_dbs()
        self._init_pra_anf_dbs()

    def _init_clean_speech_dbs(self):
        """Dynamically finds and instantiates clean speech database managers."""
        from . import sourceDBs  # Local import to access all DB classes

        self.clean_speech_dbs = {}
        database_root = self.project_root / "databases"

        for db_name, db_config in self.clean_speech_databases_configs.items():
            db_class_name = db_config.pop("class", None)
            if not db_class_name:
                raise ValueError(
                    f"Config for speech DB '{db_name}' is missing 'class' key."
                )

            db_class = getattr(sourceDBs, db_class_name, None)

            if db_class and issubclass(db_class, BaseSourceDB):
                print(
                    f"Found implementation for clean speech database: {db_class_name}"
                )
                self.clean_speech_dbs[db_name] = db_class(
                    root_dir=database_root,
                    signal_lengths=np.max(self.generation_config.signal_lengths),
                    **db_config,
                )
            else:
                raise ImportError(
                    f"Could not find a valid BaseSourceDB class named '{db_class_name}' in sourceDBs."
                )

    def _init_noise_source_dbs(self):
        """Dynamically finds and instantiates noise source database managers."""
        from . import sourceDBs  # Local import to access all DB classes

        self.noise_source_dbs = {}
        database_root = self.project_root / "databases"

        for db_name, db_config in self.noise_databases_configs.items():
            db_class_name = db_config.pop("class", None)
            if not db_class_name:
                raise ValueError(
                    f"Config for noise DB '{db_name}' is missing 'class' key."
                )

            db_class = getattr(sourceDBs, db_class_name, None)

            if db_class and issubclass(db_class, BaseSourceDB):
                print(
                    f"Found implementation for noise source database: {db_class_name}"
                )
                # Add sampling_rate if the class needs it (like DemandDatabase)
                if "sampling_rate" not in db_config:
                    db_config["sampling_rate"] = self.sampling_frequency

                self.noise_source_dbs[db_name] = db_class(
                    root_dir=database_root,
                    **db_config,
                )
            else:
                raise ImportError(
                    f"Could not find a valid BaseSourceDB class named '{db_class_name}' in sourceDBs."
                )

    def _init_pra_anf_dbs(self):
        """Initializes the PRA/ANF-specific RIR and Noise database managers."""
        self.rir_db = PraRIRsDB(root_dir=self.project_root / "databases")

        # For now, assume we use the first configured noise DB.
        # This could be extended to handle multiple noise sources.
        noise_source_db_instance = list(self.noise_source_dbs.values())[0]

        self.noise_db = AnfNoiseDB(
            root_dir=self.project_root / "databases",
            noise_source_db=noise_source_db_instance,
            sampling_frequency=self.sampling_frequency,
        )

    def _get_database_managers(self) -> list[BaseDB]:
        """Provides all database manager objects required for this datamodule."""
        managers = list(self.clean_speech_dbs.values())
        managers.extend(list(self.noise_source_dbs.values()))
        managers.append(self.rir_db)
        managers.append(self.noise_db)
        return managers

    def _get_scenario_generator(self, split: str) -> BaseScenarioGenerator | None:
        """Provides the configured PraAnfScenarioGeneratorDataset for a given split."""
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

        return PraAnfScenarioGeneratorDataset(
            id=f"{self.id}_{split}_generator",
            num_scenarios=num_scenarios,
            transform=self.transform,
            sampling_frequency=self.sampling_frequency,
            generation_config=self.generation_config,
            pra_anf_config=self.pra_anf_config,
            rir_db=self.rir_db,
            noise_db=self.noise_db,
            source_dbs=self.clean_speech_dbs,
            seed=self.seed + seed_offset if self.seed is not None else None,
            acc_device=self.acc_device,
        )
