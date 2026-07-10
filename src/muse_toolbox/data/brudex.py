from dataclasses import dataclass
import os
import torch
import torchaudio
import mat73
from torch.utils.data import Dataset
from tqdm import tqdm
from muse_toolbox.utils import *
from pathlib import Path
from typing import TypedDict, Unpack
import yaml

# Import the new base classes
from .base_dataset import (
    BaseDataModule,
    BaseSourceDB,
    BaseRIRsDB,
    BaseNoiseDB,
    BaseScenarioGenerator,
    ScenarioGenerationConfig,
)
from building_blocks.feature_extractors.base_feature import BaseFeatureExtractor
from muse_toolbox.data.sourceDBs import *  # Import all sourceDBs to be available for BrudexDataModule


class BrudexRirKwargs(TypedDict):
    """Defines the required keyword arguments for getting a BRUDEX RIR."""

    reverb_condition: str
    doa: int
    microphone_array: str


class BrudexNoiseKwargs(TypedDict):
    """Defines the required keyword arguments for getting a BRUDEX noise."""

    reverb_condition: str
    noise_type: str
    microphone_array: str


# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BRUDEX_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "databases", "brudex"))
BRUDEX_ORIGINAL_SAMPLING_FREQUENCY = 48000  # [Hz]
MICROPHONE_CHANNEL_MAPPING = {
    "BTE_IE": [0, 1, 2, 3, 4, 5],
    "BTE": [0, 1, 2, 3],
    "IE": [4, 5],
    "BTE_front": [0, 2],
    "BTE_rear": [1, 3],
    "BTE_left": [0, 1],
    "BTE_right": [2, 3],
    "BTE_front_IE": [0, 2, 4, 5],
    "BTE_rear_IE": [1, 3, 4, 5],
    "BTE_left_IE_left": [0, 1, 4],
    "BTE_right_IE_right": [2, 3, 5],
    "BTE_left_IE_right": [0, 1, 5],
    "BTE_right_IE_left": [2, 3, 4],
    "BTE_IE_EM": list(range(42)),
    "6_emics_centered": [20, 21, 26, 27, 32, 33],  # [15, 16, 21, 22, 27, 28],
}


@dataclass
class BrudexConfig:
    """Configuration specific to the BRUDEX dataset parameters."""

    reverb_conditions: list[str]
    microphone_arrays: list[str]
    noise_types: tuple[list[str], list[str], list[str]]
    doas: tuple[list[int], list[int], list[int]]


class BrudexRIRsDB(BaseRIRsDB):
    """
    Manages and provides paths to BRUDEX RIR files. Inherits from BaseRIRsDB.
    Handles pre-processing of original .mat files into specialized .pt files.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sampling_frequency: int,
        reverb_conditions: list[str],
        doas: list[int],
        microphone_arrays: list[str],
    ):
        super().__init__(root_dir=root_dir)
        self.sampling_frequency = sampling_frequency
        self._reverb_conditions = reverb_conditions
        self.doas = doas
        self.microphone_arrays = microphone_arrays
        self._mat_rirs = self._scan_for_mat_files()
        print(f"Found {len(self._mat_rirs)} RIR .mat files to consider.")

    @property
    def reverb_conditions(self) -> list[str]:
        """Returns the list of reverb conditions this database was configured with."""
        return self._reverb_conditions

    def _scan_for_mat_files(self) -> dict[tuple[str, int], Path]:
        """Scans for original RIR .mat files."""
        mat_rirs = {}
        rir_root = self.root_dir / "rir"
        for reverb in self.reverb_conditions:
            for doa in self.doas:
                stem = BrudexDatabase.construct_rir_filename_stem(reverb, doa)
                mat_path = rir_root / f"rev_{reverb}" / f"{stem}.mat"
                if mat_path.exists():
                    mat_rirs[(reverb, doa)] = mat_path
        return mat_rirs

    def prepare_data(self):
        """Converts original BRUDEX RIR .mat files into specialized .pt files."""
        print("\nStarting BRUDEX RIR .mat to .pt pre-processing...")
        if not self._mat_rirs:
            print("No RIR .mat files found to pre-process. Skipping.")
            return

        BrudexDatabase._process_mat_files(
            mat_files=set(self._mat_rirs.values()),
            sampling_frequency=self.sampling_frequency,
            microphone_arrays=self.microphone_arrays,
            desc="Pre-processing BRUDEX RIRs",
        )

    # def get_rir_path(
    #     self, reverb_condition: str, doa: int, microphone_array: str
    # ) -> Path:
    #     """Retrieves the path for a specific pre-processed RIR .pt file."""
    #     return BrudexDatabase._get_specialized_path_from_key(
    #         key=(reverb_condition, doa),
    #         mic_array=microphone_array,
    #         mat_map=self._mat_rirs,
    #         sampling_frequency=self.sampling_frequency,
    #         file_type="RIR",
    #     )

    def get_rir(
        self, **kwargs: Unpack[BrudexRirKwargs]
    ) -> torch.Tensor:  # <-- Use the specific type
        """
        Retrieves the path for a specific RIR and loads it into a tensor.
        This method is now type-safe using the BrudexRirKwargs TypedDict.
        """
        # The old try-except block can be removed.
        reverb_condition = kwargs["reverb_condition"]
        doa = kwargs["doa"]
        microphone_array = kwargs["microphone_array"]

        rir_path = BrudexDatabase._get_specialized_path_from_key(
            key=(reverb_condition, doa),
            mic_array=microphone_array,
            mat_map=self._mat_rirs,
            sampling_frequency=self.sampling_frequency,
            file_type="RIR",
        )
        return torch.load(rir_path, map_location="cpu")


class BrudexNoiseDB(BaseNoiseDB):
    """
    Manages and provides paths to BRUDEX noise files. Inherits from BaseNoiseDB.
    Handles pre-processing of original .mat files into specialized .pt files.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sampling_frequency: int,
        reverb_conditions: list[str],
        noise_types: list[str],
        microphone_arrays: list[str],
    ):
        super().__init__(root_dir=root_dir)
        self.sampling_frequency = sampling_frequency
        self.reverb_conditions = reverb_conditions
        self.noise_types = noise_types
        self.microphone_arrays = microphone_arrays
        self._mat_noises = self._scan_for_mat_files()
        print(f"Found {len(self._mat_noises)} noise .mat files to consider.")

    def _scan_for_mat_files(self) -> dict[tuple[str, str], Path]:
        """Scans for original noise .mat files."""
        mat_noises = {}
        noise_root = self.root_dir / "noise"
        for reverb in self.reverb_conditions:
            for noise_type in self.noise_types:
                stem = BrudexDatabase.construct_noise_filename_stem(reverb, noise_type)
                mat_path = noise_root / f"rev_{reverb}" / f"{stem}.mat"
                if mat_path.exists():
                    mat_noises[(reverb, noise_type)] = mat_path
        return mat_noises

    def prepare_data(self):
        """Converts original BRUDEX noise .mat files into specialized .pt files."""
        print("\nStarting BRUDEX noise .mat to .pt pre-processing...")
        if not self._mat_noises:
            print("No noise .mat files found to pre-process. Skipping.")
            return

        BrudexDatabase._process_mat_files(
            mat_files=set(self._mat_noises.values()),
            sampling_frequency=self.sampling_frequency,
            microphone_arrays=self.microphone_arrays,
            desc="Pre-processing BRUDEX Noises",
        )

    # def get_noise_path(
    #     self, reverb_condition: str, noise_type: str, microphone_array: str
    # ) -> Path:
    #     """Retrieves the path for a specific pre-processed noise .pt file."""
    #     return BrudexDatabase._get_specialized_path_from_key(
    #         key=(reverb_condition, noise_type),
    #         mic_array=microphone_array,
    #         mat_map=self._mat_noises,
    #         sampling_frequency=self.sampling_frequency,
    #         file_type="noise",
    #     )

    def get_noise(
        self, **kwargs: Unpack[BrudexNoiseKwargs]
    ) -> torch.Tensor:  # <-- Use the specific type
        """
        Retrieves the path for a specific noise file and loads it into a tensor.
        This method is now type-safe using the BrudexNoiseKwargs TypedDict.
        """
        # The old try-except block can be removed.
        reverb_condition = kwargs["reverb_condition"]
        noise_type = kwargs["noise_type"]
        microphone_array = kwargs["microphone_array"]

        noise_path = BrudexDatabase._get_specialized_path_from_key(
            key=(reverb_condition, noise_type),
            mic_array=microphone_array,
            mat_map=self._mat_noises,
            sampling_frequency=self.sampling_frequency,
            file_type="noise",
        )
        return torch.load(noise_path, map_location="cpu")


class BrudexDatabase:
    """
    A utility class containing static methods shared by BrudexRIRsDB and BrudexNoiseDB.
    This class is no longer instantiated directly. Its methods are used to avoid
    code duplication in the specialized DB classes.
    """

    @staticmethod
    def _process_mat_files(
        mat_files: set[Path],
        sampling_frequency: int,
        microphone_arrays: list[str],
        desc: str,
    ):
        """Generic logic to convert a set of .mat files to specialized .pt files."""
        resampler = torchaudio.transforms.Resample(
            orig_freq=BRUDEX_ORIGINAL_SAMPLING_FREQUENCY,
            new_freq=sampling_frequency,
        )
        converted_count = 0
        for mat_path in tqdm(mat_files, desc=desc):

            tensor_data = None
            for mic_array in microphone_arrays:
                specialized_pt_path = BrudexDatabase._get_specialized_path(
                    mat_path, sampling_frequency, mic_array
                )
                if specialized_pt_path.exists():
                    continue

                if tensor_data is None:
                    tensor_data = torch.from_numpy(
                        mat73.loadmat(mat_path)["data"]
                    ).float()

                try:
                    mic_indices = MICROPHONE_CHANNEL_MAPPING[mic_array]
                    channel_selected_data = tensor_data[:, mic_indices].mT
                    resampled_data = resampler(channel_selected_data)
                    torch.save(resampled_data, specialized_pt_path)
                    converted_count += 1
                except Exception as e:
                    print(
                        f"Error creating specialized file for {mic_array} from {mat_path}: {e}"
                    )
        if converted_count > 0:
            print(f"Created {converted_count} new specialized .pt files.")

    @staticmethod
    def _get_specialized_path(
        mat_path: Path, sampling_frequency: int, mic_array: str
    ) -> Path:
        """Constructs the path for a specialized .pt file from an original .mat path."""
        specialized_filename = f"{mat_path.stem}_{sampling_frequency}_{mic_array}.pt"
        return mat_path.parent / specialized_filename

    @staticmethod
    def _get_specialized_path_from_key(
        key: tuple,
        mic_array: str,
        mat_map: dict,
        sampling_frequency: int,
        file_type: str,
    ) -> Path:
        """Looks up a .mat path by key and constructs the specialized .pt path."""
        original_mat_path = mat_map.get(key)
        if not original_mat_path:
            raise FileNotFoundError(
                f"Original {file_type} .mat for key={key} not found."
            )
        path = BrudexDatabase._get_specialized_path(
            original_mat_path, sampling_frequency, mic_array
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Specialized {file_type} file not found: {path}. Please run prepare_data again."
            )
        return path

    @staticmethod
    def construct_rir_filename_stem(reverb_condition: str, doa: int) -> str:
        """Constructs the standard filename stem for an averaged BRUDEX RIR file."""
        return f"RIR_av_{reverb_condition}_DOA_{doa}deg"

    @staticmethod
    def construct_noise_filename_stem(reverb_condition: str, noise_type: str) -> str:
        """Constructs the standard filename stem for an averaged BRUDEX noise file."""
        return f"noise{noise_type.capitalize()}_av_{reverb_condition}"


class BrudexScenarioGeneratorDataset(BaseScenarioGenerator):
    def __init__(
        self,
        id: str,
        num_scenarios,
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
        # Store only the BRUDEX-SPECIFIC config
        self.brudex_config = brudex_config
        self.train_noise_types, self.val_noise_types, self.test_noise_types = (
            brudex_config.noise_types
        )
        self.train_doas, self.val_doas, self.test_doas = brudex_config.doas

        # Pass all GENERIC parameters to the parent constructor
        super().__init__(
            id=id,
            num_scenarios=num_scenarios,
            transform=transform,
            sampling_frequency=sampling_frequency,
            generation_config=generation_config,  # Pass the config object
            source_dbs=source_dbs,
            rir_db=rir_db,
            noise_db=noise_db,
            seed=seed,
            acc_device=acc_device,
        )

    # --- Implementation of Abstract Hooks ---

    def _sample_scenario_parameters(self) -> dict:
        """Samples BRUDEX-specific parameters for one scenario."""
        assert self.rir_db is not None, "..."

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

    def _get_sources_for_scenario(self, scenario_params: dict) -> list[dict]:
        """Selects Librispeech clips and assigns DOAs."""
        num_sources = scenario_params["num_sources"]

        # CHANGED: Get clips from the source_dbs
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

        # This logic is from your old _assign_sources method
        sources = []
        for clip_paths, doa in zip(speech_clips, doas):
            # This logic is from your old _assign_sources method
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

    # def _load_clean_speech_for_source(
    #     self, source_info: dict, required_samples: int
    # ) -> torch.Tensor:
    #     """Loads and concatenates audio files from Librispeech."""
    #     # This logic is from your old _load_clean_speeches method
    #     concatenated_audio = []
    #     current_length = 0

    #     # Ensure we have enough audio by repeating the clip if necessary
    #     all_paths = source_info["clean_speech_paths"]
    #     while current_length < required_samples:
    #         for file_path in all_paths:
    #             if current_length >= required_samples:
    #                 break
    #             audio_tensor, _ = load_audio(
    #                 file_path, sampling_frequency=self.sampling_frequency
    #             )
    #             if self.config.remove_silence and audio_tensor.numel() > 0:
    #                 vad_mask = vad_opt(
    #                     audio_tensor.unsqueeze(0),
    #                     self.sampling_frequency,
    #                     thr=-30,
    #                     min_on=self.config.neglect_silence_duration,
    #                 ).squeeze()
    #                 audio_tensor = audio_tensor[:, vad_mask]

    #             if audio_tensor.numel() > 0:
    #                 concatenated_audio.append(audio_tensor.squeeze(0))
    #                 current_length += audio_tensor.shape[-1]

    #     full_speech_signal = torch.cat(concatenated_audio)
    #     return full_speech_signal[:required_samples]


# @dataclass
# class BrudexSource:

#     def __init__(self, id: str, clean_speech: list[Path], doa: int):
#         self.id = id
#         self.clean_speech = clean_speech
#         self.doa = doa

#     def __eq__(self, other):
#         """Two sources are equal if their IDs are the same."""
#         if not isinstance(other, BrudexSource):
#             return NotImplemented
#         return self.id == other.id

#     def __hash__(self):
#         """Make the source hashable based on its unique ID."""
#         return hash(self.id)


# @dataclass
# class BrudexScenario:

#     def __init__(
#         self,
#         signal_length: float,
#         reverb_condition: str,
#         microphone_array: str,
#         noise_type: str,
#         snr: float,
#         source_activity_pattern: list[dict[str, Any]],
#     ):
#         self.signal_length = signal_length
#         self.reverb_condition = reverb_condition
#         self.microphone_array = microphone_array
#         self.noise_type = noise_type
#         self.snr = snr
#         self.source_activity_pattern = source_activity_pattern


class BrudexDataModule(BaseDataModule):
    def __init__(
        self,
        id: str,
        transform: STFTtransform,
        batch_size: int,
        num_workers: int,
        num_scenarios: list[int],
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        brudex_config: BrudexConfig,
        clean_speech_databases: list[str],
        seed: int | None,
        reset: bool,
        acc_device: torch.device = torch.device("cpu"),
        feature_extractor: BaseFeatureExtractor | None = None,
        force_load_stft: bool = False,
    ):
        # --- Step 1: Call parent constructor and perform simple assignments ---
        super().__init__(
            project_root=PROJECT_ROOT,
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
        self.clean_speech_databases_names = clean_speech_databases

        # --- Step 2: Delegate complex initialization to helper methods ---
        self._init_clean_speech_dbs()
        self._init_brudex_dbs()

        # --- Step 3: Final logging call ---
        self._verbose_parameters()

    def _init_clean_speech_dbs(self):
        """Dynamically finds and instantiates clean speech database managers."""
        self.clean_speech_dbs = {}
        database_root = os.path.join(PROJECT_ROOT, "databases")

        for db_name in self.clean_speech_databases_names:
            # Find the class object by its string name in the current module's scope
            db_class = globals().get(db_name + "Database", None)

            if db_class and isinstance(db_class, type):
                print(f"Found implementation for clean speech database: {db_name}")
                # This is where you would handle specific arguments for each class.
                self.clean_speech_dbs[db_name] = db_class(
                    root_dir=database_root,
                    signal_lengths=self.generation_config.signal_lengths,
                )
            else:
                print(
                    f"Warning: No class implementation named '{db_name}Database' found in brudex.py. It will be ignored."
                )

    def _init_brudex_dbs(self):
        """Initializes the BRUDEX-specific RIR and Noise database managers."""
        # Create a combined list of all DOAs and noise types needed across all splits
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

        self.rir_db = BrudexRIRsDB(
            root_dir=BRUDEX_PATH,
            sampling_frequency=self.sampling_frequency,
            reverb_conditions=self.brudex_config.reverb_conditions,
            doas=all_doas,
            microphone_arrays=self.brudex_config.microphone_arrays,
        )
        self.noise_db = BrudexNoiseDB(
            root_dir=BRUDEX_PATH,
            sampling_frequency=self.sampling_frequency,
            reverb_conditions=self.brudex_config.reverb_conditions,
            noise_types=all_noise_types,
            microphone_arrays=self.brudex_config.microphone_arrays,
        )

    def _verbose_parameters(self, indent: str = ""):
        print(f"{indent}Brudex Data Module Parameters:")
        print(f"{indent}  Batch Size: {self.batch_size}")
        print(f"{indent}  Num Workers: {self.num_workers}")
        print(f"{indent}  Num Scenarios: {self.num_scenarios}")
        self.transform._verbose_parameters(indent=indent + "  ")
        print(f"{indent}  Sampling Frequency: {self.sampling_frequency} [Hz]")
        print(f"{indent}  Max Sources: {self.generation_config.max_sources}")
        print(f"{indent}  Signal Lengths: {self.generation_config.signal_lengths} [s]")
        print(
            f"{indent}  Initial Noise Only Duration: {self.generation_config.initial_noise_only_duration} [s]"
        )
        print(f"{indent}  Reverb Times: {self.brudex_config.reverb_conditions} [ms]")
        print(f"{indent}  Microphone Arrays: {self.brudex_config.microphone_arrays}")
        print(f"{indent}  Train Noise Types: {self.brudex_config.noise_types[0]}")
        print(f"{indent}  Val Noise Types: {self.brudex_config.noise_types[1]}")
        print(f"{indent}  Test Noise Types: {self.brudex_config.noise_types[2]}")
        print(f"{indent}  SNRs: {self.generation_config.snrs} [dB]")
        print(f"{indent}  Clean Speech Databases: {self.clean_speech_databases_names}")
        print(f"{indent}  Train DOAs: {self.brudex_config.doas[0]} [°]")
        print(f"{indent}  Val DOAs: {self.brudex_config.doas[1]} [°]")
        print(f"{indent}  Test DOAs: {self.brudex_config.doas[2]} [°]")
        print(
            f"{indent}  Time Between Events: {self.generation_config.time_between_events} [s]"
        )
        print(f"{indent}  Activations Only: {self.generation_config.activations_only}")
        print(f"{indent}  Seed: {self.seed}")

    def _get_database_managers(self) -> list:
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


if __name__ == "__main__":
    # This block is for debugging and testing the BrudexDataModule directly.
    print("Running brudex.py in standalone debug mode...")

    # --- Add this section to teach PyYAML about custom tags like !range ---
    def tuple_constructor(loader: yaml.SafeLoader, node: yaml.nodes.Node) -> tuple:
        """Constructs a tuple, ensuring the node is a sequence."""
        if not isinstance(node, yaml.nodes.SequenceNode):
            raise yaml.constructor.ConstructorError(
                "expected a sequence node, but found %s" % type(node)
            )
        return tuple(loader.construct_sequence(node))

    yaml.add_constructor("!tuple", tuple_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!range", tuple_constructor, Loader=yaml.SafeLoader)
    # --- End of section ---

    # Load a specific debug/test config file
    config_path = Path(__file__).parent.parent / "configs/datasets/brudex.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # You can override parameters for a quick test
    config["id"] = "brudex_debug_test"  # Unique ID for this test run
    config["reset"] = True  #
    config["num_scenarios"] = [4, 2, 2]  # Use a small number for a quick run
    config["precompute_mode"] = "features"  # Use 'features' to test this class
    config["sampling_frequency"] = 8000  # Set the sampling frequency for the test

    print("Initializing BrudexDataModule with config:")
    print(config)

    generation_config = ScenarioGenerationConfig(
        max_sources=config["max_sources"],
        signal_lengths=config["signal_lengths"],
        initial_noise_only_duration=config["initial_noise_only_duration"],
        snrs=config["snrs"],
        source_power_range=config["source_power_range"],
        time_between_events=config["time_between_events"],
        fix_seglen=config["fix_seglen"],
        activations_only=config["activations_only"],
        remove_silence=config["remove_silence"],
        neglect_silence4oracle_sa=config["neglect_silence4oracle_sa"],
        bridge_clean_speech_gaps=config["bridge_clean_speech_gaps"],
        vad_threshold2define_oracle=config["vad_threshold2define_oracle"],
        vad_threshold2select_clean_speech=config["vad_threshold2select_clean_speech"],
    )

    brudex_config = BrudexConfig(
        reverb_conditions=config["reverb_conditions"],
        microphone_arrays=config["microphone_arrays"],
        noise_types=config["noise_types"],
        doas=config["doas"],
    )

    # For debugging, we need a transform. Let's create a default one.
    # In a real run, this would be defined by the algorithm config.
    debug_transform = STFTtransform(
        frame_length=0.2,
        frame_shift=0.05,
        sampling_frequency=config["sampling_frequency"],
    )

    # Instantiate the datamodule with parameters from the config file
    dm = BrudexDataModule(
        id=config["id"],
        transform=debug_transform,
        batch_size=2,
        num_workers=0,
        num_scenarios=config["num_scenarios"],
        sampling_frequency=config["sampling_frequency"],
        generation_config=generation_config,  # Pass object
        brudex_config=brudex_config,  # Pass object
        clean_speech_databases=config["clean_speech_databases"],
        seed=config["seed"],
        reset=config["reset"],
        acc_device=torch.device("cpu"),
        feature_extractor=None,
    )

    # Run the preparation step
    print("\nRunning prepare_data()...")
    dm.prepare_data()

    # Setup the datasets
    print("\nRunning setup()...")
    dm.setup(stage="fit")

    # Get one batch to verify it works
    print("\nFetching one batch from the training dataloader...")
    train_loader = dm.train_dataloader()
    first_batch = next(iter(train_loader))

    print("\nSuccessfully fetched one batch.")
    if config["precompute_mode"] == "features":
        print("Input keys:", first_batch["input"][0].keys())
        print("WGMSC shape:", first_batch["input"][0]["wgmsc_narrowband"].shape)
    else:
        print("Input shape:", first_batch["input"][0].shape)

    print("Meta keys:", first_batch["meta"].keys())
    print("Scenario ID of first item:", first_batch["meta"]["scenario_id"][0])
