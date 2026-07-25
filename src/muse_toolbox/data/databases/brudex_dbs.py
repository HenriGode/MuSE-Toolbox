import logging
import os
from pathlib import Path
from typing import TypedDict, Unpack

import mat73
import torch
import torchaudio
from tqdm import tqdm

from muse_toolbox.data.databases.base_DBs import BaseNoiseDB, BaseRIRsDB

log = logging.getLogger(__name__)


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
        log.info(f"Found {len(self._mat_rirs)} RIR .mat files to consider.")

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
        log.info("Starting BRUDEX RIR .mat to .pt pre-processing...")
        if not self._mat_rirs:
            log.info("No RIR .mat files found to pre-process. Skipping.")
            return

        BrudexDatabase._process_mat_files(
            mat_files=set(self._mat_rirs.values()),
            sampling_frequency=self.sampling_frequency,
            microphone_arrays=self.microphone_arrays,
            desc="Pre-processing BRUDEX RIRs",
        )

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
        log.info(f"Found {len(self._mat_noises)} noise .mat files to consider.")

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
        log.info("Starting BRUDEX noise .mat to .pt pre-processing...")
        if not self._mat_noises:
            log.info("No noise .mat files found to pre-process. Skipping.")
            return

        BrudexDatabase._process_mat_files(
            mat_files=set(self._mat_noises.values()),
            sampling_frequency=self.sampling_frequency,
            microphone_arrays=self.microphone_arrays,
            desc="Pre-processing BRUDEX Noises",
        )

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
                    log.error(
                        f"Error creating specialized file for {mic_array} from {mat_path}: {e}"
                    )
        if converted_count > 0:
            log.info(f"Created {converted_count} new specialized .pt files.")

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
