import logging
import random
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Union

import torchaudio
from tqdm import tqdm

from muse_toolbox.data.databases.base_DBs import BaseSourceDB

log = logging.getLogger(__name__)


class DemandDatabase(BaseSourceDB):
    """Manages the DEMAND (Diverse Environments Multichannel Acoustic Noise Database).

    Handles downloading from Zenodo, extraction, and file management.
    """

    ZENODO_RECORD_URL = "https://zenodo.org/records/1227121/files/"

    ALL_ENVIRONMENTS = [
        "DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER",
        "OHALLWAY", "OMEETING", "OOFFICE", "PCAFETER", "PRESTO", "PSTATION",
        "SCAFE", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO",
    ]

    def __init__(
        self,
        root_dir: Union[str, Path],
        sampling_rate: int = 48000,
        train_envs: Optional[List[str]] = None,
        val_envs: Optional[List[str]] = None,
        test_envs: Optional[List[str]] = None,
    ) -> None:
        """Initializes the DemandDatabase manager.

        Args:
            root_dir (Union[str, Path]): Path to the databases root directory.
            sampling_rate (int): Target sampling rate. Determines whether the 16k 
                or 48k version is downloaded. Defaults to 48000.
            train_envs (Optional[List[str]]): Environments for the train split.
            val_envs (Optional[List[str]]): Environments for the validation split.
            test_envs (Optional[List[str]]): Environments for the test split.
        """
        super().__init__(root_dir=Path(root_dir) / "demand", signal_lengths=0)

        self.download_rate_k = 16 if sampling_rate <= 16000 else 48
        log.info(f"DemandDatabase will download the {self.download_rate_k}kHz version.")

        self.split_envs = {
            "train": train_envs if train_envs is not None else self.ALL_ENVIRONMENTS,
            "val": val_envs if val_envs is not None else [],
            "test": test_envs if test_envs is not None else [],
        }

        log.info(f"DemandDatabase initialized for {self.download_rate_k}kHz.")
        for split, envs in self.split_envs.items():
            log.info(f"  - {split} split will use {len(envs)} environments.")

        self.noise_files_per_split = defaultdict(list)

    def prepare_data(self) -> None:
        """Prepares the DEMAND dataset.
        
        Ensures the DEMAND dataset is downloaded, extracted, and then scans
        files into train/val/test splits based on the configured environments.
        """
        log.info("Checking for DEMAND dataset...")
        self.root_dir.mkdir(parents=True, exist_ok=True)

        all_envs_to_download = set(sum(self.split_envs.values(), []))

        for env in tqdm(list(all_envs_to_download), desc="Processing DEMAND environments"):
            sr_dir = self.root_dir / f"Sr{self.download_rate_k}k"
            sr_dir.mkdir(exist_ok=True)
            env_dir = sr_dir / env

            if env_dir.exists() and any(env_dir.iterdir()):
                continue

            log.info(f"Environment '{env}' not found. Downloading...")
            zip_filename = f"{env}_{self.download_rate_k}k.zip"
            download_url = f"{self.ZENODO_RECORD_URL}{zip_filename}?download=1"
            zip_path = self.root_dir / zip_filename

            try:
                subprocess.run(
                    ["wget", download_url, "-O", str(zip_path)],
                    check=True,
                    capture_output=True,
                )

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(sr_dir)

                log.info(f"Successfully downloaded and extracted '{env}'.")

            except subprocess.CalledProcessError as e:
                if self.download_rate_k == 16:
                    log.warning(
                        f"Could not download 16k version for '{env}'. "
                        "Attempting to download 48k version and resample."
                    )
                    zip_filename_48k = f"{env}_48k.zip"
                    download_url_48k = f"{self.ZENODO_RECORD_URL}{zip_filename_48k}?download=1"
                    zip_path_48k = self.root_dir / zip_filename_48k
                    try:
                        subprocess.run(
                            ["wget", download_url_48k, "-O", str(zip_path_48k)],
                            check=True,
                            capture_output=True,
                        )

                        with zipfile.ZipFile(zip_path_48k, "r") as zip_ref:
                            zip_ref.extractall(sr_dir)

                        log.info(f"Resampling '{env}' files from 48k to 16k...")
                        wav_files = list(env_dir.glob("*.wav"))
                        for wav_file in wav_files:
                            waveform, sr = torchaudio.load(wav_file)
                            resampled_waveform = torchaudio.functional.resample(
                                waveform, orig_freq=sr, new_freq=16000
                            )
                            torchaudio.save(wav_file, resampled_waveform, 16000)

                        log.info(f"Successfully downloaded and resampled '{env}'.")

                    except Exception as fallback_e:
                        log.error(f"Error during 48k fallback for {env}: {fallback_e}")
                        continue
                    finally:
                        if "zip_path_48k" in locals() and zip_path_48k.exists():
                            zip_path_48k.unlink()
                else:
                    log.error(f"Error downloading {env}: {e.stderr.decode()}")
                    log.error("Please check your internet connection or the Zenodo URL.")
                    continue
            finally:
                if zip_path.exists():
                    zip_path.unlink()

        log.info("Scanning DEMAND files into splits...")
        for split, envs in self.split_envs.items():
            split_files = []
            for env in envs:
                env_dir = self.root_dir / f"Sr{self.download_rate_k}k" / env
                if env_dir.exists():
                    split_files.extend(
                        [f for f in env_dir.rglob("*.wav") if f.is_file()]
                    )
            self.noise_files_per_split[split] = split_files
            log.info(f"Found {len(split_files)} files for split: {split}")

    def get_speaker_clips(self, split: Optional[str] = None) -> Dict[str, List[List[Path]]]:
        """Returns a dictionary of noise files grouped for the given split.

        Args:
            split (Optional[str]): Generic split name ('train', 'val', 'test').

        Returns:
            Dict[str, List[List[Path]]]: Dictionary mapping noise IDs to lists of clips.

        Raises:
            ValueError: If the split is not defined or has no files.
            RuntimeError: If no noise files are loaded for the split.
        """
        if split not in self.noise_files_per_split:
            raise ValueError(f"Split '{split}' not defined or has no files.")

        file_list = self.noise_files_per_split[split]
        if not file_list:
            raise RuntimeError(
                f"No noise files loaded for split '{split}'. "
                f"Please run prepare_data() on the DataModule first."
            )

        noise_clips = {}
        for file_path in file_list:
            speaker_id = f"{file_path.parent.name}_{file_path.stem}"
            noise_clips[speaker_id] = [[file_path]]

        return noise_clips

    def get_noise_file_path(self, split: Optional[str] = None) -> Path:
        """Returns a random noise file path from the specified split.

        Args:
            split (Optional[str]): Generic split name ('train', 'val', 'test').

        Returns:
            Path: Path to a random noise file.

        Raises:
            ValueError: If the split is not defined or has no files.
            RuntimeError: If no noise files are loaded for the split.
        """
        if split not in self.noise_files_per_split:
            raise ValueError(f"Split '{split}' not defined or has no files.")

        file_list = self.noise_files_per_split[split]
        if not file_list:
            raise RuntimeError(
                f"No noise files loaded for split '{split}'. "
                f"Please run prepare_data() on the DataModule first."
            )
        return random.choice(file_list)
