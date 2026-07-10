import torch
import torchaudio
from tqdm import tqdm
from muse_toolbox.utils import *
from pathlib import Path
from collections import defaultdict

# Import the new base classes
from .base_dataset import BaseSourceDB
import subprocess
import zipfile


class TIMITDatabase:
    SPLIT_NAME_MAPPING = {
        "train": "train",
        "val": "val",
        "test": "test",
    }
    pass


class VCTKDatabase:
    SPLIT_NAME_MAPPING = {
        "train": "train",
        "val": "val",
        "test": "test",
    }
    pass


class LibrispeechDatabase(BaseSourceDB):
    """
    Manages downloading and pre-processing of the LibriSpeech dataset.
    Handles the pre-computation of 'virtual clips' by grouping consecutive utterances.
    """

    SPLIT_NAME_MAPPING = {
        "train": "train-clean-360",
        "val": "dev-clean",
        "test": "test-clean",
    }

    def __init__(
        self,
        root_dir: str | Path,
        signal_lengths: float,
        splits: list[str] | None = None,  # <-- Changed from splits_to_use to splits
    ):
        # Pass the root_dir to the parent BaseDB constructor
        super().__init__(
            root_dir=Path(root_dir) / "librispeech", signal_lengths=signal_lengths
        )

        # If specific splits are provided from the config, use them.
        # Otherwise, fall back to the default list for backward compatibility.
        if splits is not None:
            self.splits_to_use = splits
        else:
            self.splits_to_use = ["train-clean-360", "dev-clean", "test-clean"]

        print(f"LibrispeechDatabase initialized to use splits: {self.splits_to_use}")

        # # store speaker clips cache
        # self.speaker_clips_cache = {
        #     "train": self.get_speaker_clips("train"),
        #     "val": self.get_speaker_clips("val"),
        #     "test": self.get_speaker_clips("test"),
        # }

    def download(self):
        """Downloads the required LibriSpeech splits if they don't exist."""
        print("\nChecking for LibriSpeech data...")
        for split in self.splits_to_use:
            if not (self.root_dir / split).exists():
                url = split
                if not url:
                    print(
                        f"Warning: Unknown LibriSpeech split '{split}'. Cannot download."
                    )
                    continue

                print(f"Split '{split}' not found. Downloading...")
                torchaudio.datasets.LIBRISPEECH(
                    root=self.root_dir.parent, url=url, download=True
                )
                print(f"'{split}' download complete.")
        print("LibriSpeech data is ready.")

    def prepare_data(self):
        """
        Ensures that the data is downloaded and that the pre-computed speaker
        clip cache files exist for all required splits.
        """
        # Step 1: Ensure data is downloaded
        self.download()

        # Step 2: Create clip caches
        print("\nChecking/Creating Librispeech clip caches...")
        for split in self.splits_to_use:
            split_path = self.root_dir / split
            if not split_path.exists():
                print(
                    f"Warning: LibriSpeech split '{split}' not found at {split_path}. Skipping clip creation."
                )
                continue
            self._create_speaker_clips_cache(split_path)

    def _create_speaker_clips_cache(self, split_path: Path):
        """
        Walks the LibriSpeech directory structure to create clips of consecutive
        utterances and caches the result to a .pt file.
        """
        cache_file = split_path / "clean_speech_clips.pt"
        if cache_file.exists():
            print(f"Clip cache already exists for: {split_path.name}")
            return

        print(f"Creating clip cache for: {split_path.name}...")
        speaker_clips = defaultdict(list)
        target_duration = 2 * (
            max(self.signal_lengths)
            if isinstance(self.signal_lengths, (list, tuple))
            else self.signal_lengths
        )

        for speaker_path in tqdm(
            split_path.iterdir(), desc=f"Processing {split_path.name}"
        ):
            if not speaker_path.is_dir():
                continue
            for chapter_path in speaker_path.iterdir():
                if not chapter_path.is_dir():
                    continue
                utterance_files = sorted(chapter_path.glob("*.flac"))
                if not utterance_files:
                    continue

                current_clip = []
                current_duration = 0.0
                last_utterance_id = -1
                for i, file_path in enumerate(utterance_files):
                    utterance_id = int(file_path.stem.split("-")[-1])
                    is_consecutive = (last_utterance_id != -1) and (
                        utterance_id == last_utterance_id + 1
                    )
                    if not is_consecutive and i > 0:
                        current_clip = []
                        current_duration = 0.0

                    current_clip.append(file_path)
                    info = torchaudio.info(file_path)
                    current_duration += info.num_frames / info.sample_rate
                    last_utterance_id = utterance_id

                    if current_duration >= target_duration:
                        speaker_clips[speaker_path.name].append(current_clip)
                        current_clip = []
                        current_duration = 0.0
                        last_utterance_id = -1

        total_clips = sum(len(clips) for clips in speaker_clips.values())
        print(f"Created {total_clips} virtual clips for {len(speaker_clips)} speakers.")
        print(f"Saving clips to cache file: {cache_file}")
        torch.save(dict(speaker_clips), cache_file)

    def get_speaker_clips(self, split: str) -> dict[str, list[list[Path]]]:
        """
        Loads speaker clips for a given generic split ('train', 'val', 'test')
        from the pre-computed cache file.
        """
        # Translate the generic split name to the internal directory name.
        internal_split_name = self.SPLIT_NAME_MAPPING.get(split)
        if not internal_split_name:
            raise ValueError(
                f"Split '{split}' is not a valid generic split for Librispeech. "
                f"Use one of {list(self.SPLIT_NAME_MAPPING.keys())}."
            )

        split_path = self.root_dir / internal_split_name
        cache_file = split_path / "clean_speech_clips.pt"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Clip cache not found at {cache_file}. Please run prepare_data on the DataModule first."
            )

        print(f"Loading cached clips from: {cache_file}")
        speaker_clips = torch.load(cache_file, weights_only=False)
        total_clips = sum(len(clips) for clips in speaker_clips.values())
        print(
            f"Loaded {total_clips} virtual clips for {len(speaker_clips)} speakers from cache."
        )
        return speaker_clips


class DemandDatabase(BaseSourceDB):
    """
    Manages the DEMAND (Diverse Environments Multichannel Acoustic Noise Database).
    Handles downloading from Zenodo, extraction, and file management.
    """

    # Base URL for the Zenodo record
    ZENODO_RECORD_URL = "https://zenodo.org/records/1227121/files/"

    # List of all environments in the DEMAND dataset
    ALL_ENVIRONMENTS = [
        "DKITCHEN",
        "DLIVING",
        "DWASHING",
        "NFIELD",
        "NPARK",
        "NRIVER",
        "OHALLWAY",
        "OMEETING",
        "OOFFICE",
        "PCAFETER",
        "PRESTO",
        "PSTATION",
        "SCAFE",
        "SPSQUARE",
        "STRAFFIC",
        "TBUS",
        "TCAR",
        "TMETRO",
    ]

    def __init__(
        self,
        root_dir: str | Path,
        sampling_rate: int = 48000,
        train_envs: list[str] | None = None,
        val_envs: list[str] | None = None,
        test_envs: list[str] | None = None,
    ):
        # The root_dir for this DB is the 'demand' folder
        super().__init__(root_dir=Path(root_dir) / "demand", signal_lengths=0)

        # Determine which version of DEMAND to download (16k or 48k).
        # If the target rate is <= 16k, use the 16k version. Otherwise, use 48k.
        self.download_rate_k = 16 if sampling_rate <= 16000 else 48
        print(f"DemandDatabase will download the {self.download_rate_k}kHz version.")

        # Store the environments for each split
        self.split_envs = {
            "train": train_envs if train_envs is not None else self.ALL_ENVIRONMENTS,
            "val": val_envs if val_envs is not None else [],
            "test": test_envs if test_envs is not None else [],
        }

        print(f"DemandDatabase initialized for {self.download_rate_k}kHz.")
        for split, envs in self.split_envs.items():
            print(f"  - {split} split will use {len(envs)} environments.")

        # This will hold the paths to noise files, organized by split
        self.noise_files_per_split = defaultdict(list)

    def prepare_data(self):
        """
        Ensures the DEMAND dataset is downloaded, extracted, and then scans
        files into train/val/test splits based on the configured environments.
        Each environment is stored in a sample-rate-specific directory,
        e.g., 'databases/demand/DKITCHEN_16k/'.
        """
        print("\nChecking for DEMAND dataset...")
        self.root_dir.mkdir(parents=True, exist_ok=True)

        all_envs_to_download = set(sum(self.split_envs.values(), []))

        for env in tqdm(all_envs_to_download, desc="Processing DEMAND environments"):
            # Define the sample-rate-specific directory for the environment
            sr_dir = self.root_dir / f"Sr{self.download_rate_k}k"
            sr_dir.mkdir(exist_ok=True)
            env_dir = sr_dir / env

            # Check if the directory already exists and has files in it
            if env_dir.exists() and any(env_dir.iterdir()):
                continue

            # --- Download and Extract ---
            print(f"\nEnvironment '{env}' not found. Downloading...")
            zip_filename = f"{env}_{self.download_rate_k}k.zip"
            download_url = f"{self.ZENODO_RECORD_URL}{zip_filename}?download=1"
            zip_path = self.root_dir / zip_filename

            try:
                # Use wget to download
                subprocess.run(
                    ["wget", download_url, "-O", str(zip_path)],
                    check=True,
                    capture_output=True,
                )

                # Unzip into the specific environment folder
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(sr_dir)

                print(f"Successfully downloaded and extracted '{env}'.")

            except subprocess.CalledProcessError as e:
                # --- Fallback logic for missing 16k versions ---
                if self.download_rate_k == 16:
                    print(
                        f"Warning: Could not download 16k version for '{env}'. "
                        f"Attempting to download 48k version and resample."
                    )
                    zip_filename_48k = f"{env}_48k.zip"
                    download_url_48k = (
                        f"{self.ZENODO_RECORD_URL}{zip_filename_48k}?download=1"
                    )
                    zip_path_48k = self.root_dir / zip_filename_48k
                    try:
                        # Attempt to download the 48k version
                        subprocess.run(
                            ["wget", download_url_48k, "-O", str(zip_path_48k)],
                            check=True,
                            capture_output=True,
                        )

                        # Unzip the 48k files into the target 16k directory
                        with zipfile.ZipFile(zip_path_48k, "r") as zip_ref:
                            zip_ref.extractall(sr_dir)

                        # Resample all extracted .wav files from 48k to 16k in place
                        print(f"Resampling '{env}' files from 48k to 16k...")
                        wav_files = list(env_dir.glob("*.wav"))
                        for wav_file in wav_files:
                            waveform, sr = torchaudio.load(wav_file)
                            resampled_waveform = torchaudio.functional.resample(
                                waveform, orig_freq=sr, new_freq=16000
                            )
                            # Overwrite the original file with the resampled version
                            torchaudio.save(wav_file, resampled_waveform, 16000)

                        print(f"Successfully downloaded and resampled '{env}'.")

                    except Exception as fallback_e:
                        print(f"Error during 48k fallback for {env}: {fallback_e}")
                        continue
                    finally:
                        if "zip_path_48k" in locals() and zip_path_48k.exists():
                            zip_path_48k.unlink()
                else:
                    # If it wasn't a 16k download that failed, report the original error
                    print(f"Error downloading {env}: {e.stderr.decode()}")
                    print("Please check your internet connection or the Zenodo URL.")
                    continue
            finally:
                if zip_path.exists():
                    zip_path.unlink()

        # --- Scan files into splits ---
        print("\nScanning DEMAND files into splits...")
        for split, envs in self.split_envs.items():
            split_files = []
            for env in envs:
                # Point to the correct sample-rate-specific directory
                env_dir = self.root_dir / f"Sr{self.download_rate_k}k" / env
                if env_dir.exists():
                    split_files.extend(
                        [f for f in env_dir.rglob("*.wav") if f.is_file()]
                    )
            self.noise_files_per_split[split] = split_files
            print(f"Found {len(split_files)} files for split: {split}")

    def get_speaker_clips(
        self, split: str | None = None
    ) -> dict[str, list[list[Path]]]:
        """
        Implements the abstract method from BaseSourceDB.
        Returns a dictionary of "noises" (files) for the given split.
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

    def get_noise_file_path(self, split: str | None = None) -> Path:
        """
        Returns a random noise file path from the specified split.
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


class MyNoiseDB(BaseSourceDB):
    """
    Manages a database of single-channel noise files organized into splits.
    This class scans for .wav files and provides a way to retrieve a random
    noise file path for a given split.
    """

    SPLIT_NAME_MAPPING = {
        "train": "train",
        "val": "val",
        "test": "test",
    }

    def __init__(
        self,
        root_dir: str | Path,
        signal_lengths: float,
        splits: list[str] | None = None,
    ):
        # The root_dir for this DB is the 'mynoiseDB' folder
        super().__init__(
            root_dir=Path(root_dir) / "mynoiseDB", signal_lengths=signal_lengths
        )

        # If specific splits are provided from the config, use them.
        # Otherwise, fall back to a default list.
        if splits is not None:
            self.splits_to_use = splits
        else:
            self.splits_to_use = ["train", "val", "test"]

        print(f"NoiseDB initialized to use splits: {self.splits_to_use}")

        # This will hold the paths to the noise files for each split
        self.noise_files = defaultdict(list)

    def download(self):
        """No download functionality is needed for the local NoiseDB."""
        print("\nNoiseDB uses local files. No download is necessary.")
        pass

    def prepare_data(self):
        """
        Scans the specified split directories for .wav files and populates
        the internal list of available noise files.
        """
        print("\nScanning for noise files...")
        for split in self.splits_to_use:
            split_path = self.root_dir / split
            if not split_path.exists():
                print(
                    f"Warning: NoiseDB split directory '{split}' not found at {split_path}. Skipping."
                )
                continue

            # Scan for all .wav files in the directory
            files = list(split_path.glob("*.wav"))
            if not files:
                print(f"Warning: No .wav files found in {split_path}.")
                continue

            self.noise_files[split] = files
            print(f"Found {len(files)} noise files for split: {split}")

    def get_noise_file_path(self, split: str) -> Path:
        """
        Returns a random noise file path from the specified split.
        """
        # Translate the generic split name to the internal directory name.
        internal_split_name = self.SPLIT_NAME_MAPPING.get(split)
        if not internal_split_name:
            raise ValueError(
                f"Split '{split}' is not a valid generic split for NoiseDB. "
                f"Use one of {list(self.SPLIT_NAME_MAPPING.keys())}."
            )

        if not self.noise_files[internal_split_name]:
            raise RuntimeError(
                f"No noise files loaded for split '{internal_split_name}'. "
                f"Please run prepare_data() on the DataModule first."
            )

        # Return a random file path from the list for the given split
        return random.choice(self.noise_files[internal_split_name])


if __name__ == "__main__":
    pass
    # This block allows for direct testing of the database classes in this file.

    # def test_demand_database():
    #     """
    #     Tests the DemandDatabase class by initializing it, preparing the data
    #     (downloading if necessary), and fetching a few random noise files.
    #     This test uses the project's actual database directory.
    #     """
    #     print("--- Testing DemandDatabase ---")

    #     # Define the root directory for the project's databases.
    #     # The script is in 'src/datasets/', so we go up two levels to the project root.
    #     project_root = Path(__file__).parent.parent.parent
    #     db_root = project_root / "databases"

    #     print(f"Using project database root: {db_root.resolve()}")

    #     try:
    #         # 1. Initialize the database, pointing to the correct root.
    #         # The DemandDatabase class will append 'demand' to this path.
    #         # We use 16kHz for a faster download during testing.
    #         demand_db = DemandDatabase(root_dir=db_root, sampling_rate=16000)

    #         # 2. Prepare the data. This will download to the real database folder.
    #         demand_db.prepare_data()

    #         # 3. Test fetching noise files if preparation was successful.
    #         if demand_db.noise_files:
    #             print("\nFetching 5 random noise files from the prepared database:")
    #             for i in range(5):
    #                 random_file = demand_db.get_noise_file_path()
    #                 # Print the path relative to the database root for cleaner output
    #                 try:
    #                     relative_path = random_file.relative_to(db_root)
    #                     print(f"  {i+1}: {relative_path}")
    #                 except ValueError:
    #                     print(f"  {i+1}: {random_file}")
    #         else:
    #             print("\nCould not fetch noise files. The noise_files list is empty.")
    #             print(
    #                 "This might be due to a download error or if no .wav files were found."
    #             )

    #     except Exception as e:
    #         print(f"\nAn error occurred during the DemandDatabase test: {e}")

    #     print("\n--- Test Complete ---")

    # # Run the test function
    # test_demand_database()
