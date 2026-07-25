import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Union, Any

from muse_toolbox.data.databases.base_DBs import BaseSourceDB

log = logging.getLogger(__name__)


class LocalNoiseDB(BaseSourceDB):
    """Manages a database of single-channel noise files organized into splits.
    
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
        root_dir: Union[str, Path],
        signal_lengths: float,
        splits: Optional[List[str]] = None,
    ) -> None:
        """Initializes the local noise database manager.

        Args:
            root_dir (Union[str, Path]): Path to the databases root directory.
            signal_lengths (float): Default signal length.
            splits (Optional[List[str]]): List of splits to scan.
        """
        super().__init__(
            root_dir=Path(root_dir) / "mynoiseDB", signal_lengths=signal_lengths
        )

        if splits is not None:
            self.splits_to_use = splits
        else:
            self.splits_to_use = ["train", "val", "test"]

        log.info(f"NoiseDB initialized to use splits: {self.splits_to_use}")
        self.noise_files = defaultdict(list)

    def download(self) -> None:
        """No download functionality is needed for the local NoiseDB."""
        log.info("NoiseDB uses local files. No download is necessary.")
        pass

    def prepare_data(self) -> None:
        """Scans the specified split directories for .wav files."""
        log.info("Scanning for noise files...")
        for split in self.splits_to_use:
            split_path = self.root_dir / split
            if not split_path.exists():
                log.warning(f"NoiseDB split directory '{split}' not found at {split_path}. Skipping.")
                continue

            files = list(split_path.glob("*.wav"))
            if not files:
                log.warning(f"No .wav files found in {split_path}.")
                continue

            self.noise_files[split] = files
            log.info(f"Found {len(files)} noise files for split: {split}")

    def get_speaker_clips(self, split: str) -> Dict[str, List[Any]]:
        """Stub method to comply with BaseSourceDB signature.

        Args:
            split (str): Generic split name.

        Returns:
            Dict[str, List[Any]]: Empty dictionary.
        """
        return {}

    def get_noise_file_path(self, split: str) -> Path:
        """Returns a random noise file path from the specified split.

        Args:
            split (str): Generic split name ('train', 'val', 'test').

        Returns:
            Path: Path to a random noise file.

        Raises:
            ValueError: If the generic split name is invalid.
            RuntimeError: If no files are loaded for the requested split.
        """
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

        return random.choice(self.noise_files[internal_split_name])
