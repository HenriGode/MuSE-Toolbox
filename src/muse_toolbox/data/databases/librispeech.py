import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Union

import torch
import torchaudio
from tqdm import tqdm

from muse_toolbox.data.databases.base_DBs import BaseSourceDB

log = logging.getLogger(__name__)


class LibrispeechDatabase(BaseSourceDB):
    """Manages downloading and pre-processing of the LibriSpeech dataset.

    Handles the pre-computation of 'virtual clips' by grouping consecutive utterances.
    """

    SPLIT_NAME_MAPPING = {
        "train": "train-clean-360",
        "val": "dev-clean",
        "test": "test-clean",
    }

    def __init__(
        self,
        root_dir: Union[str, Path],
        signal_lengths: float,
        splits: Optional[List[str]] = None,
    ) -> None:
        """Initializes the LibrispeechDatabase.

        Args:
            root_dir (Union[str, Path]): Path to the databases root directory.
            signal_lengths (float): Required signal length for the virtual clips.
            splits (Optional[List[str]]): Specific splits to download and process. 
                If None, uses defaults.
        """
        super().__init__(
            root_dir=Path(root_dir) / "librispeech", signal_lengths=signal_lengths
        )

        if splits is not None:
            self.splits_to_use = splits
        else:
            self.splits_to_use = ["train-clean-360", "dev-clean", "test-clean"]

        log.info(f"LibrispeechDatabase initialized to use splits: {self.splits_to_use}")

    def download(self) -> None:
        """Downloads the required LibriSpeech splits if they don't exist."""
        log.info("Checking for LibriSpeech data...")
        for split in self.splits_to_use:
            if not (self.root_dir / split).exists():
                url = split
                if not url:
                    log.warning(f"Unknown LibriSpeech split '{split}'. Cannot download.")
                    continue

                log.info(f"Split '{split}' not found. Downloading...")
                self.root_dir.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.datasets.LIBRISPEECH(
                    root=self.root_dir.parent, url=url, download=True
                )
                log.info(f"'{split}' download complete.")
        log.info("LibriSpeech data is ready.")

    def prepare_data(self) -> None:
        """Prepares the dataset.
        
        Ensures that the data is downloaded and that the pre-computed speaker
        clip cache files exist for all required splits.
        """
        self.download()

        log.info("Checking/Creating Librispeech clip caches...")
        for split in self.splits_to_use:
            split_path = self.root_dir / split
            if not split_path.exists():
                log.warning(
                    f"LibriSpeech split '{split}' not found at {split_path}. "
                    "Skipping clip creation."
                )
                continue
            self._create_speaker_clips_cache(split_path)

    def _create_speaker_clips_cache(self, split_path: Path) -> None:
        """Walks the directory structure to create and cache consecutive utterance clips.

        Args:
            split_path (Path): Path to the split directory.
        """
        cache_file = split_path / "clean_speech_clips.pt"
        if cache_file.exists():
            log.info(f"Clip cache already exists for: {split_path.name}")
            return

        log.info(f"Creating clip cache for: {split_path.name}...")
        speaker_clips = defaultdict(list)
        target_duration = min(1.25 * (
            max(self.signal_lengths)
            if isinstance(self.signal_lengths, (list, tuple))
            else self.signal_lengths
        ), 170.0)

        for speaker_path in tqdm(
            list(split_path.iterdir()), desc=f"Processing {split_path.name}"
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
        log.info(f"Created {total_clips} virtual clips for {len(speaker_clips)} speakers.")
        log.info(f"Saving clips to cache file: {cache_file}")
        torch.save(dict(speaker_clips), cache_file)

    def get_speaker_clips(self, split: str) -> Dict[str, List[List[Path]]]:
        """Loads speaker clips for a generic split from the pre-computed cache.

        Args:
            split (str): Generic split name ('train', 'val', 'test').

        Returns:
            Dict[str, List[List[Path]]]: Dictionary mapping speaker IDs to lists of clips.
                Each clip is a list of Path objects pointing to consecutive utterances.

        Raises:
            ValueError: If the generic split name is invalid.
            FileNotFoundError: If the cache file does not exist.
        """
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
                f"Clip cache not found at {cache_file}. Please run prepare_data first."
            )

        log.info(f"Loading cached clips from: {cache_file}")
        speaker_clips = torch.load(cache_file, weights_only=False)
        total_clips = sum(len(clips) for clips in speaker_clips.values())
        log.info(
            f"Loaded {total_clips} virtual clips for {len(speaker_clips)} speakers from cache."
        )
        return speaker_clips
