import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from muse_toolbox.data.simulation.scenario_generation import Segment
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class DynamicChunkDataset(Dataset):
    """Universal dataset for dynamic slicing of memory-mapped recordings.

    Supports random chunk sampling with optional multi-speaker mixing augmentation
    during training (`split == "train"`), and deterministic overlapping sliding-window
    grid slicing during evaluation (`split != "train"`). Supports distinct window
    and overlap durations for training vs. evaluation so that both Cornell et al.'s
    overlapping sliding-window evaluation (e.g., 3.0 s window with 1.5 s overlap)
    and causal burn-in chunking (e.g., 7.0 s window with 2.0 s overlap) can be
    executed seamlessly on the same sample-level memory-mapped recordings.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        chunk_length_s: float,
        min_context_s: float,
        train_items_per_epoch: int,
        augmentation: dict[str, Any],
        fs: int,
        transform: STFTtransform | None = None,
        eval_chunk_length_s: float | None = None,
        eval_min_context_s: float | None = None,
    ) -> None:
        """Initializes the DynamicChunkDataset.

        Args:
            data_dir (Path): Directory containing `*_audio.npy`, `*_sad.npy`, and
                `*_meta.pt` files for the requested split.
            split (str): Dataset split identifier (`"train"`, `"val"`, or `"test"`).
            chunk_length_s (float): Training chunk duration in seconds (also used for
                evaluation when `eval_chunk_length_s` is `None`).
            min_context_s (float): Training overlap/context duration in seconds (also
                used for evaluation when `eval_min_context_s` is `None`).
            train_items_per_epoch (int): Virtual epoch length for random training sampling.
            augmentation (dict[str, Any]): On-the-fly dynamic mixing augmentation config.
            fs (int): Audio sampling rate in Hz.
            transform (STFTtransform | None): Optional STFT transform used to convert
                sample-level activity labels into frame-level ground-truth counts.
            eval_chunk_length_s (float | None): Optional evaluation sliding-window
                duration in seconds for `"val"` and `"test"` splits.
            eval_min_context_s (float | None): Optional evaluation sliding-window
                overlap duration in seconds for `"val"` and `"test"` splits.

        Raises:
            ValueError: If the active hop duration `chunk_length_s - min_context_s`
                is not strictly positive.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.train_chunk_length_s = float(chunk_length_s)
        self.train_min_context_s = float(min_context_s)
        self.eval_chunk_length_s = (
            float(eval_chunk_length_s)
            if eval_chunk_length_s is not None
            else float(chunk_length_s)
        )
        self.eval_min_context_s = (
            float(eval_min_context_s)
            if eval_min_context_s is not None
            else float(min_context_s)
        )

        if self.split == "train":
            self.chunk_length_s = self.train_chunk_length_s
            self.min_context_s = self.train_min_context_s
        else:
            self.chunk_length_s = self.eval_chunk_length_s
            self.min_context_s = self.eval_min_context_s

        self.train_items_per_epoch = int(train_items_per_epoch)
        self.augmentation = augmentation or {}
        self.fs = int(fs)
        self.transform = transform

        self.chunk_samples = int(round(self.chunk_length_s * self.fs))
        self.hop_samples = int(round((self.chunk_length_s - self.min_context_s) * self.fs))
        if self.hop_samples <= 0:
            raise ValueError(
                f"Sliding-window hop_samples must be > 0, got {self.hop_samples} "
                f"(chunk_length_s={self.chunk_length_s}, min_context_s={self.min_context_s})."
            )

        self.mmap_audio: dict[str, np.ndarray] = {}
        self.mmap_sad: dict[str, np.ndarray] = {}
        self.meta_data: dict[str, dict[str, Any]] = {}
        self.meeting_lengths: dict[str, int] = {}
        self.valid_keys: list[str] = []
        self.grid: list[tuple[str, int, bool]] = []

        self._init_mmaps()
        self._build_deterministic_grid()

    def _init_mmaps(self) -> None:
        """Initializes memory-mapped audio and SAD arrays and single-speaker lookup tables."""
        audio_files = sorted(self.data_dir.glob("*_audio.npy"))
        if not audio_files:
            log.warning(f"No *_audio.npy files found in {self.data_dir}")

        for audio_path in audio_files:
            key = audio_path.name.replace("_audio.npy", "")
            sad_path = self.data_dir / f"{key}_sad.npy"
            meta_path = self.data_dir / f"{key}_meta.pt"

            if sad_path.exists() and meta_path.exists():
                audio_mmap = np.load(audio_path, mmap_mode="r")
                sad_mmap = np.load(sad_path, mmap_mode="r")

                self.mmap_audio[key] = audio_mmap
                self.mmap_sad[key] = sad_mmap
                self.meta_data[key] = torch.load(meta_path, weights_only=False)
                self.meeting_lengths[key] = int(audio_mmap.shape[1])
                self.valid_keys.append(key)
            else:
                log.warning(f"Missing precomputed sad/meta data for {key}")

        # Load single speaker index if augmentation is enabled
        if (
            self.augmentation
            and self.augmentation.get("enabled", False)
            and self.split == "train"
        ):
            idx_path = self.data_dir.parent / f"single_speaker_index_{self.fs}Hz.npy"
            if idx_path.exists():
                self.single_speaker_index = np.load(idx_path).tolist()

                self.single_speaker_dict: dict[str, dict[int, list[int]]] = {}
                for item in self.single_speaker_index:
                    m_id, a_id, s_idx = item.split("_")
                    key = f"{m_id}_{a_id}"
                    start_idx = int(s_idx)
                    end_idx = start_idx + self.chunk_samples

                    if key in self.mmap_sad and end_idx <= self.mmap_sad[key].shape[-1]:
                        sad_chunk = self.mmap_sad[key][:, start_idx:end_idx]
                        active_speaker = int(np.argmax(sad_chunk.sum(axis=1)))

                        if key not in self.single_speaker_dict:
                            self.single_speaker_dict[key] = {}
                        if active_speaker not in self.single_speaker_dict[key]:
                            self.single_speaker_dict[key][active_speaker] = []

                        self.single_speaker_dict[key][active_speaker].append(start_idx)
            else:
                self.single_speaker_index = []
                self.single_speaker_dict = {}
                log.warning(
                    f"Augmentation enabled but no single_speaker_index.npy found at {idx_path}"
                )

    def _build_deterministic_grid(self) -> None:
        """Builds the ordered sliding-window chunk grid for validation and test splits."""
        self.grid = []
        if self.split != "train":
            for key in self.valid_keys:
                length = self.meeting_lengths[key]
                starts: list[int] = []
                start_idx = 0
                while start_idx + self.chunk_samples <= length:
                    starts.append(start_idx)
                    start_idx += self.hop_samples
                if not starts and length > 0:
                    starts.append(0)
                num_chunks = len(starts)
                for chunk_pos, s_idx in enumerate(starts):
                    is_last_chunk = chunk_pos == (num_chunks - 1)
                    self.grid.append((key, s_idx, is_last_chunk))

    def __len__(self) -> int:
        """Returns the number of chunks in the dataset epoch or evaluation grid.

        Returns:
            int: Dataset length.
        """
        if self.split == "train":
            return self.train_items_per_epoch
        return len(self.grid)

    def _get_random_single_speaker_chunk(
        self, key: str, speaker_id: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Samples a random single-speaker audio and SAD slice for a given recording and speaker.

        Args:
            key (str): Recording identifier (`"{meeting_id}_{array_id}"`).
            speaker_id (int): Speaker index within the recording's SAD matrix.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: Tuple of `(audio_chunk, sad_row)`.
        """
        if not hasattr(self, "single_speaker_dict") or not self.single_speaker_dict:
            return None, None

        start_idx = random.choice(self.single_speaker_dict[key][speaker_id])
        end_idx = start_idx + self.chunk_samples

        audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
        sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()

        return audio_chunk, sad_chunk[speaker_id, :]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retrieves a waveform chunk and its associated frame-level and sample-level metadata.

        Args:
            idx (int): Dataset item index.

        Returns:
            dict[str, Any]: Dictionary with `"input"` (`torch.Tensor` of shape `(M, T_samples)`)
                and `"meta"` containing ground-truth labels and sliding-window coordinates.
        """
        apply_aug = False
        if self.split == "train" and self.augmentation.get("enabled", False):
            if hasattr(self, "single_speaker_dict") and len(self.single_speaker_dict) > 0:
                prob_aug = self.augmentation.get("prob_augmentation", 0.5)
                apply_aug = random.random() < prob_aug

        # Augmentation logic
        if apply_aug:
            probs = [
                self.augmentation.get("prob_1spk", 0.25),
                self.augmentation.get("prob_2spk", 0.25),
                self.augmentation.get("prob_3spk", 0.25),
                self.augmentation.get("prob_4spk", 0.25),
            ]
            num_speakers = random.choices([1, 2, 3, 4], weights=probs, k=1)[0]

            valid_keys = [
                k
                for k, spk_dict in self.single_speaker_dict.items()
                if len(spk_dict) >= num_speakers
            ]
            if not valid_keys:
                valid_keys = [
                    k for k, spk_dict in self.single_speaker_dict.items() if len(spk_dict) > 0
                ]
                if valid_keys:
                    key = random.choice(valid_keys)
                    num_speakers = min(num_speakers, len(self.single_speaker_dict[key]))
                else:
                    apply_aug = False
            else:
                key = random.choice(valid_keys)

        if apply_aug:
            speaker_ids = random.sample(
                list(self.single_speaker_dict[key].keys()), num_speakers
            )
            first_aud, first_act = self._get_random_single_speaker_chunk(
                key, speaker_ids[0]
            )

            mixed_audio = np.zeros_like(first_aud, dtype=np.float32)
            mixed_sad = np.zeros(
                (4, self.chunk_samples), dtype=np.bool_
            )  # Max 4 speakers

            mixed_audio += first_aud
            mixed_sad[0, :] = first_act

            sir_min, sir_max = self.augmentation.get("relative_sir_range", [-10.0, 10.0])
            sir_std = self.augmentation.get("sir_std", 4.0)

            for i in range(1, num_speakers):
                aud, act = self._get_random_single_speaker_chunk(key, speaker_ids[i])
                db_shift = random.gauss(0.0, sir_std)
                db_shift = max(sir_min, min(db_shift, sir_max))
                gain = 10.0 ** (db_shift / 20.0)
                mixed_audio += aud * gain
                mixed_sad[i, :] = act

            max_peak = np.max(np.abs(mixed_audio))
            if max_peak > 0.99:
                target_peak = random.uniform(0.5, 0.99)
                mixed_audio = mixed_audio * (target_peak / max_peak)

            audio_tensor = torch.from_numpy(mixed_audio).float()
            sad_tensor = torch.from_numpy(mixed_sad)
            start_idx = 0
            is_first_chunk = True
            is_last_chunk = True
            recording_id = f"aug_{key}_{idx}"

            base_meta = self.meta_data[key].copy()
            base_meta["scenario_params"] = {"augmentation_applied": True}
        else:
            if self.split == "train":
                key = random.choice(self.valid_keys)
                length = self.meeting_lengths[key]
                if length > self.chunk_samples:
                    start_idx = random.randint(0, length - self.chunk_samples)
                else:
                    start_idx = 0
                is_first_chunk = start_idx == 0
                is_last_chunk = (start_idx + self.chunk_samples) >= length
                recording_id = key
            else:
                key, start_idx, is_last_chunk = self.grid[idx]
                is_first_chunk = start_idx == 0
                recording_id = key

            end_idx = start_idx + self.chunk_samples
            audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
            sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()

            audio_tensor = torch.from_numpy(audio_chunk).float()
            sad_tensor = torch.from_numpy(sad_chunk)

            base_meta = self.meta_data[key].copy()
            base_meta["scenario_params"] = {"augmentation_applied": False}

        # Shift and crop the segments relative to the chunk timeframe
        start_time = start_idx / self.fs
        end_time = start_time + self.chunk_length_s

        if "segments" in base_meta:
            new_segments = []
            for seg in base_meta["segments"]:
                if seg.start < end_time and seg.end > start_time:
                    new_start = max(0.0, seg.start - start_time)
                    new_end = min(self.chunk_length_s, seg.end - start_time)
                    new_segments.append(
                        Segment(
                            start=new_start,
                            end=new_end,
                            num_sources=seg.num_sources,
                            event_type=seg.event_type,
                        )
                    )
            base_meta["segments"] = new_segments

        # Generate downstream ground-truth labels mathematically from sample-level SAD
        if self.transform is not None:
            sad_frames = self.transform.samples2frames_quantity(sad_tensor, dim=1)
            source_count = sad_frames.sum(dim=0)
        else:
            sad_frames = None
            source_count = sad_tensor.sum(dim=0)

        base_meta.update(
            {
                "scenario_id": f"{key}_{start_time:.2f}",
                "recording_id": recording_id,
                "start_time": start_time,
                "start_sample": int(start_idx),
                "hop_samples": int(self.hop_samples),
                "chunk_samples": int(audio_tensor.shape[-1]),
                "is_first_chunk": bool(is_first_chunk),
                "is_last_chunk": bool(is_last_chunk),
                "sad_samples": sad_tensor,
                "sad_frames": sad_frames,
                "source_count": source_count,
            }
        )

        return {"input": audio_tensor, "meta": base_meta}
