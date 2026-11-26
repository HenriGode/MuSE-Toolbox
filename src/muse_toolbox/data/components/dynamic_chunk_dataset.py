import hashlib
import logging
import random
from pathlib import Path
from typing import Any, Mapping

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
        train_items_per_epoch: int | None,
        augmentation: dict[str, Any],
        fs: int,
        transform: STFTtransform | None = None,
        eval_chunk_length_s: float | None = None,
        eval_min_context_s: float | None = None,
        allowed_keys: set[str] | list[str] | None = None,
        mic_count_probs: Mapping[int, float] | None = None,
        single_channel_expansion: bool = False,
        seed: int = 777,
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
            train_items_per_epoch (int | None): Base virtual epoch length in chunks for
                unaugmented training sampling, or ``None`` to automatically use 100% of
                the unique corpus duration per epoch. When additive augmentation
                (``augmentation_factor``) is enabled on `"train"`, the total epoch
                length becomes ``base + round(base * augmentation_factor)``.
            augmentation (dict[str, Any]): On-the-fly dynamic mixing augmentation config.
            fs (int): Audio sampling rate in Hz.
            transform (STFTtransform | None): Optional STFT transform used to convert
                sample-level activity labels into frame-level ground-truth counts.
            eval_chunk_length_s (float | None): Optional evaluation sliding-window
                duration in seconds for `"val"` and `"test"` splits.
            eval_min_context_s (float | None): Optional evaluation sliding-window
                overlap duration in seconds for `"val"` and `"test"` splits.
            allowed_keys (set[str] | list[str] | None): Optional collection of
                recording keys (``"{meeting_id}_{array_id}"``) to include.
            mic_count_probs (Mapping[int, float] | None, optional): Optional mapping from
                microphone counts ``M in {1..8}`` to sampling probabilities summing to
                ``1.0`` for dynamic subsampling of precomputed arrays (e.g. ``Rand3-8``
                from ``Rand8``). Defaults to ``None`` (use all channels).
            single_channel_expansion (bool, optional): If ``True``, serves single-channel
                streams: random microphone per training chunk, one deterministic-random
                microphone per recording on `"val"`, and exhaustive expansion over all
                microphones on `"test"`. Defaults to ``False``.
            seed (int, optional): Base seed used for deterministic per-recording channel
                selection on evaluation splits. Defaults to ``777``.

        Raises:
            ValueError: If the active hop duration `chunk_length_s - min_context_s`
                is not strictly positive, or if `mic_count_probs` or
                `augmentation_factor` is invalid.
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

        self.augmentation = augmentation or {}
        self.fs = int(fs)
        self.transform = transform
        self.allowed_keys = set(allowed_keys) if allowed_keys is not None else None
        self.mic_count_probs = (
            self._validate_and_normalize_mic_count_probs(mic_count_probs)
            if mic_count_probs is not None
            else None
        )
        self.single_channel_expansion = bool(single_channel_expansion)
        self.seed = int(seed)

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
        self.recording_num_mics: dict[str, int] = {}
        self.valid_keys: list[str] = []
        self.recording_weights: list[float] = []
        self.grid: list[tuple[str, int, bool]] = []
        self._grid_mic_override: list[list[int] | None] = []
        self._grid_rec_id_override: list[str | None] = []
        self._eval_recording_mic_indices: dict[str, list[int]] = {}

        self._init_mmaps()
        self.base_train_items_per_epoch = (
            int(train_items_per_epoch)
            if train_items_per_epoch is not None
            else self._compute_default_train_items_per_epoch()
        )
        aug_active = (
            self.split == "train"
            and bool(self.augmentation.get("enabled", False))
            and hasattr(self, "single_speaker_dict")
            and len(self.single_speaker_dict) > 0
        )
        aug_factor_raw = self.augmentation.get("augmentation_factor", None)
        if aug_active and aug_factor_raw is not None:
            aug_factor = float(aug_factor_raw)
            if aug_factor < 0.0:
                raise ValueError(
                    f"augmentation_factor must be >= 0.0, got {aug_factor}."
                )
            self._use_additive_aug_partition = True
            self.num_augmented_items = int(
                round(self.base_train_items_per_epoch * aug_factor)
            )
            self.train_items_per_epoch = (
                self.base_train_items_per_epoch + self.num_augmented_items
            )
        else:
            self._use_additive_aug_partition = False
            self.num_augmented_items = 0
            self.train_items_per_epoch = self.base_train_items_per_epoch
        self._build_deterministic_grid()

    @staticmethod
    def _validate_and_normalize_mic_count_probs(
        mic_count_probs: Mapping[int, float],
    ) -> dict[int, float]:
        """Validates and normalizes a discrete microphone-count probability table.

        Args:
            mic_count_probs (Mapping[int, float]): Mapping from integer microphone
                count ``M in {1..8}`` to probability mass summing to ``1.0``.

        Returns:
            dict[int, float]: Sorted normalized mapping `{M: probability}`.

        Raises:
            ValueError: If the mapping is empty, contains keys outside ``1..8``,
                contains negative probabilities, or does not sum to ``1.0``.
        """
        if not mic_count_probs:
            raise ValueError("mic_count_probs must be a non-empty dictionary.")

        parsed: dict[int, float] = {}
        for raw_m, raw_p in mic_count_probs.items():
            if isinstance(raw_m, bool):
                raise ValueError(f"Microphone count key must be int, got bool {raw_m}.")
            try:
                m_int = int(raw_m)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Microphone count key '{raw_m}' is not a valid integer."
                ) from exc
            if float(m_int) != float(raw_m) if isinstance(raw_m, float) else False:
                raise ValueError(
                    f"Microphone count key '{raw_m}' must be an integer."
                )
            if not (1 <= m_int <= 8):
                raise ValueError(
                    f"Microphone count key {m_int} out of allowed range [1, 8]."
                )
            p_val = float(raw_p)
            if p_val < 0.0 or not np.isfinite(p_val):
                raise ValueError(
                    f"Probability for {m_int} mics must be finite and >= 0, got {raw_p}."
                )
            parsed[m_int] = parsed.get(m_int, 0.0) + p_val

        total_prob = sum(parsed.values())
        if abs(total_prob - 1.0) > 1e-3:
            raise ValueError(
                f"mic_count_probs must sum to 1.0 (got {total_prob:.6f})."
            )

        return {m: parsed[m] / total_prob for m in sorted(parsed.keys())}

    @staticmethod
    def _make_recording_rng(seed: int, split: str, key: str) -> random.Random:
        """Builds a reproducible PRNG seeded deterministically by split and recording key.

        Args:
            seed (int): Base dataset seed.
            split (str): Dataset split identifier.
            key (str): Recording key.

        Returns:
            random.Random: Seeded random generator independent of global PRNG state.
        """
        digest = hashlib.sha256(f"{seed}:{split}:{key}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _sample_mic_subset(
        self, m_avail: int, rng: random.Random | Any = random
    ) -> list[int]:
        """Samples an ordered subset of microphone indices according to ``mic_count_probs``.

        Args:
            m_avail (int): Number of physical channels available in the recording.
            rng (random.Random | Any, optional): PRNG instance supporting
                ``choices`` and ``sample``. Defaults to ``random``.

        Returns:
            list[int]: Ascending sorted list of selected channel indices.
        """
        if self.mic_count_probs is None:
            return list(range(m_avail))

        eligible = [(m, p) for m, p in self.mic_count_probs.items() if m <= m_avail and p > 0.0]
        if not eligible:
            m_target = m_avail
        else:
            counts = [item[0] for item in eligible]
            weights = [item[1] for item in eligible]
            m_target = int(rng.choices(counts, weights=weights, k=1)[0])

        if m_target >= m_avail:
            return list(range(m_avail))
        return sorted(rng.sample(range(m_avail), m_target))

    @staticmethod
    def _slice_mic_pos(
        mic_pos: Any, mic_indices: list[int], total_mics: int
    ) -> Any:
        """Slices stored array microphone coordinates to match subsampled channel indices.

        Args:
            mic_pos (Any): Coordinates tensor, array, or nested list.
            mic_indices (list[int]): Selected microphone indices.
            total_mics (int): Original microphone count prior to subsampling.

        Returns:
            Any: Coordinate structure sliced along its microphone axis.
        """
        if isinstance(mic_pos, (np.ndarray, torch.Tensor)):
            if mic_pos.ndim == 2 and mic_pos.shape[1] == total_mics:
                return mic_pos[:, mic_indices]
            if mic_pos.ndim == 2 and mic_pos.shape[0] == total_mics:
                return mic_pos[mic_indices, :]
        elif isinstance(mic_pos, list) and mic_pos:
            if (
                isinstance(mic_pos[0], list)
                and len(mic_pos) == 3
                and len(mic_pos[0]) == total_mics
            ):
                return [[row[i] for i in mic_indices] for row in mic_pos]
            if len(mic_pos) == total_mics:
                return [mic_pos[i] for i in mic_indices]
        return mic_pos

    def _init_mmaps(self) -> None:
        """Initializes memory-mapped audio and SAD arrays and single-speaker lookup tables."""
        audio_files = sorted(self.data_dir.glob("*_audio.npy"))
        if not audio_files:
            log.warning(f"No *_audio.npy files found in {self.data_dir}")

        for audio_path in audio_files:
            key = audio_path.name.replace("_audio.npy", "")
            if self.allowed_keys is not None and key not in self.allowed_keys:
                continue
            sad_path = self.data_dir / f"{key}_sad.npy"
            meta_path = self.data_dir / f"{key}_meta.pt"

            if sad_path.exists() and meta_path.exists():
                audio_mmap = np.load(audio_path, mmap_mode="r")
                sad_mmap = np.load(sad_path, mmap_mode="r")

                self.mmap_audio[key] = audio_mmap
                self.mmap_sad[key] = sad_mmap
                self.meta_data[key] = torch.load(meta_path, weights_only=False)
                self.meeting_lengths[key] = int(audio_mmap.shape[1])
                self.recording_num_mics[key] = int(audio_mmap.shape[0])
                self.valid_keys.append(key)
            else:
                log.warning(f"Missing precomputed sad/meta data for {key}")

        self.recording_weights = [
            float(max(1, self.meeting_lengths[k] - self.chunk_samples + 1))
            for k in self.valid_keys
        ]

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

    def _compute_default_train_items_per_epoch(self) -> int:
        """Computes 100% unique corpus duration per epoch in training chunks.

        Multi-array recordings sharing a ``meeting_id`` in metadata are grouped
        so an epoch equals one full pass over unique session audio rather than
        multiplying epoch size by the number of simultaneous arrays.

        Returns:
            int: Computed virtual epoch length (at least ``1``).
        """
        if not self.valid_keys:
            return 1
        grouped_lengths: dict[str, int] = {}
        for key in self.valid_keys:
            meta = self.meta_data.get(key, {})
            group_id = str(meta.get("meeting_id", key))
            grouped_lengths[group_id] = max(
                grouped_lengths.get(group_id, 0), self.meeting_lengths[key]
            )
        total_chunks = sum(
            max(1, length // self.chunk_samples)
            for length in grouped_lengths.values()
        )
        return max(1, total_chunks)

    def _build_deterministic_grid(self) -> None:
        """Builds the ordered sliding-window chunk grid for validation and test splits."""
        self.grid = []
        self._grid_mic_override = []
        self._grid_rec_id_override = []
        self._eval_recording_mic_indices = {}
        if self.split != "train":
            min_valid_samples = (
                int(self.transform.hop_length) if self.transform is not None else 1
            )
            for key in self.valid_keys:
                length = self.meeting_lengths[key]
                m_avail = self.recording_num_mics[key]
                starts: list[int] = []
                start_idx = 0
                while start_idx + self.chunk_samples <= length:
                    starts.append(start_idx)
                    start_idx += self.hop_samples
                covered_end = (starts[-1] + self.chunk_samples) if starts else 0
                if (length - covered_end) >= min_valid_samples and start_idx < length:
                    starts.append(start_idx)
                num_chunks = len(starts)

                if self.single_channel_expansion:
                    if self.split == "val":
                        rng = self._make_recording_rng(self.seed, self.split, key)
                        mic_list = [int(rng.randrange(m_avail))]
                        self._eval_recording_mic_indices[key] = mic_list
                        rec_id = f"{key}_ch{mic_list[0]}"
                        for chunk_pos, s_idx in enumerate(starts):
                            is_last_chunk = chunk_pos == (num_chunks - 1)
                            self.grid.append((key, s_idx, is_last_chunk))
                            self._grid_mic_override.append(mic_list)
                            self._grid_rec_id_override.append(rec_id)
                    else:
                        for ch_idx in range(m_avail):
                            mic_list = [int(ch_idx)]
                            rec_id = f"{key}_ch{ch_idx}"
                            for chunk_pos, s_idx in enumerate(starts):
                                is_last_chunk = chunk_pos == (num_chunks - 1)
                                self.grid.append((key, s_idx, is_last_chunk))
                                self._grid_mic_override.append(mic_list)
                                self._grid_rec_id_override.append(rec_id)
                else:
                    if self.mic_count_probs is not None:
                        rng = self._make_recording_rng(self.seed, self.split, key)
                        mic_list = self._sample_mic_subset(m_avail, rng=rng)
                    else:
                        mic_list = list(range(m_avail))
                    self._eval_recording_mic_indices[key] = mic_list
                    for chunk_pos, s_idx in enumerate(starts):
                        is_last_chunk = chunk_pos == (num_chunks - 1)
                        self.grid.append((key, s_idx, is_last_chunk))
                        self._grid_mic_override.append(mic_list)
                        self._grid_rec_id_override.append(key)

    def __len__(self) -> int:
        """Returns the number of chunks in the dataset epoch or evaluation grid.

        Returns:
            int: Dataset length.
        """
        if self.split == "train":
            return self.train_items_per_epoch
        return len(self.grid)

    def _get_random_single_speaker_chunk(
        self,
        key: str,
        speaker_id: int,
        mic_indices: list[int] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Samples a random single-speaker audio and SAD slice for a given recording and speaker.

        Args:
            key (str): Recording identifier (`"{meeting_id}_{array_id}"`).
            speaker_id (int): Speaker index within the recording's SAD matrix.
            mic_indices (list[int] | None, optional): Subset of channel indices to copy
                after slicing the temporal window. Defaults to ``None`` (all channels).

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: Tuple of `(audio_chunk, sad_row)`.
        """
        if not hasattr(self, "single_speaker_dict") or not self.single_speaker_dict:
            return None, None

        start_idx = random.choice(self.single_speaker_dict[key][speaker_id])
        end_idx = start_idx + self.chunk_samples

        # Slice temporal window first before indexing channels to avoid full-mmap reads on long files
        audio_window = self.mmap_audio[key][:, start_idx:end_idx]
        m_avail = self.recording_num_mics.get(key, int(audio_window.shape[0]))
        if mic_indices is not None and len(mic_indices) != m_avail:
            audio_chunk = audio_window[mic_indices, :].copy()
        else:
            audio_chunk = audio_window.copy()
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
        if (
            self.split == "train"
            and bool(self.augmentation.get("enabled", False))
            and hasattr(self, "single_speaker_dict")
            and len(self.single_speaker_dict) > 0
        ):
            if self._use_additive_aug_partition:
                apply_aug = idx >= self.base_train_items_per_epoch
            else:
                prob_aug = float(self.augmentation.get("prob_augmentation", 0.5))
                apply_aug = random.random() < prob_aug

        # Augmentation logic (length-weighted across meetings having sufficient speakers)
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
                    aug_weights = [
                        float(max(1, self.meeting_lengths.get(k, 1))) for k in valid_keys
                    ]
                    key = random.choices(valid_keys, weights=aug_weights, k=1)[0]
                    num_speakers = min(num_speakers, len(self.single_speaker_dict[key]))
                else:
                    apply_aug = False
            else:
                aug_weights = [
                    float(max(1, self.meeting_lengths.get(k, 1))) for k in valid_keys
                ]
                key = random.choices(valid_keys, weights=aug_weights, k=1)[0]

        if apply_aug:
            m_avail = self.recording_num_mics[key]
            if self.single_channel_expansion:
                mic_indices = [int(random.randrange(m_avail))]
            elif self.mic_count_probs is not None:
                mic_indices = self._sample_mic_subset(m_avail, rng=random)
            else:
                mic_indices = list(range(m_avail))

            speaker_ids = random.sample(
                list(self.single_speaker_dict[key].keys()), num_speakers
            )
            first_aud, first_act = self._get_random_single_speaker_chunk(
                key, speaker_ids[0], mic_indices=mic_indices
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
                aud, act = self._get_random_single_speaker_chunk(
                    key, speaker_ids[i], mic_indices=mic_indices
                )
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
            recording_id = (
                f"aug_{key}_ch{mic_indices[0]}_{idx}"
                if self.single_channel_expansion
                else f"aug_{key}_{idx}"
            )
            base_meta = self.meta_data[key].copy()
        else:
            if self.split == "train":
                key = random.choices(
                    self.valid_keys, weights=self.recording_weights, k=1
                )[0]
                m_avail = self.recording_num_mics[key]
                length = self.meeting_lengths[key]
                if length > self.chunk_samples:
                    start_idx = random.randint(0, length - self.chunk_samples)
                else:
                    start_idx = 0
                is_first_chunk = start_idx == 0
                is_last_chunk = (start_idx + self.chunk_samples) >= length

                if self.single_channel_expansion:
                    mic_indices = [int(random.randrange(m_avail))]
                    recording_id = f"{key}_ch{mic_indices[0]}"
                elif self.mic_count_probs is not None:
                    mic_indices = self._sample_mic_subset(m_avail, rng=random)
                    recording_id = key
                else:
                    mic_indices = list(range(m_avail))
                    recording_id = key
            else:
                key, start_idx, is_last_chunk = self.grid[idx]
                m_avail = self.recording_num_mics[key]
                mic_override = self._grid_mic_override[idx]
                mic_indices = (
                    list(mic_override)
                    if mic_override is not None
                    else list(range(m_avail))
                )
                recording_id = self._grid_rec_id_override[idx] or key
                is_first_chunk = start_idx == 0

            end_idx = min(start_idx + self.chunk_samples, self.meeting_lengths[key])
            # Slice temporal window first before indexing channels (safe for long AMI mmap files)
            audio_window = self.mmap_audio[key][:, start_idx:end_idx]
            if len(mic_indices) != m_avail or mic_indices != list(range(m_avail)):
                audio_chunk = audio_window[mic_indices, :].copy()
            else:
                audio_chunk = audio_window.copy()
            sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()

            audio_tensor = torch.from_numpy(audio_chunk).float()
            sad_tensor = torch.from_numpy(sad_chunk)
            base_meta = self.meta_data[key].copy()

        num_mics = len(mic_indices)
        scenario_params = dict(base_meta.get("scenario_params", {}))
        scenario_params["augmentation_applied"] = bool(apply_aug)
        scenario_params["num_mics"] = num_mics
        scenario_params["mic_indices"] = list(mic_indices)
        scenario_params["recording_id"] = recording_id
        if "mic_pos" in scenario_params and scenario_params["mic_pos"] is not None:
            if len(mic_indices) != m_avail or mic_indices != list(range(m_avail)):
                scenario_params["mic_pos"] = self._slice_mic_pos(
                    scenario_params["mic_pos"], mic_indices, m_avail
                )
        base_meta["scenario_params"] = scenario_params

        # Shift and crop the segments relative to the chunk timeframe
        actual_chunk_s = float(audio_tensor.shape[-1]) / float(self.fs)
        start_time = start_idx / self.fs
        end_time = start_time + actual_chunk_s

        if "segments" in base_meta:
            new_segments = []
            for seg in base_meta["segments"]:
                if seg.start < end_time and seg.end > start_time:
                    new_start = max(0.0, seg.start - start_time)
                    new_end = min(actual_chunk_s, seg.end - start_time)
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
                "scenario_id": f"{recording_id}_{start_time:.2f}",
                "recording_id": recording_id,
                "num_mics": num_mics,
                "mic_indices": list(mic_indices),
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
