import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from muse_toolbox.data.datamodules.base_datamodule import BaseDataModule

log = logging.getLogger(__name__)


class AMIDataModule(BaseDataModule):
    """DataModule for the AMI Meeting Corpus.

    Overrides `prepare_data` to precompute full continuous meeting recordings and
    sample-level speaker activity vectors into memory-mappable NumPy arrays split
    into `train/`, `val/`, and `test/` directories.
    """

    def __init__(
        self,
        database: Any,
        train_meetings: list[str],
        val_meetings: list[str],
        test_meetings: list[str],
        chunk_length_s: float,
        min_context_s: float,
        train_items_per_epoch: int | None,
        arrays: list[str],
        augmentation: dict[str, Any],
        sampling_frequency: int,
        label_parser: Any = None,
        id: str = "ami",
        only_8ch_circular: bool = False,
        eval_chunk_length_s: float | None = None,
        eval_min_context_s: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the AMIDataModule.

        Args:
            database (Any): Instantiated AMI database manager.
            train_meetings (list[str]): List of training meeting IDs.
            val_meetings (list[str]): List of validation meeting IDs.
            test_meetings (list[str]): List of test meeting IDs.
            chunk_length_s (float): Training chunk duration in seconds.
            min_context_s (float): Training context/overlap duration in seconds.
            train_items_per_epoch (int | None): Number of random training chunks
                per epoch, or ``None`` to match 100% unique corpus duration.
            arrays (list[str]): Microphone array identifiers (e.g., `["Array1", "Array2"]`).
            augmentation (dict[str, Any]): On-the-fly multi-speaker mixing config.
            sampling_frequency (int): Audio sampling rate in Hz.
            label_parser (Any): Optional label parser passed from Hydra.
            id (str): Dataset configuration identifier.
            only_8ch_circular (bool, optional): If ``True``, restricts AMI to the
                homogeneous 8-channel circular tabletop arrays (``Array1`` in all
                rooms plus Edinburgh ``Array2``, excluding Idiap 4-ch ``Array2``
                and TNO 10-ch linear ``Array2``). Defaults to ``False``.
            eval_chunk_length_s (float | None): Evaluation sliding-window duration in seconds.
            eval_min_context_s (float | None): Evaluation sliding-window overlap in seconds.
            *args (Any): Positional arguments forwarded to `BaseDataModule`.
            **kwargs (Any): Keyword arguments forwarded to `BaseDataModule`.
        """
        super().__init__(
            sampling_frequency=sampling_frequency,
            chunk_length_s=chunk_length_s,
            min_context_s=min_context_s,
            eval_chunk_length_s=eval_chunk_length_s,
            eval_min_context_s=eval_min_context_s,
            train_items_per_epoch=train_items_per_epoch,
            augmentation=augmentation,
            id=id,
            num_scenarios=None,
            *args,
            **kwargs,
        )
        self.database = database
        self.train_meetings = train_meetings
        self.val_meetings = val_meetings
        self.test_meetings = test_meetings
        self.arrays = arrays
        self.only_8ch_circular = bool(only_8ch_circular)

    def _is_allowed_meeting_array(self, meeting_id: str, array_id: str) -> bool:
        """Checks whether a ``(meeting_id, array_id)`` pair passes array filtering.

        Args:
            meeting_id (str): AMI meeting identifier (e.g., ``"EN2002a"``).
            array_id (str): Array identifier (e.g., ``"Array1"`` or ``"Array2"``).

        Returns:
            bool: ``True`` if the recording should be included.
        """
        if not self.only_8ch_circular:
            return True
        if array_id == "Array1":
            return True
        if array_id == "Array2":
            return meeting_id.startswith(("EN", "ES"))
        return False

    def _get_allowed_keys_for_split(self, split: str) -> set[str]:
        """Returns the allowed ``"{meeting_id}_{array_id}"`` keys for a split.

        Args:
            split (str): Dataset split (``"train"``, ``"val"``, or ``"test"``).

        Returns:
            set[str]: Allowed recording keys respecting ``only_8ch_circular``.
        """
        meetings = getattr(self, f"{split}_meetings")
        return {
            f"{meeting_id}_{array_id}"
            for meeting_id in meetings
            for array_id in self.arrays
            if self._is_allowed_meeting_array(meeting_id, array_id)
        }

    def _get_scenario_generator_dataset(self, split: str) -> Dataset:
        """Instantiates ``DynamicChunkDataset`` for an AMI split with array filtering.

        Args:
            split (str): Dataset split (``"train"``, ``"val"``, or ``"test"``).

        Returns:
            Dataset: Configured ``DynamicChunkDataset``.
        """
        from muse_toolbox.data.components.dynamic_chunk_dataset import (
            DynamicChunkDataset,
        )

        allowed_keys = (
            self._get_allowed_keys_for_split(split) if self.only_8ch_circular else None
        )
        return DynamicChunkDataset(
            data_dir=self.precomputed_dir / split,
            split=split,
            chunk_length_s=self.chunk_length_s,
            min_context_s=self.min_context_s,
            train_items_per_epoch=self.train_items_per_epoch,
            augmentation=self.augmentation,
            fs=self.sampling_frequency,
            transform=self.transform,
            eval_chunk_length_s=self.eval_chunk_length_s,
            eval_min_context_s=self.eval_min_context_s,
            allowed_keys=allowed_keys,
            mic_count_probs=self.mic_count_probs,
            single_channel_expansion=self.single_channel_expansion,
            seed=self.seed if self.seed is not None else 777,
        )

    def _get_database_managers(self) -> list[Any]:
        """Returns the list of underlying database managers.

        Returns:
            list[Any]: List containing the AMI database instance.
        """
        return [self.database]

    def _get_scenario_generator(self, split: str) -> Dataset | None:
        """Returns `None` as AMI uses direct continuous meeting precomputation.

        Args:
            split (str): Dataset split name.

        Returns:
            Dataset | None: Always `None`.
        """
        return None

    def prepare_data(self) -> None:
        """Precomputes continuous AMI meeting recordings and sample-level SAD arrays."""
        if self.data_is_prepared:
            return

        self._handle_reset()
        self.precomputed_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"Precomputing continuous AMI meetings for '{self.id}' to memory-mappable numpy arrays..."
        )

        single_speaker_chunks: list[str] = []
        chunk_samples = int(round(self.chunk_length_s * self.sampling_frequency))
        hop_samples = int(
            round((self.chunk_length_s - self.min_context_s) * self.sampling_frequency)
        )
        lp = getattr(self.database, "label_parser", None)
        parser_config: dict[str, Any] | None = (
            lp.get_config() if lp is not None and hasattr(lp, "get_config") else None
        )

        for split in ["train", "val", "test"]:
            meetings = getattr(self, f"{split}_meetings")
            split_dir = self.precomputed_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for meeting_id in tqdm(meetings, desc=f"Precomputing AMI {split}"):
                for array_id in self.arrays:
                    if not self._is_allowed_meeting_array(meeting_id, array_id):
                        continue
                    key = f"{meeting_id}_{array_id}"
                    audio_path = split_dir / f"{key}_audio.npy"
                    sad_path = split_dir / f"{key}_sad.npy"
                    meta_path = split_dir / f"{key}_meta.pt"

                    files_exist = (
                        audio_path.exists() and sad_path.exists() and meta_path.exists()
                    )
                    needs_audio_recompute = not files_exist
                    needs_sad_recompute = not files_exist
                    if files_exist:
                        cached_meta = torch.load(meta_path, weights_only=False)
                        if (
                            cached_meta.get("sampling_frequency")
                            != self.sampling_frequency
                        ):
                            log.info(
                                f"Regenerating {key} for sampling_frequency={self.sampling_frequency} Hz "
                                f"(cached={cached_meta.get('sampling_frequency')} Hz)."
                            )
                            needs_audio_recompute = True
                            needs_sad_recompute = True
                        elif (
                            parser_config is not None
                            and cached_meta.get("label_parser_config") != parser_config
                        ):
                            log.info(
                                f"Regenerating SAD labels for {key} with updated label_parser {parser_config} "
                                f"(cached={cached_meta.get('label_parser_config')})."
                            )
                            needs_sad_recompute = True

                    if needs_audio_recompute:
                        try:
                            audio, sad_dict = self.database.get_meeting(
                                meeting_id, array_id
                            )
                        except Exception as e:
                            log.error(f"Failed to process {meeting_id} {array_id}: {e}")
                            continue

                        sad_keys = list(sad_dict.keys())
                        if len(sad_keys) > 0:
                            sad_tensor = torch.stack(
                                [sad_dict[k] for k in sad_keys], dim=0
                            ).bool()
                        else:
                            sad_tensor = torch.zeros(
                                1, audio.shape[1], dtype=torch.bool
                            )

                        np.save(audio_path, audio.numpy())
                        np.save(sad_path, sad_tensor.numpy())
                        torch.save(
                            {
                                "meeting_id": meeting_id,
                                "array_id": array_id,
                                "sampling_frequency": self.sampling_frequency,
                                "label_parser_config": parser_config,
                            },
                            meta_path,
                        )
                    elif needs_sad_recompute:
                        try:
                            total_samples = int(
                                np.load(audio_path, mmap_mode="r").shape[1]
                            )
                            sad_dict = self.database.label_parser.get_sad_samples(
                                meeting_id, total_samples
                            )
                        except Exception as e:
                            log.error(
                                f"Failed to regenerate SAD labels for {meeting_id} {array_id}: {e}"
                            )
                            continue

                        sad_keys = list(sad_dict.keys())
                        if len(sad_keys) > 0:
                            sad_tensor = torch.stack(
                                [sad_dict[k] for k in sad_keys], dim=0
                            ).bool()
                        else:
                            sad_tensor = torch.zeros(
                                1, total_samples, dtype=torch.bool
                            )

                        np.save(sad_path, sad_tensor.numpy())
                        torch.save(
                            {
                                "meeting_id": meeting_id,
                                "array_id": array_id,
                                "sampling_frequency": self.sampling_frequency,
                                "label_parser_config": parser_config,
                            },
                            meta_path,
                        )

                    # Build single-speaker index for augmentation
                    if split == "train" and self.augmentation.get("enabled", False):
                        sad_mmap = np.load(sad_path, mmap_mode="r")
                        length = sad_mmap.shape[-1]
                        start_idx = 0
                        while start_idx + chunk_samples <= length:
                            chunk_sad = sad_mmap[:, start_idx : start_idx + chunk_samples]
                            is_single_speaker_chunk = chunk_sad.any(axis=-1).sum() == 1
                            chunk_counts = chunk_sad.sum(axis=0)
                            if (
                                np.sum(chunk_counts == 1) >= chunk_samples // 2
                                and is_single_speaker_chunk
                            ):
                                single_speaker_chunks.append(
                                    f"{meeting_id}_{array_id}_{start_idx}"
                                )
                            start_idx += hop_samples

        if self.augmentation.get("enabled", False):
            index_path = (
                self.precomputed_dir
                / f"single_speaker_index_{self.sampling_frequency}Hz.npy"
            )
            np.save(index_path, np.array(single_speaker_chunks, dtype=str))

        self.data_is_prepared = True
