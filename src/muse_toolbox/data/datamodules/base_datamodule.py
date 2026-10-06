"""Base DataModule for MuSE-Toolbox.

Provides the generic abstraction for all scenario-based data generation,
saving raw audio/SAD to memory-mappable .npy files, and PyTorch Lightning DataLoaders.
"""

import logging
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sized
from pathlib import Path
import numpy as np

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from muse_toolbox.data.simulation.base_scenario_generator import ScenarioGenerationConfig
from muse_toolbox.utils import STFTtransform
from muse_toolbox.data.components.collate import raw_audio_collate_fn
from muse_toolbox.data.components.dynamic_chunk_dataset import DynamicChunkDataset

log = logging.getLogger(__name__)


class BaseDataModule(pl.LightningDataModule, ABC):
    """Abstract base class for all data modules in this project.

    It defines the common structure and provides generic dataloader methods
    to reduce boilerplate code in specific implementations. Subclasses are
    expected to define `self.train_ds`, `self.val_ds`, and `self.test_ds`
    in their `setup()` method.
    """

    def __init__(
        self,
        data_dir: str | Path,
        id: str,
        transform: STFTtransform,
        batch_size: int,
        num_workers: int,
        num_scenarios: list[int] | None,
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        seed: int | None,
        reset: bool,
        acc_device: torch.device,
        chunk_length_s: float,
        min_context_s: float,
        train_items_per_epoch: int,
        augmentation: dict,
        **kwargs
    ) -> None:
        """Initializes the BaseDataModule.

        Args:
            data_dir (str | Path): The base directory for all data.
            id (str): Unique identifier for this dataset configuration.
            transform (STFTtransform): The STFT transform configuration.
            batch_size (int): The batch size for the dataloaders.
            num_workers (int): The number of worker processes for data loading.
            num_scenarios (List[int]): List of scenarios to generate [train, val, test].
            sampling_frequency (int): Sampling rate for audio processing.
            generation_config (ScenarioGenerationConfig): Configuration for scenario mixing.
            seed (Optional[int]): Random seed for reproducibility.
            reset (bool): If True, forces deletion of previously generated data for this ID.
            acc_device (torch.device): Device to use for precomputation acceleration.
            chunk_length_s (float): Length of the chunks to load in seconds.
            min_context_s (float): Overlap context in seconds.
            train_items_per_epoch (int): Number of training items per epoch.
            augmentation (dict): Augmentation configuration dictionary.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.id = id
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_scenarios = num_scenarios
        self.sampling_frequency = sampling_frequency
        self.generation_config = generation_config
        self.seed = seed
        self.reset = reset
        self.acc_device = acc_device
        self.chunk_length_s = chunk_length_s
        self.min_context_s = min_context_s
        self.train_items_per_epoch = train_items_per_epoch
        self.augmentation = augmentation
        
        self.data_is_prepared = False
        self.precomputed_dir = self.data_dir / "datasets" / self.id

    # --- Abstract methods for subclasses to implement ---
    @abstractmethod
    def _get_database_managers(self) -> list:
        """Provides all database manager objects (e.g., for speech, RIRs, noise).

        Returns:
            list: List of instantiated database managers.
        """
        pass

    @abstractmethod
    def _get_scenario_generator(self, split: str) -> Dataset | None:
        """Provides the configured scenario generator for a given split.

        Args:
            split (str): The data split (e.g., 'train', 'val', 'test').

        Returns:
            Optional[Dataset]: The scenario generator, or None if unused.
        """
        pass

    # --- Main Orchestration Method ---
    def prepare_data(self) -> None:
        """Generic data preparation pipeline. Orchestrates all generation steps."""
        if self.data_is_prepared:
            log.info("Data is already prepared. Skipping preparation step.")
        else:
            log.info("Starting data preparation...")
            self._handle_reset()
            self._prepare_source_databases()
            with torch.no_grad():
                self._generate_scenarios()
            log.info("Data preparation finished.")
            self.data_is_prepared = True

    # --- Helper Functions for the Pipeline ---
    def _handle_reset(self) -> None:
        """Checks and executes the reset logic.
        
        Warning: Because this is an HPC-compatible script, setting reset=True 
        will forcefully delete existing cached datasets without prompting `input()`.
        """
        if not getattr(self, "reset", False):
            return

        predictions_base_path = self.data_dir / "predictions"
        checkpoints_base_path = self.data_dir / "checkpoints"
        
        dir_to_delete = self.precomputed_dir
        dependent_dirs_to_delete = list(predictions_base_path.glob(f"{self.id}*")) + \
                                   list(checkpoints_base_path.glob(f"{self.id}*"))

        if dir_to_delete.exists():
            log.warning("=" * 50)
            log.warning("!! WARNING: RESET FLAG IS ENABLED !!")
            log.warning(f"Deleting precomputed database at: {dir_to_delete}")
            log.warning("=" * 50)
            try:
                shutil.rmtree(dir_to_delete)
                log.info("Database deletion successful.")
            except OSError as e:
                log.error(f"Error deleting directory {dir_to_delete}: {e}")
        else:
            log.info(f"Reset flag is True, but no existing directory found at: {dir_to_delete}")

        if dependent_dirs_to_delete:
            log.warning("Deleting dependent prediction and checkpoint directories...")
            for d in dependent_dirs_to_delete:
                try:
                    shutil.rmtree(d)
                    log.info(f"Deleted dependent dir: {d}")
                except OSError as e:
                    log.error(f"Error deleting directory {d}: {e}")

    def _prepare_source_databases(self) -> None:
        """Calls the prepare_data method on all registered database managers."""
        log.info("--- Preparing all source databases ---")
        for db_manager in self._get_database_managers():
            if hasattr(db_manager, "prepare_data"):
                db_manager.prepare_data()

    def _generate_scenarios(self) -> None:
        """Generates and saves scenario files as memory-mappable .npy files."""
        log.info("--- Starting scenario pre-computation ---")

        # 1. Setup Base Paths
        database_path = self.precomputed_dir
        database_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Scenarios will be saved to: {database_path}")

        # Set random global seed for reproducibility
        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)

        # 3. Iterate over Splits
        for split in ["train", "val", "test"]:
            generator = self._get_scenario_generator(split)
            if not generator or not isinstance(generator, Sized) or len(generator) == 0:
                log.info(f"No scenarios to generate for '{split}' split. Skipping.")
                continue

            # Create Split-Specific Directory
            split_dir = database_path / split
            split_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                for i in tqdm(range(len(generator)), desc=f"Generating {split} scenarios"):
                    audio_path = split_dir / f"scenario_{i}_audio.npy"
                    sad_path = split_dir / f"scenario_{i}_sad.npy"
                    meta_path = split_dir / f"scenario_{i}_meta.pt"
                    
                    if audio_path.exists() and sad_path.exists() and meta_path.exists():
                        continue

                    data = generator.__getitem__(i)
                    input_tensor = data["input"]
                    meta = data["meta"]
                    
                    sad_dict = meta.get("sad_samples", {})
                    speaker_ids = [k for k in sad_dict.keys() if k != "noise"]
                    
                    if speaker_ids:
                        sad_tensor = torch.stack([sad_dict[k] for k in speaker_ids], dim=0)
                    else:
                        sad_tensor = torch.zeros(1, input_tensor.shape[1], dtype=torch.bool)
                        
                    meta["speaker_ids"] = speaker_ids
                    
                    for key in ["sad_samples", "sad_frames", "source_count"]:
                        if key in meta:
                            del meta[key]

                    np.save(audio_path, input_tensor.numpy())
                    np.save(sad_path, sad_tensor.numpy())
                    torch.save(meta, meta_path)

    def setup(self, stage: str | None = None) -> None:
        if not self.precomputed_dir.exists():
            raise FileNotFoundError(f"Base data directory not found: {self.precomputed_dir}. Run prepare_data first.")

        if stage == "fit" or stage is None:
            self.train_ds = self._get_scenario_generator_dataset('train')
            self.val_ds = self._get_scenario_generator_dataset('val')
        if stage in ["test"] or stage is None:
            self.test_ds = self._get_scenario_generator_dataset('test')
            
    def _get_scenario_generator_dataset(self, split: str) -> Dataset:
        return DynamicChunkDataset(
            data_dir=self.precomputed_dir / split,
            split=split,
            chunk_length_s=self.chunk_length_s,
            min_context_s=self.min_context_s,
            train_items_per_epoch=self.train_items_per_epoch,
            augmentation=self.augmentation,
            fs=self.sampling_frequency,
            transform=self.transform
        )

    def train_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the training set.

        Returns:
            DataLoader: Training set PyTorch DataLoader.
            
        Raises:
            NotImplementedError: If `self.train_ds` is not set.
        """
        if not hasattr(self, "train_ds") or self.train_ds is None:
            raise NotImplementedError("self.train_ds must be set in the setup() method.")

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=raw_audio_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the validation set.

        Returns:
            DataLoader: Validation set PyTorch DataLoader.
        """
        if not hasattr(self, "val_ds") or self.val_ds is None:
            raise NotImplementedError("self.val_ds must be set in the setup() method.")

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=raw_audio_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the test set.

        Returns:
            DataLoader: Test set PyTorch DataLoader.
        """
        if not hasattr(self, "test_ds") or self.test_ds is None:
            raise NotImplementedError("self.test_ds must be set in the setup() method.")

        return DataLoader(
            self.test_ds,
            batch_size=1, # Inference is done with batch size 1 to avoid mixed meeting boundaries
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=raw_audio_collate_fn,
        )
