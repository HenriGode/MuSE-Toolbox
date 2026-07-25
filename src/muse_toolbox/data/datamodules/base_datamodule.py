"""Base DataModule for MuSE-Toolbox.

Provides the generic abstraction for all scenario-based data generation,
feature pre-computation, and PyTorch Lightning DataLoaders.
"""

import logging
import shutil
import functools
from abc import ABC, abstractmethod
from collections.abc import Sized
from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from muse_toolbox.data.simulation.base_scenario_generator import ScenarioGenerationConfig
from muse_toolbox.data.components.precomputed_dataset import PrecomputedDataset
from muse_toolbox.models.components.feature_extractors.base_feature import (
    BaseFeatureExtractor,
)
from muse_toolbox.utils import STFTtransform, move2device

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
        num_scenarios: list[int],
        sampling_frequency: int,
        generation_config: ScenarioGenerationConfig,
        seed: int | None,
        reset: bool,
        acc_device: torch.device,
        feature_extractor: BaseFeatureExtractor | None,
        force_load_stft: bool,
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
            feature_extractor (Optional[BaseFeatureExtractor]): Optional feature extractor.
            force_load_stft (bool): If True, forces loading of STFT data even if features are precomputed.
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
        self.feature_extractor = feature_extractor
        if isinstance(self.feature_extractor, functools.partial):
            self.feature_extractor = self.feature_extractor(transform=self.transform)
            
        self.force_load_stft = force_load_stft
        self.acc_device = acc_device
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
        """Generates and saves scenario files and pre-computes features."""
        log.info("--- Starting scenario pre-computation ---")

        # 1. Setup Base Paths
        database_path = self.precomputed_dir
        database_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Scenarios will be saved to: {database_path}")

        # 2. Analyze Feature Extractor Capabilities
        feat_extr = self.feature_extractor
        feature_base_dir = None
        stft_base_dir = None

        # Capture original training state to restore later
        feat_extr_was_training = True

        if isinstance(feat_extr, BaseFeatureExtractor):
            feat_extr_was_training = feat_extr.training
            feat_extr.eval()
            
            # Determine device for precomputation once
            trainer = getattr(self, "trainer", None)
            try:
                if trainer is not None and getattr(trainer, "device_ids", None):
                    self._precompute_dev = torch.device(f"cuda:{trainer.device_ids[0]}")
                elif torch.cuda.is_available():
                    # self._precompute_dev = torch.device("cuda:0")
                    log.warning("CUDA is avaiable but on a different device. Now processing on CPU.")
                    self._precompute_dev = torch.device("cpu")
                else:
                    self._precompute_dev = torch.device("cpu")
            except Exception:
                self._precompute_dev = torch.device("cpu")
            
            # Move feature extractor to the device
            feat_extr = feat_extr.to(self._precompute_dev)


            # Condition A: Precompute Features
            if feat_extr.precompute_type == "features":
                feature_base_dir = database_path / "features" / feat_extr.signature
                feature_base_dir.mkdir(parents=True, exist_ok=True)

                with open(feature_base_dir / "signature.txt", "w") as f:
                    f.write(feat_extr.full_signature)
                log.info(f"Pre-computing features for: {feat_extr.full_signature}")

            # Condition B: Precompute STFT
            if getattr(feat_extr, "uses_stft", False) and isinstance(feat_extr.transform, STFTtransform):
                stft_signature = feat_extr.transform.signature
                stft_base_dir = database_path / "stft" / stft_signature
                stft_base_dir.mkdir(parents=True, exist_ok=True)
                log.info(f"Pre-computing STFTs for: {stft_signature}")

        # Set random global seed for reproducibility
        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)

        # 3. Iterate over Splits
        for split in ["train", "val", "test"]:
            generator = self._get_scenario_generator(split)
            if not generator or not isinstance(generator, Sized) or len(generator) == 0:
                log.info(f"No scenarios to generate for '{split}' split. Skipping.")
                continue

            # Prepare directories
            dirs = {"raw": database_path / split}
            dirs["raw"].mkdir(parents=True, exist_ok=True)

            if feature_base_dir:
                dirs["feats"] = feature_base_dir / split
                dirs["feats"].mkdir(parents=True, exist_ok=True)
            if stft_base_dir:
                dirs["stft"] = stft_base_dir / split
                dirs["stft"].mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                for i in tqdm(range(len(generator)), desc=f"Pre-computing {split} scenarios"):
                    paths = {k: d / f"scenario_{i}.pt" for k, d in dirs.items()}
                    missing = {k for k, p in paths.items() if not p.exists()}
                    
                    if not missing:
                        continue

                    # Load or Generate Raw Audio
                    if "raw" in missing:
                        data = generator.__getitem__(i)
                        input_tensor = data["input"]
                        meta = data["meta"]
                        
                        raw_save_dict = {"meta": meta, "input_type": "raw_audio"}
                        if "references" not in meta:
                            raw_save_dict["raw_audio"] = input_tensor
                            
                        torch.save(raw_save_dict, paths["raw"])
                    else:
                        loaded = torch.load(paths["raw"], weights_only=False)
                        meta = loaded["meta"]
                        if "raw_audio" in loaded:
                            input_tensor = loaded["raw_audio"]
                        else:
                            input_tensor = torch.stack(list(meta["references"].values())).sum(dim=0)

                    # Compute Derived Data
                    if missing and isinstance(feat_extr, BaseFeatureExtractor):
                        dev = self._precompute_dev
                        
                        derived = move2device(
                            feat_extr.precompute(input_tensor.to(dev)),
                            torch.device("cpu"),
                        )

                        if "stft" in missing and "stft" in derived and isinstance(feat_extr.transform, STFTtransform):
                            torch.save(
                                {
                                    "stft": derived["stft"],
                                    "input_type": "stft",
                                    "stft_info": feat_extr.transform.signature,
                                },
                                paths["stft"]
                            )

                        has_features = "features" in derived
                        has_stacked = any(k[0].isdigit() and "_features" in k for k in derived.keys())

                        if "feats" in missing and (has_features or has_stacked):
                            payload = derived["features"] if has_features else {
                                k: v for k, v in derived.items() if k[0].isdigit() and "_features" in k
                            }
                            torch.save(
                                {
                                    "features": payload,
                                    "input_type": "features",
                                    "feature_info": feat_extr.full_signature,
                                },
                                paths["feats"]
                            )

        # Restore original training state and move back to CPU
        if isinstance(feat_extr, BaseFeatureExtractor):
            feat_extr.train(feat_extr_was_training)
            feat_extr = feat_extr.to("cpu")

    def setup(self, stage: str | None = None) -> None:
        """Smart setup method that assigns train/val/test datasets.
        
        Args:
            stage (Optional[str]): Lightning stage ('fit', 'test', None).
        """
        base_root = self.precomputed_dir

        if not base_root.exists():
            raise FileNotFoundError(f"Base data directory not found: {base_root}. Run prepare_data first.")

        feature_extractor = self.feature_extractor

        if isinstance(feature_extractor, BaseFeatureExtractor):
            precompute_type = feature_extractor.precompute_type
            load_features = (precompute_type == "features") and (not self.force_load_stft)
            load_stft = (precompute_type == "stft") or self.force_load_stft

            if load_features:
                log.info("Loading precomputed features.")
                data_root = base_root / "features" / feature_extractor.signature
            elif load_stft and isinstance(feature_extractor.transform, STFTtransform):
                stft_dir = base_root / "stft" / feature_extractor.transform.signature
                if stft_dir.exists():
                    log.info("Loading precomputed STFTs.")
                    data_root = stft_dir
                else:
                    log.info("STFT directory not found. Loading raw audio.")
                    data_root = base_root
            else:
                log.info("Unknown precompute_type. Loading raw audio.")
                data_root = base_root
        else:
            log.info("No feature extractor defined. Loading raw audio.")
            data_root = base_root

        data_roots = [data_root]

        if stage == "fit" or stage is None:
            self.train_ds = PrecomputedDataset(
                precomputed_dir=[dr / "train" for dr in data_roots],
                preload_to_ram=False,
            )
            self.val_ds = PrecomputedDataset(
                precomputed_dir=[dr / "val" for dr in data_roots],
                preload_to_ram=False,
            )
        if stage in ["test"] or stage is None:
            self.test_ds = PrecomputedDataset(
                precomputed_dir=[dr / "test" for dr in data_roots],
                preload_to_ram=False,
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
            collate_fn=self.train_ds.collate_fn,
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
            collate_fn=self.val_ds.collate_fn,
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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.test_ds.collate_fn,
        )
