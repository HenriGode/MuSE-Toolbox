"""Precomputed dataset loader for MuSE-Toolbox.

Provides the dataset class for loading and collating pre-computed 
scenario tensors from disk into `HeterogeneousBatch` objects.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch

log = logging.getLogger(__name__)


class PrecomputedDataset(Dataset):
    """A dataset for loading pre-computed scenario tensors from disk.

    This dataset loads pre-computed .pt files from a standardized directory structure.
    It handles heterogeneous data types and collates inputs into a HeterogeneousBatch.
    """

    def __init__(
        self,
        precomputed_dir: str | Path | list[str | Path],
        preload_to_ram: bool = False,
    ) -> None:
        """Initializes the PrecomputedDataset.

        Args:
            precomputed_dir (Union[str, Path, List[Union[str, Path]]]): Directory or 
                list of directories containing pre-computed 'scenario_*.pt' files.
            preload_to_ram (bool): If True, pre-loads all dataset tensors into RAM 
                during initialization for faster access. Defaults to False.

        Raises:
            TypeError: If `precomputed_dir` is not of the expected type.
            FileNotFoundError: If the specified directory or files are not found.
        """
        # Handle both single path and list of paths
        if isinstance(precomputed_dir, (str, Path)):
            self.precomputed_dirs = [Path(precomputed_dir)]
        elif isinstance(precomputed_dir, list):
            self.precomputed_dirs = [Path(p) for p in precomputed_dir]
        else:
            raise TypeError(
                "precomputed_dir must be a string, Path, or list of strings/Paths."
            )

        self.files: list[Path] = []
        for d in self.precomputed_dirs:
            if not d.exists():
                raise FileNotFoundError(
                    f"Precomputed directory not found: {d}. "
                    "Run the DataModule's prepare_data step first."
                )
            self.files.extend(d.glob("scenario_*.pt"))

        # Sorting ensures that index 'i' always maps to the same file,
        # which is required for reproducibility across different OS file systems.
        self.files = sorted(self.files)

        if not self.files:
            raise FileNotFoundError(
                f"No 'scenario_*.pt' files found in {self.precomputed_dirs}."
            )

        log.info(
            f"Initialized dataset with {len(self.files)} files from {len(self.precomputed_dirs)} directories."
        )

        self.data_in_ram: list[dict[str, Any]] = []
        if preload_to_ram:
            log.info(f"Pre-loading {len(self.files)} files into RAM...")
            # Since self.files is sorted, data_in_ram will be in the exact same order.
            for f in tqdm(self.files, desc="Pre-loading data"):
                self.data_in_ram.append(self._load_scenario(f))
            log.info("...pre-loading complete.")

    def _load_scenario(self, file_path: Path) -> dict[str, Any]:
        """Loads a single scenario from disk.

        Args:
            file_path (Path): Path to the scenario '.pt' file.

        Returns:
            Dict[str, Any]: The loaded scenario dictionary.
        """
        scenario = torch.load(file_path, weights_only=False)
        if scenario["input_type"] == "raw_audio":
            if "raw_audio" not in scenario:
                scenario["raw_audio"] = torch.stack(
                    list(scenario["meta"]["references"].values())
                ).sum(dim=0)
        elif scenario["input_type"] in ["stft", "features"]:
            meta_filename = (
                file_path.parent.parent.parent.parent
                / file_path.parent.name
                / file_path.name
            )
            scenario["meta"] = torch.load(meta_filename, weights_only=False)["meta"]
        return scenario

    def __len__(self) -> int:
        """Gets the total number of scenarios.

        Returns:
            int: The number of files in the dataset.
        """
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Retrieves a single scenario by index.

        Since both self.files and self.data_in_ram (if populated) are sorted
        identically, index always refers to the same scenario.

        Args:
            index (int): The index of the scenario.

        Returns:
            Dict[str, Any]: The scenario data.
        """
        if self.data_in_ram:
            return self.data_in_ram[index]
        else:
            return self._load_scenario(self.files[index])

    def collate_fn(self, batch: list[dict[str, Any]]) -> HeterogeneousBatch:
        """Collates items into a HeterogeneousBatch object.
        
        Crucially, it sorts items so that 'raw_audio' comes first, then 'stft', 
        then 'features'. This aligns with the processing order in 
        `HeterogeneousBatch.unpack_and_process`.
        
        Args:
            batch (List[Dict[str, Any]]): A list of loaded scenario dictionaries.
            
        Returns:
            HeterogeneousBatch: The collated batch ready for model consumption.
                
        Raises:
            ValueError: If there is inconsistent feature info across the batch.
        """
        if not batch:
            return HeterogeneousBatch([], [], [], {})

        # 1. Separate items by type
        raw_items = [x for x in batch if x.get("input_type") == "raw_audio"]
        stft_items = [x for x in batch if x.get("input_type") == "stft"]
        feat_items = [x for x in batch if x.get("input_type") == "features"]

        # 2. Create the sorted list for metadata extraction
        sorted_batch = raw_items + stft_items + feat_items

        # 3. Extract Data
        raw_audio_list = [x["raw_audio"] for x in raw_items]
        stft_audio_list = [x["stft"] for x in stft_items]
        features_list = [x["features"] for x in feat_items]

        # 4a. Extract STFT Info
        stft_info = None
        if stft_items:
            stft_info = stft_items[0].get("stft_info")
        if stft_info is not None:
            for idx, item in enumerate(stft_items[1:], start=1):
                current_info = item.get("stft_info")
                if current_info != stft_info:
                    raise ValueError(
                        f"Inconsistent stft_info in batch. "
                        f"Item 0 has {stft_info}, but item {idx} has {current_info}."
                    )

        # 4b. Extract Feature Info
        feature_info = None
        if feat_items:
            feature_info = feat_items[0].get("feature_info")
            for idx, item in enumerate(feat_items[1:], start=1):
                current_info = item.get("feature_info")
                if current_info != feature_info:
                    raise ValueError(
                        f"Inconsistent feature_info in batch. "
                        f"Item 0 has {feature_info}, but item {idx} has {current_info}."
                    )

        # 5. Collate Metadata
        collated_meta = defaultdict(list)
        if sorted_batch:
            keys = sorted_batch[0]["meta"].keys()
            for item in sorted_batch:
                for k in keys:
                    collated_meta[k].append(item["meta"][k])

        return HeterogeneousBatch(
            raw_audio=raw_audio_list,
            stft_audio=stft_audio_list,
            features=features_list,
            meta=dict(collated_meta),
            stft_info=stft_info,
            feature_info=feature_info,
        )
