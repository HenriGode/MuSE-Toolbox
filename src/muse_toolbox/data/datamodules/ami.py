import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import logging
from tqdm import tqdm

from muse_toolbox.data.datamodules.base_datamodule import BaseDataModule

log = logging.getLogger(__name__)

class AMIDataModule(BaseDataModule):
    """
    DataModule for AMI. Overrides prepare_data to precompute full 1-hour 
    recordings into memory-mappable NumPy arrays split into train/val/test folders.
    """
    def __init__(
        self,
        database,
        train_meetings: list[str],
        val_meetings: list[str],
        test_meetings: list[str],
        chunk_length_s: float,
        min_context_s: float,
        train_items_per_epoch: int,
        arrays: list[str],
        augmentation: dict,
        sampling_frequency: int,
        label_parser=None, # Passed from hydra
        *args, **kwargs
    ):
        super().__init__(
            sampling_frequency=sampling_frequency,
            chunk_length_s=chunk_length_s,
            min_context_s=min_context_s,
            train_items_per_epoch=train_items_per_epoch,
            augmentation=augmentation,
            id="ami",
            num_scenarios=None,
            *args, **kwargs
        )
        self.database = database
        self.train_meetings = train_meetings
        self.val_meetings = val_meetings
        self.test_meetings = test_meetings
        self.arrays = arrays
        
    def _get_database_managers(self) -> list:
        return [self.database]

    def _get_scenario_generator(self, split: str) -> Dataset | None:
        # AMI bypasses dynamic scenario generation. It uses direct precomputation.
        return None
        
    def prepare_data(self):
        if self.data_is_prepared:
            return
            
        self.precomputed_dir.mkdir(parents=True, exist_ok=True)
        log.info("Precomputing continuous AMI meetings to memory-mappable numpy arrays...")
        
        single_speaker_chunks = []
        chunk_samples = int(self.chunk_length_s * self.sampling_frequency)
        hop_samples = int((self.chunk_length_s - self.min_context_s) * self.sampling_frequency)
        
        for split in ['train', 'val', 'test']:
            meetings = getattr(self, f"{split}_meetings")
            split_dir = self.precomputed_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for meeting_id in tqdm(meetings, desc=f"Precomputing AMI {split}"):
                for array_id in self.arrays:
                    # Naming strictly matched by DynamicChunkDataset ("*_audio.npy")
                    key = f"{meeting_id}_{array_id}"
                    audio_path = split_dir / f"{key}_audio.npy"
                    sad_path = split_dir / f"{key}_sad.npy"
                    meta_path = split_dir / f"{key}_meta.pt"
                    
                    if audio_path.exists() and sad_path.exists() and meta_path.exists():
                        pass
                    else:
                        try:
                            audio, sad_dict = self.database.get_meeting(meeting_id, array_id)
                        except Exception as e:
                            log.error(f"Failed to process {meeting_id} {array_id}: {e}")
                            continue
                        
                        # Stack SAD into boolean tensor
                        sad_keys = list(sad_dict.keys())
                        if len(sad_keys) > 0:
                            sad_tensor = torch.stack([sad_dict[k] for k in sad_keys], dim=0).bool()
                        else:
                            sad_tensor = torch.zeros(1, audio.shape[1], dtype=torch.bool)
                        
                        # Save
                        np.save(audio_path, audio.numpy())
                        np.save(sad_path, sad_tensor.numpy())
                        torch.save({"meeting_id": meeting_id, "array_id": array_id}, meta_path)
                    
                    # Build single-speaker index for augmentation
                    if split == 'train' and self.augmentation.get('enabled', False):
                        sad_mmap = np.load(sad_path, mmap_mode='r')                            
                        length = sad_mmap.shape[-1]
                        start_idx = 0
                        while start_idx + chunk_samples <= length:
                            chunk_sad = sad_mmap[:, start_idx:start_idx+chunk_samples]
                            is_single_speaker_chunk = chunk_sad.any(axis=-1).sum() == 1
                            chunk_counts = chunk_sad.sum(axis=0)
                            if np.sum(chunk_counts == 1) >= chunk_samples // 2 and is_single_speaker_chunk:
                                single_speaker_chunks.append(f"{meeting_id}_{array_id}_{start_idx}")
                            start_idx += hop_samples

        if self.augmentation.get('enabled', False):
            index_path = self.precomputed_dir / f"single_speaker_index_{self.sampling_frequency}Hz.npy"
            np.save(index_path, np.array(single_speaker_chunks, dtype=str))
                
        self.data_is_prepared = True
