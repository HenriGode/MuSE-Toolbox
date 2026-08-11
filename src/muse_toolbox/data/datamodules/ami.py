import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import logging
from tqdm import tqdm
from hydra.utils import instantiate
import math
import random

from muse_toolbox.data.components.collate import raw_audio_collate_fn
from muse_toolbox.data.datamodules.base_datamodule import BaseDataModule
from muse_toolbox.data.databases.ami import AMIDatabase
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)

class AMIScenarioGeneratorDataset(Dataset):
    """
    Dataset for dynamic slicing of AMI meetings. 
    Supports random sampling (train) and deterministic grid (test/val).
    """
    collate_fn = staticmethod(raw_audio_collate_fn)
    
    def __init__(
        self, 
        meeting_ids: list[str],
        data_dir: Path,
        split: str,
        chunk_length_s: float,
        min_context_s: float,
        steps_per_epoch: int,
        augmentation: dict,
        arrays: list[str],
        fs: int,
        transform: STFTtransform | None = None
    ):
        self.meeting_ids = meeting_ids
        self.data_dir = data_dir
        self.split = split
        self.chunk_length_s = chunk_length_s
        self.min_context_s = min_context_s
        self.steps_per_epoch = steps_per_epoch
        self.augmentation = augmentation
        self.arrays = arrays
        self.fs = fs
        self.transform = transform
        
        self.chunk_samples = int(chunk_length_s * fs)
        self.hop_samples = int((chunk_length_s - min_context_s) * fs)
        
        # We will hold references to mmap'd numpy arrays for fast loading
        self.mmap_audio = {}
        self.mmap_sad = {}
        self.meeting_lengths = {}
        self.valid_meetings = []
        
        self._init_mmaps()
        self._build_deterministic_grid()

    def _init_mmaps(self):
        for meeting_id in self.meeting_ids:
            for array_id in self.arrays:
                audio_path = self.data_dir / f"{meeting_id}_{array_id}_audio.npy"
                sad_path = self.data_dir / f"{meeting_id}_{array_id}_sad.npy"
                
                if audio_path.exists() and sad_path.exists():
                    audio_mmap = np.load(audio_path, mmap_mode='r')
                    sad_mmap = np.load(sad_path, mmap_mode='r')
                    
                    key = f"{meeting_id}_{array_id}"
                    self.mmap_audio[key] = audio_mmap
                    self.mmap_sad[key] = sad_mmap
                    self.meeting_lengths[key] = audio_mmap.shape[1]
                    self.valid_meetings.append(key)
                else:
                    log.warning(f"Missing precomputed data for {meeting_id} {array_id}")
                    
        # Load single speaker index
        if self.augmentation and self.augmentation.get("enabled", False) and self.split == 'train':
            idx_path = self.data_dir / "single_speaker_index.npy"
            if idx_path.exists():
                self.single_speaker_index = np.load(idx_path).tolist()
                
                self.single_speaker_dict = {}
                for item in self.single_speaker_index:
                    m_id, a_id, s_idx = item.split('_')
                    key = f"{m_id}_{a_id}"
                    start_idx = int(s_idx)
                    end_idx = start_idx + self.chunk_samples
                    
                    if key in self.mmap_sad:
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
                log.warning("Augmentation enabled but no single_speaker_index.npy found.")

    def _build_deterministic_grid(self):
        self.grid = []
        if self.split != 'train':
            for key in self.valid_meetings:
                meeting_id, array_id = key.split('_')
                length = self.meeting_lengths[key]
                start_idx = 0
                while start_idx + self.chunk_samples <= length:
                    self.grid.append((key, start_idx))
                    start_idx += self.hop_samples

    def __len__(self):
        if self.split == 'train':
            return self.steps_per_epoch
        return len(self.grid)

    def _get_random_single_speaker_chunk(self, key, speaker_id):
        if not hasattr(self, 'single_speaker_dict') or not self.single_speaker_dict:
            return None, None
            
        start_idx = random.choice(self.single_speaker_dict[key][speaker_id])
        end_idx = start_idx + self.chunk_samples
        
        # Audio is [M, T], SAD is [speakers, T]
        audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
        sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()
        
        return audio_chunk, sad_chunk[speaker_id, :]

    def __getitem__(self, idx):
        # Determine if we should apply augmentation for this specific item
        apply_aug = False
        if self.split == 'train' and self.augmentation.get("enabled", False):
            if hasattr(self, 'single_speaker_dict') and len(self.single_speaker_dict) > 0:
                prob_aug = self.augmentation.get("prob_augmentation", 0.5)
                apply_aug = random.random() < prob_aug

        # If aug is requested, try to configure it
        if apply_aug:
            probs = [
                self.augmentation.get("prob_1spk", 0.25),
                self.augmentation.get("prob_2spk", 0.25),
                self.augmentation.get("prob_3spk", 0.25),
                self.augmentation.get("prob_4spk", 0.25)
            ]
            num_speakers = random.choices([1, 2, 3, 4], weights=probs, k=1)[0]
            
            valid_keys = [k for k, spk_dict in self.single_speaker_dict.items() if len(spk_dict) >= num_speakers]
            
            if not valid_keys:
                # Fallback to max available speakers if no key has enough
                valid_keys = [k for k, spk_dict in self.single_speaker_dict.items() if len(spk_dict) > 0]
                if valid_keys:
                    key = random.choice(valid_keys)
                    num_speakers = min(num_speakers, len(self.single_speaker_dict[key]))
                else:
                    apply_aug = False
            else:
                key = random.choice(valid_keys)

        if apply_aug:
            # Sample distinct speaker IDs without replacement
            speaker_ids = random.sample(list(self.single_speaker_dict[key].keys()), num_speakers)
            
            # Fetch first chunk to get the correct audio shape for this array
            first_aud, first_act = self._get_random_single_speaker_chunk(key, speaker_ids[0])
            
            mixed_audio = np.zeros_like(first_aud, dtype=np.float32)
            mixed_sad = np.zeros((4, self.chunk_samples), dtype=np.bool_) # Max 4 speakers
            
            mixed_audio += first_aud
            mixed_sad[0, :] = first_act
            
            # Mix in the rest of the distinct speakers
            for i in range(1, num_speakers):
                aud, act = self._get_random_single_speaker_chunk(key, speaker_ids[i])
                mixed_audio += aud
                mixed_sad[i, :] = act
                    
            audio_tensor = torch.from_numpy(mixed_audio).float()
            sad_tensor = torch.from_numpy(mixed_sad)
            meeting_id, array_id = key.split('_')
            start_idx = 0
        elif self.split == 'train':
            # Random sampling without mix
            key = random.choice(self.valid_meetings)
            length = self.meeting_lengths[key]
            if length > self.chunk_samples:
                start_idx = random.randint(0, length - self.chunk_samples)
            else:
                start_idx = 0
            meeting_id, array_id = key.split('_')
            end_idx = start_idx + self.chunk_samples
            
            audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
            sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()
            audio_tensor = torch.from_numpy(audio_chunk).float()
            sad_tensor = torch.from_numpy(sad_chunk)
        else:
            # Deterministic grid
            key, start_idx = self.grid[idx]
            meeting_id, array_id = key.split('_')
            end_idx = start_idx + self.chunk_samples
            
            audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
            sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()
            audio_tensor = torch.from_numpy(audio_chunk).float()
            sad_tensor = torch.from_numpy(sad_chunk)
            
        # Downsample to sad_frames using STFT transform if available
        if self.transform is not None:
            sad_frames = self.transform.samples2frames_quantity(sad_tensor, dim=1)
            source_count = sad_frames.sum(dim=0)
        else:
            sad_frames = None # Fallback
            source_count = sad_tensor.sum(dim=0)
            
        meta = {
            "scenario_id": f"{meeting_id}_{array_id}_{start_idx / self.fs:.2f}",
            "scenario_params": {},
            "meeting_id": meeting_id,
            "array_id": array_id,
            "start_time": start_idx / self.fs,
            "sad_samples": sad_tensor,
            "sad_frames": sad_frames,
            "source_count": source_count
        }
        
        return {"input": audio_tensor, "meta": meta}


class AMIDataModule(BaseDataModule):
    """
    DataModule for AMI. Overrides prepare_data to precompute full 1-hour 
    recordings into memory-mappable NumPy arrays instead of static chunks.
    """
    def __init__(
        self,
        database,
        train_meetings: list[str],
        val_meetings: list[str],
        test_meetings: list[str],
        chunk_length_s: float,
        min_context_s: float,
        steps_per_epoch: int,
        arrays: list[str],
        augmentation: dict,
        sampling_frequency: int,
        label_parser=None, # Passed from hydra
        *args, **kwargs
    ):
        super().__init__(sampling_frequency=sampling_frequency, *args, **kwargs)
        self.database = database
        self.train_meetings = train_meetings
        self.val_meetings = val_meetings
        self.test_meetings = test_meetings
        self.chunk_length_s = chunk_length_s
        self.min_context_s = min_context_s
        self.steps_per_epoch = steps_per_epoch
        self.arrays = arrays
        self.augmentation = augmentation
        self.sampling_frequency = sampling_frequency
        self.precomputed_dir = self.data_dir / "ami_precomputed"
        
    def _get_database_managers(self) -> list:
        return [self.database]
        
    def prepare_data(self):
        if self.data_is_prepared:
            return
            
        self.precomputed_dir.mkdir(parents=True, exist_ok=True)
        log.info("Precomputing continuous AMI meetings to memory-mappable numpy arrays...")
        
        # Combine all meetings for precomputation
        all_meetings = np.unique(
            self.train_meetings + 
            self.val_meetings + 
            self.test_meetings
        )
        
        for meeting_id in tqdm(all_meetings, desc="Precomputing AMI"):
            for array_id in self.arrays:
                audio_path = self.precomputed_dir / f"{meeting_id}_{array_id}_audio.npy"
                sad_path = self.precomputed_dir / f"{meeting_id}_{array_id}_sad.npy"
                
                if audio_path.exists() and sad_path.exists():
                    continue
                    
                try:
                    audio, sad_dict = self.database.get_meeting(meeting_id, array_id)
                except Exception as e:
                    log.error(f"Failed to process {meeting_id} {array_id}: {e}")
                    continue
                
                # Stack SAD into tensor [num_speakers, num_samples]
                sad_keys = list(sad_dict.keys())
                if len(sad_keys) > 0:
                    sad_tensor = torch.stack([sad_dict[k] for k in sad_keys], dim=0)
                else:
                    sad_tensor = torch.zeros(1, audio.shape[1], dtype=torch.bool)
                
                # Save as numpy for mmap_mode
                np.save(audio_path, audio.numpy())
                np.save(sad_path, sad_tensor.numpy())
                
        # Build single-speaker index for augmentation
        if self.augmentation.get('enabled', False):
            index_path = self.precomputed_dir / "single_speaker_index.npy"
            if not index_path.exists():
                log.info("Building single-speaker index for data augmentation...")
                single_speaker_chunks = []
                chunk_samples = int(self.chunk_length_s * self.sampling_frequency)
                hop_samples = int((self.chunk_length_s - self.min_context_s) * self.sampling_frequency)
                
                for meeting_id in tqdm(all_meetings, desc="Building single-speaker index"):
                    for array_id in self.arrays:
                        sad_path = self.precomputed_dir / f"{meeting_id}_{array_id}_sad.npy"
                        if sad_path.exists():
                            sad_mmap = np.load(sad_path, mmap_mode='r')                            
                            length = sad_mmap.shape[-1]
                            start_idx = 0
                            while start_idx + chunk_samples <= length:
                                chunk_sad = sad_mmap[:, start_idx:start_idx+chunk_samples]
                                is_single_speaker_chunk = chunk_sad.any(axis=-1).sum() == 1
                                chunk_counts = chunk_sad.sum(axis=0)
                                # If speaker count is exactly 1 for at least the half of the samples in the chunk and never >1
                                if np.sum(chunk_counts == 1) >= chunk_samples // 2 and is_single_speaker_chunk:
                                    single_speaker_chunks.append(f"{meeting_id}_{array_id}_{start_idx}")
                                start_idx += hop_samples
                
                # Save as string array
                np.save(index_path, np.array(single_speaker_chunks, dtype=str))
                
        self.data_is_prepared = True

    def _get_scenario_generator(self, split: str) -> Dataset:
        # We override standard pipeline to use our dynamic chunking dataset
        meetings = getattr(self, f"{split}_meetings")
        
        return AMIScenarioGeneratorDataset(
            meeting_ids=meetings,
            data_dir=self.precomputed_dir,
            split=split,
            chunk_length_s=self.chunk_length_s,
            min_context_s=self.min_context_s,
            steps_per_epoch=self.steps_per_epoch,
            augmentation=self.augmentation,
            arrays=self.arrays,
            fs=self.sampling_frequency,
            transform=self.feature_extractor.transform if hasattr(self, 'feature_extractor') else None
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self._get_scenario_generator('train')
            self.val_dataset = self._get_scenario_generator('val')
        if stage == 'test' or stage is None:
            self.test_dataset = self._get_scenario_generator('test')
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=raw_audio_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=raw_audio_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=raw_audio_collate_fn)
