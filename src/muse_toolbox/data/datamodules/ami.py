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

from muse_toolbox.data.datamodules.base_datamodule import BaseDataModule
from muse_toolbox.data.databases.ami import AMIDatabase
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)

class AMIScenarioGeneratorDataset(Dataset):
    """
    Dataset for dynamic slicing of AMI meetings. 
    Supports random sampling (train) and deterministic grid (test/val).
    """
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
        fs: int = 16000,
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
        
        self.chunk_frames = int(chunk_length_s * fs)
        self.hop_frames = int((chunk_length_s - min_context_s) * fs)
        
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
            else:
                self.single_speaker_index = []
                log.warning("Augmentation enabled but no single_speaker_index.npy found.")

    def _build_deterministic_grid(self):
        self.grid = []
        if self.split != 'train':
            for key in self.valid_meetings:
                meeting_id, array_id = key.split('_')
                length = self.meeting_lengths[key]
                start_idx = 0
                while start_idx + self.chunk_frames <= length:
                    self.grid.append((key, start_idx))
                    start_idx += self.hop_frames

    def __len__(self):
        if self.split == 'train':
            return self.steps_per_epoch
        return len(self.grid)

    def _get_random_single_speaker_chunk(self):
        if not hasattr(self, 'single_speaker_index') or not self.single_speaker_index:
            return None, None
            
        choice = random.choice(self.single_speaker_index)
        meeting_id, array_id, start_idx_str = choice.split('_')
        key = f"{meeting_id}_{array_id}"
        start_idx = int(start_idx_str)
        end_idx = start_idx + self.chunk_frames
        
        # Audio is [8, T], SAD is [speakers, T]
        audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
        sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()
        
        # Find the active speaker index
        active_speaker = np.argmax(sad_chunk.sum(axis=1))
        
        return audio_chunk, active_speaker

    def __getitem__(self, idx):
        if self.split == 'train' and self.augmentation.get("enabled", False) and hasattr(self, 'single_speaker_index') and len(self.single_speaker_index) > 0:
            # Determine how many speakers to mix based on probabilities
            probs = [
                self.augmentation.get("prob_1spk", 0.25),
                self.augmentation.get("prob_2spk", 0.25),
                self.augmentation.get("prob_3spk", 0.25),
                self.augmentation.get("prob_4spk", 0.25)
            ]
            num_speakers = random.choices([1, 2, 3, 4], weights=probs, k=1)[0]
            
            mixed_audio = np.zeros((len(self.arrays)*4, self.chunk_frames), dtype=np.float32) # Assume max 8 mics
            mixed_sad = np.zeros((4, self.chunk_frames), dtype=np.bool_) # Max 4 speakers
            
            # Use random sampling for the mix
            for i in range(num_speakers):
                aud, act = self._get_random_single_speaker_chunk()
                if aud is not None:
                    mixed_audio[:aud.shape[0], :] += aud
                    mixed_sad[i, :] = True
                    
            audio_tensor = torch.from_numpy(mixed_audio).float()
            sad_tensor = torch.from_numpy(mixed_sad)
            meeting_id, array_id, start_idx = "mixed", "mixed", 0
        elif self.split == 'train':
            # Random sampling without mix
            key = random.choice(self.valid_meetings)
            length = self.meeting_lengths[key]
            if length > self.chunk_frames:
                start_idx = random.randint(0, length - self.chunk_frames)
            else:
                start_idx = 0
            meeting_id, array_id = key.split('_')
            end_idx = start_idx + self.chunk_frames
            
            audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
            sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()
            audio_tensor = torch.from_numpy(audio_chunk).float()
            sad_tensor = torch.from_numpy(sad_chunk)
        else:
            # Deterministic grid
            key, start_idx = self.grid[idx]
            meeting_id, array_id = key.split('_')
            end_idx = start_idx + self.chunk_frames
            
            audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
            sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()
            audio_tensor = torch.from_numpy(audio_chunk).float()
            sad_tensor = torch.from_numpy(sad_chunk)
            
        # Calculate speaker counts
        # sad_chunk is [speakers, time]. Sum over speakers to get counts
        sad_counts = sad_tensor.sum(dim=0)
        
        # Downsample to sad_frames using STFT transform if available
        if self.transform is not None:
            # Usually STFTtransform requires specific shapes. For now we mock it if needed
            # In PRA_ANF, they usually take max over the frame window.
            hop_length = self.transform.hop_length
            frames = sad_counts.unfold(0, self.transform.nfft, hop_length)
            sad_frames, _ = frames.max(dim=-1)
        else:
            sad_frames = sad_counts # Fallback
            
        meta = {
            "meeting_id": meeting_id,
            "array_id": array_id,
            "start_time": start_idx / self.fs,
            "sad_samples": sad_tensor,
            "sad_frames": sad_frames
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
        self.database = instantiate(database, sampling_frequency=sampling_frequency)
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
        
    def prepare_data(self):
        if self.data_is_prepared:
            return
            
        self.precomputed_dir.mkdir(parents=True, exist_ok=True)
        log.info("Precomputing continuous AMI meetings to memory-mappable numpy arrays...")
        
        # Combine all meetings for precomputation
        all_meetings = (
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
                chunk_frames = int(self.chunk_length_s * self.sampling_frequency)
                hop_frames = int((self.chunk_length_s - self.min_context_s) * self.sampling_frequency)
                
                for meeting_id in all_meetings:
                    for array_id in self.arrays:
                        sad_path = self.precomputed_dir / f"{meeting_id}_{array_id}_sad.npy"
                        if sad_path.exists():
                            sad_mmap = np.load(sad_path, mmap_mode='r')
                            sad_counts = sad_mmap.sum(axis=0)
                            
                            length = sad_counts.shape[0]
                            start_idx = 0
                            while start_idx + chunk_frames <= length:
                                chunk_counts = sad_counts[start_idx:start_idx+chunk_frames]
                                # If speaker count is exactly 1 for the whole chunk
                                if np.all(chunk_counts == 1):
                                    single_speaker_chunks.append(f"{meeting_id}_{array_id}_{start_idx}")
                                start_idx += hop_frames
                
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
