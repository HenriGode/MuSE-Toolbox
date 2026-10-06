import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import logging
import random

from muse_toolbox.utils import STFTtransform
from muse_toolbox.data.simulation.scenario_generation import Segment

log = logging.getLogger(__name__)

class DynamicChunkDataset(Dataset):
    """
    Universal dataset for dynamic slicing of memory-mapped recordings.
    Supports random sampling (train) and deterministic grid (test/val).
    """
    def __init__(
        self, 
        data_dir: Path,
        split: str,
        chunk_length_s: float,
        min_context_s: float,
        train_items_per_epoch: int,
        augmentation: dict,
        fs: int,
        transform: STFTtransform | None = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.chunk_length_s = chunk_length_s
        self.min_context_s = min_context_s
        self.train_items_per_epoch = train_items_per_epoch
        self.augmentation = augmentation
        self.fs = fs
        self.transform = transform
        
        self.chunk_samples = int(chunk_length_s * fs)
        self.hop_samples = int((chunk_length_s - min_context_s) * fs)
        
        self.mmap_audio = {}
        self.mmap_sad = {}
        self.meta_data = {}
        self.meeting_lengths = {}
        self.valid_keys = []
        
        self._init_mmaps()
        self._build_deterministic_grid()

    def _init_mmaps(self):
        audio_files = list(self.data_dir.glob("*_audio.npy"))
        if not audio_files:
            log.warning(f"No *_audio.npy files found in {self.data_dir}")
            
        for audio_path in audio_files:
            key = audio_path.name.replace("_audio.npy", "")
            sad_path = self.data_dir / f"{key}_sad.npy"
            meta_path = self.data_dir / f"{key}_meta.pt"
            
            if sad_path.exists() and meta_path.exists():
                audio_mmap = np.load(audio_path, mmap_mode='r')
                sad_mmap = np.load(sad_path, mmap_mode='r')
                
                self.mmap_audio[key] = audio_mmap
                self.mmap_sad[key] = sad_mmap
                self.meta_data[key] = torch.load(meta_path, weights_only=False)
                self.meeting_lengths[key] = audio_mmap.shape[1]
                self.valid_keys.append(key)
            else:
                log.warning(f"Missing precomputed sad/meta data for {key}")
                
        # Load single speaker index if augmentation is enabled
        if self.augmentation and self.augmentation.get("enabled", False) and self.split == 'train':
            # For AMI, the index is usually saved in the parent precomputed dir, not split dir
            idx_path = self.data_dir.parent / f"single_speaker_index_{self.fs}Hz.npy"
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
                log.warning(f"Augmentation enabled but no single_speaker_index.npy found at {idx_path}")

    def _build_deterministic_grid(self):
        self.grid = []
        if self.split != 'train':
            for key in self.valid_keys:
                length = self.meeting_lengths[key]
                start_idx = 0
                while start_idx + self.chunk_samples <= length:
                    self.grid.append((key, start_idx))
                    start_idx += self.hop_samples

    def __len__(self):
        if self.split == 'train':
            return self.train_items_per_epoch
        return len(self.grid)

    def _get_random_single_speaker_chunk(self, key, speaker_id):
        if not hasattr(self, 'single_speaker_dict') or not self.single_speaker_dict:
            return None, None
            
        start_idx = random.choice(self.single_speaker_dict[key][speaker_id])
        end_idx = start_idx + self.chunk_samples
        
        audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
        sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()
        
        return audio_chunk, sad_chunk[speaker_id, :]

    def __getitem__(self, idx):
        apply_aug = False
        if self.split == 'train' and self.augmentation.get("enabled", False):
            if hasattr(self, 'single_speaker_dict') and len(self.single_speaker_dict) > 0:
                prob_aug = self.augmentation.get("prob_augmentation", 0.5)
                apply_aug = random.random() < prob_aug

        # Augmentation logic
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
                valid_keys = [k for k, spk_dict in self.single_speaker_dict.items() if len(spk_dict) > 0]
                if valid_keys:
                    key = random.choice(valid_keys)
                    num_speakers = min(num_speakers, len(self.single_speaker_dict[key]))
                else:
                    apply_aug = False
            else:
                key = random.choice(valid_keys)

        if apply_aug:
            speaker_ids = random.sample(list(self.single_speaker_dict[key].keys()), num_speakers)
            first_aud, first_act = self._get_random_single_speaker_chunk(key, speaker_ids[0])
            
            mixed_audio = np.zeros_like(first_aud, dtype=np.float32)
            mixed_sad = np.zeros((4, self.chunk_samples), dtype=np.bool_) # Max 4 speakers
            
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
            
            base_meta = self.meta_data[key].copy()
            base_meta["scenario_params"] = {"augmentation_applied": True}
        else:
            if self.split == 'train':
                key = random.choice(self.valid_keys)
                length = self.meeting_lengths[key]
                if length > self.chunk_samples:
                    start_idx = random.randint(0, length - self.chunk_samples)
                else:
                    start_idx = 0
            else:
                key, start_idx = self.grid[idx]
                
            end_idx = start_idx + self.chunk_samples
            audio_chunk = self.mmap_audio[key][:, start_idx:end_idx].copy()
            sad_chunk = self.mmap_sad[key][:, start_idx:end_idx].copy()
            
            audio_tensor = torch.from_numpy(audio_chunk).float()
            sad_tensor = torch.from_numpy(sad_chunk)
            
            base_meta = self.meta_data[key].copy()
            base_meta["scenario_params"] = {"augmentation_applied": False}

        # Fix Catch 2: Shift and crop the segments relative to the new chunk timeframe
        start_time = start_idx / self.fs
        end_time = start_time + self.chunk_length_s
        
        if "segments" in base_meta:
            new_segments = []
            for seg in base_meta["segments"]:
                # Check for overlap
                if seg.start < end_time and seg.end > start_time:
                    new_start = max(0.0, seg.start - start_time)
                    new_end = min(self.chunk_length_s, seg.end - start_time)
                    new_segments.append(Segment(start=new_start, end=new_end, num_sources=seg.num_sources, event_type=seg.event_type))
            base_meta["segments"] = new_segments

        # Generate downstream ground-truth labels mathematically
        if self.transform is not None:
            sad_frames = self.transform.samples2frames_quantity(sad_tensor, dim=1)
            source_count = sad_frames.sum(dim=0)
        else:
            sad_frames = None
            source_count = sad_tensor.sum(dim=0)
            
        base_meta.update({
            "scenario_id": f"{key}_{start_time:.2f}",
            "start_time": start_time,
            "sad_samples": sad_tensor,
            "sad_frames": sad_frames,
            "source_count": source_count
        })
        
        return {"input": audio_tensor, "meta": base_meta}
