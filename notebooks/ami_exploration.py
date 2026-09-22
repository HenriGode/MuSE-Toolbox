# %% [markdown]
# # AMI Dataset Exploration & Statistics
# This notebook provides a comprehensive look into the AMI dataset's continuous scenarios, chunking behavior, data augmentation mechanics, and label statistics.

# %%
import os
import sys
from pathlib import Path
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio

plt.style.use('seaborn-v0_8-darkgrid')

# Setup project root
PROJECT_ROOT = Path(os.getcwd()).parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from hydra import initialize, compose
from hydra.utils import instantiate

# %% [markdown]
# ## 1. Setup & Initialization
# Instantiate the AMI DataModule via Hydra configuration to access the precomputed arrays and datasets.

# %%
# Load configuration
with initialize(version_base="1.3", config_path="../configs"):
    cfg = compose(config_name="default", overrides=["experiment=data_check"])

print("Instantiating AMIDataModule...")
datamodule = instantiate(cfg.dataset)
datamodule.prepare_data()
datamodule.setup('fit')

train_dataset = datamodule.train_dataset
val_dataset = datamodule.val_dataset

print(f"Train Dataset size: {len(train_dataset)} chunks")
print(f"Val Dataset size: {len(val_dataset)} chunks")

# %% [markdown]
# ## 2. Standard Scenario Visualization
# We load a continuous meeting directly from the precomputed memory-mapped arrays and visualize the full timeline.

# %%
# Select a random valid meeting from the validation set to avoid augmented noise
valid_keys = val_dataset.valid_meetings
scenario_key = valid_keys[0] # Deterministic for reproducibility
print(f"Visualizing Scenario: {scenario_key}")

audio_mmap = val_dataset.mmap_audio[scenario_key]
sad_mmap = val_dataset.mmap_sad[scenario_key]

# We will plot the first 60 seconds
fs = val_dataset.fs
print(f"Sampling Rate: {fs} Hz")
duration_s = 60
num_samples = min(int(duration_s * fs), audio_mmap.shape[1])

audio_segment = audio_mmap[:, :num_samples]
sad_segment = sad_mmap[:, :num_samples]
time_axis = np.arange(num_samples) / fs
source_count = sad_segment.sum(axis=0)

fig, axes = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 2, 1]}, sharex=True)

# 1. Audio Waveform
# We plot the first channel for simplicity
axes[0].plot(time_axis, audio_segment[0, :], color='gray', alpha=0.8)
axes[0].set_title(f"Audio Waveform (Channel 0) - {scenario_key}")
axes[0].set_ylabel("Amplitude")

# 2. Individual VADs
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i in range(sad_segment.shape[0]):
    # Offset each speaker vertically for visibility
    color = colors[i % len(colors)]
    axes[1].plot(time_axis, sad_segment[i, :] * 0.8 + i, label=f"Speaker {i+1}", color=color)
axes[1].set_title("Individual Speaker Activities (VAD)")
axes[1].set_yticks(np.arange(0.4, sad_segment.shape[0], 1))
axes[1].set_yticklabels([f"Spk {i+1}" for i in range(sad_segment.shape[0])])

# 3. Source Count
axes[2].plot(time_axis, source_count, color='purple', drawstyle='steps-post')
axes[2].set_title("Aggregate Source Count")
axes[2].set_ylabel("Concurrent Speakers")
axes[2].set_xlabel("Time (s)")
axes[2].set_yticks([0, 1, 2, 3, 4])

plt.tight_layout()
plt.show()

# Audio playback (downmix to mono for playback)
mono_audio = audio_segment.mean(axis=0)
display(Audio(mono_audio, rate=fs))

# %% [markdown]
# ## 3. Chunking Visualization
# Visualize how the continuous audio is sliced into overlapping chunks using the deterministic grid from the validation set.

# %%
chunk_length_s = val_dataset.chunk_length_s
min_context_s = val_dataset.min_context_s
chunk_samples = val_dataset.chunk_samples
hop_samples = val_dataset.hop_samples

print(f"Chunk Length: {chunk_length_s} s")
print(f"Hop Length: {(hop_samples) / fs} s")
print(f"Overlap: {(chunk_samples - hop_samples) / fs} s")

fig, ax = plt.subplots(figsize=(15, 3))
ax.plot(time_axis, source_count, color='gray', drawstyle='steps-post', alpha=0.5, label="Source Count")

# Plot the boundaries of the first 5 chunks
num_chunks_to_plot = 5
colors = plt.cm.Set2(np.linspace(0, 1, num_chunks_to_plot))

for i in range(num_chunks_to_plot):
    # This assumes the grid starts at 0 for this scenario
    start_time = (i * hop_samples) / fs
    end_time = start_time + chunk_length_s
    
    # Draw a shaded span for each chunk
    ax.axvspan(start_time, end_time, color=colors[i], alpha=0.3, label=f"Chunk {i+1}")
    ax.text(start_time + 0.1, 4.2 - (i*0.4), f"Chunk {i+1}", color=colors[i][:3], fontweight='bold')

ax.set_title("Deterministic Grid Chunking Visualization")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Source Count")
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlim(0, 30)
plt.show()

# %% [markdown]
# ## 4. Augmented Chunks Visualization
# Fetch dynamically augmented chunks from the training dataloader to verify the temporal offsets, overlap mechanics, and relative dB scaling.

# %%
# Force augmentation to be true for this test
train_dataset.augmentation["prob_augmentation"] = 1.0

# Fetch an augmented chunk
item = train_dataset[random.randint(0, len(train_dataset)-1)]

print("--- Chunk Statistics ---")
scenario_params = item["meta"].get("scenario_params", {})
if scenario_params.get("augmentation_applied", False):
    print("Augmentation: APPLIED")
    speaker_ids = scenario_params.get("speaker_ids", [])
    gains = scenario_params.get("gains", [])
    print(f"Number of Speakers: {len(speaker_ids)}")
    for i, (sid, gain) in enumerate(zip(speaker_ids, gains)):
        print(f"  Speaker {i+1} ({sid}): Gain = {gain:.3f} (linear)")
    
    clip_ratio = scenario_params.get("clipping_ratio", 1.0)
    if clip_ratio < 1.0:
        print(f"Global Clipping Protection Applied (Ratio: {clip_ratio:.3f})")
else:
    print("Augmentation: NONE (Original Chunk)")

aug_audio = item["input"].numpy()
aug_sad = item["meta"]["sad_samples"].numpy()
aug_source_count = aug_sad.sum(axis=0)
constituent_audio = item["meta"].get("single_source_components", None)
if constituent_audio is not None:
    constituent_audio = constituent_audio.numpy()

time_axis_chunk = np.arange(aug_audio.shape[1]) / fs

# Determine number of active speakers to plot
num_active = constituent_audio.shape[0] if constituent_audio is not None else 1

fig, axes = plt.subplots(3 + num_active, 1, figsize=(15, 10 + 2*num_active), sharex=True)

# 1. Mixture Audio Waveform
axes[0].plot(time_axis_chunk, aug_audio[0, :], color='black', alpha=0.8)
axes[0].set_title("Augmented Mixture Waveform (Channel 0)")
axes[0].set_ylabel("Amplitude")

# 2. Constituent Audio Waveforms
if constituent_audio is not None:
    for i in range(num_active):
        color = colors[i % len(colors)]
        axes[1+i].plot(time_axis_chunk, constituent_audio[i, 0, :], color=color, alpha=0.8)
        axes[1+i].set_title(f"Single Speaker {i+1} Waveform (Channel 0)")
        axes[1+i].set_ylabel("Amplitude")

# 3. Individual VADs
vad_ax = axes[1 + num_active]
for i in range(aug_sad.shape[0]):
    color = colors[i % len(colors)]
    vad_ax.plot(time_axis_chunk, aug_sad[i, :] * 0.8 + i, label=f"Speaker {i+1}", color=color)
vad_ax.set_title("Individual Speaker Activities in Mixture")
vad_ax.set_yticks(np.arange(0.4, aug_sad.shape[0], 1))
vad_ax.set_yticklabels([f"Spk {i+1}" for i in range(aug_sad.shape[0])])

# 4. Source Count
count_ax = axes[2 + num_active]
count_ax.plot(time_axis_chunk, aug_source_count, color='purple', drawstyle='steps-post')
count_ax.set_title("Augmented Source Count")
count_ax.set_ylabel("Concurrent Speakers")
count_ax.set_xlabel("Time (s)")
count_ax.set_yticks([0, 1, 2, 3, 4])

plt.tight_layout()
plt.show()

print("--- Augmented Mixture Audio ---")
mono_aug_audio = aug_audio.mean(axis=0)
display(Audio(mono_aug_audio, rate=fs))

if constituent_audio is not None:
    for i in range(num_active):
        print(f"--- Single Speaker {i+1} Audio ---")
        mono_single = constituent_audio[i].mean(axis=0)
        display(Audio(mono_single, rate=fs))

# %% [markdown]
# ## 5. Dataset Statistics & Transition Analysis
# Compute source count distribution and state transition statistics over a subset of the dataset.

# %%
print("Computing Dataset Statistics...")

total_samples = 0
class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
transitions = {}
time_since_last_event = {f"{i}->{j}": [] for i in range(5) for j in range(5) if i != j}

# To speed up, we'll process 10 random full meetings from the training set
num_meetings_to_analyze = min(10, len(train_dataset.valid_meetings))
sample_meetings = random.sample(train_dataset.valid_meetings, num_meetings_to_analyze)

all_events_per_sec = []

for key in sample_meetings:
    sad_matrix = train_dataset.mmap_sad[key]
    source_count = sad_matrix.sum(axis=0)
    
    # Class distribution
    unique, counts = np.unique(source_count, return_counts=True)
    for u, c in zip(unique, counts):
        if u in class_counts:
            class_counts[u] += c
    
    total_samples += len(source_count)
    
    # Transitions
    # Find indices where state changes
    changes = np.where(source_count[:-1] != source_count[1:])[0]
    
    num_events = len(changes)
    duration_s = len(source_count) / fs
    all_events_per_sec.append(num_events / duration_s)
    
    last_change_idx = 0
    for idx in changes:
        from_state = int(source_count[idx])
        to_state = int(source_count[idx + 1])
        trans_key = f"{from_state}->{to_state}"
        
        if trans_key not in transitions:
            transitions[trans_key] = 0
        transitions[trans_key] += 1
        
        time_elapsed_s = (idx - last_change_idx) / fs
        if trans_key in time_since_last_event:
            time_since_last_event[trans_key].append(time_elapsed_s)
        
        last_change_idx = idx

print("\n--- Overall Class Distribution ---")
for k, v in class_counts.items():
    percentage = (v / total_samples) * 100 if total_samples > 0 else 0
    print(f"{k} Speakers: {percentage:.2f}%")

print(f"\n--- Event Frequency ---")
print(f"Average Source Count Changes per Second: {np.mean(all_events_per_sec):.3f} ± {np.std(all_events_per_sec):.3f}")

all_intervals = []
for intervals in time_since_last_event.values():
    all_intervals.extend(intervals)
    
if all_intervals:
    print(f"Average Time Between Events: {np.mean(all_intervals):.3f}s ± {np.std(all_intervals):.3f}s")

# %%
# Plot Transition Histogram
trans_keys = list(transitions.keys())
trans_counts = list(transitions.values())

# Sort by count for better visualization
sorted_indices = np.argsort(trans_counts)[::-1]
trans_keys = [trans_keys[i] for i in sorted_indices]
trans_counts = [trans_counts[i] for i in sorted_indices]

plt.figure(figsize=(12, 5))
plt.bar(trans_keys[:15], trans_counts[:15], color='teal') # Plot top 15 transitions
plt.title("Top Source Count Transitions")
plt.xlabel("Transition (From -> To)")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# %%
# Plot Time Since Last Event for top 4 transitions
top_4_trans = trans_keys[:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, t_key in enumerate(top_4_trans):
    times = time_since_last_event[t_key]
    # Filter extreme outliers for visualization
    times = [t for t in times if t < 10.0] 
    
    axes[i].hist(times, bins=50, color='coral', alpha=0.7, density=True)
    axes[i].set_title(f"Time Elapsed Before Transition: {t_key}")
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Density")

plt.tight_layout()
plt.show()

# %%
