#!/usr/bin/env python
# coding: utf-8

# # VAD Configuration Investigation
# This notebook allows you to quickly generate a small raw dataset with your VAD settings and visualize the sources and ground truth labels (SAD).

# In[3]:


import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Setup project root
PROJECT_ROOT = Path(os.getcwd()).parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from hydra import initialize, compose
from hydra.utils import instantiate


# In[ ]:


# Initialize Hydra and instantiate the Datamodule.
# We override the number of scenarios to just 10 in the train split,
# and we disable the feature extractor and transform to just get raw data quickly.
with initialize(version_base="1.3", config_path="../configs"):
    cfg = compose(config_name="default", overrides=[
        "experiment=sanity_check",
        "dataset.num_scenarios=[10, 0, 0]",
        "dataset.reset=True",
        "model.feature_extractor=null"
        "dataset.generation_config.remove_silence=False",
        "dataset.generation_config.vad_threshold2select_clean_speech=-40",
        "dataset.generation_config.bridge_clean_speech_gaps=0.25",
        "dataset.generation_config.vad_threshold2define_oracle=-60",
        "dataset.generation_config.neglect_silence4oracle_sa=0.4"
    ])

print("Instantiating datamodule...")
datamodule = instantiate(cfg.dataset)

print("Generating 10 raw scenarios (this will use your new VAD settings)...")
datamodule.prepare_data()


# In[5]:


# Load the generated dataset
from muse_toolbox.data.components.precomputed_dataset import PrecomputedDataset

split_dir = datamodule.precomputed_dir / "train"
dataset = PrecomputedDataset(precomputed_dir=[split_dir], preload_to_ram=False)
print(f"Loaded dataset with {len(dataset)} scenarios.")


# In[6]:


def preprocess_item(item):
    meta = item['meta']
    raw_audio = item.get('raw_audio', None)
    if isinstance(raw_audio, torch.Tensor): raw_audio = raw_audio[0,:].float().cpu().numpy()
    params = meta.get('scenario_params', {})
    references = meta.get('references', {})
    sad_samples = meta.get('sad_samples', {})
    
    fs = 8000
    hop = 128
            
    speaker_start_times = []
    for k in references.keys():
        if k == 'noise' or k == 'noisy': continue
        start_idx = float('inf')
        if k in sad_samples:
            v = sad_samples[k]
            if isinstance(v, torch.Tensor): v = v.cpu().numpy()
            active = np.where(v > 0.5)[0]
            if len(active) > 0: start_idx = active[0]
        speaker_start_times.append((k, start_idx))
    speaker_start_times.sort(key=lambda x: x[1])
    sorted_speakers = [x[0] for x in speaker_start_times]
    
    ordered_labels = ['noisy']
    if 'noise' in references: ordered_labels.append('noise')
    ordered_labels.extend(sorted_speakers)
    
    signals = {}
    vads = {}
    
    # Extract mono channel (channel 0) for each signal
    for k, sig in references.items():
        if sig.ndim > 1:
            sig = sig[0]  # Mono
        signals[k] = sig.float().cpu().numpy()
        
        v = None
        if k in sad_samples:
            v = sad_samples[k]
            if isinstance(v, torch.Tensor): v = v.float().cpu().numpy()
        vads[k] = v
        
    # If noisy not in references, sum components
    if 'noisy' not in signals:
        if len(references) > 0:
            mix = sum(signals[k] for k in signals)
            signals['noisy'] = mix
            vads['noisy'] = None
        elif raw_audio is not None:
            signals['noisy'] = raw_audio
            vads['noisy'] = None

    num_samples = len(signals['noisy'])
    time_wave = np.arange(num_samples) / fs
    
    valid_sads = []
    for k, v in sad_samples.items():
        if k != 'noise' and k != 'noisy':
            if isinstance(v, torch.Tensor): v = v.float().cpu().numpy()
            valid_sads.append(v)
    gt_count_samples = np.sum(valid_sads, axis=0) if len(valid_sads) > 0 else np.zeros_like(time_wave)

    return {
        'fs': fs, 'hop': hop,
        'signals': signals, 'vads': vads,
        'labels': ordered_labels,
        'time_wave': time_wave,
        'gt_count_samples': gt_count_samples,
    }


# In[7]:


def create_plot(processed_data, scenario_idx):
    if not processed_data: return
    labels = processed_data['labels']
    sigs = processed_data['signals']
    vads = processed_data['vads']
    time_wave = processed_data['time_wave']
    
    num_src = len(labels)
    total_rows = num_src * 2 + 1
    
    all_sigs_vals = [sigs[lbl] for lbl in labels if lbl in sigs]
    max_amp = max([np.max(np.abs(s)) for s in all_sigs_vals if len(s) > 0], default=1.0)
    if max_amp == 0: max_amp = 1.0
    wave_ylim = [-max_amp * 1.1, max_amp * 1.1]
    
    height_ratios = []
    for _ in range(num_src):
        height_ratios.extend([2, 1])
    height_ratios.append(1) # For GT count plot
    
    fig_h = num_src * 1.5 + 1.5
    fig = plt.figure(figsize=(12, fig_h))
    gs = fig.add_gridspec(total_rows, 2, width_ratios=[50, 1], height_ratios=height_ratios, hspace=0.0, wspace=0.1)
    
    axes = []
    for r in range(total_rows):
        sharex = axes[0] if r > 0 else None
        ax = fig.add_subplot(gs[r, 0], sharex=sharex)
        axes.append(ax)
        
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    label_colors = {lbl: colors[i % 10] for i, lbl in enumerate(labels)}
    
    spec_ims = []
    global_spec_max_db = -np.inf
    
    for i, lbl in enumerate(labels):
        ax_spec = axes[i*2]
        ax_wave = axes[i*2 + 1]
        
        if lbl not in sigs: continue
        s = sigs[lbl]
        v = vads.get(lbl, None)
        c = label_colors[lbl]
        disp_lbl = lbl if not lbl.startswith("source") else f"S{lbl.split()[-1]}"
        
        Pxx, freqs, bins, im = ax_spec.specgram(s, NFFT=512, Fs=processed_data['fs'], noverlap=256, cmap='inferno')
        spec_ims.append(im)
        
        if len(Pxx) > 0:
             pxx_safe = np.maximum(Pxx, 1e-10)
             curr_max = 10 * np.log10(np.max(pxx_safe))
             if curr_max > global_spec_max_db: global_spec_max_db = curr_max
                 
        ax_spec.set_ylabel("Freq")
        ax_spec.text(0.01, 0.8, f"{disp_lbl}", transform=ax_spec.transAxes, color=c, fontweight='bold')
        
        ax_wave.plot(time_wave, s, color=c, lw=0.8)
        if v is not None:
            if len(v) != len(time_wave):
                  v_t = torch.tensor(v).float().view(1,1,-1)
                  v_up = F.interpolate(v_t, size=len(time_wave), mode='nearest')
                  v = v_up.squeeze().numpy()
            ax_wave.fill_between(time_wave, -1e9, 1e9, where=(v > 0.5), color=c, alpha=0.2, transform=ax_wave.get_xaxis_transform())
        ax_wave.set_ylabel("Amp")
        ax_wave.set_ylim(wave_ylim)
        
    if global_spec_max_db == -np.inf: global_spec_max_db = 0
    spec_vmax = global_spec_max_db
    spec_vmin = spec_vmax - 80 
    for im in spec_ims: im.set_clim(spec_vmin, spec_vmax)
        
    if spec_ims:
        cax_spec = fig.add_subplot(gs[0 : num_src*2, 1])
        fig.colorbar(spec_ims[0], cax=cax_spec, label='dB')

    # GT count plot
    ax_gt = axes[-1]
    gt_count = processed_data['gt_count_samples']
    mx_c = max(np.max(gt_count) if len(gt_count) > 0 else 0, 3)
    ax_gt.plot(time_wave, gt_count, color='cyan', linestyle='-', lw=2, label='GT Source Count')
    ax_gt.set_ylabel("Count")
    ax_gt.set_ylim(-0.5, mx_c + 0.5)
    ax_gt.legend(loc='upper right', frameon=True, facecolor='black', framealpha=0.6, labelcolor='white')
    ax_gt.set_xlabel("Time [s]")
    
    for i, ax in enumerate(axes):
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)
        if i != len(axes) - 1:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
    axes[0].set_xlim(time_wave[0], time_wave[-1])
    fig.suptitle(f"Scenario {scenario_idx}", fontsize=14, y=0.92)
    plt.show()


# In[8]:


# Visualize all 10 scenarios
for i in range(len(dataset)):
    item = dataset[i]
    processed = preprocess_item(item)
    create_plot(processed, i)

