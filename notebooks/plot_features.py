import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

# --- Configuration ---
DATASET = "J3_PRAANF_circ_8ch_20cm_SIR10dB"
SPLIT = "test"
SCENARIO_IDX = 97
TIME_WINDOW = [0,30]
BASE_DIR = f"/data4/Henri/MuSE-Toolbox/data/datasets/{DATASET}"
OUTPUT_DIR = "/data4/Henri/MuSE-Toolbox/notebooks/feature_plots"
# ---------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_params(dirname):
    params = {}
    for part in dirname.split('_'):
        if part.startswith('fs'):
            try:
                params['fs'] = float(part[2:])
            except ValueError:
                pass
        elif part.startswith('sf'):
            try:
                params['sf'] = float(part[2:])
            except ValueError:
                pass
        elif part.startswith('freq'):
            # e.g., freq0.0-4000.0
            try:
                freqs = part[4:].split('-')
                params['fmin'] = float(freqs[0])
                params['fmax'] = float(freqs[1])
            except (ValueError, IndexError):
                pass
    return params

def plot_feature(tensor, name, params, output_path, gt_count=None):
    if isinstance(tensor, dict):
        if 'stft' in tensor:
            tensor = tensor['stft']
        elif 'features' in tensor:
            tensor = tensor['features']
        else:
            for v in tensor.values():
                if isinstance(v, torch.Tensor):
                    tensor = v
                    break

    print(f"Plotting {name} with shape {tensor.shape}")
    
    if tensor.is_complex():
        tensor = tensor.abs()
    tensor = tensor.squeeze()
    
    # Ensure tensor is (channel, freq, time) if 3D
    if tensor.dim() == 3:
        # If dim 0 is larger than dim 1, it's likely (freq, channel, time) like in STFT
        if tensor.shape[0] > tensor.shape[1]:
            tensor = tensor.transpose(0, 1)
            
    # Now we can safely take the first channel/batch
    while tensor.dim() > 2:
        tensor = tensor[0]
        
    data = tensor.detach().cpu().numpy()
    
    fs = params.get('fs', 0.016)
    sf = params.get('sf', 8000.0)
    hop_length = int(fs * sf)
    
    # The golden ratio is ~1.618. For a tall plot, height = width * 1.618
    width = 5
    # fig, ax = plt.subplots(figsize=(width, width * 1.618))
    fig, ax = plt.subplots(figsize=(width, width * 1.309))
    
    is_logmel = 'logmel' in name.lower()
    is_stft = 'stft' in name.lower()
    
    if is_stft:
        # Convert to dB scale for better visualization
        data = 20 * np.log10(np.clip(data, a_min=1e-8, a_max=None))
        vmin, vmax = np.percentile(data, 2), np.percentile(data, 99.9)
        im = librosa.display.specshow(data, sr=sf, hop_length=hop_length, 
                                 x_axis='time', y_axis='linear', cmap='inferno', ax=ax,
                                 vmin=vmin, vmax=vmax)
        # fig.colorbar(im, ax=ax, format='%+2.0f dB')
    elif is_logmel:
        fmin = params.get('fmin', 0.0)
        fmax = params.get('fmax', sf / 2)
        vmin, vmax = np.percentile(data, 2), np.percentile(data, 99.9)
        im = librosa.display.specshow(data, sr=sf, hop_length=hop_length, 
                                 x_axis='time', y_axis='mel', 
                                 fmin=fmin, fmax=fmax, cmap='inferno', ax=ax,
                                 vmin=vmin, vmax=vmax)
        # fig.colorbar(im, ax=ax, format='%+2.0f dB')
    else:
        # Generic plot for other features (e.g., GMSC, IPD)
        if 'gmsc' in name.lower():
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = None, None
            
        im = librosa.display.specshow(data, sr=sf, hop_length=hop_length, 
                                 x_axis='time', cmap='inferno',
                                 vmin=vmin, vmax=vmax, ax=ax)
        # For non-spectrogram features, it is better to label the Y-axis explicitly
        ax.set_ylabel('Feature Dimension')
        # fig.colorbar(im, ax=ax)
        
    if gt_count is not None:
        ax2 = ax.twinx()
        time_vector = np.arange(len(gt_count)) * fs
        ax2.plot(time_vector, gt_count, color='cyan', linewidth=1.5, label='Active Sources')
        ax2.set_ylim(-0.5, 4.5)
        ax2.set_yticks([0, 1, 2, 3, 4])
        ax2.set_ylabel('Source Count')
        
    if TIME_WINDOW is not None:
        ax.set_xlim(TIME_WINDOW[0], TIME_WINDOW[1])
        
    ax.set_title(f"{name} (Scenario {SCENARIO_IDX})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    print(f"Processing dataset: {DATASET}, split: {SPLIT}, scenario: {SCENARIO_IDX}")
    
    # Load ground truth source count
    gt_path = os.path.join(BASE_DIR, SPLIT, f"scenario_{SCENARIO_IDX}.pt")
    gt_count = None
    if os.path.exists(gt_path):
        try:
            main_data = torch.load(gt_path, weights_only=False)
            if 'meta' in main_data and 'sad_frames' in main_data['meta']:
                sad_frames = main_data['meta']['sad_frames']
                first_val = next(iter(sad_frames.values()))
                gt_count = torch.zeros(first_val.shape, dtype=torch.int32)
                for k, v in sad_frames.items():
                    if k.lower() != 'noise':
                        gt_count += v.int()
                gt_count = gt_count.numpy()
                print("Loaded ground truth source counts successfully.")
        except Exception as e:
            print(f"Warning: Could not load ground truth from {gt_path}: {e}")
    else:
        print(f"Ground truth file not found at {gt_path}")
    
    # 1. Plot STFT first
    stft_base = os.path.join(BASE_DIR, "stft")
    if os.path.exists(stft_base):
        stft_dirs = os.listdir(stft_base)
        for stft_dir in stft_dirs:
            stft_path = os.path.join(stft_base, stft_dir, SPLIT, f"scenario_{SCENARIO_IDX}.pt")
            if os.path.exists(stft_path):
                tensor = torch.load(stft_path)
                params = parse_params(stft_dir)
                out_path = os.path.join(OUTPUT_DIR, f"STFT_scenario_{SCENARIO_IDX}.png")
                plot_feature(tensor, "STFT", params, out_path, gt_count=gt_count)
                break
        else:
            print(f"Could not find a valid STFT file for scenario {SCENARIO_IDX} in {stft_base}")
    else:
        print(f"No stft directory found at {stft_base}")
    
    # 2. Plot all other features
    features_base = os.path.join(BASE_DIR, "features")
    if os.path.exists(features_base):
        for feat_dir in os.listdir(features_base):
            if 'pure_stft' in feat_dir.lower() or 'purestft' in feat_dir.lower():
                print(f"Skipping pure stft directory: {feat_dir}")
                continue
            
            feat_path = os.path.join(features_base, feat_dir, SPLIT, f"scenario_{SCENARIO_IDX}.pt")
            if os.path.exists(feat_path):
                tensor = torch.load(feat_path)
                params = parse_params(feat_dir)
                # Feature name is the part before the first underscore
                feat_name = feat_dir.split('_')[0]
                out_path = os.path.join(OUTPUT_DIR, f"{feat_name}_scenario_{SCENARIO_IDX}.png")
                plot_feature(tensor, feat_name, params, out_path, gt_count=gt_count)
            else:
                print(f"Feature file not found: {feat_path}")
    else:
        print(f"No features directory found at {features_base}")

if __name__ == "__main__":
    main()
