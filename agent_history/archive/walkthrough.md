# AMI Dataset Exploration Script Walkthrough

I have successfully created the exploration script to help you investigate the AMI Dataset datamodule's behavior and the mechanics of the data augmentation we implemented.

## What Was Done
I created a new Python script formatted with `# %%` cell markers at [notebooks/ami_exploration.py](file:///data4/Henri/MuSE-Toolbox/notebooks/ami_exploration.py). You can open this file in VS Code and run it interactively exactly like a Jupyter Notebook, which satisfies your preference for version-control-friendly files while retaining inline plotting and audio playback.

## Features Included

### 1. Standard Scenario Visualization
- The script loads a full continuous meeting from the precomputed memory-mapped validation set arrays.
- It plots the first 60 seconds across three aligned panels:
  - **Raw Audio Waveform** (Channel 0)
  - **Individual Speaker Activities (VAD)** (Staggered vertically for clarity)
  - **Aggregate Source Count** (Step-function plot)
- It provides an `IPython.display.Audio` player so you can listen to the downmixed audio while looking at the timeline.

### 2. Chunking Visualization
- Visualizes how the continuous validation scenario is deterministically sliced into fixed 7-second chunks.
- Overlays shaded regions representing consecutive chunks on top of the Source Count plot to clearly show the overlap behavior resulting from your `hop_samples` configuration.

### 3. Augmented Chunks Visualization
- Fetches a dynamically augmented chunk from the training dataloader (forcing `prob_augmentation = 1.0`).
- Plots the resulting synthetic mixture, individual speaker activities, and source count to visually verify our recent implementation of overlap mechanics and relative dB scaling.
- Includes an `Audio` player to listen to the augmented mixture.

### 4. Dataset Statistics & Transition Analysis
- Scans a representative subset of the training set (to keep execution fast) and computes:
  - **Class Distribution:** The percentage of time where 0, 1, 2, 3, or 4 speakers are simultaneously active.
  - **Event Frequency:** The average number of source count changes per second (e.g., 0.8 events/sec $\pm$ 0.1).
  - **Transition Histogram:** A bar chart of the absolute frequencies for all specific state transitions (e.g., $0 \rightarrow 1$, $1 \rightarrow 2$, $2 \rightarrow 3$).
  - **Time Elapsed Distributions:** A grid of density histograms plotting the distribution of time elapsed *before* a specific transition occurred (e.g., how long do we stay in state 2 before transitioning to state 3?).

## Next Steps
You can open [ami_exploration.py](file:///data4/Henri/MuSE-Toolbox/notebooks/ami_exploration.py) in your editor and execute the cells interactively. Let me know if the statistics or plots reveal any unexpected behavior!
