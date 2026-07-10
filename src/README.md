# Source Code (src)
**The Core Logic of MuSE-Toolbox**

## Purpose
This directory contains the `muse_toolbox` python package. This is the horizontal slice of the project where reusable logic lives.

## Structure
- **muse_toolbox/**: The main Python package.
  - **models/**: Neural network architectures (RNNs, Transformers, Conformers).
  - **rtf_estimation/**: Signal processing logic for Spatial Audio and RTF tasks.
  - **beamforming/**: Speech enhancement and beamforming logic.
  - **counting/**: Logic specific to speaker counting tasks.
  - **data/**: PyTorch Datasets and DataModules.
  - **utils/**: Shared utilities (STFT, I/O, Logging).
  - **pipelines/**: High-level logic connecting models and data to run experiments.

## Usage
Do not run scripts directly from here. This package is meant to be imported.
Example: `from muse_toolbox.rtf_estimation import rtf_solver`
