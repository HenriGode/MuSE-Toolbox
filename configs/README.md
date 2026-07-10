# Configurations (configs)
**The "Brain" of the Framework**

## Purpose
This directory contains all hyperparameter definitions, model specifications, and environment settings using Hydra.

## Vertical Structure
Configuration files are organized hierarchically:
1.  **Topic** (e.g., `rtf_estimation`)
2.  **Subtopic** (e.g., `journal1_ebop`)
3.  **Experiment** (The actual .yaml file, e.g., `best_model.yaml`)

## Usage
These files define **WHAT** to run.
Pass these to the main runner:
`python -m muse_toolbox.main experiment=topic/subtopic/experiment_name`
