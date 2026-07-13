# MuSE-Toolbox Development Status & Agent History
**Last Updated:** 2026-07-13 18:34:53 (CEST)

*This document captures the entire history of the automated migration and refactoring of the `framewiseSpeakerCounting` project into the modular `MuSE-Toolbox` package. It consolidates all past agent plans, historical roadmaps, and configuration setups into one master ledger.*

---

## 🎯 What Has Been Accomplished (The "Done" List)

### 1. The Massive Data Architecture Split
- **De-monolithing:** We broke down the gigantic 1,350-line `base_dataset.py` into strictly decoupled, cohesive modules:
  - `data/base_DBs.py` (managing raw data loading)
  - `data/base_scenario_generator.py` (the core mixing engine)
  - `data/precomputed_dataset.py` (handles pre-cached datasets)
  - `data/base_datamodule.py` (the PyTorch Lightning data bindings)

### 2. Model Refactoring (The 5-Point SOP)
We strictly applied our **5-Point Standard Operating Procedure (SOP)** across the core modeling code to enforce rigid software engineering standards:
1. **Structural Fitness:** Eliminated relative import bugs; forced absolute paths (`from muse_toolbox...`).
2. **Professional Logging:** Eradicated `print()` statements in favor of Python's `logging` module (`log.info`, `log.warning`).
3. **Strict Type-Hinting:** Enforced `typing` modules on all signatures.
4. **Google-Style Docstrings:** Normalized all ad-hoc comments into standardized Google format.
5. **Architectural Cohesion:** Extracted estimators from generic directories into task-specific homes, enforcing `BaseRTFestimator` and `BaseSourceCountEstimator` interfaces (guaranteeing `forward_` and `get_config()`). Deprecated PyTorch `Variable` patterns were removed.

**Refactored Directories:**
- `models/components/nn_blocks/` (e.g., `conv_tasnet.py`, `causal_conv1d.py`)
- `models/components/feature_extractors/` (`WGMSC`, `IPD`, `LogMel`)
- `models/rtf_estimation/estimators/` (BOP-based, Covariance-based, Oracle)
- `models/source_counting/estimators/` (CoSAD, TCN, GRU, PrecomputedSAD)

### 3. Hydra Configuration & HPC Integration
- **Config Directory:** Translated all legacy setup scripts into a highly modular `configs/` directory (`dataset/`, `model/`, `trainer/`, `experiment/`, etc.).
- **Magic Instantiation:** Implemented `_target_` mapping so `hydra.utils.instantiate()` automatically builds objects.
- **Multirun & Parallel Sweeps:** Enabled native Hydra multirun (`-m`). You can now sweep parameters or run multiple experiments in parallel (via `hydra-joblib-launcher` or `hydra-submitit-launcher` for SLURM). Example: `python main.py -m experiment=journal1_ebop model.lr=0.001,0.0001`.
- **HPC Runner:** `scripts/main.py` is the single point-of-entry and includes `torch.set_float32_matmul_precision("highest")` to maximize Tensor Core utilization on Ampere+ GPUs.

### 4. Pipelines
- **PyTorch Lightning Wrappers:** Stubbed out `pipelines/source_counting_pipeline.py` and `pipelines/rtf_estimation_pipeline.py` as clean orchestrators capable of dynamically loading Oracle parameters or pre-computed SAD files.

---

## 🚧 What Is Open (The Roadmap)

If you are resuming development, here are the direct next steps needed to complete the project functionality:

- [ ] **1. Apply the 5-Point SOP to the `utils` Directory**
  - Files like `data_utils.py` and `util_classes.py` still contain dense matrix logic and native `print` statements. They need strict typing, Google-style docstrings, and `logging`.
  
- [ ] **2. Formalize the PyTorch Lightning Modules**
  - Extract the specific training step logic from the legacy codebase into `COSADmodule` and `RTFmodule`. Plug these completed LitModules into the new skeletal `pipelines/` scripts.

- [ ] **3. Setup the AMI Dataset**
  - Introduce the full AMI corpus integration inside the `data/` directory. Build out the `Database`, `Dataset`, and `DataModule` wrappers.

- [ ] **4. Migrate Remaining Experiments to Hydra**
  - Use the `configs/experiment/J1_seglen.yaml` template to port the rest of the legacy experiment scripts (like `J2_deactivation` or `PRA_ANF`) into the `configs/experiment/` directory.

- [ ] **5. Setup the Testing Framework**
  - Establish a formal `pytest` suite. Write unit tests for the core feature extractors, TCN blocks, and estimators to rigorously lock down their tensor shapes and behavior.
