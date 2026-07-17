# MuSE-Toolbox Development Status & Agent History
**Last Updated:** 2026-07-15 22:52:00 (CEST)

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

### 5. Utilities Refactoring (Math & DSP)
- **Dismantled Monoliths:** Split `math4torch.py` into 7 modular files and `sigproc4torch.py` into 9 modular files inside `utils/math/` and `utils/dsp/`.
- **Strict Explicit Imports (Option 3):** Eliminated all wildcard (`*`) imports for these modules. The `__init__.py` files now act as clean facades defining strict `__all__` exports.
- **Dependency Cleanup:** Updated `data_utils.py` and `util_classes.py` to use explicit, targeted imports from the new `math` and `dsp` modules.

### 6. IDE Standardization
- **Type Checking & Formatting:** Configured the VS Code workspace (`.vscode/settings.json` and `pyrightconfig.json`) to enforce Black formatting on save and strict Pyright/Pylance type checking and import validation.

### 7. Core Directories SOP Application (`metrics`, `losses`, `utils`)
- **Metrics Refactoring:** Flattened the `metrics/common/` directory. Applied strict Python 3.10+ typing, absolute imports, and Google-style docstrings to `BaseMetric`, `RefMetric`, and all RTF/SAD metrics (e.g., `fwssnr.py`).
- **Losses Refactoring:** Flattened `losses/common/`, moving `base_loss.py` directly into `losses/`. Modernized typing across `cross_entropy.py` and removed relative imports.
- **Utils Cleanup:** Applied the SOP to `debug_utils.py`, `profiling_utils.py`, `system.py`, `tensor_ops.py`, etc., aggressively replacing `print()` statements with standard `logging.getLogger(__name__)` calls. Deleted the unused `model_utils.py`.
- **Codebase Assessment:** Generated a formal architecture review now stored in `agent_history/codebase_assessment.md`.

### 8. Data Directory Architecture Refactoring
- Decoupled the entire dataset generation and resolution logic from the package source code.
- Removed fragile `__file__`-based global path constants like `PROJECT_ROOT` and `BRUDEX_PATH` from data modules.
- Re-routed all pathing dynamically through Hydra injection via `${paths.data_dir}`.
- Segregated data into `/data/databases/` (for raw downloaded corpora like LibriSpeech and BRUDEX) and `/data/datasets/` (for precomputed scenario tensors).

---

## 🚧 What Is Open (The Roadmap)

If you are resuming development, here are the direct next steps needed to complete the project functionality:

- [x] **1. Formalize the PyTorch Lightning Modules**
  - *Completed:* Extracted diagnostic logic from `BaseLitModel` into Callbacks (`nan_guard`, `complexity_profiler`, `save_results`, `causality_check`).
  - *Completed:* Ported `COSADmodule` and `RTFmodule`.
  - *Completed:* Generated the Hydra configuration glue (`configs/task/`, `configs/model/`) to properly inject dependencies into the `pipelines/` scripts.

- [x] **1b. Integration Testing & Debugging**
  - *Completed:* Resolved Hydra interpolation bugs, fixed Torchaudio dataset downloading bugs, successfully migrated `brudex/` base data.
  - *Completed:* End-to-end `main.py` test run executed successfully without errors.

- [ ] **1c. Pipeline Validation & End-to-End Sanity Checks**
  - **Status:** *In Progress.*
  - **Next Step:** Design a specific minimal test experiment to validate the internal logic of the pipeline at every step (data loading, feature extraction, predictions, loss computation).

- [ ] **1d. Output Directory Architecture Refactor**
  - Ensure the `.hydra`, `wandb`, `results`, `predictions`, and `audio` all log correctly inside the new `outputs/<task>/<experiment>/<timestamp>/<split>/` directory structure. Refactor Callbacks and Hydra config logging rules to achieve this.

- [ ] **1e. HeterogeneousBatch Architectural Refactor**
  - Strip all DSP/Transform logic out of `HeterogeneousBatch` and move it directly into the PyTorch Lightning module `forward()` passes, ensuring it acts solely as a dumb data container. Detailed in `agent_history/HETEROGENEOUS_BATCH_REFACTOR.md`.

- [ ] **1f. Algorithmic Latency & Real-Time Causal STFT Redesign**
  - Transition STFT parameters to `center=False` to eliminate artificial future lookahead.
  - Refactor Ground Truth framewise alignment to anchor on the end of the STFT window.
  - Rewrite `CausalityCheckCallback` into `AlgorithmicLatencyCallback` with static perturbation-based testing. Detailed in `agent_history/algorithmic_latency_design.md`.

- [ ] **2. Setup the AMI Dataset**
  - Introduce the full AMI corpus integration inside the `data/` directory. Build out the `Database`, `Dataset`, and `DataModule` wrappers.

- [ ] **3. Migrate Remaining Experiments to Hydra**
  - Use the `configs/experiment/J1_seglen.yaml` template to port the rest of the legacy experiment scripts (like `J2_deactivation` or `PRA_ANF`) into the `configs/experiment/` directory.

- [ ] **4. Setup the Testing Framework**
  - Establish a formal `pytest` suite. Write unit tests for the core feature extractors, TCN blocks, and estimators to rigorously lock down their tensor shapes and behavior.
