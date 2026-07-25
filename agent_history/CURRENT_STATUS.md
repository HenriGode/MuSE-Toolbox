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

## 🧠 Architecture Decisions & Processing Flow Blueprint

We have firmly established the ideal processing pipeline for the toolbox, designed to be strictly array-agnostic, permutation-invariant, and highly scalable. The pipeline follows this explicit sequence:

1. **Raw Audio:** Multi-channel waveform input.
2. **Global STFT:** A unified time-frequency transform for all features (e.g., 64ms window, 16ms shift at 8kHz) to eliminate heterogeneous batch padding issues.
3. **Feature Extraction:** Extracts spectral (e.g., Log-Mel, dependent on $M$ mics) and spatial (e.g., IPD, dependent on $P$ pairs) domains.
4. **Channel Combinators (Per Feature):** Safely condenses the spatial dimension of *each* feature down to a fixed dimension $J$. 
   - **Crucial Decision (Late Stacking):** We use individual combinators (like Self-Attention or Circular Mean) *before* stacking. This preserves the mathematical purity of the data (circular phase math vs. linear energy math) and solves the geometry mismatch between $M$ mics and $P$ pairs.
   - **Advanced Pattern:** This stage can leverage **Global-to-Local Cross-Attention** (where a pooled global context vector queries the individual modality features) to allow power-based features to guide spatial-based features without mathematically corrupting them.
5. **Feature Stacking:** Concatenates the now-fixed dimension vectors ($J_{Mel} + J_{IPD}$) into a unified feature vector.
6. **Source Count Estimator:** Temporal classifier (GRU / TCN) predicting frame-wise activity.
7. **Output:** Discrete source count predictions.

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

- [x] **1c. Pipeline Validation & End-to-End Sanity Checks**
  - *Completed:* Runtime sanity checks with `brudex` and `pra_anf` datasets executed successfully. The code and Hydra configs run end-to-end without errors.
  - *Pending (Ongoing):* A thorough mathematical/behavioral investigation to ensure all internal logic (data loading, feature extraction, predictions, loss computation) performs exactly as intended.

- [x] **1d. Output Directory Architecture Refactor**
  - *Completed:* The `.hydra`, `wandb`, `results`, `predictions`, and `audio` all log correctly inside the new `outputs/<task>/<experiment>/<timestamp>/<split>/` directory structure. Refactored Callbacks and Hydra config logging rules to achieve this.



- [x] **1g. wandb logger set name dynamically (mabye even project) in config**

- [x] **1h. Dataset source power rations adjust to 5dB instead fo 0**
  - Adjust sourc epower range to 5 db maybe instead of 0 db as it is now

- [x] **1i. Set up random channel permutation training to not learn any kind of channel position information**
  - Added a permutation layer at the beginning of the training directly in HeterogeneousBatch

- [x] **1j. add the wandb run name to the timestamp level of the outputs dir strucutre**


- [ ] **2. Implement the Decoupled Channel Combinator Architecture**
  - [x] Implement the new processing flow by formally decoupling the channel condensation logic from the raw feature extraction.
  - [x] Create a new `channel_combinators` subdirectory in both `src/muse_toolbox/models/components/` and `configs/model/`.
  - [ ] Refactor `StackedFeatureExtractor` to execute the late-stacking paradigm: it must route raw spatial/spectral features through their respective combinators before concatenating them.
  - [x] Explore and validate the Global-to-Local Cross-Attention block as an advanced combinator option and also still teh Self attention channel combinator.
  - [x] integrate the new logic into the COSAD and RTF pipelines and validate it.


- [ ] **3. Directory-level README Documentation**
  - Expand the practice of including contextual `README.md` files in both the `src/` subdirectories (detailing architectural data-flow) and `configs/` subdirectories (detailing hyperparameter tuning).
  - Propagate this documentation style to all other component directories (e.g., `estimators`, `data`, `metrics`), matching what was done for `feature_extractors`.

- [ ] **4. Algorithmic Latency & Real-Time Causal STFT Redesign**
  - Transition STFT parameters to `center=False` to eliminate artificial future lookahead.
  - Refactor Ground Truth framewise alignment to anchor on the end of the STFT window.
  - Rewrite `CausalityCheckCallback` into `AlgorithmicLatencyCallback` with static perturbation-based testing. Detailed in `agent_history/algorithmic_latency_design.md`.

- [ ] **5. HeterogeneousBatch Architectural Refactor**
  - Strip all DSP/Transform logic out of `HeterogeneousBatch` and move it directly into the PyTorch Lightning module `forward()` passes, ensuring it acts solely as a dumb data container. Detailed in `agent_history/HETEROGENEOUS_BATCH_REFACTOR.md`.

- [ ] **5b. Real-Time Causal Training Refactoring (60s Signal Optimization)**
  - Retain the 60-second scenario generation for realistic transitions, but resolve the start-at-zero bias and memory constraints for recurrent models (e.g., GRU).
  - Implement **Stateful Truncated BPTT (TBPTT)** to train on shorter chunks (e.g., 5s) while detaching and passing the hidden state between chunks. ( I guess i am not doing this!, maybe training on short chunks but without plassing the hidden state.)
  - Implement a **Burn-in Loss Masking** period (e.g., ignoring loss for the first 1-2 seconds) to allow the GRU hidden state and GMSC smoothing buffers to initialize properly. (this seems for me the most interesting and promising. to determine the burn in time we would need to condier parameters liek smoothing constant and so on but much more the bias behavior of the class imbalance at the beginning of the signals due to the activation pattern generation.)
  - Implement **Weighted Cross-Entropy Loss** to naturally balance the over-represented 0-source class across the 100-hour dataset.

- [ ] **6. Setup the Testing Framework**
  - Establish a formal `pytest` suite. Write unit tests for the core feature extractors, TCN blocks, and estimators to rigorously lock down their tensor shapes and behavior.

- [ ] **7. Setup the AMI Dataset**
  - Introduce the full AMI corpus integration inside the `data/` directory. Build out the `Database`, `Dataset`, and `DataModule` wrappers.




