# MuSE-Toolbox – Journal 3 Project Overview, Status & Execution Plan

**Project Title:** *Geometry-Agnostic and Geometry-Informed Online Sound Source Counting*  
**Authors:** Henri Gode and Simon Doclo (University of Oldenburg)  
**Primary Specification & Draft Notes:** [`documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex)  
**Writing Style Guidelines:** [`documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/scientific_writing_instructions.txt`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/scientific_writing_instructions.txt)  
**Last Updated:** 2026-11-13

---

## 1. Scientific Objective & Core Research Scope

This project investigates **strictly STFT-frame causal, online DNN-based acoustic sound source counting** ($\widehat{K}_t \in \{0, 1, \dots, K_{\max}\}$ with $K_{\max}=4$) and its **generalization across unseen microphone array geometries and variable numbers of microphones ($M$)**.

Generalization to arbitrary arrays is investigated along three complementary axes:
1. **Feature-Inherent Generalization:** Exploiting multi-channel spatial coherence features that inherently condense $M$ channels into a single representation ($\mathbf{Z}_{\mathrm{GMSC}}$, bidirectional $\mathbf{Z}_{\mathrm{wGMSC}}$), compared against pairwise spatial features ($\mathbf{Z}_{\mathrm{CSIPD}}$, Polar $\mathbf{Z}_{\mathrm{PSC}}$) and per-channel spectral features ($\mathbf{Z}_{\mathrm{STFT}}$, $\mathbf{Z}_{\mathrm{mel}}$).
2. **Architecture-Inherent Generalization:** Adapting the Cornell et al. (2022) Transformer baseline into a strictly causal multi-channel architecture that processes arbitrary channel counts $C$ via parallel per-channel Transformer streams interleaved with permutation-invariant cross-channel sharing layers (e.g., Transform-Average-Concatenate / TAC or self-attention) and masked channel pooling.
3. **Explicit Geometry Conditioning (Geometry-Informed):** Providing the network with relative 3D physical microphone coordinates ($\mathbf{G} \in \mathbb{R}^{M \times 3}$, or pairwise baseline vectors $\mathbf{v}_{ij} = \mathbf{p}_j - \mathbf{p}_i \in \mathbb{R}^3$) via early input concatenation or late conditioning (e.g., Feature-wise Linear Modulation / FiLM or layer-wise spatial positional encodings inside interleaved TAC layers).

---

## 2. Macro-Level Project Lifecycle & Workflow

| Step | Phase Description | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Step 1** | **Discuss Project Idea** (with supervisor) | ✅ **Done** | Core scope and scientific questions established. |
| **Step 2** | **Literature Review & Research Gap** (with supervisor) | ✅ **Done** | Completed; occasionally revisited for specific baseline mechanics (e.g., Cornell et al., TAC, FiLM). |
| **Step 3** | **Detailed Execution Plan & Specification** (`journal3.tex`) | 🔄 **In Progress** | Sections 2, 3, and 5 finished; Sections 4 and 6 in progress. To be discussed with supervisor once complete. |
| **Step 4** | **Codebase Implementation & Validation** (`MuSE-Toolbox`) | 🔄 **In Progress** | Data pipeline, features, and sanity tests done; multi-channel architecture & feature fusion refactoring in progress. |
| **Step 5** | **Running Experiments** (Training & Evaluation Matrix) | ⏳ **Not Started** | Starts once Step 3 & Step 4 are completed and verified. |
| **Step 6** | **Evaluating Results & Research Story** (with supervisor) | ⏳ **Not Started** | Analyze ablations and zero-shot generalization results. |
| **Step 7** | **Writing Final Paper Draft & Supervisor Feedback** | ⏳ **Not Started** | Distill `journal3.tex` notes and experimental findings into final IEEE manuscript. |

### Coupled Iterative Methodology (Steps 3 & 4)
We intentionally execute **Step 3 (`journal3.tex` notes)** and **Step 4 (Code Implementation)** in lockstep:
1. **Plan & Formulate:** Design a module/method and write its exact mathematical and algorithmic description into [`journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex) (including implementation details that may later be condensed for the final paper).
2. **Implement & Test:** Implement the corresponding code in `src/muse_toolbox/` and `configs/`, then run verification scripts.
3. **Synchronize:** Refine [`journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex) if practical implementation or testing reveals necessary adjustments, ensuring the specification document always mirrors the actual codebase 1:1.

---

## 3. Where We Come From (Completed Milestones)

### 3.1 Modular Software Architecture (`MuSE-Toolbox`)
* **Package Refactoring (5-Point SOP):** Migrated legacy code from `framewiseSpeakerCounting` into `src/muse_toolbox/` enforcing: (1) absolute imports (`from muse_toolbox...`), (2) Python `logging`, (3) strict Python 3.10+ type hints, (4) Google-style docstrings, and (5) standardized base interfaces (`BaseFeatureExtractor`, `BaseChannelCombinator`, `BaseSourceCountEstimator`, `BaseRTFestimator`).
* **Decoupled Utilities & Callbacks:** Modularized `utils/math/` and `utils/dsp/`; isolated PyTorch Lightning diagnostics into callbacks (`nan_guard`, `complexity_profiler`, `save_results`, `causality_check`).
* **Hydra Configuration & Output Hierarchy:** All datasets, models, trainers, and sweeps instantiate dynamically via Hydra (`configs/`). Experiment outputs route into a split-first hierarchy: `outputs/<task>/<experiment>/<timestamp>/{logging,checkpoints,train,val,test}/`.

### 3.2 Unified Data Pipeline & Datasets (Section 5 of `journal3.tex`)
* **Six Datasets Configured & Specified (`f_s = 8\text{ kHz}`, $K_{\max} = 4$):**
  1. **`Circ8` (Synthetic PRA-ANF):** Image source method (`pyroomacoustics`, rooms $[4,8]\times[4,8]\times[2.5,4]\text{ m}$, $RT_{60} \in [0.2, 0.8]\text{ s}$) + diffuse DEMAND noise (`ANF-Generator`, $\text{SNR} \in [5, 20]\text{ dB}$) + LibriSpeech (`train-clean-360` / `dev-clean` / `test-clean`, relative power $[-5, 5]\text{ dB}$, $120\text{ s}$ scenarios, random speaker transitions every $[1, 10]\text{ s}$). Fixed 8-ch uniform circular array ($r = 10\text{ cm}$).
  2. **`Circ3-8` (Synthetic PRA-ANF):** Same as `Circ8`, with channel count $M \in \{3, \dots, 8\}$.
  3. **`Rand8` (Synthetic PRA-ANF):** Fixed 8-ch randomized planar array ($20\text{ cm} \times 20\text{ cm}$).
  4. **`Rand3-8` (Synthetic PRA-ANF):** Randomized planar array with $M \in \{3, \dots, 8\}$.
  5. **`BRUDEX` (Measured RIRs & Noise):** Dummy head BTE + in-ear microphones ($M \in \{2, \dots, 6\}$ across 7 symmetric/asymmetric configurations), low reverberation ($RT_{60} \approx 310\text{ ms}$), disjoint DOAs and noise splits (train: babble/white; val: babble; test: cafeteria). Acoustic power ground truth ($-30\text{ dB}$ threshold on isolated clean speech, ignoring silences $< 1.0\text{ s}$).
  6. **`AMI` (Real-World Meetings):** ~100 hours across Edinburgh, Idiap, and TNO rooms ($M \in \{4, 8, 10\}$ circular and linear arrays), official Full-corpus-ASR split, manual word-level headset annotations (`ami_public_manual_1.6.2`). Uses Cornell's dynamic training augmentation: 50% unaltered $7.0\text{ s}$ chunks and 50% synthetic mixtures of 2–4 isolated single-speaker chunks mixed with Gaussian relative SIR shifts $\mathcal{N}(0, 4^2)\text{ dB} \in [-10, 10]\text{ dB}$ and peak scaling to $[0.5, 0.99]$ when peak $> 0.99$.
* **Removal of `HeterogeneousBatch` & Offline Precomputation:**
  * Deleted `HeterogeneousBatch` and `PrecomputedDataset`.
  * Standardized disk storage to `/data4/Henri/MuSE-Toolbox/data/datasets/<id>/` containing memory-mapped `*_audio.npy` and `*_meta.pt` files.
  * Implemented [`DynamicChunkDataset`](../src/muse_toolbox/data/components/dynamic_chunk_dataset.py): samples random $7.0\text{ s}$ chunks during training (`train_items_per_epoch`) and deterministic sequential $7.0\text{ s}$ chunks with a $5.0\text{ s}$ hop ($2.0\text{ s}$ overlap context) during validation and testing.
  * Implemented [`raw_audio_collate_fn`](../src/muse_toolbox/data/components/collate.py): zero-pads variable-channel audio into a standard dictionary (`input`: `[B, M_max, T]`, `mic_lengths`: `[B]`, `time_lengths`: `[B]`, `meta`: dict).
* **Sanity Datasets & Pipeline Verification:**
  * Created [`scripts/prepare_sanity_datasets.py`](../scripts/prepare_sanity_datasets.py) (`sanity_brudex`, `sanity_pra_anf`, `sanity_ami`) and [`tests/test_model_pipeline.py`](../tests/test_model_pipeline.py).
  * Verified end-to-end forward pass, loss calculation, backpropagation gradients, validation metrics, and heterogeneous microphone count batches (e.g., 3 mics + 8 mics in one batch).

### 3.3 Signal Model & Feature Extraction (Sections 2 & 3 of `journal3.tex`)
* **Section 2 (Signal Model & Problem Formulation):** Completed in [`journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex). Formulates the time and STFT signal model ($F = L_{\mathrm{win}}/2 - 1$ discarding DC and Nyquist), MTF/ATF mixture model, second-order spatial covariance rank relation ($\mathrm{rank}\{\mathbf{R}_{x,t}\} = \min(M, K_t)$), strict STFT frame-level causality ($\tau \le t$), and acoustic power presence vs. linguistic alignment.
* **Section 3 (Feature Extraction):** Completed in [`journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex) under the unified tensor convention $C \times D \times T$:
  * **Spectral Features:**
    * $\mathbf{Z}_{\mathrm{STFT}} \in \mathbb{R}^{M \times 2F \times T}$: Concatenated real and imaginary STFT coefficients (Cartesian form).
    * $\mathbf{Z}_{\mathrm{mel}} \in \mathbb{R}^{M \times D_{\mathrm{mel}} \times T}$ (and single-channel baseline $\mathbf{Z}_{\mathrm{mel}}^{(\mathrm{ref})} \in \mathbb{R}^{1 \times D_{\mathrm{mel}} \times T}$): Log-Mel filterbank energies.
  * **Spatial Features:**
    * $\mathbf{Z}_{\mathrm{CSIPD}} \in \mathbb{R}^{P \times 2F \times T}$ with $P = M(M-1)/2$: Cosine and sine of pairwise inter-channel phase differences.
    * $\mathbf{Z}_{\mathrm{PSC}} \in \mathbb{R}^{P \times 3F \times T}$: Pairwise spatial coherence in polar form ($[|\bm{\gamma}_{p,t}|^{\mathrm{T}}, \cos(\bm{\psi}_{p,t})^{\mathrm{T}}, \sin(\bm{\psi}_{p,t})^{\mathrm{T}}]^{\mathrm{T}}$) derived from recursively smoothed spatial covariance $\widehat{\mathbf{R}}_{y,t,f}$.
    * $\mathbf{Z}_{\mathrm{GMSC}} \in \mathbb{R}^{1 \times F \times T}$: Principal eigenvalue coherence metric $(\lambda_{\max}\{\bm{\Gamma}\{\widehat{\mathbf{R}}_{y,t,f}\}\} - 1)/(M - 1)$.
    * $\mathbf{Z}_{\mathrm{wGMSC}} \in \mathbb{R}^{1 \times 2F \times T}$: Bidirectional whitened GMSC stacking causal activation coherence (whitened with $\widehat{\mathbf{R}}_{y,t-t_v,f}$) and causal time-reversed deactivation coherence (moving-average covariance $\widehat{\mathbf{R}}_{y,t-t_v,f}^{(\mathrm{ma})}$ whitened with $\widehat{\mathbf{R}}_{y,t,f}^{(\mathrm{ma})}$).

---

## 4. Where We Are Right Now (Current State & Open Implementation Gaps)

We are currently in the middle of **coupled Steps 3 & 4**, transitioning from **Feature Extraction (Section 3)** to **Proposed Architecture / Model Design (Section 4)** and its corresponding Python implementation.

### Current Gaps Between Planned Design and Codebase

1. **Estimator Architecture (Mono-Channel vs. Multi-Channel Interleaved):**
   * *Current Code:* [`COSADmodule`](../src/muse_toolbox/models/source_counting/cosad.py) applies an early `channel_combinator` to collapse $C \to 1$ before the estimator. [`BaseSourceCountEstimator`](../src/muse_toolbox/models/source_counting/estimators/base_estimator.py) and [`TransformerEstimator`](../src/muse_toolbox/models/source_counting/estimators/transformer_estimator.py) enforce `C == 1` (`[B, F, T]`).
   * *Needed:* Refactor the architecture so the Cornell Transformer backbone operates on arbitrary channel streams `[B, C, E, T]` (per-channel Transformer blocks interleaved with masked cross-channel TAC / Self-Attention layers, followed by final masked channel pooling before the output classification head), while supporting a clean config toggle to recover Cornell's exact mono-channel baseline ($C=1$ or identity spatial block) and strictly frame-by-frame processing at the native STFT resolution $T$ (no temporal subsampling, with configurable past context concatenation via `chunk_size`).
2. **Spectral + Spatial Feature Combination:**
   * *Current Code:* [`StackedFeatureExtractor`](../src/muse_toolbox/models/components/feature_extractors/stacked_features.py) raises a `ValueError` when stacking features with different channel dimensions ($M$ vs. $P = M(M-1)/2$ vs. $1$).
   * *Needed:* Finalize and implement our multi-feature fusion strategy (handling Cornell's early concatenation vs. late FiLM conditioning across mismatched channel dimensions $M$, $P$, and $1$).
3. **Array Geometry Conditioning:**
   * *Current Code:* Microphone coordinates (`mic_pos`) are stored in dataset metadata, and [`COSADmodule.forward()`](../src/muse_toolbox/models/source_counting/cosad.py) randomly permutes audio channels during training without permuting `mic_pos` or passing geometry to the model.
   * *Needed:* Synchronize channel permutation augmentation between `x` and `mic_pos`, and implement the geometry conditioning mechanism (relative 3D coordinates $\tilde{\mathbf{p}}_m \in \mathbb{R}^3$ / pairwise baseline vectors $\mathbf{v}_{ij} \in \mathbb{R}^3$ injected via early fusion or late FiLM / TAC positional encoding).
4. **Pairwise Spatial Coherence (`PSC`) Extractor in Code:**
   * *Current Code:* `gmsc.py`, `wgmsc.py`, `ipd.py`, `log_mel.py`, and `pure_stft.py` exist in [`src/muse_toolbox/models/components/feature_extractors/`](../src/muse_toolbox/models/components/feature_extractors/), alongside DC/Nyquist bin handling (`F = L_win/2 - 1`).
   * *Needed:* Implement the polar `PSC` feature extractor ($\mathbf{Z}_{\mathrm{PSC}} \in \mathbb{R}^{P \times 3F \times T}$) and verify DC/Nyquist bin slicing across all extractors to match Section 2 & 3 of `journal3.tex`.
5. **Evaluation & Causality Details:**
   * *Sequential Chunk Burn-In Trimming:* Verify/implement discarding the first $2.0\text{ s}$ (`min_context_s`) of each $7.0\text{ s}$ validation/test chunk so predictions are stitched strictly over the valid $5.0\text{ s}$ hop region.
   * *AMI Multi-Array Probability Fusion:* Implement probability-level averaging across simultaneous arrays in the same room before $\argmax$ decision.
   * *Strict Causality Verification:* Ensure STFT framing, causal past-context frame concatenation (`chunk_size`), and causal self-attention masking strictly prevent future lookahead across the frame-wise pipeline.

---

## 5. Next Steps & Roadmap (Where We Are Going)

### Phase A: Finalize Architecture Design & Specification (Step 3 + Step 4)
1. **Align on Architecture & Feature/Geometry Fusion Mechanics:**
   * Decide the exact mathematical formulation for:
     * **(a) Interleaved Multi-Channel Transformer:** Per-channel causal Transformer blocks + interleaved cross-channel sharing (TAC / Self-Attention) + final channel pooling.
     * **(b) Spectral & Spatial Feature Combination:** Early concatenation vs. FiLM conditioning, and how $M$-channel, $P$-pair, and $1$-channel features interact.
     * **(c) Geometry Conditioning:** Early vs. Late (FiLM / TAC) injection of relative 3D coordinates.
2. **Write Section 4 (Proposed Architecture / Method) in [`journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex):**
   * Document the complete architecture, feature fusion, geometry conditioning, and loss functions in `journal3.tex` following [`scientific_writing_instructions.txt`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/scientific_writing_instructions.txt).
3. **Implement & Refactor Model Components in `MuSE-Toolbox`:**
   * Implement the `PSC` feature extractor and verify `F = L_win/2 - 1` consistency.
   * Refactor `TransformerEstimator`, cross-channel combinators/TAC blocks, multi-feature fusion/conditioning modules, and `COSADmodule` (including synchronized coordinate permutation).
   * Add corresponding Hydra configuration files in `configs/model/`.

### Phase B: Pipeline Completion & End-to-End Verification (Step 4 + Step 3)
4. **Complete Evaluation Pipeline Details:**
   * Implement/verify the $2.0\text{ s}$ overlap burn-in discard for sequential validation/test chunks and multi-array posterior probability averaging for AMI.
   * Verify strict frame-level causality across all model configurations.
5. **Run Comprehensive Sanity Tests:**
   * Extend and run [`tests/test_model_pipeline.py`](../tests/test_model_pipeline.py) across `sanity_brudex`, `sanity_pra_anf`, and `sanity_ami` covering single-feature, multi-feature, geometry-agnostic, and geometry-conditioned configurations.
6. **Complete Section 6 (Experimental Setup) in [`journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex) & Review with Supervisor:**
   * Finalize all experimental setup details, metrics, and training hyperparameters in `journal3.tex`, then discuss the complete plan with Simon Doclo.

### Phase C: Experiment Execution, Analysis & Paper Writing (Steps 5, 6 & 7)
7. **Generate Production Datasets & Execute Experiment Matrix (Step 5):**
   * Precompute full-scale datasets (`Circ8`, `Circ3-8`, `Rand8`, `Rand3-8`, `BRUDEX`, `AMI`).
   * Run the systematic experimental campaigns:
     * *Part 0:* Feature ablation (Spectral vs. Spatial vs. Combined;Early vs. Late/FiLM fusion).
     * *Part 1:* Generalization across matched vs. unmatched array geometries (`Circ` vs. `Rand`) and fixed vs. variable microphone counts ($8$ vs. $3\text{--}8$).
     * *Part 2:* Impact of explicit 3D geometry conditioning on unseen arrays.
     * *Part 3:* Zero-shot and matched evaluation on `BRUDEX` and `AMI`, plus causality/latency analysis.
8. **Evaluate Results & Write Final Manuscript (Steps 6 & 7):**
   * Generate publication figures/tables, discuss the experimental findings and research story with supervisor, and write Section 1 (Introduction), Section 7 (Results & Discussion), and Section 8 (Conclusion) of [`journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex).

