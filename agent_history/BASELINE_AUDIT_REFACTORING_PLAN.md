# Baseline Audit & Pipeline Refactoring Plan (`A7–F30`)

**Created:** 2026-11-26  
**Status:** ✅ Completed & Verified  
**Context:** Outcome of the three-way consistency audit between Cornell et al. (2022) (`CORNEL~1.PDF`), [`journal3.tex`](../documents/source_counting/journal3_wGMSC_Transformer/paper_draft_notes/journal3.tex), and `src/muse_toolbox/` + `configs/`.

---

## Verified / No-Action Topics (Intentional Parity or Already Aligned)
- **`A1`–`A6`:** Unified `25 ms` Hann / `10 ms` shift causal left-padded STFT (`center=False`), DC & Nyquist removal (`F=199`), all-pairs `CSIPD(mode="all")` (`P=M(M-1)/2`), IPD sign convention ($\varphi_i - \varphi_j$), and block-concatenated CSIPD layout (`[cos, sin]`) are intentional and documented.
- **`B9`–`B11`, `B15`:** Cat-Pool context stacking (`C=7`), stride subsampling (`S=10` with right-padding), input `LayerNorm` + $1\times 1$ bottleneck (`D_emb=384`), and temporal upsampling of **unnormalized logits** (linear interpolation when `causal=False`, causal ZOH when `causal=True`) are consistent across all three sources.
- **`C16`–`C18`:** Dual `CatPoolNorm` stems, Early Fusion concatenation before bottleneck, and Late FiLM placement before each of the $R$ TE blocks match across all three sources.
- **`D24`:** Gaussian notation $\mathcal{N}(0, 4^2)$ ($\sigma = 4.0\text{ dB}$ clamped to $[-10, 10]\text{ dB}$) and peak normalization $> 0.99 \to \mathcal{U}(0.5, 0.99)$ match Cornell's relative SIR spread.

---

## Actionable Topics & Live Progress Checklist

### Phase 1: Project Documentation & AMI Annotation Diagnostic (`A7`, `D21`, `E29`)
- [x] **Step 1.1:** Create this live tracking file (`agent_history/BASELINE_AUDIT_REFACTORING_PLAN.md`).
- [x] **Step 1.2 (`A7`, `E29`):** Update `agent_history/PROJECT_OVERVIEW.md` with:
  - `A7` (deferred): all three `wGMSC` discrepancies (`whiteidx_rev` double-subtraction giving $2T_v$ lookback instead of $T_v$, `start_rev = 2*(M-1)` zero-padding, `regularize(..., 1e-6)` + eigendecomposition whitening alignment with `journal3.tex`).
  - `E29` (deferred optional toggle): independent per-array evaluation as the intentional default, with `eval_array_fusion` multi-array averaging noted as a future option for Cornell Table 12 comparison.
- [x] **Step 1.3 (`D21`):** Extend `AMIWordsXMLParser` in `src/muse_toolbox/data/databases/ami_label_parsers.py` with `include_vocalsounds: bool = False` and `bridge_gaps_s: float = 0.0`, and expose both in `configs/dataset/ami.yaml`.
- [x] **Step 1.4 (`D21`):** Create interactive `#%%` script `scripts/check_ami_annotations.py` comparing `AMISegmentsXMLParser`, `AMIWordsXMLParser(<w>)`, and `<w>+<vocalsound>` (+ gap bridging) against Cornell Table 1 (`dev` & `eval`), and run `--no-plot`.

### Phase 2: AMI Loader, Filtering, Augmentation, Corpus Epochs & Trailing Chunks (`D20`, `D23`, `E25`, `E28`-Grid)
- [x] **Step 2.1 (`D20`):** Fix `AMIDatabase.get_meeting` (`src/muse_toolbox/data/databases/ami.py`) to load channels `01..10` (`f"{meeting_id}.{array_id}-{i:02d}.wav"`) so TNO `Array2` loads all 10 microphones.
- [x] **Step 2.2 (`D20`, `D23`):** Add `only_8ch_circular: bool = False` to `AMIDataModule` (`src/muse_toolbox/data/datamodules/ami.py`) and `configs/dataset/ami.yaml` (filtering `prepare_data()` and `DynamicChunkDataset` so fixed-8ch baselines exclude Idiap 4-ch `Array2` and TNO 10-ch linear `Array2`), and set `augmentation.prob_augmentation: 0.412` (`D23`).
- [x] **Step 2.3 (`E25`, `E28`-Grid):** Update `DynamicChunkDataset`, `BaseDataModule`, and `configs/dataset/base_dataset.yaml`:
  - Support `train_items_per_epoch: null` (auto 100% unique corpus duration per epoch, grouped by `meeting_id` when present).
  - Use length-weighted recording sampling (`max(1, length - chunk_samples + 1)`) for unaugmented training chunks.
  - Retain trailing evaluation chunks with `length - covered_end >= hop_length` (dropping only partial-hop remainders `< hop_length`) without trailing zero-padding.

### Phase 3: FiLM Zero-Bias Init, Variable-Length Padding Masks & Optimizer (`C19`, `B14`, `E26`, `E28`-Model)
- [x] **Step 3.1 (`C19`, `B14`, `E28`-Model):** Update `src/muse_toolbox/models/source_counting/estimators/transformer_estimator.py` (and `base_estimator.py` / `osc.py`):
  - Initialize `FiLMLayer` `gamma_proj.bias` and `beta_proj.bias` to `0.0` (preserving identity at step 0 while keeping small Gaussian weights and strictly non-zero gradients).
  - Add `self.out_norm = nn.LayerNorm(embed_dim)` prior to `self.output` in `TransformerEstimator` and `FusionTransformerEstimator` (`B14`).
  - Thread `valid_frames` into `TransformerEstimator` and `FusionTransformerEstimator` for `MHALayer` `key_padding_mask` and exact per-item linear upsampling on variable-length validation batches.
- [x] **Step 3.2 (`E26`):** Update `BaseLitModel.configure_optimizers` (`src/muse_toolbox/models/base_model.py`) to split 2D+ weight matrices (`weight_decay`) vs. 1D biases/norms (`weight_decay=0.0`), and update configs:
  - `configs/model/osc.yaml`: `RAdam(lr=0.0001, weight_decay=0.0001, decoupled_weight_decay=true)` + `ReduceLROnPlateau.patience: 5`.
  - `configs/trainer/default.yaml`: `gradient_clip_val: 5.0`, `gradient_clip_algorithm: "norm"`.
  - `configs/callbacks/default.yaml`: `early_stopping.patience: 15`.
  - `configs/dataset/base_dataset.yaml`: `batch_size: 16`, `train_items_per_epoch: null`.

### Phase 4: Continuous Stitched Evaluation Loss & Exact Burn-In Trimming (`E27`, `E28`-Stitching)
- [x] **Step 4.1 (`E28`-Stitching):** Fix `"causal_burnin"` in `OSCmodule._prepare_eval_estimates_and_targets` (`osc.py`) to track `emitted_end_frame` per `recording_id` (`burnin_frames = min(valid_len, max(0, emitted_end_frame - start_frame))`) and guarantee `recording_id` in `filtered_meta`.
- [x] **Step 4.2 (`E27`):** Update `BaseLitModel.validation_step`, `test_step`, and `_metric_step` (`base_model.py`) to stitch once per step and compute frame-weighted `val/loss` and `test/loss` directly on stitched continuous outputs.

### Phase 5: `AveragePrecision`, Full-Recording `TolerantEventF1` & Transition-Stratified `DetectionDelay` (`F30`)
- [x] **Step 5.1 (`F30`):** Create `src/muse_toolbox/metrics/source_counting/average_precision.py` (`AveragePrecision` computing corpus-pooled per-class `AP_0spk..AP_Kspk` + `mAP` with zero-shot guard & zero-positive guard) and export in `metrics/__init__.py`.
- [x] **Step 5.2 (`F30`):** Refactor `src/muse_toolbox/metrics/source_counting/eventF1.py` (`TolerantEventF1`) to:
  - Accumulate stitched sequences by `recording_id` (using `prepend=seq[:1]`) and match once per continuous recording.
  - Expand multi-step jumps $|b - a|$ into multiplicity-aware unit transitions carrying both `step_transition` (`0->1`, etc.) and `raw_jump` (`0->2`, `4->0`, etc.).
  - Solve optimal 1D bipartite matching (`scipy.optimize.linear_sum_assignment`) within `tolerance_time = 1.0 s`.
  - Compute `delay_{mean,median,iqr,abs_mean}_{act,deact,overall}` and expose `get_events_dataframe()`.
- [x] **Step 5.3 (`F30`):** Update `src/muse_toolbox/callbacks/save_results.py` to save `test_events.csv` (and `test_events_<ds_name>.csv`), and register `AveragePrecision` + `tolerance_time: 1.0` in `configs/model/osc.yaml`.

### Phase 6: Synchronize `journal3.tex`, Build PDF & Run Full Test Suite (`A8`, `B12/B13`, `B14`, `C19`, `D20–D23`, `E25–E29`, `F30`)
- [x] **Step 6.1:** Add `\newacronym{AP}`, citations (`park2019specaugment`, `liu2020radam`), and synchronize `journal3.tex` Sections II-C, IV-A, IV-B2, V-C, and VII with the codebase, adhering strictly to `scientific_writing_instructions.txt`.
- [x] **Step 6.2:** Compile `journal3.tex` via `latexmk -pdf -interaction=nonstopmode journal3.tex` cleanly with zero errors.
- [x] **Step 6.3:** Write `tests/test_audit_refactorings.py`, update `tests/test_cornell_baselines.py` (`test_film_layer_initialization_and_gradients`), and verify all tests + `tests/test_model_pipeline.py` pass cleanly.

---

## Execution Log & Deviations

### 1. AMI XML vs. Cornell Table 1 Diagnostic (`scripts/check_ami_annotations.py --no-plot`)
Executed via `/home/henrig/.conda/envs/j3/bin/python scripts/check_ami_annotations.py --no-plot` (exit code `0`):
- **`dev` (18 meetings):**
  - Cornell Table 1: `[15.87, 67.17, 13.95, 2.59, 0.42]%`
  - `AMISegmentsXMLParser` (`segments.xml`): `[17.90, 65.55, 13.61, 2.53, 0.41]%` (`L1 = 4.07`)
  - `AMIWordsXMLParser (<w> only)`: `[21.52, 67.39, 10.05, 0.98, 0.07]%` (`L1 = 11.74`)
  - `AMIWordsXMLParser (<w> + <vocalsound>)`: `[20.84, 66.11, 11.19, 1.63, 0.22]%` (`L1 = 9.94`)
  - `bridge_gaps <= 0.10 s` / `<= 0.20 s`: `L1 = 9.94` / `9.93`
- **`eval` (16 meetings):**
  - Cornell Table 1: `[15.12, 68.39, 12.63, 3.09, 0.76]%`
  - `AMISegmentsXMLParser` (`segments.xml`): `[16.65, 67.16, 12.40, 3.04, 0.75]%` (`L1 = 3.05`)
  - `AMIWordsXMLParser (<w> only)`: `[19.55, 68.72, 9.89, 1.71, 0.13]%` (`L1 = 9.51`)
  - `AMIWordsXMLParser (<w> + <vocalsound>)`: `[19.03, 67.72, 10.72, 2.20, 0.33]%` (`L1 = 7.82`)
  - `bridge_gaps <= 0.10 s` / `<= 0.20 s`: `L1 = 7.81` / `7.81`

### 2. Design Refinements & Deviations
- **`B14` (`out_norm`):** Added `self.out_norm = nn.LayerNorm(embed_dim)` prior to `self.output` in both `TransformerEstimator` and `FusionTransformerEstimator`, and synchronized `journal3.tex` (Fig. 2, Eq. `eq:output_conv`, and Alg. 1) rather than removing `out_norm` from the paper.
- **`E28` (`overlap_avg`, `MHALayer`, `valid_frames`):** Beyond fixing `"causal_burnin"` via per-recording `emitted_end_frame` tracking, also generalized `"overlap_avg"` using `n_ov = min(prev_tail.shape[0], valid_len)` so shorter trailing evaluation chunks combine cleanly without shape mismatches. Updated `MHALayer.forward` to pass `is_causal=(key_padding_mask is None)` with an explicit boolean causal `attn_mask` for PyTorch SDPA compatibility.
- **`F30` (`TolerantEventF1` DataFrame separation):** Kept `TolerantEventF1.get_dataframe()` returning `None` so per-scenario `test_metrics.csv` rows are unchanged, while exposing `get_events_dataframe()` for `SaveTestResultsCallback` to write `test_events.csv` (and `test_events_<ds_name>.csv`). Note that `TolerantEventF1` uses `torch.Tensor.median()`, which selects the lower middle order statistic on even-length delay arrays.

### 3. Final Verification Summary
- **Unit Tests (`/home/henrig/.conda/envs/j3/bin/python`):** `21/21` test functions passed (`test_audit_refactorings.py`: 10/10, `test_cornell_baselines.py`: 3/3, `test_feature_extractors.py`: 4/4, `test_multi_dataset_eval.py`: 4/4 including end-to-end Trainer multi-dataset evaluation). Note: `j3` conda env does not ship a `pytest` binary, so all `test_*` functions were executed directly in `j3`.
- **Pipeline Script (`/home/henrig/.conda/envs/j3/bin/python tests/test_model_pipeline.py`):** Exited `0`; verified `brudex`, `pra_anf_circ_8ch`, and `ami` (forward, loss, backward gradients, validation metrics, and heterogeneous 3+8 mic batching).
- **LaTeX Build (`latexmk -pdf -g -interaction=nonstopmode journal3.tex`):** Exited `0` (15 pages, 0 errors, 0 prose `---` em-dashes, 1 pre-existing undefined citation `mcauliffe2017montreal` at line 288).

### 4. Follow-Up Refinements
- **Synthetic Duration Alignment:** Updated `configs/dataset/base_simulated_dataset.yaml` `num_scenarios` from `[1200, 150, 150]` (`40 h / 5 h / 5 h`) to **`[2400, 300, 300]`** (`80 h / 10 h / 10 h` at `120 s` per scene), matching the unique recording duration of AMI splits (`79.65 h / 9.67 h / 9.06 h`), and documented this in `journal3.tex` Sec. V-A & V-B.
- **AMI Default Parser & Cache Invalidation:** Switched `configs/dataset/ami.yaml` `label_parser` to `AMISegmentsXMLParser` (matching Cornell et al. Table 1; documented `AMIWordsXMLParser` options in comments), added `get_config()` on `BaseAMILabelParser` / `AMIWordsXMLParser`, and updated `AMIDataModule.prepare_data()` to store `"label_parser_config"` in `*_meta.pt` and selectively regenerate stale `*_sad.npy` files without re-loading raw audio when only `label_parser` settings change. Synchronized `journal3.tex` Sec. V-C.
- **Causal vs. Non-Causal / Asymmetric `TolerantEventF1`:** Extended `TolerantEventF1` (`src/muse_toolbox/metrics/source_counting/eventF1.py`) with `past_tolerance_time: float = 0.0`, `future_tolerance_time: float | None = None`, `causal_mode: bool | str = "auto"`, and `estimator_causal: bool = False`. In `OSCmodule` and `configs/model/osc.yaml` (`causal_mode: "auto"`), non-causal models evaluate with symmetric `[-1.0, +1.0] s`, whereas strictly causal models evaluate with asymmetric `[-past_tolerance_time, +future_tolerance_time]` (`[0.0, +1.0] s` by default), treating premature transitions ($\hat{t}_e < t_e$) as false positives. Synchronized `journal3.tex` Sec. VII-D.
- **Docstring Cleanup (`C19`):** Updated `FiLMLayer` class docstring in `src/muse_toolbox/models/source_counting/estimators/transformer_estimator.py` to reflect `0.0` bias initialization cleanly.

---

## Round 2 Refactoring (`#1`–`#18`) Checklist
- [x] **Step 1 (`#6`, `#16`):** Update `agent_history/PROJECT_OVERVIEW.md` (5 datasets, zero-storage `Rand3-8`, single-channel train/val/test protocol, deferred multi-channel permutation + `mic_pos` sync).
- [x] **Step 2 (`#2`, `#3`, `#6`):** Fix `gmsc()` unsquared formula (`signal_stats.py`), add missing `coherenceMatrix(Ry)` call (`gmsc.py`), and add `is_single_channel_input` introspection (`base_feature.py`, `log_mel.py`, `pure_stft.py`, `stacked_features.py`).
- [x] **Step 3 (`#6`, `#7`, `#12`, `#13`, `#16`):** Update `DynamicChunkDataset`, DataModules, and YAML configs (`170%` epoch scaling, length-weighted aug sampling, single-channel train/val/test expansion, dynamic `mic_count_probs` with `precomputed_id`, drop `Circ3-8`, BRUDEX DOAs).
- [x] **Step 4 (`#1`, `#6`, `#11`, `#16`, `#17`):** Add `is_single_channel_system` & `predict_step` stitching (`osc.py`), `AP_VAD`/`AP_OSD` and per-`num_mics` stratification (`AveragePrecision`, `TolerantEventF1`, `META2DF`, `SaveTestResultsCallback` -> `test_metrics_by_num_mics.csv`), fix `trainer.test(ckpt_path="best")`, and update `PrecomputedSAD`.
- [x] **Step 5 (`#5`–`#14`, `#16`, `#18`):** Update `J3bib.bib` (`mcauliffe2017montreal`) and synchronize `journal3.tex`.
- [x] **Step 6:** Extend unit tests (`26/26` unit tests + `test_model_pipeline.py` passed) and compile `journal3.tex` cleanly via `latexmk -pdf -g`.
