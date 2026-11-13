# Grand Unification Plan: Unified Dataloading & Chunking

The goal of this architectural refactor is to perfectly unify the data logistics for all datasets (AMI, BRUDEX, and PRA_ANF). This plan replaces the legacy full-sequence loading paradigm of synthetic datasets with the high-performance dynamic chunking paradigm designed for AMI, ensuring that all models are trained and evaluated identically, regardless of the underlying dataset.

## Proposed Implementation Steps

### 1. Configuration Unification
Currently, chunking and augmentation parameters are hardcoded only for AMI. We need to promote them to the base level.
- **[MODIFY]** `configs/dataset/base_dataset.yaml`:
  - Add `chunk_length_s: 7.0`
  - Add `min_context_s: 2.0`
  - Add the base `augmentation` dictionary (with `enabled: false` by default).
- **[MODIFY]** `configs/dataset/ami.yaml`:
  - Remove duplicate definitions and set `augmentation.enabled: true` explicitly for AMI.

### 2. Standardization to Memory-Mapped `.npy` & Removal of Precomputation
To make 7s slicing lightning fast for synthetic datasets (BRUDEX/PRA_ANF) without allocating full 120s tensors into RAM, we will change how they are cached, and **completely remove STFT/feature precomputation** from the dataloaders.
- **[MODIFY]** `src/muse_toolbox/data/datamodules/base_datamodule.py`:
  - In `_generate_scenarios()`, delete all logic related to computing and saving `stft` and `feats`.
  - Instead of saving a single `scenario_{i}.pt` containing the raw audio, we will exclusively save the audio to `scenario_{i}_audio.npy` and the metadata (like SAD labels and parameters) to `scenario_{i}_meta.pt`. All STFT and feature extraction will now happen natively on the GPU during the model's `forward()` pass.

### 3. The Universal Dynamic Chunking Dataset
We will abstract the fantastic slicing logic you built for AMI into a universal dataset class that handles both 1-hour meetings and 120s synthetic scenarios seamlessly.
- **[ADD]** `src/muse_toolbox/data/components/dynamic_chunk_dataset.py`:
  - Create a `DynamicChunkDataset` class that maps `.npy` files.
  - **Training Logic:** Grabs a random 7s slice. Applies the overlap augmentation *only* if the config says `augmentation.enabled: true`.
  - **Validation/Testing Logic (Stateless Overlap-Add):** Generates a deterministic grid of sequential 7s chunks separated by the hop size (`chunk_length_s - min_context_s`). 

### 4. Legacy Code Execution (Phase A)
With the new universal dataset ready and precomputation abandoned, we can delete the old pipelines.
- **[MODIFY]** `src/muse_toolbox/data/datamodules/base_datamodule.py`:
  - Update `setup()` to completely bypass the old dataset and instantiate the new `DynamicChunkDataset` for train/val/test.
  - Update the dataloader functions to strictly use `raw_audio_collate_fn`.
  - Remove any `force_load_stft` or `feature_extractor` attributes from the DataModule layer, as they belong strictly to the Model layer now.
- **[DELETE]** `src/muse_toolbox/data/components/precomputed_dataset.py`: Delete entirely.
- **[DELETE]** `src/muse_toolbox/data/components/heterogeneous_batch.py`: Delete entirely.

### 5. Stateless Inference Wrapper
Because the models evaluate 7s chunks statelessly, the output of each chunk must be stitched together properly, throwing away the first 2s (which lacks historical context).
- **[MODIFY]** Base evaluation code (e.g., `BaseModel` or validation loop):
  - Ensure that during testing/validation, when we receive the batched predictions for a 7s chunk, we discard the `0s-2s` region (where context is weak) and only keep the `2s-7s` region, stitching them together to rebuild the full 120s or 1-hour prediction.

## Review & Approval
Please review this unified implementation plan. If it perfectly aligns with your vision for the architecture, click **Proceed**, and I will begin refactoring the configuration files and `BaseDataModule` caching logic!

## Current Status (End of Day Update)
- **Phase 1-4 (Dataloaders & Configs):** ✅ **COMPLETED**. `DynamicChunkDataset` is functional. `HeterogeneousBatch` is fully removed. Datasets correctly yield `[Batch, Mics, Time]` dense audio tensors and metadata dictionaries. Tests for `brudex`, `pra_anf`, and `ami` datasets all pass perfectly.
- **Phase 5 (Model Refactoring & Testing):** 🔄 **IN PROGRESS**. `BaseLitModel`, `gss.py`, `rtf_module.py`, `precomputed_sad.py`, and `cosad.py` have been refactored to accept dictionary batches. Tomorrow, we will verify the models' execution and address the fact that different items in a batch could have a different number of microphones.
- **CHiME-6 Implementation:** ⏳ **PENDING**.
