# HeterogeneousBatch Architectural Refactor

This document outlines the planned refactoring of the `HeterogeneousBatch` class, to be executed **after** the first successful end-to-end integration test run of the pipeline.

## The Problem
Currently, `HeterogeneousBatch` is a massive data container doing significantly more than it should. It encapsulates logic for:
- Initializing and executing STFT transformations.
- Computing array geometries and performing spatial operations.
- Interfacing directly with `torch.device` casting.

This tightly couples the data pipeline directly to the DSP/Transform code and to the Modeling code. In fact, it was the root cause of the severe circular import issues encountered earlier in development, because the models needed the batch definitions, but the batch definitions were importing the models and their transform tools.

## The Goal
The data layer should be entirely agnostic of what the model plans to do with it.

1. **Strip `HeterogeneousBatch`:** Refactor it to act as a pure, "dumb" data class (or a `typing.NamedTuple` / `@dataclass`). It should strictly be a container carrying `audio_signals`, `labels`, and `metadata` from the DataLoader to the model.
2. **Move Processing Logic:** All STFT generation, windowing logic, and feature extraction (LogMel, etc.) must be relocated entirely inside the `forward()` pass of the PyTorch Lightning Modules (e.g., `COSADmodule.forward()`).
3. **Resolve Decoupling:** By stripping the logic out of `HeterogeneousBatch`, the `data/` modules will no longer depend on the `utils/dsp` or `models/` layers, establishing a clean, unidirectional flow of dependencies:
   `Data Layer -> Model Layer -> Utility Layer`

## Execution Trigger
This refactoring should begin immediately after the first successful end-to-end run of the `source_counting` pipeline has been verified.
