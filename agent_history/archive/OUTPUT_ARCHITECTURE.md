# Output Directory Architecture

This document outlines the agreed-upon architecture for organizing experiment outputs, predictions, logs, and checkpoints in the MuSE-Toolbox.

## Core Philosophy
The hierarchy prioritizes the **Dataset Split** (Train vs Val vs Test) over the **Data Type** (Results vs Predictions). This allows for easy archiving, manipulation, and clarity of exactly which split generated a specific artifact.

## Directory Tree

```text
outputs/
└── <task_name>/                                   <-- Task Level (e.g. source_counting)
    └── <experiment_id>/                           <-- Experiment Level (e.g. J3_BXLS_test)
        └── <timestamp>/                           <-- Timestamp Level (e.g. 2026-07-16_17-30-00)
            │
            ├── logging/                           <-- Consolidated logs
            │   ├── .hydra/                        <-- Config snapshots
            │   └── wandb/                         <-- WandB telemetry
            │
            ├── checkpoints/                       <-- Model weights
            │
            ├── train/                             <-- Training Split
            │   ├── results/
            │   ├── predictions/
            │   └── output_audio/
            │
            ├── val/                               <-- Validation Split
            │   ├── results/
            │   ├── predictions/
            │   └── output_audio/
            │
            └── test/                              <-- Testing Split
                ├── results/
                ├── predictions/
                └── output_audio/
```

## Implementation Strategy

1. **Hydra Setup:** 
   - `hydra.run.dir` dynamically constructs `outputs/${task}/${experiment}/${now:%Y-%m-%d_%H-%M-%S}`.
   - `hydra.output_subdir` is set to `logging/.hydra` to move the Hydra configs into the consolidated logging folder.
2. **WandB & PyTorch Lightning:** 
   - `WandbLogger(save_dir=...)` targets the `logging/` directory.
   - `ModelCheckpoint(dirpath=...)` targets the `checkpoints/` directory.
3. **Custom Callbacks & Metrics:**
   - Callbacks like `SaveTestResultsCallback` or `BasePredictionWriter` will interrogate PyTorch Lightning's state flags (`trainer.training`, `trainer.validating`, `trainer.testing`) during execution.
   - Based on the active flag, they will dynamically route their file outputs into the corresponding split subdirectory (e.g., if `trainer.testing` is active, save to `${hydra:runtime.output_dir}/test/results/`).
