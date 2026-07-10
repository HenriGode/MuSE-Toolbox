# Data
**Pure Inputs**

## Purpose
This directory stores raw data and processed datasets meant for model ingestion. **Files here should be treated as Read-Only.**

## Structure
- **raw_databases/**: Original audio corpora (LibriSpeech, WSJ). Usually symlinks.
- **datamodules/**: Organized by DataModule name.
  - **<name>/original/**: The effective "source" audio for this datamodule. This is often .wav files that have been cut/manipulated from the raw database, or file lists (JSON/CSV) pointing to them.
  - **<name>/precomputed/**: Pre-processed tensors (.pt files) ready for PyTorch loading.

## Rules
1. **Never commit large files** to Git.
2. Use symlinks for huge datasets to avoid duplication.
