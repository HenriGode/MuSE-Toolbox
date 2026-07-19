# Feature Extractors in MuSE-Toolbox

This document provides a deep dive into the feature extraction logic within the `COSAD` architecture. It covers the data processing flow, the implemented feature extractors, how they manipulate tensor dimensions, and crucially, how they handle a variable number of input microphone channels ($M$).

## 1. High-Level Data Flow

The `HeterogeneousBatch` pipeline unifies the pre-processing. The flow from raw audio to estimates is as follows:

1. **Raw Audio**: The pipeline starts with a batch of raw waveforms of shape `(Batch, M, Samples)`.
2. **STFT Transform**: The `STFTtransform` module converts the raw audio into the time-frequency domain resulting in complex tensors of shape `(Batch, F, M, T)`, where:
   - $F$ = Number of Frequency Bins
   - $M$ = Number of Microphones (Channels)
   - $T$ = Number of Time Frames
3. **Feature Extraction**: The specific feature extractor (subclassing `BaseFeatureExtractor`) consumes this STFT and outputs a real-valued 2D representation per frame. 
   - **Output Shape**: `(Batch, J, T)` where $J$ is the final condensed feature dimension.
4. **Source Counting Estimator**: A sequential model (like a GRU) takes `(Batch, J, T)` and predicts the source count `(Batch, max_sources+1, T)`.

---

## 2. Implemented Feature Extractors

### 2.1. Log-Mel Spectrograms (`log_mel.py`)
Extracts standard spectral power characteristics.

- **`LogMel_Feature_Extractor`**:
  - Takes STFT `(B, F, M, T)` $\rightarrow$ Calculates Power $\rightarrow$ Applies MelScale filterbanks.
  - **Dimensionality**: Outputs `(B, M, n_mels, T)`. 
  - **Handling $M$**: If `mode="ref"`, it simply selects the reference channel resulting in `(B, n_mels, T)`. If `mode="all"`, it flattens the channels into the feature dimension resulting in `(B, M * n_mels, T)`.

- **`Condensed_LogMel_Feature_Extractor`**:
  - Handles the variable $M$ dimension from the "all" mode to enforce a fixed output dimension $J = n\_mels$.
  - **Handling Variable $M$**: 
    - **`condense_method="mean"`**: Simply averages across all channels, reducing `(B, M, n_mels, T)` $\rightarrow$ `(B, n_mels, T)`.
    - **`condense_method="conv"`**: Maintains a `nn.ModuleDict` containing a separate causal 1D CNN for every possible microphone count (up to `max_channels`). It dynamically routes the flattened `(B, M * n_mels, T)` tensor into the CNN matching $M$, which projects it down to `(B, n_mels, T)`.

### 2.2. Inter-channel Phase Difference (IPD / CSIPD) (`ipd.py`)
Captures spatial information by evaluating the phase differences between microphones.

- **`IPD_Feature_Extractor` & `CSIPD_Feature_Extractor`**:
  - Evaluates phase angle differences.
  - **Handling $M$**: Evaluates $P$ microphone pairs. If `mode="ref"`, $P = M-1$. If `mode="all"`, it evaluates all unique pairs $P = M(M-1)/2$.
  - **Dimensionality**: IPD outputs `(B, P * F, T)`. CSIPD splits angles into Cosine and Sine components, doubling the dimension to `(B, 2 * P * F, T)`.

- **`Condensed_IPD_Feature_Extractor` & `Condensed_CSIPD_Feature_Extractor`**:
  - Condenses the variable $P$-dependent dimension into a fixed dimension $J = F$ (or $J=2F$ for CSIPD).
  - **Handling Variable $M$**:
    - **`condense_method="circular_mean"` (IPD)**: Computes the circular mean of angles across the $P$ pairs. Output is `(B, F, T)`.
    - **`condense_method="vector_mean"` (CSIPD)**: Averages the Cos/Sin vectors across the $P$ pairs. Output is `(B, 2F, T)`.
    - **`condense_method="conv"`**: Similar to LogMel, utilizes a dictionary of causal CNNs tailored to the number of input pairs $P$, projecting the flat input down to the fixed $F$ (or $2F$) dimension.

### 2.3. STFT Convolutional Encoder (`stft_conv.py`)
A purely learnable feature extractor operating directly on the complex STFT.

- **`STFT_Conv_Feature_Encoder`**:
  - Takes STFT `(B, F, M, T)`.
  - **Dimensionality**: Stacks the Real and Imaginary components $\rightarrow$ `(B, 2F, M, T)`. Flattens channels and frequency $\rightarrow$ `(B, M * 2F, T)`.
  - **Handling Variable $M$**: Because the flattened dimension $M \times 2F$ changes with the number of microphones, this module maintains a `nn.ModuleDict` of causal 1D CNNs (one for every $M \in [1, max\_channels]$). It dynamically routes the tensor to the matching CNN, which maps it to a fixed `out_channels` dimension $J$. Output: `(B, J, T)`.

### 2.4. Generalized Magnitude-Squared Coherence (WGMSC / GMSC) (`wgmsc.py` / `gmsc.py`)
Advanced spatial coherence features specifically designed to be robust to noise.

- **WGMSC Mechanism**:
  - Calculates a recursively smoothed covariance matrix of the mixture signal across the $M$ channels.
  - Estimates a noise covariance matrix from a past look-back window (Whitening).
  - Computes the Whitened GMSC from the whitened coherence matrix.
- **Handling Variable $M$**: The elegance of GMSC/WGMSC is that the spatial coherence metric is a singular scalar property calculated from the determinant/trace of the $M \times M$ spatial covariance matrix. Therefore, **it is inherently invariant to the number of microphones $M$**. The operation collapses $M$ internally.
- **Dimensionality**:
  - **Narrowband**: Evaluates coherence per frequency bin $\rightarrow$ `(B, F, T)`.
  - **Wideband**: Combines frequencies using the trace of the whitened covariance as weights $\rightarrow$ `(B, 1, T)`.
  - **Reverse Features (`rev_features`)**: If enabled, computes an additional look-forward covariance estimation, appending it to the forward estimation $\rightarrow$ `(B, 2F, T)` or `(B, 2, T)`.

### 2.5. Stacked Features (`stacked_features.py`)
A meta-extractor that allows arbitrary combinations of the above extractors.

- **Mechanism**: Takes a list of feature extractor configurations. It instantiates them, runs them in parallel (either using raw audio or STFT), and concatenates their outputs along the feature dimension $J$.
- **Dimensionality**: Output $J = \sum J_i$ (the sum of the feature dimensions of all sub-extractors).
