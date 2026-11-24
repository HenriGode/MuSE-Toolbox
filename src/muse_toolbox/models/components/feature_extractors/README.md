# Feature Extractors in MuSE-Toolbox

This document summarizes the feature extraction layer within the **Online Source Counting (`OSC`)** architecture (`OSCmodule`).

## 1. High-Level Data Flow

1. **Raw Audio**: Batched dictionary waveforms of shape `(B, M, T_samples)` with `mic_lengths` and `time_lengths`.
2. **STFT Transform**: `STFTtransform` converts raw audio into complex STFT tensors `(B, F, M, T)` (transposed internally to `(B, M, F, T)` inside `BaseFeatureExtractor`), discarding DC and Nyquist bins (`F = L_win / 2 - 1`).
3. **Feature Extraction**: Every `BaseFeatureExtractor` outputs real-valued 4D tensors of shape **`(B, C, F_feat, T)`**, where `C` is the channel/pair count (`1`, `M`, or `P`) and `F_feat` is the block-concatenated per-channel feature dimension.
4. **Source Counting Estimator**: Fixed-array baseline estimators (`TransformerEstimator`, `FusionTransformerEstimator`) flatten `(B, C, F_feat, T) -> (B, C * F_feat, T)` in their input pre-stage before `CatPoolNorm`.

## 2. Implemented Feature Extractors

- **`PureSTFTFeatureExtractor` (`pure_stft.py`)**: Polar (`[|Y|; cos(phi); sin(phi)]`, `3*F`) or Cartesian (`[Re; Im]`, `2*F`), `mode="ref" | "all"`.
- **`LogMel_Feature_Extractor` (`log_mel.py`)**: Log-Mel filterbank energies (`n_mels=80` default), `mode="ref" | "all"`.
- **`IPD_Feature_Extractor` & `CSIPD_Feature_Extractor` (`ipd.py`)**: Pairwise IPD (`F`) and block-concatenated `[cos(ipd); sin(ipd)]` (`2*F`) across `P = M - 1` (`"ref"`) or `P = M*(M-1)//2` (`"all"`) pairs.
- **`PSC_Feature_Extractor` (`psc.py`)**: Recursively smoothed Pairwise Spatial Coherence in block-concatenated polar form `[|Gamma|; cos(psi); sin(psi)]` (`3*F`, default `smoothing_time_constant=0.1` s).
- **`GMSC_Feature_Extractor` & `WGMSC_Feature_Extractor` (`gmsc.py`, `wgmsc.py`)**: Array-condensed coherence features (`C = 1`, `F` or `2*F`).
- **`STFT_Conv_Feature_Encoder` (`stft_conv.py`)**: Learnable causal 1D convolutional STFT encoder.
