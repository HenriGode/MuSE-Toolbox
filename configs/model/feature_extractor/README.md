# Feature Extractor Configurations

This directory contains the default Hydra configuration blocks for instantiating feature extractors in MuSE-Toolbox. Each file uses `_partial_: true` so `OSCmodule` injects the runtime `transform` object during instantiation in `configs/model/osc.yaml` or `configs/model/osc_fusion.yaml`.

## Available Configs

- **`log_mel.yaml`**: Single-channel (`mode: "ref"`) or multi-channel (`mode: "all"`) Log-Mel filterbanks (`n_mels: 80`).
- **`pure_stft.yaml`**: Polar (`representation: "polar"`, `3*F`) or Cartesian (`representation: "cartesian"`, `2*F`) pure STFT features (`mode: "ref" | "all"`).
- **`ipd.yaml` / `csipd.yaml`**: Pairwise IPD (`F`) and block-concatenated CSIPD (`2*F`, `mode: "ref" | "all"`).
- **`psc.yaml`**: Pairwise Spatial Coherence (`3*F`, `smoothing_time_constant: 0.1`, `mode: "ref" | "all"`).
- **`gmsc.yaml` / `wgmsc.yaml`**: Array-condensed GMSC (`F`) and bidirectional whitened GMSC (`2*F`).
- **`stft_conv.yaml`**: Learnable causal 1D CNN STFT encoder.
- **`stacked.yaml`**: Meta-extractor concatenating extractors that share the same channel dimension `C`.
