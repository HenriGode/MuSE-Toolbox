# Feature Extractor Configurations

This directory contains the default Hydra configuration blocks for instantiating the various feature extractors in MuSE-Toolbox. Each file is designed to be dynamically injected into a model configuration (e.g., `cosad.yaml`) via the `defaults` list.

## 1. How to use these configurations

In your main model config (e.g., `configs/model/cosad.yaml`), you can compose the feature extractor dynamically:

```yaml
defaults:
  - feature_extractor: condensed_logmel  # Points to condensed_logmel.yaml in this directory
```

Because these files use `_partial_: true`, Hydra will inject the configuration parameters as a factory function. The `COSADmodule` then finishes the instantiation by passing the runtime `transform` object.

## 2. Configuration Parameters by Extractor

### 2.1. Log-Mel Spectrograms

**`log_mel.yaml`**
- `mode` (`"ref"` or `"all"`): Whether to extract from a single reference channel or keep all channels.
- `ref_channel` (`int`): Which channel to use if `mode="ref"`.
- `n_mels` (`int`): Number of Mel filterbank bands.
- `f_min` (`float`): Minimum frequency for the Mel scale.
- `f_max` (`float` | `null`): Maximum frequency for the Mel scale.
- `log_offset` (`float`): Small constant added to avoid log(0).

**`condensed_logmel.yaml`**
Inherits all parameters from `log_mel.yaml`, and adds condensation parameters to handle the variable $M$ dimension when `mode="all"`:
- `condense_method` (`"conv"` or `"mean"`): Method to condense the channels. `conv` uses a learned CNN, `mean` simply averages.
- `max_channels` (`int`): Maximum number of microphones expected. The model pre-builds CNNs up to this count.
- `num_layers` (`int`): Number of causal CNN layers (if `condense_method="conv"`).
- `kernel_size` (`int`): Temporal kernel size of the CNN.
- `dropout` (`float`): Dropout probability.

### 2.2. Inter-channel Phase Differences (IPD / CSIPD)

**`ipd.yaml`** and **`csipd.yaml`**
- `mode` (`"ref"` or `"all"`): `"ref"` compares all channels against `ref_channel`. `"all"` computes pairwise differences between all unique combinations.
- `ref_channel` (`int`): Reference channel if `mode="ref"`.

**`condensed_ipd.yaml`** and **`condensed_csipd.yaml`**
Inherits parameters from the basic IPD/CSIPD, plus condensation parameters:
- `condense_method` (`"conv"`, `"circular_mean"`, or `"vector_mean"`): 
  - `"circular_mean"` (IPD) or `"vector_mean"` (CSIPD) are deterministic unlearnable condensation methods.
  - `"conv"` uses learned 1D CNNs to condense the pair dimensions.
- `max_channels`, `num_layers`, `kernel_size`, `dropout`: Same as `condensed_logmel`.

### 2.3. STFT Convolutional Encoder

**`stft_conv.yaml`**
A fully learnable stack that projects directly from the complex STFT.
- `out_channels` (`int`): The final condensed feature dimension.
- `kernel_size` (`int`): Temporal kernel size of the causal CNN.
- `num_layers` (`int`): Number of layers in the CNN stack.
- `dropout` (`float`): Dropout probability.
- `max_channels` (`int`): Maximum expected microphones to pre-build models for.

### 2.4. Spatial Coherence (GMSC / WGMSC)

**`gmsc.yaml`**
- `smoothing_time_constant` (`float`): Time constant in seconds for recursively smoothing the signal covariance matrices.

**`wgmsc.yaml`**
Advanced noise-whitened coherence.
- `smoothing_time_constant` (`float`): Forward smoothing time constant [s].
- `whitening_time_constant` (`float`): Look-back window size for estimating the noise covariance matrix [s].
- `rev_features` (`bool`): If `true`, also runs the estimations backwards in time.
- `smoothing_time_constant_rev` (`float`): Reverse smoothing time constant [s].
- `whitening_time_constant_rev` (`float`): Look-forward window size for estimating the reverse noise covariance [s].
- `wideband_features` (`bool`): If `true`, averages coherence across all frequency bins. If `false`, outputs narrowband coherence per frequency bin.

### 2.5. Meta-Extractor: Stacked Features

**`stacked.yaml`**
Allows concatenating multiple feature extractors together.
- `extractors` (`list`): A list of dictionaries. Each dictionary must have a single key (the class name, e.g. `Condensed_LogMel_Feature_Extractor`) and its value is a block of configuration parameters for that specific extractor.
