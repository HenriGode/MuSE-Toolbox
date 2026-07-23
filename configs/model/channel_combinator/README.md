# Channel Combinators

This directory contains configuration files for the Channel Combinator subsystem.
The Channel Combinator is responsible for transforming a variable number of input channels 
(e.g., from different microphones or microphone pairs) into a fixed-size output dimension, 
so the subsequent source count estimator (classifier) can operate on a fixed shape.

All inputs to Channel Combinators have the shape `(Batch, Channels, Features, Time)`
and produce an output of shape `(Batch, C_out, Features, Time)`.

## Available Combinators

- **identity.yaml**: The 'Do Nothing' combinator. Useful for features that are already channel-independent (e.g. GMSC).
- **average.yaml**: Averages across the channel dimension to produce exactly 1 output channel.
- **select.yaml**: Selects a specific reference channel (default `ref_channel: 0`).
- **mlp.yaml**: A learnable network that maps variable input channels to a fixed output channel size using a separate small MLP for each possible number of input channels up to `max_channels`.
- **tac.yaml**: Transform-Average-Concatenate (TAC) layer that processes each channel independently and then pools them.
- **self_attention.yaml**: Treats each channel as a token in a sequence, applies attention over channels, and averages the representations.
- **cross_attention.yaml**: Computes a global query (mean of channels) and attends to individual channels as keys/values.

## Usage

To use a specific channel combinator, include it in your model's yaml configuration:

```yaml
defaults:
  - channel_combinator: average
```

Or from the command line:

```bash
python scripts/main.py model.channel_combinator=cross_attention
```

## Adding a New Combinator

1. Create a new python file in `src/muse_toolbox/models/components/channel_combinator/` or add to an existing file.
2. Create a class that inherits from `BaseChannelCombinator`.
3. Implement `is_trainable` and `forward(self, x: torch.Tensor) -> torch.Tensor`.
4. Export your class in `src/muse_toolbox/models/components/channel_combinator/__init__.py`.
5. Create a default config file in this directory (`configs/model/channel_combinator/`) using Hydra syntax.
