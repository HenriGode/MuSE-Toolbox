import torch
from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch
from muse_toolbox.models.components.feature_extractors.gmsc import GMSC_Feature_Extractor
from muse_toolbox.utils import STFTtransform

transform = STFTtransform(frame_length=512, frame_shift=256, sampling_frequency=16000, window_type='hann')
extractor = GMSC_Feature_Extractor(transform=transform, smoothing_time_constant=0.1)

# Case 1: Raw Audio (what the pra_anf dataset actually outputs)
raw_audio = [torch.randn(8, 16000)] # M=8, N=16000
batch = HeterogeneousBatch(raw_audio=raw_audio, meta={'sad_frames': [{}], 'rtfs': [{}], 'segments': [[]]})
batch.apply_feature_extractor(extractor)
print("Raw Audio processed feature shape:", batch.processed_features[0].shape)
batch._pad_features()
print("Raw Audio padded feature shape:", batch.padded_features.shape)
