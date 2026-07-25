import torch
import traceback
from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch
from muse_toolbox.models.components.feature_extractors.wgmsc import WGMSC_Feature_Extractor
from muse_toolbox.utils import STFTtransform
from muse_toolbox.models.components.channel_combinator.attention import SelfAttentionChannelCombinator

try:
    transform = STFTtransform(frame_length=512, frame_shift=256, sampling_frequency=16000, window_type='hann')
    extractor = WGMSC_Feature_Extractor(transform=transform, smoothing_time_constant=0.1, whitening_time_constant=0.2, smoothing_time_constant_rev=0.1, whitening_time_constant_rev=0.2, wideband_features=False)

    # Let's mock a HeterogeneousBatch with raw audio
    batch = HeterogeneousBatch(
        raw_audio=[torch.randn(8, 16000)], # 1 second audio
        meta={'sad_frames': [{'A': torch.zeros(60)}], 'rtfs': [{'A': torch.zeros(257, 1, 8, 1)}], 'segments': [[ ]]},
    )
    # We don't care about meta for this test
    batch.meta['source_count'] = [torch.tensor([1])]
    
    print("Applying extractor...")
    batch.apply_feature_extractor(extractor)
    print("Processed feature shape:", batch.processed_features[0].shape)
    
    combinator = SelfAttentionChannelCombinator(input_feature_dim=257)
    print("Applying combinator...")
    batch.apply_channel_combinator(combinator)
    print("Combinator Success! Padded features shape:", batch.padded_features.shape)
    
except Exception as e:
    traceback.print_exc()

