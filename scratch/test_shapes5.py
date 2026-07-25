import torch
import traceback
from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch
from muse_toolbox.models.components.feature_extractors.wgmsc import WGMSC_Feature_Extractor
from muse_toolbox.utils import STFTtransform

try:
    class DummyTransform(STFTtransform):
        def encode(self, x):
            shape = list(x.shape)
            if len(shape) == 2:
                return torch.randn(257, shape[0], 100, dtype=torch.complex64)
            else:
                return torch.randn(shape[0], 257, shape[1], 100, dtype=torch.complex64)

    transform = DummyTransform(frame_length=512, frame_shift=256, sampling_frequency=16000, window_type='hann')
    extractor = WGMSC_Feature_Extractor(transform=transform, smoothing_time_constant=0.1, whitening_time_constant=0.2, smoothing_time_constant_rev=0.1, whitening_time_constant_rev=0.2, wideband_features=False)

    class MockBatch(HeterogeneousBatch):
        def __init__(self):
            self.raw_audio = [torch.randn(8, 16000)]
            self.stft_audio = []
            self.features = []
            self.meta = {}
            self.batch_size = 1
            self.device = torch.device('cpu')
            self.status = 'input'
            self.padded_features = None
            self.mask = None

    batch = MockBatch()

    batch.apply_feature_extractor(extractor)
    print("After apply_feature_extractor:", batch.processed_features[0].shape)
    batch._pad_features()
    print("After padding:", batch.padded_features.shape)

    class DummyCombinator:
        def forward(self, x):
            b, c, f, t = x.shape
            return x

    batch.apply_channel_combinator(DummyCombinator())
    print("Combinator Success!")
except Exception as e:
    traceback.print_exc()
