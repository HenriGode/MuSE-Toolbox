import torch
from muse_toolbox.models.components.feature_extractors.gmsc import GMSC_Feature_Extractor
from muse_toolbox.models.components.feature_extractors.wgmsc import WGMSC_Feature_Extractor
from muse_toolbox.utils import STFTtransform

transform = STFTtransform(frame_length=512, frame_shift=256, sampling_frequency=16000, window_type='hann')
extractor_gmsc = GMSC_Feature_Extractor(transform=transform, smoothing_time_constant=0.1)
extractor_wgmsc = WGMSC_Feature_Extractor(transform=transform, smoothing_time_constant=0.1, whitening_time_constant=0.2, smoothing_time_constant_rev=0.1, whitening_time_constant_rev=0.2, wideband_features=False)

raw_audio = torch.randn(1, 8, 16000)

print("--- GMSC ---")
feat = extractor_gmsc.forward_raw_audio(raw_audio)
print("Forward out:", feat.shape)
feat = feat.squeeze(0)
print("Squeeze(0) out:", feat.shape)

print("--- WGMSC (Narrowband) ---")
feat = extractor_wgmsc.forward_raw_audio(raw_audio)
print("Forward out:", feat.shape)
feat = feat.squeeze(0)
print("Squeeze(0) out:", feat.shape)

