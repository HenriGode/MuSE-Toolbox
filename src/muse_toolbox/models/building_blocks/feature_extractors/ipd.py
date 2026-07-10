import torch
from .base_feature import BaseFeatureExtractor
from muse_toolbox.utils import STFTtransform
import torch.nn as nn
from building_blocks import CausalConv1d
from collections import defaultdict


class IPD_Feature_Extractor(BaseFeatureExtractor):
    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
    ):
        super().__init__(transform=transform)
        self.transform = transform
        self.mode = mode
        self.ref_channel = ref_channel

        if mode not in ["ref", "all"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'ref' or 'all'.")

    def get_config(self) -> dict:
        return {
            "mode": self.mode,
            "ref_channel": self.ref_channel,
        }

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def signature(self) -> str:
        sign = (
            f"IPD_Mdependent"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"
        return sign

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError(
            "feature_dim is M-dependent for IPD features. Use Condensed_IPD_Feature_Extractor for fixed dimension."
        )

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, F, M, T)
        phase = torch.angle(batch)
        B, F, M, T = phase.shape

        if self.mode == "ref":
            ref_phase = phase[:, :, self.ref_channel : self.ref_channel + 1, :]
            diff = phase - ref_phase
            # Remove the ref channel
            indices = [i for i in range(M) if i != self.ref_channel]
            if not indices:
                return torch.empty(B, F, 0, T, device=phase.device)
            diff = diff[:, :, indices, :]

        elif self.mode == "all":
            # Compute all pairs (i, j) where i < j
            diffs = []
            for i in range(M):
                for j in range(i + 1, M):
                    diffs.append(phase[:, :, i : i + 1, :] - phase[:, :, j : j + 1, :])
            if not diffs:
                return torch.empty(B, 0, F, T, device=phase.device)
            diff = torch.cat(diffs, dim=-2)

        else:
            raise ValueError(f"Invalid mode '{self.mode}'.")

        # Wrap to [-pi, pi]
        ipd = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi

        # Flatten channels and freq -> (B, J, T)
        B, F, P, T = ipd.shape
        J = P * F
        return ipd.swapaxes(-2, -3).reshape(B, J, T)


class CSIPD_Feature_Extractor(IPD_Feature_Extractor):
    @property
    def signature(self) -> str:
        sign = (
            f"CSIPD_Mdependent"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"
        return sign

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError(
            "feature_dim is M-dependent for CSIPD features. Use Condensed_CSIPD_Feature_Extractor for fixed dimension."
        )

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # 1. Get standard IPD features: (B, J, T)
        ipd_flat = super().forward_stft(batch)
        B, J, T = ipd_flat.shape

        c = torch.cos(ipd_flat)
        s = torch.sin(ipd_flat)

        # Interleave cos and sin: (B, J, 2, T)
        csipd = torch.stack((c, s), dim=2)

        # Flatten to (B, 2*J, T)
        return csipd.reshape(B, J * 2, T)


class Condensed_IPD_Feature_Extractor(IPD_Feature_Extractor):
    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
        condense_method: str = "conv",  # "conv" or "circular_mean"
        max_channels: int = 6,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        # We pass num_channels=None because feature_dim is now fixed (F)
        super().__init__(transform=transform, mode=mode, ref_channel=ref_channel)
        self.condense_method = condense_method
        self.max_channels = max_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.freq_dim = self.transform.num_freq_bins

        if self.condense_method == "conv":
            self.models = nn.ModuleDict()
            for m in range(2, max_channels + 1):
                # Calculate number of pairs P based on mode
                if self.mode == "ref":
                    p = m - 1
                else:
                    p = m * (m - 1) // 2

                # Map (P*F) -> F
                self.models[str(m)] = self._build_model(input_dim=p * self.freq_dim)

    def _build_model(self, input_dim: int) -> nn.Sequential:
        """Builds the causal CNN stack for a specific input dimension."""
        layers = []
        for i in range(self.num_layers):
            in_ch = input_dim if i == 0 else self.freq_dim

            # Use custom CausalConv1d
            layers.append(CausalConv1d(in_ch, self.freq_dim, self.kernel_size))
            layers.append(nn.BatchNorm1d(self.freq_dim))
            layers.append(nn.ReLU())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    @property
    def is_trainable(self) -> bool:
        return self.condense_method == "conv"

    @property
    def signature(self) -> str:
        sign = (
            f"IPD"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"

        sign += f"_{self.condense_method}"

        if self.condense_method == "conv":
            sign += f"_nl{self.num_layers}_ks{self.kernel_size}_do{self.dropout}"
        return sign

    @property
    def feature_dim(self) -> int:
        # Always returns F, independent of M
        return self.freq_dim

    @property
    def precompute_type(self) -> str:
        return "features"

    def precompute(
        self, batch: torch.Tensor, input_type: str = "raw_audio"
    ) -> dict[str, torch.Tensor]:
        precomputedict = defaultdict(torch.Tensor)

        # Get STFT first
        if input_type == "raw_audio":
            stft = self.transform.encode(batch)
        elif input_type == "stft":
            stft = batch
        else:
            raise ValueError(f"Invalid input_type {input_type}")

        # Compute raw IPD (using parent class)
        if stft.ndim == 3:
            stft = stft.unsqueeze(0)  # (1, F, M, T)
        else:
            raise ValueError(
                "Expected stft input to have 3 dimensions (F, M, T). No batch dimension in the precompute step allowed!"
            )

        raw_ipd = super().forward_stft(stft)

        raw_ipd = raw_ipd.squeeze(0)  # (J, T)

        precomputedict["features"] = raw_ipd
        return precomputedict

    def forward_precomputed_features(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, J, T)
        ipd_flat = batch
        B, J, T = ipd_flat.shape

        # Infer M from J
        P = J // self.freq_dim
        if self.mode == "ref":
            M = P + 1
        else:
            # P = M*(M-1)/2 => M^2 - M - 2P = 0
            M = int((1 + (1 + 8 * P) ** 0.5) / 2)

        if self.condense_method == "conv":
            if str(M) not in self.models:
                raise ValueError(
                    f"No model found for M={M} (max_channels={self.max_channels})"
                )
            return self.models[str(M)](ipd_flat)

        elif self.condense_method == "circular_mean":
            # Reshape to (B, F, P, T) to average over P
            ipd = ipd_flat.reshape(B, self.freq_dim, P, T)

            # Circular Mean: atan2( sum(sin), sum(cos) )
            sin_sum = torch.sum(torch.sin(ipd), dim=-2)
            cos_sum = torch.sum(torch.cos(ipd), dim=-2)
            mean_ipd = torch.atan2(sin_sum, cos_sum)  # (B, F, T)

            return mean_ipd

        else:
            raise ValueError(f"Unknown condense_method: {self.condense_method}")

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # 1. Get standard IPD features: (B, P*F, T)
        ipd_flat = super().forward_stft(batch)
        return self.forward_precomputed_features(ipd_flat)


class Condensed_CSIPD_Feature_Extractor(CSIPD_Feature_Extractor):
    def __init__(
        self,
        transform: STFTtransform,
        mode: str = "ref",
        ref_channel: int = 0,
        condense_method: str = "conv",  # "conv" or "vector_mean"
        max_channels: int = 8,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__(transform=transform, mode=mode, ref_channel=ref_channel)
        self.condense_method = condense_method
        self.max_channels = max_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.freq_dim = self.transform.num_freq_bins

        if self.condense_method == "conv":
            self.models = nn.ModuleDict()
            for m in range(2, max_channels + 1):
                if self.mode == "ref":
                    p = m - 1
                else:
                    p = m * (m - 1) // 2

                # Map (2*P*F) -> 2*F
                self.models[str(m)] = self._build_model(input_dim=2 * p * self.freq_dim)

    def _build_model(self, input_dim: int) -> nn.Sequential:
        """Builds the causal CNN stack for a specific input dimension."""
        layers = []
        for i in range(self.num_layers):
            in_ch = input_dim if i == 0 else 2 * self.freq_dim

            # Use custom CausalConv1d
            layers.append(CausalConv1d(in_ch, 2 * self.freq_dim, self.kernel_size))
            layers.append(nn.BatchNorm1d(2 * self.freq_dim))
            layers.append(nn.ReLU())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    @property
    def is_trainable(self) -> bool:
        return self.condense_method == "conv"

    @property
    def signature(self) -> str:
        sign = (
            f"CSIPD"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"

        sign += f"_{self.condense_method}"

        if self.condense_method == "conv":
            sign += f"_nl{self.num_layers}_ks{self.kernel_size}_do{self.dropout}"
        return sign

    @property
    def feature_dim(self) -> int:
        # Always returns 2*F
        return 2 * self.freq_dim

    @property
    def precompute_type(self) -> str:
        return "features"

    def precompute(
        self, batch: torch.Tensor, input_type: str = "raw_audio"
    ) -> dict[str, torch.Tensor]:
        precomputedict = defaultdict(torch.Tensor)

        # Get STFT first
        if input_type == "raw_audio":
            stft = self.transform.encode(batch)
        elif input_type == "stft":
            stft = batch
        else:
            raise ValueError(f"Invalid input_type {input_type}")

        # Compute raw CSIPD (using parent class)
        if stft.ndim == 3:
            stft = stft.unsqueeze(0)  # (1, F, M, T)
        else:
            raise ValueError(
                "Expected stft input to have 3 dimensions (F, M, T). No batch dimension in the precompute step allowed!"
            )

        raw_csipd = super().forward_stft(stft)

        raw_csipd = raw_csipd.squeeze(0)  # (J, T)

        precomputedict["features"] = raw_csipd
        return precomputedict

    def forward_precomputed_features(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, 2*P*F, T)
        csipd_flat = batch
        B, PF2, T = csipd_flat.shape

        # Infer M
        P = (PF2 // 2) // self.freq_dim
        if self.mode == "ref":
            M = P + 1
        else:
            M = int((1 + (1 + 8 * P) ** 0.5) / 2)

        if self.condense_method == "conv":
            if str(M) not in self.models:
                raise ValueError(f"No model found for M={M}")
            return self.models[str(M)](csipd_flat)

        elif self.condense_method == "vector_mean":
            # Reshape to (B, P, F, 2, T)
            csipd_unflat = csipd_flat.reshape(B, P, self.freq_dim, 2, T)

            # Average over P (dim 1) -> (B, F, 2, T)
            mean_csipd = torch.mean(csipd_unflat, dim=1)

            # Flatten F and 2 -> (B, 2*F, T)
            return mean_csipd.reshape(B, self.freq_dim * 2, T)

        else:
            raise ValueError(f"Unknown condense_method: {self.condense_method}")

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        # 1. Get standard CSIPD features: (B, 2*P*F, T)
        csipd_flat = super().forward_stft(batch)
        return self.forward_precomputed_features(csipd_flat)


if __name__ == "__main__":
    transform = STFTtransform(
        frame_length=0.04,
        frame_shift=0.01,
        sampling_frequency=8000,
        window_type="sqrt-hann",
    )

    def test_condensed_ipd():
        stft = transform
        # Test Learnable
        extractor = Condensed_IPD_Feature_Extractor(
            transform=stft, mode="ref", condense_method="conv", max_channels=6
        )

        # Batch with M=3
        B, M, T_samples = 2, 6, 480000
        x = torch.randn(B, M, T_samples)

        out = extractor(x)
        print(f"IPD Learnable Output shape: {out.shape}")
        assert out.shape[1] == stft.nfft // 2 + 1

        # Test Circular Mean
        extractor_mean = Condensed_IPD_Feature_Extractor(
            transform=stft, mode="ref", condense_method="circular_mean"
        )
        out_mean = extractor_mean(x)
        print(f"IPD Mean Output shape: {out_mean.shape}")
        assert out_mean.shape[1] == stft.nfft // 2 + 1

    def test_condensed_csipd():
        stft = transform

        # Test Learnable
        extractor = Condensed_CSIPD_Feature_Extractor(
            transform=stft, mode="ref", condense_method="conv", max_channels=6
        )

        # Batch with M=3
        B, M, T_samples = 2, 6, 480000
        x = torch.randn(B, M, T_samples)

        out = extractor(x)
        print(f"CSIPD Learnable Output shape: {out.shape}")
        assert out.shape[1] == 2 * (stft.nfft // 2 + 1)

        # Test Vector Mean
        extractor_mean = Condensed_CSIPD_Feature_Extractor(
            transform=stft, mode="ref", condense_method="vector_mean"
        )
        out_mean = extractor_mean(x)
        print(f"CSIPD Mean Output shape: {out_mean.shape}")
        assert out_mean.shape[1] == 2 * (stft.nfft // 2 + 1)

    test_condensed_ipd()
    test_condensed_csipd()
    print("All tests passed!")
