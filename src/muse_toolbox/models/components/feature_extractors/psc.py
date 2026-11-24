import logging
from typing import Any

import torch

from muse_toolbox.models.components.feature_extractors.base_feature import (
    BaseFeatureExtractor,
)
from muse_toolbox.utils import STFTtransform, exp_windowing_recursive, get_real_dtype

log = logging.getLogger(__name__)


class PSC_Feature_Extractor(BaseFeatureExtractor):
    """Extracts Pairwise Spatial Coherence (PSC) features from a multi-channel STFT.

    Recursively smooths per-channel auto-powers and selected pair cross-powers
    in double precision, normalizes them into complex pairwise coherence
    coefficients ``Gamma_{p,t,f}``, and represents each coefficient in polar
    format ``[|Gamma|, cos(psi), sin(psi)]`` of dimension ``3 * F``.
    """

    def __init__(
        self,
        transform: STFTtransform,
        smoothing_time_constant: float = 0.1,
        mode: str = "all",
        ref_channel: int = 0,
        eps: float = 1e-12,
    ) -> None:
        """Initializes the Pairwise Spatial Coherence feature extractor.

        Args:
            transform (STFTtransform): The STFT transformation config.
            smoothing_time_constant (float, optional): Recursive exponential
                smoothing time constant in seconds. Setting ``0.0`` disables
                temporal smoothing (instantaneous coherence). Defaults to ``0.1``.
            mode (str, optional): Pair selection mode, either ``"ref"`` (all
                channels paired against ``ref_channel``, ``P = M - 1``) or
                ``"all"`` (upper-triangular pairs ``i < j``,
                ``P = M * (M - 1) // 2``). Defaults to ``"all"``.
            ref_channel (int, optional): Reference channel index when
                ``mode="ref"``. Defaults to ``0``.
            eps (float, optional): Numerical floor applied when normalizing
                powers and coherence magnitudes. Defaults to ``1e-12``.

        Raises:
            ValueError: If ``mode``, ``smoothing_time_constant``, or ``eps`` is
                invalid.
        """
        super().__init__(transform=transform)
        if mode not in ("ref", "all"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'ref' or 'all'.")
        if smoothing_time_constant < 0.0:
            raise ValueError(
                f"smoothing_time_constant must be non-negative, got {smoothing_time_constant}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.transform = transform
        self.smoothing_time_constant = smoothing_time_constant
        self.mode = mode
        self.ref_channel = ref_channel
        self.eps = eps

    def get_config(self) -> dict[str, Any]:
        """Returns the feature extractor configuration dictionary.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {
            "smoothing_time_constant": self.smoothing_time_constant,
            "mode": self.mode,
            "ref_channel": self.ref_channel,
            "eps": self.eps,
        }

    @property
    def is_trainable(self) -> bool:
        """Indicates that this feature extractor has no learnable parameters."""
        return False

    @property
    def signature(self) -> str:
        """Returns a unique identifier string for caching and logging."""
        sign = (
            f"PSC_Mdependent"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_stc{self.smoothing_time_constant}_{self.mode}"
        )
        if self.mode == "ref":
            sign += f"_ref{self.ref_channel}"
        return sign

    @property
    def feature_dim(self) -> int:
        """Returns the output feature dimension ``3 * F`` per microphone pair."""
        return 3 * self.transform.num_freq_bins

    def output_channels(self, num_mics: int | None = None) -> int:
        """Returns the number of microphone pairs ``P`` for an ``M``-microphone array."""
        if num_mics is None:
            raise ValueError(
                f"{self.__class__.__name__} produces an M-dependent number of pairs (mode='{self.mode}'); "
                "set `num_mics` when pairing with a fixed-width flattening estimator."
            )
        if num_mics < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires at least 2 microphones, got num_mics={num_mics}."
            )
        if self.mode == "ref":
            return int(num_mics) - 1
        return int(num_mics) * (int(num_mics) - 1) // 2

    def _get_pair_indices(
        self, num_channels: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds row and column index tensors for the configured microphone pairs.

        Args:
            num_channels (int): Number of microphone channels ``M``.
            device (torch.device): Target device for index tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 1D index tensors ``(row, col)`` of
                length ``P``.

        Raises:
            IndexError: If ``mode="ref"`` and ``ref_channel`` is out of bounds
                for non-empty multi-channel input.
        """
        if num_channels <= 1:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        if self.mode == "ref":
            if not (0 <= self.ref_channel < num_channels):
                raise IndexError(
                    f"ref_channel={self.ref_channel} out of range for {num_channels} channels."
                )
            row = torch.tensor(
                [i for i in range(num_channels) if i != self.ref_channel],
                dtype=torch.long,
                device=device,
            )
            col = torch.full_like(row, self.ref_channel)
            return row, col

        return torch.triu_indices(
            num_channels, num_channels, offset=1, device=device
        )

    def forward_stft(
        self, batch: torch.Tensor, valid_mics: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Computes polar PSC features from a multi-channel complex STFT tensor.

        Args:
            batch (torch.Tensor): Complex STFT tensor of shape ``(B, M, F, T)``.
            valid_mics (torch.Tensor | None, optional): Unused valid microphone
                counts preserved for interface compatibility. Defaults to ``None``.

        Returns:
            torch.Tensor: Real-valued PSC tensor of shape ``(B, P, 3*F, T)``,
                containing ``[|Gamma|, cos(psi), sin(psi)]`` stacked along the
                frequency axis.
        """
        batch_size, num_channels, num_freqs, num_frames = batch.shape
        real_dtype = get_real_dtype(batch)
        row, col = self._get_pair_indices(num_channels, batch.device)
        if row.numel() == 0:
            return torch.empty(
                batch_size,
                0,
                3 * num_freqs,
                num_frames,
                dtype=real_dtype,
                device=batch.device,
            )

        x = batch.to(dtype=torch.complex128)
        auto_power = x.real.pow(2) + x.imag.pow(2)
        cross_power = x[:, row] * x[:, col].conj()

        if self.smoothing_time_constant > 0.0:
            alpha = self.transform.timeConstant2smoothingFactor(
                self.smoothing_time_constant
            )
            psd = exp_windowing_recursive(
                auto_power, smoothing_factor=alpha, dim=-1
            )
            cross = exp_windowing_recursive(
                cross_power, smoothing_factor=alpha, dim=-1
            )
        else:
            psd = auto_power
            cross = cross_power

        denom = torch.sqrt((psd[:, row] * psd[:, col]).clamp(min=0.0))
        safe_denom = denom.clamp(min=self.eps)
        gamma = torch.where(
            denom > self.eps,
            cross / safe_denom,
            torch.zeros_like(cross),
        )

        # Normalize trig components by the unclamped raw magnitude so cos^2 + sin^2 == 1.
        raw_mag = torch.abs(gamma)
        safe_mag = raw_mag.clamp(min=self.eps)
        valid_mag = raw_mag > self.eps
        cos_psi = torch.where(
            valid_mag, gamma.real / safe_mag, torch.zeros_like(raw_mag)
        ).clamp(-1.0, 1.0)
        sin_psi = torch.where(
            valid_mag, gamma.imag / safe_mag, torch.zeros_like(raw_mag)
        ).clamp(-1.0, 1.0)
        mag = raw_mag.clamp(max=1.0)

        return torch.cat([mag, cos_psi, sin_psi], dim=2).to(dtype=real_dtype)

    def get_valid_feature_mask(
        self, valid_mics_count: torch.Tensor | None, max_M: int
    ) -> torch.Tensor | None:
        """Returns a boolean mask indicating valid microphone pairs per batch item.

        Args:
            valid_mics_count (torch.Tensor | None): Tensor of shape ``(B,)``
                giving valid microphone counts per batch item.
            max_M (int): Maximum number of microphones in the batch.

        Returns:
            torch.Tensor | None: Boolean mask of shape ``(B, P)`` where a pair
                is valid iff both constituent microphones are valid.
        """
        if valid_mics_count is None:
            return None

        batch_size = valid_mics_count.shape[0]
        device = valid_mics_count.device
        mic_mask = torch.arange(max_M, device=device).expand(
            batch_size, max_M
        ) < valid_mics_count.unsqueeze(1)

        row, col = self._get_pair_indices(max_M, device)
        if row.numel() == 0:
            return torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        return mic_mask[:, row] & mic_mask[:, col]

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs the feature extractor configuration parameters."""
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(
            f"{indent}  Smoothing Time Constant: {self.smoothing_time_constant} s"
        )
        log.info(f"{indent}  Mode: {self.mode}")
        if self.mode == "ref":
            log.info(f"{indent}  Reference Channel: {self.ref_channel}")
        log.info(f"{indent}  Epsilon: {self.eps}")
