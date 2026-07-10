import torch
from typing import Union
from muse_toolbox.utils import STFTtransform, exp_windowing, to_one_hot
from .base_estimator import BaseSourceCountEstimator


class WGMSC_Threshold_Detector(BaseSourceCountEstimator):
    """
    Detects source activation events based on a wideband coherence measure
    using a simple thresholding and refractory period logic.
    """

    def __init__(
        self,
        input_dim: int,
        transform: STFTtransform,
        max_sources: int = 4,
        smoothing_time_constant: Union[float, None] = 1.0,  # [s]
        coherence_threshold: float = 0.5,
        min_activation_time_difference: Union[float, None] = 1.0,  # [s]
        smoothing_time_constant_rev: Union[float, None] = 0.5,  # [s]
        coherence_threshold_rev: float = 0.6,
        min_deactivation_time_difference: Union[float, None] = 0.5,
        detect_deactivations: bool = False,
    ):
        """
        Initializes the WGMSC_Threshold_Detector module.
        """
        super().__init__(
            input_dim=input_dim, transform=transform, max_sources=max_sources
        )

        self.smoothing_time_constant = smoothing_time_constant
        self.coherence_threshold = coherence_threshold
        if isinstance(min_activation_time_difference, float):
            assert (
                min_activation_time_difference > 0
            ), "Min activation time difference must be positive."
        self.min_activation_time_difference = min_activation_time_difference
        self.smoothing_time_constant_rev = smoothing_time_constant_rev
        self.coherence_threshold_rev = coherence_threshold_rev
        if isinstance(min_deactivation_time_difference, float):
            assert (
                min_deactivation_time_difference > 0
            ), "Min deactivation time difference must be positive."
        self.min_deactivation_time_difference = min_deactivation_time_difference
        self.detect_deactivations = detect_deactivations

    def get_config(self) -> dict:
        return {
            "max_sources": self.max_sources,
            "smoothing_time_constant": self.smoothing_time_constant,
            "coherence_threshold": self.coherence_threshold,
            "min_activation_time_difference": self.min_activation_time_difference,
            "smoothing_time_constant_rev": self.smoothing_time_constant_rev,
            "coherence_threshold_rev": self.coherence_threshold_rev,
            "min_deactivation_time_difference": self.min_deactivation_time_difference,
            "detect_deactivations": self.detect_deactivations,
        }

    def _verbose_parameters(self, indent: str = "") -> None:
        super()._verbose_parameters(indent)
        print(f"{indent}  Smoothing Time Constant: {self.smoothing_time_constant} s")
        print(f"{indent}  Coherence Threshold: {self.coherence_threshold}")
        print(
            f"{indent}  Min Activation Time Difference: {self.min_activation_time_difference} s"
        )
        print(
            f"{indent}  Smoothing Time Constant (Reverse): {self.smoothing_time_constant_rev} s"
        )
        print(
            f"{indent}  Coherence Threshold (Reverse): {self.coherence_threshold_rev}"
        )
        print(
            f"{indent}  Min Deactivation Time Difference: {self.min_deactivation_time_difference} s"
        )
        print(f"{indent}  Detect Deactivations: {self.detect_deactivations}")

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Detects source activations.
        Args:
            features (torch.Tensor): (B, C, T)
        Returns:
            torch.Tensor: (B, T, C_out)
        """
        wgmsc_wideband = features[:, 0:1, :]
        num_frames = wgmsc_wideband.shape[-1]
        batch_size = wgmsc_wideband.shape[0]

        # 1. Detect activations
        activations = self._wgmsc_wideband_thresholding(wgmsc_wideband)

        # 2. Detect deactivations
        if self.detect_deactivations and features.shape[1] > 1:
            wgmsc_wideband_rev = features[:, 1:2, :]
            deactivations = self._wgmsc_wideband_thresholding_rev(wgmsc_wideband_rev)
        else:
            deactivations = [
                torch.tensor([], device=wgmsc_wideband.device)
                for _ in range(batch_size)
            ]

        # 3. Convert to source activity
        logits = self._events_to_source_activity(
            activations, deactivations, num_frames, batch_size, wgmsc_wideband.device
        )
        return logits

    def _events_to_source_activity(
        self,
        activations: list[torch.Tensor],
        deactivations: list[torch.Tensor],
        num_frames: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts lists of activation/deactivation events into a one-hot encoded
        frame-wise source activity tensor. This logic is specific to this detector.
        """
        source_count_delta = torch.zeros(
            (batch_size, num_frames), device=device, dtype=torch.float32
        )

        for i in range(batch_size):
            if activations[i].numel() > 0:
                act_indices = (
                    self.transform.times2frames(activations[i])
                    .long()
                    .clamp(max=num_frames - 1)
                )
                updates = torch.ones_like(act_indices, dtype=torch.float32)
                source_count_delta[i].scatter_add_(0, act_indices, updates)

            if deactivations[i].numel() > 0:
                deact_indices = (
                    self.transform.times2frames(deactivations[i])
                    .long()
                    .clamp(max=num_frames - 1)
                )
                updates = -torch.ones_like(deact_indices, dtype=torch.float32)
                source_count_delta[i].scatter_add_(0, deact_indices, updates)

        source_count = torch.cumsum(source_count_delta, dim=1)
        source_count_clamped = source_count.clamp(min=0, max=self.max_sources).long()

        logits = to_one_hot(source_count_clamped, self.max_sources + 1)  # (B, T, C)

        return logits

    def _wgmsc_wideband_thresholding(
        self,
        wgmsc_wideband: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Detects source activations from wideband WGMSC by finding rising edges
        above a coherence threshold.
        """
        wgmsc_squeezed = wgmsc_wideband.squeeze()
        if wgmsc_squeezed.dim() == 1:
            wgmsc_squeezed = wgmsc_squeezed.unsqueeze(0)

        if self.smoothing_time_constant is not None:
            wgmsc_smoothed = exp_windowing(
                wgmsc_squeezed,
                self.transform.timeConstant2smoothingFactor(
                    self.smoothing_time_constant
                ),
                dim=-1,
            )
        else:
            wgmsc_smoothed = wgmsc_squeezed

        above_threshold = wgmsc_smoothed > self.coherence_threshold
        rising_edges = torch.cat(
            [
                torch.zeros_like(above_threshold[:, :1], dtype=torch.bool),
                above_threshold[:, 1:] & ~above_threshold[:, :-1],
            ],
            dim=1,
        )

        batch_activations = []
        for i in range(rising_edges.shape[0]):
            activations_indices = rising_edges[i].nonzero(as_tuple=True)[0]
            activation_times = self.transform.frames2times(activations_indices)

            if self.min_activation_time_difference is not None:
                activation_times = activation_times[
                    activation_times >= self.min_activation_time_difference
                ]
                if len(activation_times) > 1:
                    final_activation_times = activation_times[:1]
                    for time in activation_times[1:]:
                        if (
                            time - final_activation_times[-1]
                        ) >= self.min_activation_time_difference:
                            final_activation_times = torch.cat(
                                [final_activation_times, time.unsqueeze(dim=0)], dim=-1
                            )
                    batch_activations.append(final_activation_times)
                else:
                    batch_activations.append(activation_times)
            else:
                batch_activations.append(activation_times)

        return batch_activations

    def _wgmsc_wideband_thresholding_rev(
        self,
        wgmsc_wideband_rev: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Detects source activations from wideband WGMSC by finding rising edges
        above a coherence threshold.
        """
        wgmsc_squeezed_rev = wgmsc_wideband_rev.squeeze()
        if wgmsc_squeezed_rev.dim() == 1:
            wgmsc_squeezed_rev = wgmsc_squeezed_rev.unsqueeze(0)

        if self.smoothing_time_constant_rev is not None:
            wgmsc_smoothed_rev = exp_windowing(
                wgmsc_squeezed_rev,
                self.transform.timeConstant2smoothingFactor(
                    self.smoothing_time_constant_rev
                ),
                dim=-1,
            )
        else:
            wgmsc_smoothed_rev = wgmsc_squeezed_rev

        above_threshold = wgmsc_smoothed_rev > self.coherence_threshold_rev
        rising_edges = torch.cat(
            [
                torch.zeros_like(above_threshold[:, :1], dtype=torch.bool),
                above_threshold[:, 1:] & ~above_threshold[:, :-1],
            ],
            dim=1,
        )

        batch_deactivations = []
        for i in range(rising_edges.shape[0]):
            deactivations_indices = rising_edges[i].nonzero(as_tuple=True)[0]
            deactivation_times = self.transform.frames2times(deactivations_indices)

            if self.min_deactivation_time_difference is not None:
                deactivation_times = deactivation_times[
                    deactivation_times >= self.min_deactivation_time_difference
                ]
                if len(deactivation_times) > 1:
                    final_deactivation_times = deactivation_times[:1]
                    for time in deactivation_times[1:]:
                        if (
                            time - final_deactivation_times[-1]
                        ) >= self.min_deactivation_time_difference:
                            final_deactivation_times = torch.cat(
                                [final_deactivation_times, time.unsqueeze(dim=0)],
                                dim=-1,
                            )
                    batch_deactivations.append(final_deactivation_times)
                else:
                    batch_deactivations.append(deactivation_times)
            else:
                batch_deactivations.append(deactivation_times)

        return batch_deactivations
