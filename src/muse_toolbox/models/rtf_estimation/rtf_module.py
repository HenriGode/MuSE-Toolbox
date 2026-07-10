import torch
import math
import matplotlib
from muse_toolbox.models.building_blocks.rtf_estimators import BaseRTFestimator, Oracle
from muse_toolbox.models.common.base_model import BaseLitModel
from typing import Optional
from muse_toolbox.utils import (
    HeterogeneousBatch,
    STFTtransform,
    weighted_SCM,
    smoothCovarianceMatrix,
    hermitian_angle,
    Beamformer,
    growing_average_SCM,
    Segment,
    identify_segments,
    covariance_SCM,
    exp_windowing_recursive_changing_factor,
    db2amp,
)
from dataclasses import dataclass

matplotlib.use("agg")

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class RTFmodule(BaseLitModel):

    def __init__(
        self,
        transform: STFTtransform,
        smoothing_time_constant: float,  # [s]
        segment_forgetting_factor: float,  # in the intervall [0,1]
        fix_prev_rel_time: float,  # ['last', 'half', '2/3']
        noisy_cov_init_time: float,  # [s]
        registry_HA_threshold: float,  # [rad]
        rtf_estimator: BaseRTFestimator,
        interferer_gain: float,  # [dB]
        bf_type: str,
        refchannels: list[int],
        max_sources: Optional[int] = None,
        source_activity_method: Optional[torch.nn.Module] = None,  # None = "oracle"
        batch_size: int = 1,
        loss_config: dict = {"CrossEntropy": None},
        optimizer_config: Optional[dict] = None,
        lr_scheduler_config: Optional[dict] = None,
        metrics_train: Optional[dict] = None,
        metrics_val: Optional[dict] = None,
        metrics_test: Optional[dict] = None,
        compute_complexity_metrics: bool = False,
        check_causality: bool = False,
    ):
        super().__init__(
            model_name=f"RTF_Estimator_{rtf_estimator.__class__.__name__}",
            batch_size=batch_size,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            metrics_test=metrics_test,
            compute_complexity_metrics=compute_complexity_metrics,
            check_causality=check_causality,
            transform=transform,
            sad_model_name=(
                source_activity_method.__class__.__name__
                if source_activity_method is not None
                else "oracle"
            ),
        )

        ## Processing parameters
        self.source_activity_method = source_activity_method

        self.processor = RTFScenarioProcessor(
            rtf_estimator=rtf_estimator,
            transform=transform,
            smoothing_time_constant=smoothing_time_constant,
            segment_forgetting_factor=segment_forgetting_factor,
            fix_prev_rel_time=fix_prev_rel_time,
            noisy_cov_init_time=noisy_cov_init_time,  # [s]
            registry_HA_threshold=registry_HA_threshold,  # [rad]
            max_sources=max_sources,
            interferer_gain=interferer_gain,  # [dB]
            bf_type=bf_type,
            ref_channels=refchannels,
        )

        self.num_params = self.count_parameters()
        self.save_hyperparameters(ignore=["source_activity_method", "rtf_estimator"])

    def get_resampled_source_count(
        self, batch: HeterogeneousBatch
    ) -> list[torch.Tensor]:
        """
        Resample source count sequence from one STFT hop size to another, using nearest neighbor interpolation.

        Args:
            source_count: [T_in] Tensor of source counts at input hop size

        Returns:
            source_count_out: [T_out] Tensor of source counts at output hop size
        """
        source_count = []
        for i in range(batch.batch_size):
            if (
                batch.meta["scenario_params"][i]["transform"].signature
                != self.transform.signature
            ):
                source_count.append(
                    self.transform.samples2frames_quantity(
                        (
                            batch.meta["scenario_params"][i][
                                "transform"
                            ].frames2samples_quantity(
                                batch.meta["source_count"][i].float()
                            )
                        )
                    ).int()
                )
            else:
                source_count.append(batch.meta["source_count"][i])

        return source_count

    def forward_(self, batch: HeterogeneousBatch) -> HeterogeneousBatch:
        # 1. Get Source Activity (Oracle or Estimated)
        # Shape: [Batch, Time] (counts) ### Future-TODO: also allow [Batch, Time, Sources] (activity flags)
        source_activity = (
            self.source_activity_method(batch)
            if self.source_activity_method is not None
            else self.get_resampled_source_count(batch)
        )

        results = []

        # 2. Level 1 Loop
        # Iterate over each scenario in the batch
        for b_idx in range(batch.batch_size):
            # Extract single scenario data (handling padding if necessary)
            # You might need lengths from batch.meta['seq_len'] to slice correctly

            scenario_kwargs = {}
            # Pass oracle metadata if available
            if isinstance(self.processor.rtf_estimator, Oracle):
                scenario_kwargs["sad_frames"] = batch.meta["sad_frames"][b_idx]
                scenario_kwargs["oracle_rtfs"] = batch.meta["rtfs"][b_idx]

            # Delegate to Level 2
            scenario_result = self.processor.process_scenario(
                stft=batch.stft_audio[b_idx],  # [F, M, T]
                source_activity=source_activity[b_idx],  # [T]
                **scenario_kwargs,
            )
            results.append(scenario_result)

        batch.estimates = results  # List of list of tensors

        # batch.print_summary()

        return batch

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        processed_batch = self(batch)
        self._metric_step(processed_batch, dataloader_idx, "test")

    def _metric_step(
        self, processed_batch: HeterogeneousBatch, dataloader_idx, step_type
    ):
        meta_dict = processed_batch.meta.copy()
        meta_dict["dataloader_idx"] = self.batch_size * [dataloader_idx]
        targets = (meta_dict["rtfs"], meta_dict["references"])
        self.metric_collections[step_type].update(
            processed_batch.estimates, targets, meta_dict, dataloader_idx
        )

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> HeterogeneousBatch:
        return self(batch)


@dataclass
class RTFState:
    noise_cov: torch.Tensor
    noise_cov_est_frames: int
    noisy_cov: torch.Tensor
    prev_noisy_cov: torch.Tensor
    prev_act_rtfs: torch.Tensor
    prev_act_ids: torch.Tensor


@dataclass
class SourceRegistry:
    # Stores global source identities
    # registry_rtfs: [F, M, N_reg]

    def __init__(self, Kmax, F, M, device, dtype):
        self.registry_rtfs = torch.zeros((F, M, 0), device=device, dtype=dtype)
        self.registry_counts = torch.zeros((0,), device=device, dtype=torch.long)
        self.global_ids = torch.empty((0,), device=device, dtype=torch.long)
        self.next_global_id = 0
        self.max_sources = Kmax

    def match_framewise(
        self, rtf_seq: torch.Tensor, threshold: float, active_known_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Matches a sequence of RTFs against the registry using a growing average distance.

        Args:
            rtf_seq: [F, T, M, 1] Candidate RTF sequence
            threshold: Distance threshold for matching

        Returns:
            matched_ids: [T] Global IDs for each frame (-1 if new)
        """
        F, T, M, _ = rtf_seq.shape
        device = rtf_seq.device
        K_reg = self.registry_rtfs.shape[-1]

        canidate_mask = ~torch.isin(self.global_ids, active_known_ids)
        canidate_ids = self.global_ids[canidate_mask]
        canidate_rtfs = self.registry_rtfs[..., canidate_mask]
        K_can = canidate_rtfs.shape[-1]

        if K_can == 0:
            return torch.full((T,), -1, dtype=torch.long, device=device)
        if K_can == 1 and K_reg == self.max_sources:
            # If only one candidate and at max capacity, always match to it
            return canidate_ids.expand(T)

        # 1. Prepare for broadcasting and compute Hermitian angles
        # rtf_seq: [F, T, M, 1] -> [F, T, 1, M] (after squeezing last dim and unsqueezing for N)
        # registry: [F, M, N] -> [F, 1, N, M] (permute and unsqueeze for T)

        # Candidates A: [F, T, 1, M]
        A = rtf_seq.squeeze(-1).unsqueeze(-2)

        # Registry B: [F, 1, N, M]
        B = canidate_rtfs.permute(0, 2, 1).unsqueeze(1)

        # Compute Angles: [F, T, N]
        # hermitian_angle computes angle between complex vectors in last dimension
        angles = hermitian_angle(A, B, dim=-1).squeeze(-1)

        # 2. Average over frequency -> [T, N]
        dist_inst = angles.mean(dim=0)

        # 3. Growing Average over Time -> [T, N]
        # "growing average logic should happen after hermitian angles are computed"
        cum_dist = torch.cumsum(dist_inst, dim=0)
        counts = torch.arange(1, T + 1, device=device).unsqueeze(-1)
        avg_dist = cum_dist / counts

        # 4. Decision
        min_vals, min_idxs = avg_dist.min(dim=-1)  # [T]

        # Map index to global ID if under threshold
        if K_reg == self.max_sources:
            # If at max capacity, never register new sources
            threshold = 3 * PI  # Large value to prevent new registrations

        matched_ids = torch.where(
            min_vals < threshold,
            canidate_ids[min_idxs],
            torch.tensor(-1, device=device),
        )

        return matched_ids

    def register_new(self, rtf: torch.Tensor):
        """
        Register a new source.
        rtf: [F, M, 1] or [F, M]
        """
        assert rtf.ndim == 3, "RTF must be a 3D tensor."

        # Add to registry tensor
        self.registry_rtfs = torch.cat([self.registry_rtfs, rtf], dim=-1)

        # Add count (initialized to 1 update)
        self.registry_counts = torch.cat(
            [
                self.registry_counts,
                torch.tensor([1], device=rtf.device, dtype=torch.long),
            ]
        )

        # Assign new Global ID
        new_id = self.next_global_id
        self.global_ids = torch.cat(
            [self.global_ids, torch.tensor([new_id], device=rtf.device)]
        )
        self.next_global_id += 1

        return new_id

    def update_entry(self, global_id: int, rtf: torch.Tensor):
        """
        Update an existing source's average RTF.
        """
        # smooth update? Or just replace?
        # Usually for registry we might want a running average.
        # Let's do a simple running average based on counts.

        idx = (self.global_ids == global_id).nonzero().squeeze()
        assert idx.ndim == 0, "Global ID must be unique."
        self.registry_rtfs[..., idx] = rtf.squeeze(-1)
        self.registry_counts[idx] += 1


class RTFScenarioProcessor:
    def __init__(
        self,
        rtf_estimator: BaseRTFestimator,
        transform: STFTtransform,
        smoothing_time_constant,
        segment_forgetting_factor,
        noisy_cov_init_time,
        fix_prev_rel_time,
        bf_type,
        seg_cov_win=lambda x: torch.hann_window(
            x.shape[-1] + 1, device=x.device
        ).sqrt()[1:],
        registry_HA_threshold: float = 0.25,
        max_sources: Optional[int] = None,
        interferer_gain: float = 0.0,  # [dB]
        ref_channels: list[int] = [0],
    ):
        self.rtf_estimator = rtf_estimator
        self.transform = transform
        self.smoothing_time_constant = smoothing_time_constant
        self.forgetting_factor = self.transform.timeConstant2smoothingFactor(
            self.smoothing_time_constant
        )
        self.segment_forgetting_factor = segment_forgetting_factor
        self.fix_prev_rel_time = fix_prev_rel_time
        self.noisy_cov_init_time = noisy_cov_init_time
        self.seg_cov_win = seg_cov_win
        self.registry_HA_threshold = registry_HA_threshold
        self.max_sources = max_sources
        self.interferer_gain = interferer_gain
        self.bf_types = [bf_type] if isinstance(bf_type, str) else bf_type
        self.ref_channels = ref_channels

    def process_scenario(
        self, stft: torch.Tensor, source_activity: torch.Tensor, **kwargs
    ) -> tuple[
        dict[str, torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """
        stft: [F, M, T]
        source_activity: [T] (counts)
        """

        # 1. Identify Segments
        # Breaks time axis into chunks where activity is constant
        segments = identify_segments(source_activity)

        # 2. Initialize State
        F, M, T = stft.shape
        # noise_cov: Initialized to Identity * small epsilon
        state = RTFState(
            noise_cov=torch.zeros(F, M, M, dtype=stft.dtype, device=stft.device),
            noise_cov_est_frames=0,
            noisy_cov=torch.zeros(F, 0, M, M, dtype=stft.dtype, device=stft.device),
            prev_noisy_cov=torch.zeros(F, M, M, dtype=stft.dtype, device=stft.device),
            prev_act_rtfs=torch.zeros(F, M, 0, dtype=stft.dtype, device=stft.device),
            prev_act_ids=torch.zeros(0, dtype=torch.long, device=stft.device),
        )

        # Initialize Source Registry for this scenario
        registry = SourceRegistry(self.max_sources, F, M, stft.device, stft.dtype)

        rtf_estimates = []  # List of [F, T_seg, M, K] tensors
        id_estimates = []  # List of [T_seg, K] tensors

        # noisy_cov_mat_all_segments = self._precompute_noisy_cov_mat(stft, segments)
        # noisy_cov_mat_all_segments2 = self._precompute_noisy_cov_mat2(stft, segments)

        Rn4bf = None

        # fix_prev_idx = 0
        # 3. Level 2 Loop (Over Segments)
        for seg_idx, seg in enumerate(segments):
            assert (
                seg.num_sources == 0 if seg_idx == 0 else True
            ), "First segment must have 0 sources."

            # if len(id_estimates) == 8:
            #     a = 1  # Debugging breakpoint

            # Extract STFT for this segment
            # seg_stft: [F, M, T_seg]
            seg_stft = stft[:, :, seg.start : seg.end]
            T_seg = seg.end - seg.start

            # Determine fixed previous noisy covariance index
            # fix_prev_idx = int(self.fix_prev_rel_time * T_seg) - 1

            # Update noisy covariance matrix (uses growing/smooth average logic)
            state = self._update_noisy_cov(state, seg_stft)

            # A. Update Noise Covariance and reset Noisy Covariance (if 0 sources)
            if seg.num_sources == 0:
                assert seg.event_type in [
                    "init",
                    "deactivation",
                ], "0-source segment must be init or deactivation."
                state = self._update_noise_cov_and_reset(state, seg_stft)
                rtf_estimates.append(
                    torch.zeros(F, T_seg, M, 0, dtype=stft.dtype, device=stft.device)
                )
                # IDs: [T_seg, 0]
                id_estimates.append(
                    torch.zeros(T_seg, 0, dtype=torch.long, device=stft.device)
                )

                if seg_idx == 0:
                    Rn4bf = state.noise_cov.clone().unsqueeze(-3)

            # B. Handle Activation and Process Frames
            elif seg.event_type == "activation":
                assert seg_idx > 0, "First segment cannot be activation."

                estimator_kwargs = {}
                if isinstance(self.rtf_estimator, Oracle):
                    estimator_kwargs.update(kwargs)
                    estimator_kwargs["segment"] = seg

                # 1. Estimate New RTF Sequence
                new_rtf_seq = self.rtf_estimator(
                    noise_cov=state.noise_cov.unsqueeze(-3),
                    noisy_cov=state.noisy_cov,
                    old_rtfs=state.prev_act_rtfs.unsqueeze(-3),
                    old_noisy_cov=state.prev_noisy_cov.unsqueeze(-3),
                    **estimator_kwargs,
                )  # [F, T_seg, M, 1]

                # 2. Combine with Old RTFs
                old_rtf_seq = state.prev_act_rtfs.unsqueeze(-3).repeat(
                    1, T_seg, 1, 1
                )  # [F, T_seg, M, K_old]
                all_rtf_seq = torch.cat(
                    [old_rtf_seq, new_rtf_seq], dim=-1
                )  # [F, T_seg, M, K_old + 1]
                rtf_estimates.append(all_rtf_seq)

                # 3. Match Framing against Registry
                # matched_ids: [T_seg]
                matched_ids_seq = registry.match_framewise(
                    new_rtf_seq, self.registry_HA_threshold, state.prev_act_ids
                )

                # 4. Update Registry at END of segment (to refine the stored model)
                # Use the "fixed" estimate for validation
                batch_noisy_cov = self._segment_batch_cov(seg_stft)
                fixed_new_rtf = self.rtf_estimator(
                    noise_cov=state.noise_cov.unsqueeze(-3),
                    noisy_cov=batch_noisy_cov.unsqueeze(-3),
                    old_rtfs=state.prev_act_rtfs.unsqueeze(-3),
                    old_noisy_cov=state.prev_noisy_cov.unsqueeze(-3),
                    **estimator_kwargs,
                )[
                    ..., 0, :, :
                ]  # [F, M, 1]

                # Decide final ID based on last frame (most evidence)
                final_decision_id = registry.match_framewise(
                    fixed_new_rtf.unsqueeze(-3),
                    self.registry_HA_threshold,
                    state.prev_act_ids,
                )

                if final_decision_id.item() == -1:
                    # It finalized as NEW.

                    # Incorporate into registry.
                    final_decision_id = torch.tensor(
                        [registry.register_new(fixed_new_rtf)],
                        device=final_decision_id.device,
                    )
                    # Update match_ids_seq to reflect this final decision (all -1 -> new ID)
                    matched_ids_seq = torch.where(
                        matched_ids_seq == -1,
                        final_decision_id,
                        matched_ids_seq,
                    )
                else:
                    # It matched an existing logic. Update that entry.
                    registry.update_entry(int(final_decision_id.item()), fixed_new_rtf)

                # 5. Combine with Old IDs
                # state.prev_act_ids contains IDs of [Source 0, Source 1 ... Source K_old-1]
                # We append the new one.

                if state.prev_act_ids.numel() != 0:
                    # Repeat for time [T_seg, K_old]
                    old_ids_seq = state.prev_act_ids.expand(T_seg, -1)
                    # Combine: [T_seg, K_old + 1]
                    current_ids = torch.cat(
                        [old_ids_seq, matched_ids_seq.unsqueeze(-1)], dim=-1
                    )
                else:
                    current_ids = matched_ids_seq.unsqueeze(-1)
                id_estimates.append(current_ids)

                # Update State
                state.prev_act_rtfs = torch.cat(
                    [state.prev_act_rtfs, fixed_new_rtf], dim=-1
                )  # [F, M, K_new]
                state.prev_noisy_cov = batch_noisy_cov.clone()
                state.prev_act_ids = torch.cat(
                    [
                        state.prev_act_ids,
                        final_decision_id,
                    ]
                )

            # C. Handle Deactivation and Process Frames
            elif seg.event_type == "deactivation":
                assert seg_idx > 1, "First and second segment cannot be deactivation."

                deactivation_kwargs = {}
                if isinstance(self.rtf_estimator, Oracle):
                    deactivation_kwargs.update(kwargs)
                    deactivation_kwargs["segment"] = seg

                # Identify deactivated source (local index) [T_seg]
                indices_of_removed_rtf = self._estimate_deactivated_source(
                    segment_stft=seg_stft,
                    noise_cov=state.noise_cov,
                    noisy_cov=state.noisy_cov,
                    old_rtfs=state.prev_act_rtfs,
                    old_noisy_cov=state.prev_noisy_cov,
                    **deactivation_kwargs,
                )

                # RTFs
                old_rtf_seq = state.prev_act_rtfs.unsqueeze(-3).repeat(
                    1, T_seg, 1, 1
                )  # [F, T_seg, M, K_old]
                remaining_rtf_seq = self._removing_deactivated_rtfs(
                    old_rtf_seq, indices_of_removed_rtf
                )
                rtf_estimates.append(remaining_rtf_seq)

                # IDs

                # Expand to [T_seg, K_old]
                current_ids_expanded = state.prev_act_ids.unsqueeze(0).expand(T_seg, -1)

                # Remove the one specified by indices_of_removed_rtf[t]
                # Helper function similar to _removing_deactivated_rtfs but for 1D ID list
                remaining_ids_seq = self._remove_ids_framewise(
                    current_ids_expanded, indices_of_removed_rtf
                )
                id_estimates.append(remaining_ids_seq)

                # Update State logic (End of Segment)
                batch_noisy_cov = self._segment_batch_cov(seg_stft)
                state.prev_noisy_cov = batch_noisy_cov.clone()

                # Determine final removal for state update
                final_removed_idx = indices_of_removed_rtf[-1].item()

                # Remove from Active IDs list
                if 0 <= final_removed_idx < len(state.prev_act_ids):
                    state.prev_act_ids = torch.cat(
                        [
                            state.prev_act_ids[:final_removed_idx],
                            state.prev_act_ids[final_removed_idx + 1 :],
                        ],
                        dim=-1,
                    )

                # Remove from RTFs
                if state.prev_act_rtfs.shape[-1] > 1:
                    state.prev_act_rtfs = torch.cat(
                        [
                            state.prev_act_rtfs[..., :final_removed_idx],
                            state.prev_act_rtfs[..., final_removed_idx + 1 :],
                        ],
                        dim=-1,
                    )
                else:
                    state.prev_act_rtfs = state.prev_act_rtfs[..., :0]

            if -1 in id_estimates[-1]:
                a = 1  # Debugging breakpoint

        target = {}
        target_bf = {}
        for bft in self.bf_types:

            if bft in ["LCMV", "MVDR"]:
                assert (
                    Rn4bf is not None
                ), "Noise covariance for beamforming not initialized."
                R4bf = Rn4bf
            elif bft in ["LCMP", "MPDR"]:
                R4bf = smoothCovarianceMatrix(
                    stft_signal=stft, smoothing_factor=self.forgetting_factor
                )
            elif bft in ["LCMP2", "MPDR2"]:
                # Ry_reinit_4bf = torch.cat(
                #     self._precompute_noisy_cov_mat(stft, segments), dim=-3
                # )
                R4bf = self._precompute_noisy_cov_mat2(stft, segments)
            else:
                raise ValueError(f"Unknown bf_type: {bft}")

            if bft in ["MVDR", "MPDR", "MPDR2"]:
                rtfs4bf = [rtfs[..., -1:] for rtfs in rtf_estimates]
            elif bft in ["LCMV", "LCMP", "LCMP2"]:
                rtfs4bf = rtf_estimates
            else:
                raise ValueError(f"Unknown bf_type: {bft}")

            target[bft], target_bf[bft] = self._extract_target(rtfs4bf, stft, R4bf)
            target[bft] = target[bft][:, self.ref_channels]
            target_bf[bft] = target_bf[bft][..., self.ref_channels]

        return (
            target,  # [F, 1, T]
            rtf_estimates,
            id_estimates,
            target_bf,  # [F, T, M, 1]
        )

    def _extract_target(
        self, rtf_estimates: list[torch.Tensor], stft: torch.Tensor, R4bf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract target RTF estimates from the list based on some criteria.
        The criterium is that we use the latest activated source in each segment as target.
        We do this by simply enhancing the last rtf in each segment and collecting the outputs.

            Args:
                rtf_estimates: List of [F, T_seg, M, K] tensors for each segment
                stft: [F, M, T] STFT of the entire signal (for beamforming)
                R4bf: [F, 1, M, M] Noise covariance matrix for beamforming

            Returns:
                target_estimates: [F, M, T] tensor
        """
        target_estimates = []
        target_bfs = []
        current_frame = 0
        for rtf_seg in rtf_estimates:
            F, T_seg, M, K = rtf_seg.shape
            if K == 0:
                # No sources in this segment
                target_estimates.append(
                    torch.zeros(F, M, T_seg, dtype=stft.dtype, device=stft.device)
                )
                target_bfs.append(
                    torch.zeros(F, T_seg, M, M, dtype=stft.dtype, device=stft.device)
                )
            else:
                # Beamform to get enhanced signal
                gains = torch.ones(K, 1, device=stft.device, dtype=stft.dtype) * db2amp(
                    torch.tensor(self.interferer_gain)
                )
                gains[-1] = 1  # Actual target source gain, last activated source
                framewise_stft = stft[
                    :, :, current_frame : current_frame + T_seg
                ].transpose(-1, -2)[
                    ..., None
                ]  # [F, T_seg, M, 1]

                if R4bf.shape[-3] == 1:
                    R4bf_segment = R4bf
                else:
                    R4bf_segment = R4bf[:, current_frame : current_frame + T_seg, :, :]

                w = Beamformer(
                    covMat2min=R4bf_segment,
                    RTFs4constraints=rtf_seg,
                    gains=gains,
                )
                out = (w.mH @ framewise_stft)[..., 0].transpose(-1, -2)  # [F, M, T_seg]
                target_bfs.append(w)
                target_estimates.append(out)
            current_frame += T_seg

        return (
            torch.cat(target_estimates, dim=-1),  # [F, M, T]
            torch.cat(target_bfs, dim=-3),  # [F, T, M, M]
        )

    def _precompute_noisy_cov_mat(self, stft: torch.Tensor, segments: list[Segment]):
        F, M, T = stft.shape
        noisy_cov_mat_all_segments = []
        seg_noisy_cov = None
        for seg in segments:
            seg_stft = stft[:, :, seg.start : seg.end]
            seg_noisy_cov = smoothCovarianceMatrix(
                stft_signal=seg_stft,
                smoothing_factor=self.forgetting_factor,
                init_cov=(
                    seg_noisy_cov[..., -1:, :, :].clone()
                    if seg_noisy_cov is not None
                    else None
                ),
                init_smoothing_factor=self.segment_forgetting_factor,
            )  # [F, M, M]
            noisy_cov_mat_all_segments.append(seg_noisy_cov)
        return noisy_cov_mat_all_segments  # [S][F, Tseg, M, M]

    def _precompute_noisy_cov_mat2(self, stft: torch.Tensor, segments: list[Segment]):
        F, M, T = stft.shape
        # [F, T, M, M]
        instant_noisy_cov = covariance_SCM(stft.transpose(-2, -1)[..., None])
        smoothing_factor = self.forgetting_factor * torch.ones(T, device=stft.device)
        for seg in segments:
            smoothing_factor[seg.start] = self.segment_forgetting_factor
        smoothing_factor[0] = 0.0  # No previous data for first frame
        smoothed_noisy_cov = exp_windowing_recursive_changing_factor(
            instant_noisy_cov,
            smoothing_factor,
            dim=-3,
        )  # [F, T, M, M]
        return smoothed_noisy_cov

    def _remove_ids_framewise(self, current_ids, remove_indices):
        """
        current_ids: [T, K]
        remove_indices: [T]
        Returns: [T, K-1]
        """
        T, K = current_ids.shape
        if K == 1:
            return torch.zeros(T, 0, device=current_ids.device, dtype=current_ids.dtype)
        elif K == 0:
            raise ValueError("No IDs to remove from.")

        # Create mask
        # indices 0..K-1
        k_idx = torch.arange(K, device=current_ids.device).unsqueeze(0)  # [1, K]
        rem_idx = remove_indices.unsqueeze(1)  # [T, 1]
        mask = k_idx != rem_idx  # [T, K]

        # Select
        flat_rem = torch.masked_select(current_ids, mask)
        return flat_rem.view(T, K - 1)

    def _update_noisy_cov(
        self, state: RTFState, segment_stft  # , fix_prev_idx
    ) -> RTFState:
        # assert state.noisy_cov.shape[-3] > fix_prev_idx, "fix_prev_idx out of bounds."
        # state.prev_noisy_cov = state.noisy_cov[..., fix_prev_idx, :, :].clone()
        state.noisy_cov = smoothCovarianceMatrix(
            stft_signal=segment_stft,
            smoothing_factor=self.forgetting_factor,
            init_cov=(
                state.noisy_cov[..., -1:, :, :].clone()
                if state.noisy_cov.shape[-3] > 0
                else None
            ),
            init_smoothing_factor=self.segment_forgetting_factor,
        )
        return state

    def _segment_batch_cov(self, seg_stft: torch.Tensor) -> torch.Tensor:
        # segment_stft: [F, M, T_seg]
        # Returns: [F, M, M]
        return weighted_SCM(
            data=seg_stft,
            weights=self.seg_cov_win(seg_stft),  # Full weightes average over segment
        )

    def _update_noise_cov_and_reset(self, state: RTFState, segment_stft) -> RTFState:
        old_noise_cov = state.noise_cov
        T_old = state.noise_cov_est_frames
        T_seg = segment_stft.shape[2]
        T_total = T_old + T_seg

        # Update the noise covariance matrix
        new_noise_cov = self._segment_batch_cov(segment_stft)
        state.noise_cov = (old_noise_cov * T_old + new_noise_cov * T_seg) / T_total
        state.noise_cov_est_frames = T_total

        # Reset previous noisy covariance (Rv) matrix with noise covariance
        state.prev_noisy_cov = new_noise_cov.clone()

        state.prev_act_rtfs = torch.zeros(
            state.prev_act_rtfs.shape[0],
            state.prev_act_rtfs.shape[1],
            0,
            dtype=state.prev_act_rtfs.dtype,
            device=state.prev_act_rtfs.device,
        )

        state.prev_act_ids = torch.zeros(
            0, dtype=torch.long, device=state.prev_act_ids.device
        )

        return state

    def _pool_estimates(self, rtf_seq: torch.Tensor, index: int) -> torch.Tensor:
        # rtf_seq: [F, T, M, 1]
        # Simple mean pooling over time
        return rtf_seq[..., index, :, :]  # [F, M, 1]

    def _estimate_deactivated_source(
        self, segment_stft, noise_cov, noisy_cov, old_rtfs, old_noisy_cov, **kwargs
    ):
        # Check oracle mode
        if isinstance(self.rtf_estimator, Oracle):
            return self._estimate_deactivated_source_oracle(old_rtfs, **kwargs)

        K_old = old_rtfs.shape[-1]

        gains = torch.eye(K_old, device=old_rtfs.device, dtype=old_rtfs.dtype)[
            None, None, None, ...
        ].transpose(0, -1)
        # Beamform to get estimates for all old sources
        # segment_stft: [F, M, T_seg]
        # noisy_cov: [F, T_seg, M, M]
        # old_rtfs: [F, M, K_old]
        # gains: [K_old, 1, 1, K_old, 1]
        framewise_stft = segment_stft.transpose(-1, -2)[..., None]  # [F, T_seg, M, 1]

        gainsLCMX = torch.eye(K_old, device=old_rtfs.device, dtype=old_rtfs.dtype)[
            None, None, None, ...
        ].transpose(0, -1)
        gainsMXDR = torch.tensor([[1.0]], device=old_rtfs.device, dtype=old_rtfs.dtype)
        rtfsLCMX = old_rtfs.unsqueeze(-3)
        rtfsMXDR = old_rtfs.unsqueeze(-3).unsqueeze(0).transpose(0, -1)

        Ry_grow = growing_average_SCM(segment_stft)
        # Ry_grow[..., :6, :, :] = regularize(Ry_grow[..., :6, :, :], reg_factor=1e-1)
        c = 0
        meas1 = None
        for covMat2min in [noisy_cov]:  # [noise_cov.unsqueeze(-3), Ry_grow, noisy_cov]:
            c += 1
            for gains, rtfs_constraint, b in [
                # (gainsMXDR, rtfsMXDR, "M"),
                (gainsLCMX, rtfsLCMX, "L"),
            ]:
                out = Beamformer(covMat2min, rtfs_constraint, gains, framewise_stft)[
                    ..., 0
                ].transpose(-1, -2)
                meas0 = torch.sum(abs(out**2), dim=(-2, -3))
                meas1 = torch.cumsum(meas0, dim=-1) / (
                    torch.arange(meas0.shape[-1], device=meas0.device) + 1
                )

                if False:
                    import matplotlib.pyplot as plt

                    plt.figure()
                    plt.plot(meas0.mT.cpu().numpy())
                    plt.savefig(f"Playground/plot0_{c}_{b}.png")
                    plt.close()
                    plt.figure()
                    plt.plot(meas1.mT.cpu().numpy())
                    plt.savefig(f"Playground/plot1_{c}_{b}.png")
                    plt.close()

        if meas1 is not None:
            return meas1.argmin(dim=0)
        else:
            raise RuntimeError("No measurement computed for deactivation estimation.")

    def _estimate_deactivated_source_oracle(
        self,
        old_rtfs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Oracle implementation for deactivation.
        Identifies which of the old_rtfs corresponds to the source that became inactive
        by comparing against the ground-truth active sources for the current segment.
        """
        if "segment" in kwargs and "sad_frames" in kwargs and "oracle_rtfs" in kwargs:
            segment: Segment = kwargs["segment"]
            sad_frames: dict[str, torch.Tensor] = kwargs["sad_frames"]
            oracle_rtfs: dict[str, torch.Tensor] = kwargs["oracle_rtfs"]
        else:
            raise ValueError(
                "Missing required keyword arguments 'segment', 'sad_frames', and 'oracle_rtfs' for Oracle."
            )

        # old_rtfs: [F, M, K_old]
        F, M, K_old = old_rtfs.shape
        device = old_rtfs.device
        T_seg = segment.end - segment.start

        # --- Edge Case 1: Trivial Removal ---
        # If there is only 1 (or 0) old source, result is index 0.
        if K_old == 1:
            return torch.zeros(T_seg, dtype=torch.long, device=device)
        elif K_old == 0:
            raise ValueError("No old RTFs to remove from in Oracle deactivation.")

        # --- Identify Oracle Active Sources (The "Stayers") ---

        # 1. Compute Activity Counts & Filter Zeros
        # Extract counts for current segment, explicitly filtering out zero-activity sources
        active_sources = []
        for k, sf in sad_frames.items():
            if k == "noise":
                continue

            # Sum activity over the segment
            count = sf[segment.start : segment.end].sum()

            # CRITICAL FIX: Only consider sources with actual activity > 0
            if count > 0:
                active_sources.append((k, count))

        if not active_sources:
            # Oracle says everything is silent -> any old source is valid to remove.
            # Usually implies K_active=0, so removing index 0 is a safe fallback for "clearing".
            return torch.zeros(T_seg, dtype=torch.long, device=device)

        # 2. Unzip safely
        source_ids = [k for k, v in active_sources]
        counts = torch.stack([v for k, v in active_sources])

        # 3. Determine how many sources are truly active now
        # We rely on the segment info (which dictates the "new" count S_new)
        # We assume the top S_new most active sources are the ones staying.
        k_active = min(segment.num_sources, len(source_ids))

        if k_active == 0:
            # If oracle active set is empty, return 0 (removing anyone is correct)
            return torch.zeros(T_seg, dtype=torch.long, device=device)

        # 4. Get active Source IDs
        _, topk_indices = torch.topk(counts, k_active)
        top_k_source_ids = [source_ids[i] for i in topk_indices.tolist()]

        # 5. Extract "Stayed" Candidates
        # Stack candidates: [F, 1, M, k_active] (assuming oracle_rtfs entries are [F, 1, M, 1])
        # Note: logic matches oracle.py concat dim=-1
        active_rtfs = torch.cat([oracle_rtfs[sid] for sid in top_k_source_ids], dim=-1)

        # --- Find the Outlier ---
        # We compare 'old_rtfs' (Candidates for removal) vs 'active_rtfs' (Anchors that stayed).
        # The 'old_rtf' that is furthest from *all* anchors is the one that left.

        # 1. Prepare for broadcasting
        # Old RTFs (The Query): [F, M, K_old] -> [F, K_old, 1, M]
        r = old_rtfs.permute(0, 2, 1).unsqueeze(2)

        # Active RTFs (The References): [F, 1, M, k_active] -> [F, 1, k_active, M]
        c = active_rtfs.squeeze(1).permute(0, 2, 1).unsqueeze(1)

        # 2. Compute Match Distances [F, K_old, k_active]
        # Angles between every old source and every currently active source
        angles = hermitian_angle(r, c, dim=-1)

        # 3. Aggregation
        # Mean over frequency -> [K_old, k_active]
        mean_angles = angles.mean(dim=0).squeeze(-1)

        # 4. Nearest Neighbor Distance
        # For each old source, how close is it to the set of current sources?
        # min_dist[i] = distance from old[i] to its closest match in active set
        min_dist_to_active, _ = mean_angles.min(dim=-1)  # [K_old]

        # 5. Outlier Selection
        # The old source with the LARGEST distance to the active set is the one that was removed.
        _, removed_idx = min_dist_to_active.max(dim=0)

        # Return as framewise indices
        return removed_idx.expand(T_seg)

    def _removing_deactivated_rtfs(
        self, old_rtf_seq: torch.Tensor, indices_of_removed_rtf: torch.Tensor
    ) -> torch.Tensor:
        """
        Removes the deactivated RTF vector from the set of old RTF vectors for each frame.

        Args:
            old_rtf_seq: [F, T, M, K_old]
            indices_of_removed_rtf: [T] - The index of the source to remove for each frame

        Returns:
            remaining_rtf_seq: [F, T, M, K_new] where K_new = K_old - 1
        """
        F, T, M, K_old = old_rtf_seq.shape
        K_new = K_old - 1

        if K_new == 0:
            return torch.zeros(
                F, T, M, 0, dtype=old_rtf_seq.dtype, device=old_rtf_seq.device
            )

        # We need to select K_new elements for each time frame T.
        # Since the index to remove varies with T, we can't just slice.
        # We'll create a mask.

        # indices_of_removed_rtf: [T] -> expand to [F, T, M, K_old]
        # We want a mask where mask[..., k] is True if k != removed_index[t]

        # Create a range tensor for K: [1, 1, 1, K_old]
        k_indices = torch.arange(K_old, device=old_rtf_seq.device).view(1, 1, 1, K_old)

        # Expand removed indices to match K dimension for comparison: [1, T, 1, 1]
        removed_indices_expanded = indices_of_removed_rtf.view(1, T, 1, 1)

        # Create boolean mask: True where we KEEP the element
        # [1, T, 1, K_old] -> broadcasts to [F, T, M, K_old]
        mask = k_indices != removed_indices_expanded

        # Expand mask to full shape [F, T, M, K_old]
        mask = mask.expand(F, T, M, K_old)

        # Select elements using the mask
        # masked_select returns a 1D tensor, so we must reshape carefully.
        # We know exactly K_new elements are kept per (f, t, m) tuple.
        remaining_flat = torch.masked_select(old_rtf_seq, mask)

        # Reshape back to [F, T, M, K_new]
        remaining_rtf_seq = remaining_flat.view(F, T, M, K_new)

        return remaining_rtf_seq
