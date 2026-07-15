import logging

import torch
from torchaudio.transforms import SoudenMVDR

from muse_toolbox.models.base_model import BaseLitModel
from muse_toolbox.data.components import HeterogeneousBatch
from muse_toolbox.utils import (
    STFTtransform,
    activity_dict2tensor,
    get_real_dtype,
    makeHermitian,
    norm_by_sum,
    peigvech,
    regularize,
    trace,
    wmean,
    zero2identity,
)

log = logging.getLogger(__name__)


class BlockOnlineGSS(BaseLitModel):
    """
    A PyTorch Lightning module for Block-Online Guided Source Separation (GSS).

    This module orchestrates the processing of heterogeneous scenarios by breaking them 
    down into utterances and subsequently into STFT blocks. It delegates the core block-level 
    separation to `OnlineGSS_Block_Processor`.
    """

    def __init__(
        self,
        transform: STFTtransform,
        max_sources: int,
        block_size: float,  # [seconds]
        pre_context: float,  # [seconds]
        ref_channels: list[int] = [0],
        latency_constraint: bool = True,
        batch_size: int = 1,
        loss_config: dict = {"CrossEntropy": None},
        optimizer_config: dict | None = None,
        lr_scheduler_config: dict | None = None,
        metrics_train: dict | None = None,
        metrics_val: dict | None = None,
        metrics_test: dict | None = None,
        compute_complexity_metrics: bool = False,
        check_causality: bool = False,
    ):
        """
        Initializes the BlockOnlineGSS module.

        Args:
            transform (STFTtransform): The STFT configuration for transforming audio signals.
            max_sources (int): The maximum number of expected sources.
            block_size (float): The length of the processing block in seconds.
            pre_context (float): The length of the pre-context window in seconds.
            ref_channels (list[int]): List of reference channels for target extraction.
            latency_constraint (bool): Whether to enforce latency constraints during beamforming.
            batch_size (int): Batch size for processing.
            loss_config (dict): Configuration for the loss function.
            optimizer_config (Optional[dict]): Configuration for the optimizer.
            lr_scheduler_config (Optional[dict]): Configuration for the learning rate scheduler.
            metrics_train (Optional[dict]): Metrics to track during training.
            metrics_val (Optional[dict]): Metrics to track during validation.
            metrics_test (Optional[dict]): Metrics to track during testing.
            compute_complexity_metrics (bool): Whether to profile computational complexity.
            check_causality (bool): Whether to enforce causality checks on the model.
        """
        super().__init__(
            model_name="BlockOnlineGSS",
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
            block_size_frames=int(transform.times2frames(block_size)),
            pre_context_frames=int(transform.times2frames(pre_context)),
        )

        self.max_sources = max_sources
        self.block_size_frames = int(transform.times2frames(block_size))
        self.pre_context_frames = int(transform.times2frames(pre_context))
        self.ref_channels = ref_channels

        self.processor = OnlineGSS_Block_Processor(
            Kmax=self.max_sources + 1,
            L=self.block_size_frames,
            C=self.pre_context_frames,
            ref_channels=self.ref_channels,
            latency_constraint=latency_constraint,
        )

    def forward_(self, batch: HeterogeneousBatch) -> HeterogeneousBatch:
        """
        Executes the forward pass for a batch of heterogeneous scenarios.

        Args:
            batch (HeterogeneousBatch): The input batch containing STFT audio and metadata.

        Returns:
            HeterogeneousBatch: The same batch, populated with the `estimates` attribute.
        """

        results = []

        # 2. Level 1 Loop
        # Iterate over each scenario in the batch
        for b_idx in range(batch.batch_size):
            # Extract single scenario data (handling padding if necessary)
            # You might need lengths from batch.meta['seq_len'] to slice correctly

            # Delegate to Level 2
            scenario_result = self._forward_scenario(
                stft=batch.stft_audio[b_idx],
                source_activity=batch.meta["sad_frames"][b_idx],
                source_id_map=batch.meta["id_map"][b_idx],
            )
            results.append(scenario_result)

        batch.estimates = results  # List of list of tensors

        return batch

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Executes a single test step, computing metrics for the given batch.

        Args:
            batch (dict): The test batch data.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.
        """
        processed_batch = self(batch)
        self._metric_step(processed_batch, dataloader_idx, "test")

    def _metric_step(
        self, processed_batch: HeterogeneousBatch, dataloader_idx: int, step_type: str
    ) -> None:
        """
        Updates the metric collections based on the estimates from the forward pass.

        Args:
            processed_batch (HeterogeneousBatch): The processed batch containing estimates and metadata.
            dataloader_idx (int): The index of the dataloader.
            step_type (str): The step type (e.g., 'val', 'test').
        """
        meta_dict = processed_batch.meta.copy()
        meta_dict["dataloader_idx"] = self.batch_size * [dataloader_idx]
        targets = (meta_dict["rtfs"], meta_dict["references"])
        self.metric_collections[step_type].update(
            processed_batch.estimates, targets, meta_dict, dataloader_idx
        )

    def _forward_scenario(
        self,
        stft: torch.Tensor,
        source_activity: dict[str, torch.Tensor],
        source_id_map: dict[int, str],
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], list[list[int]], dict[str, torch.Tensor]]:
        """
        Processes a single scenario (utterance) through the Block-Online GSS logic.

        Args:
            stft (torch.Tensor): The mixture STFT of shape `[F, M, T]`.
            source_activity (dict[str, torch.Tensor]): A dictionary mapping source IDs to their activity tensors over time.
            source_id_map (dict[int, str]): A mapping from local integer IDs to global string IDs.

        Returns:
            tuple: A 4-element tuple containing:
                - dict[str, torch.Tensor]: Target signal estimates (e.g., SMVDR output).
                - list[torch.Tensor]: Segment-wise RTF estimates.
                - list[list[int]]: Segment-wise active source IDs.
                - dict[str, torch.Tensor]: Segment-wise target beamformer weights.
        """
        source_ids, activity_tensor, target_id_stream, _ = activity_dict2tensor(
            source_activity, source_id_map
        )

        enhanced_segments, rtfs_segments, id_segments, beamformer_segments = (
            self.processor.process_utterance(
                stft, activity_tensor, source_ids=source_ids
            )
        )

        target, target_bf = self._extract_target(
            enhanced_segments, id_segments, target_id_stream, beamformer_segments
        )  # (F, M, T)

        return (
            {"SMVDR": target[:, self.ref_channels]},
            rtfs_segments,
            id_segments,
            {"SMVDR": target_bf[..., self.ref_channels]},
        )

    def _extract_target(
        self,
        enhanced_segments: list[torch.Tensor],
        id_segments: list[list[int]],
        target_id_stream: torch.Tensor,
        beamformer_segments: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts a single channel taget signal from the enhances segments.
            Here we decided to always extract the latest activatec source in each segment.

        Args:
            enhanced_segments (list[torch.Tensor]): List of enhanced segments.
            id_segments (list[list[int]]): List of active source IDs for each segment.

        Returns:
            torch.Tensor: Target enhanced segments concatenated along the time dimension.
        """
        F, M = enhanced_segments[0].shape[-3], enhanced_segments[0].shape[-2]
        target_segments = []
        target_bf_segments = []
        start = 0
        for seg_idx, seg_ids in enumerate(id_segments):
            enhanced_seg = enhanced_segments[seg_idx]  # (B, Kt, F, M, T_seg)
            T_seg = enhanced_seg.shape[-1]
            end = start + T_seg
            if len(seg_ids) == 0:
                # No active sources in this segment
                target_segments.append(
                    torch.zeros(
                        (F, M, T_seg),
                        device=enhanced_seg.device,
                        dtype=enhanced_seg.dtype,
                    )
                )
                target_bf_segments.append(
                    torch.zeros(
                        (F, T_seg, M, M),
                        device=enhanced_seg.device,
                        dtype=enhanced_seg.dtype,
                    )
                )
            else:
                target_id_seg = target_id_stream[start:end]  # (T_seg,)
                # test uniques of target_id within the segment
                unique_target_ids = torch.unique(target_id_seg)
                if len(unique_target_ids) != 1:
                    raise ValueError(
                        "Multiple target IDs found in a single segment. This is not supported."
                    )
                target_id = unique_target_ids[0].item()
                if target_id not in seg_ids:
                    raise ValueError("Target source is not active in this segment.")
                    # Target source is not active in this segment
                    target_segments.append(
                        torch.zeros(
                            (F, M, T_seg),
                            device=enhanced_seg.device,
                            dtype=enhanced_seg.dtype,
                        )
                    )
                    target_bf_segments.append(
                        torch.zeros(
                            (F, T_seg, M, M),
                            device=enhanced_seg.device,
                            dtype=enhanced_seg.dtype,
                        )
                    )
                else:
                    # Target source is active in this segment
                    target_idx_in_seg = seg_ids.index(target_id)
                    target_segments.append(
                        enhanced_seg[
                            0,
                            target_idx_in_seg,
                        ]
                    )  # (F, M, T_seg)
                    target_bf_segments.append(
                        beamformer_segments[seg_idx][target_idx_in_seg,]
                    )  # (F, T_seg, M, M)
            start = end

        return (
            torch.cat(target_segments, dim=-1),
            torch.cat(target_bf_segments, dim=-3),
        )  # (F, M, T_total), (F, T_total, M, M)


class OnlineGSS_Block_Processor:
    """
    A processor class that executes the Block-Online Guided Source Separation (GSS) algorithm.

    This class maintains the state (e.g., spatial covariance matrices, spatial filter weights) 
    across overlapping blocks of an STFT utterance, updating them recursively based on 
    provided source activity information.
    """
    def __init__(
        self,
        Kmax: int,
        L: int,
        C: int,
        eta: float | None = None,
        ref_channels: list[int] = [0],
        latency_constraint: bool = True,
    ):
        """
        Initializes the OnlineGSS_Block_Processor.

        Args:
            Kmax (int): The maximum number of sources to track.
            L (int): The block length in frames.
            C (int): The number of pre-context frames per block.
            eta (Optional[float]): Decay factor for moving sources. Defaults to None.
            ref_channels (list[int]): List of reference channels.
            latency_constraint (bool): Whether to enforce a causal latency constraint.
        """
        self.Kmax = Kmax  # Number of sources
        self.L = L  # Block length (in frames)
        self.C = C  # Number of pre-context frames
        self.eta = eta  # factor of decay for moving sources
        # reference channel for Souden MVDR
        self.ref_channels = (
            ref_channels  # Not used in current implementation, MIMO output
        )
        self.SMVDR = SoudenMVDR()
        self.latency_constraint = latency_constraint

    def process_utterance(
        self, stft: torch.Tensor, activity: torch.Tensor, source_ids: list[int]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[int]], list[torch.Tensor]]:
        """Processes an entire utterance block by block.

        Args:
            stft (torch.Tensor): The mixture STFT of shape `(F, M, T)`.
            activity (torch.Tensor): Source activity tensor of shape `(Kmax, T)`.
            source_ids (list[int]): List of active source IDs.

        Returns:
            tuple: A 4-element tuple containing:
                - list[torch.Tensor]: Segment-wise enhanced STFT outputs.
                - list[torch.Tensor]: Segment-wise framewise RTF vectors.
                - list[list[int]]: Segment-wise active source IDs.
                - list[torch.Tensor]: Segment-wise beamformer weight matrices.
        """
        F, M, T = stft.shape  # (F, M, T)
        stft = stft.to(torch.complex64)
        activity = activity.to(torch.float32)

        stft_blocks = self._segment_into_blocks(stft)  # (N, F, M, C+L)
        activity_blocks = self._segment_into_blocks(activity)  # (N, K, C+L)

        Tpad = stft_blocks.shape[0] * self.L + self.C  # N * L + C

        # L1: Initialize quantities to store results // Initial #sources
        self.source_ids = source_ids
        self.Kn = set()  # set of seen source indices
        self.gamma = torch.zeros(
            self.Kmax, F, Tpad, device=stft.device, dtype=get_real_dtype(stft)
        )
        self.Gamma = torch.zeros(
            self.Kmax, F, device=stft.device, dtype=get_real_dtype(stft)
        )
        self.B = torch.zeros(self.Kmax, F, M, M, device=stft.device, dtype=stft.dtype)
        self.Bn_plus = torch.zeros(
            self.Kmax, F, M, M, device=stft.device, dtype=stft.dtype
        )
        if self.latency_constraint:
            self.w_old = torch.zeros(
                self.Kmax, F, M, M, device=stft.device, dtype=stft.dtype
            )
            self.R_speech_old = torch.zeros(
                self.Kmax, F, M, M, device=stft.device, dtype=stft.dtype
            )
            self.R_noise_old = torch.zeros(
                self.Kmax, F, M, M, device=stft.device, dtype=stft.dtype
            )

        # Get number of blocks
        N = stft_blocks.shape[0]

        # L2: Loop over blocks
        enhanced_frames = []
        rtfs_frames = []
        Kbf_frames = []
        beamformer_frames = []
        for n in range(N):
            enhanced_block, block_rtfs, block_Kbf, beamformer_block = (
                self.process_block(
                    n=n,
                    Xn_plus=stft_blocks[n],
                    dn_plus=activity_blocks[n],
                )
            )
            enhanced_frames.extend(enhanced_block)
            rtfs_frames.extend(block_rtfs)
            Kbf_frames.extend(block_Kbf)
            beamformer_frames.extend(beamformer_block)

        return self._frame2segment(
            enhanced_frames, rtfs_frames, Kbf_frames, beamformer_frames, T
        )

    def _frame2segment(
        self,
        enhanced_frames: list[torch.Tensor],
        rtfs_frames: list[torch.Tensor],
        Kbf_frames: list[list[int]],
        beamformer_frames: list[torch.Tensor],
        T: int,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[list[int]], list[torch.Tensor]
    ]:
        """Converts block-wise outputs to segment-wise outputs.
            A segment is defined as the connected part of an utterance where the same sources are active.

        Args:
            enhanced_frames (list[torch.Tensor]): List of enhanced frames of shape (Kbf, F, M, 1)
            rtfs_frames (list[torch.Tensor]): List of RTFs for each frame of shape (F, M, Kbf)
            Kbf_frames (list[list[int]]): List of active source ids for each frame.
            beamformer_frames (list[torch.Tensor]): List of beamformer weight matrices for each frame.
            T (int): Total number of frames in the utterance.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor], list[list[int]], list[torch.Tensor]]:
                - List of enhanced segments.
                - List of RTFs for each segment.
                - List of active source ids for each segment.
                - List of beamformer weight matrices for each segment.
        """
        enhanced_segments = []
        rtfs_segments = []
        id_segments = []
        beamformer_segments = []

        for t in range(T):
            if t == 0:
                # Start new segment
                enhanced_segments.append([enhanced_frames[t]])
                rtfs_segments.append([rtfs_frames[t]])
                id_segments.append(Kbf_frames[t])
                beamformer_segments.append([beamformer_frames[t]])
            else:
                if Kbf_frames[t] == Kbf_frames[t - 1]:
                    # Continue current segment
                    enhanced_segments[-1].append(enhanced_frames[t])
                    rtfs_segments[-1].append(rtfs_frames[t])
                    beamformer_segments[-1].append(beamformer_frames[t])
                else:
                    # Start new segment
                    enhanced_segments.append([enhanced_frames[t]])
                    rtfs_segments.append([rtfs_frames[t]])
                    id_segments.append(Kbf_frames[t])
                    beamformer_segments.append([beamformer_frames[t]])

        # Convert lists of frames to tensors for each segment
        enhanced_segments = [
            torch.cat(segment, dim=-1).to(torch.complex64)
            for segment in enhanced_segments
        ]
        rtfs_segments = [
            torch.stack(segment, dim=-3).to(torch.complex64)
            for segment in rtfs_segments
        ]
        beamformer_segments = [
            torch.cat(segment, dim=-3).to(torch.complex64)
            for segment in beamformer_segments
        ]

        return enhanced_segments, rtfs_segments, id_segments, beamformer_segments

    def process_block(
        self,
        n: int,
        Xn_plus: torch.Tensor,
        dn_plus: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[list[int]], list[torch.Tensor]
    ]:
        """Processes a single block of STFT data.

        Args:
            Xn_plus (torch.Tensor): STFT block of shape (F, M, C+L)
            dn_plus (torch.Tensor): Activity block of shape (Kmax, C+L)

        Returns:
            torch.Tensor: Processed output for the block.
            list[torch.Tensor]: List of framewise RTFs for the block.
            list[list[int]]: List of active source ids for the block.
            list[torch.Tensor]: List of weight matrices for the block.
        """
        # Extract relevant indices and data for the current block
        Tn_plus = (
            n * self.L + torch.arange(self.C + self.L, device=Xn_plus.device)
        ).unsqueeze(
            0
        )  # (1, C+L)
        Tn = Tn_plus[:, self.C :]  # (1, L)
        Tn_c = Tn_plus[:, : self.C]  # (1, C)
        # Tn_plus = n * self.L + torch.arange(self.C + self.L)  # (C+L,)
        # Tn = Tn_plus[self.C :]  # (L,)
        # Tn_c = Tn_plus[: self.C]  # (C,)
        F, M, _ = Xn_plus.shape  # (F, M, C+L)
        Xn = Xn_plus[..., self.C :]  # (F, M, L)
        dn = dn_plus[..., self.C :]  # (Kmax, L)
        Xn_c = Xn_plus[..., : self.C]  # (F, M, C)
        dn_c = dn_plus[..., : self.C]  # (Kmax, C)

        # L3: Block-Online WPE -> We ommit WPE for now

        # L4 and L5: Check for active sources in this block // Silent block
        if dn.sum() == 0:
            # No active sources in this block
            enhanced_block = self.L * [
                torch.zeros((0, F, M, 1), device=Xn_plus.device, dtype=Xn_plus.dtype)
            ]
            block_rtfs = self.L * [
                torch.zeros(
                    (F, M, 0),
                    device=Xn_plus.device,
                    dtype=Xn_plus.dtype,
                )
            ]
            block_w_store = self.L * [
                torch.zeros(
                    (0, F, 1, M, M),
                    device=Xn_plus.device,
                    dtype=Xn_plus.dtype,
                )
            ]
            self._zero_buffers()
            return (
                enhanced_block,
                block_rtfs,
                self.L * [[]],
                block_w_store,
            )

        # Process active sources
        # L6: create a set of active source indices // Set of active sources
        Kact_ai = torch.nonzero(dn_plus.sum(dim=1) > 0)
        Kact = Kact_ai.squeeze(-1)  # (Kact,)

        # L8: Determine whether new sources have appeared // New sources
        # flatten Kact to a list of integers for set operations
        Kact_list = Kact.flatten().tolist()

        Knew_list = [idx for idx in Kact_list if idx not in self.Kn]
        Kold_list = [idx for idx in Kact_list if idx in self.Kn]

        # update the set
        self.Kn.update(Knew_list)

        # Convert back to 1D index tensors matched to Kact's structure and device
        Knew = torch.tensor(Knew_list, device=dn_plus.device, dtype=torch.long)
        Kold = torch.tensor(Kold_list, device=dn_plus.device, dtype=torch.long)

        Knew_ai = Knew.unsqueeze(-1)

        # L9 - L11: Initialize parameters for new sources
        if len(Knew) > 0:
            self.gamma[Knew_ai, :, Tn_c] = 0
            self.Gamma[Knew, :] = 0
            self.B[Knew, :, :, :] = 0

        # L12: Initialize gamma for (t,k) in Tn x Kact by (8)
        self.gamma[Kact_ai, :, Tn] = (
            norm_by_sum(dn[Kact], dims=0)
            .nan_to_num(nan=0.0)
            .unsqueeze(-1)
            .expand(-1, -1, F)
        )  # (Kact, F, L)



        # L13: Update alpha for k in Kact using Xn_plus by (5)
        # Time context clarified by mail from author: "I think Xn^plus in Line 13 should be Tn^plus."
        alpha = self.gamma[Kact_ai, :, Tn_plus].mean(dim=1).unsqueeze(-1)  # (Kact, 1)

        # L14: Calculate Bn_plus for k in Kact using Xn_plus by (14)
        Xn_plus_rs = Xn_plus.permute(0, 2, 1).unsqueeze(-1)  # (F, C+L, M, 1)
        gamma_rs = (
            self.gamma[:, :, Tn_plus.squeeze(0)].unsqueeze(-1).unsqueeze(-1)
        )  # (Kmax, F, C+L, 1, 1)
        xxH = makeHermitian(Xn_plus_rs @ Xn_plus_rs.mH)  # (F, C+L, M, M)

        if len(Kold) > 0:
            B_rs_14 = self.B[Kold].unsqueeze(-3)  # (Kold, F, 1, M, M)
            xHBx_14 = (
                Xn_plus_rs.mH @ self._solve(B_rs_14, Xn_plus_rs)
            ).real  # (Kold, F, C+L, 1, 1)
            self.Bn_plus[Kold] = makeHermitian(
                M
                * wmean(
                    xxH / xHBx_14,
                    weights=gamma_rs[Kold],
                    dims=-3,
                    keepdim=False,
                ).nan_to_num(nan=0.0)
            )  # (Kold, F, M, M)

        if len(Knew) > 0:
            self.Bn_plus[Knew] = makeHermitian(
                M
                * wmean(
                    xxH,
                    weights=gamma_rs[Knew],
                    dims=-3,
                    keepdim=False,
                ).nan_to_num(nan=0.0)
            )  # (Knew, F, M, M)



        # L15: Update B for k in Kact using Xn_plus by (15)-(16) or (17)
        # We use (15)-(16) here, since sources are not moving in our scenairo
        # Confusion about "using Xn_plus" here -> it was likely a typo and should be "using Tn", clarified by mail from the author
        if self.eta == None:
            # (15)
            gamma_Tn_sum = self.gamma[Kact_ai, :, Tn].sum(dim=1)  # (Kact, F)
            denominator_15 = self.Gamma[Kact] + gamma_Tn_sum  # (Kact, F)
            factor_B = (
                (self.Gamma[Kact] / denominator_15).unsqueeze(-1).unsqueeze(-1)
            )  # (Kact, F, 1, 1)
            factor_Bn_plus = (
                (gamma_Tn_sum / denominator_15).unsqueeze(-1).unsqueeze(-1)
            )  # (Kact, F, 1, 1)
            self.B[Kact] = regularize(
                makeHermitian(
                    factor_B * self.B[Kact] + factor_Bn_plus * self.Bn_plus[Kact]
                ),
                reg_factor=1e-6,
            )  # (Kact, F, M, M)
            # (16)
            self.Gamma[Kact] += gamma_Tn_sum  # (Kact, F)
        else:
            # (17) for moving sources or sampling frequency mismatch
            # not used here
            self.B[Kact] = makeHermitian(
                self.eta * self.B[Kact] + self.Bn_plus[Kact]
            )  # (Kact, F, M, M)

        # L16: Update gamma for (t,k) in Tn_plus x Kact by (7)
        dn_plus_rs = dn_plus[Kact].unsqueeze(-2)  # (Kact, 1, C+L)
        B_rs_7 = self.B[Kact].unsqueeze(-3)  # (Kact, F, 1, M, M)
        xHBx_7 = (
            (Xn_plus_rs.mH @ self._solve(B_rs_7, Xn_plus_rs))
            .real.squeeze(-1)
            .squeeze(-1)
        )  # (Kact, F, C+L)
        detB = torch.linalg.det(B_rs_7.to(torch.complex128)).real  # (Kact, F, 1)



        a_d_detB_xHbx = (alpha * dn_plus_rs / detB / xHBx_7).nan_to_num(
            nan=0.0
        )  # (Kact, F, C+L)



        self.gamma[Kact_ai, :, Tn_plus] = self._num_stability4gamma(
            norm_by_sum(a_d_detB_xHbx, dims=0)
            .nan_to_num(nan=0.0)
            .to(self.gamma.dtype)
            .permute(0, 2, 1)
        )  # (Kact, F, C+L)



        # L17: Define Kbf as the set of sources to be beamformed // All active speech sources
        Kbf = Kact[torch.tensor(self.source_ids, device=Kact.device)[Kact] != -2]
        Kbf_ai = Kbf.unsqueeze(-1)

        if len(Kbf) == 0:
            # No active speech sources to beamform
            enhanced_block = self.L * [
                torch.zeros((0, F, M, 1), device=Xn_plus.device, dtype=Xn_plus.dtype)
            ]
            block_rtfs = self.L * [
                torch.zeros(
                    (F, M, 0),
                    device=Xn_plus.device,
                    dtype=Xn_plus.dtype,
                )
            ]
            block_w_store = self.L * [
                torch.zeros(
                    (0, F, 1, M, M),
                    device=Xn_plus.device,
                    dtype=Xn_plus.dtype,
                )
            ]
            self._zero_buffers()
            return (
                enhanced_block,
                block_rtfs,
                self.L * [[]],
                block_w_store,
            )

        # L18: Calculate R_speech and R_noise from Xn_plus by (10)-(11)
        gamma_rs_10_11 = (
            self.gamma[Kbf_ai, :, Tn_plus].permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        )  # (Kbf, F, C+L, 1, 1)
        R_speech = wmean(
            xxH, weights=gamma_rs_10_11, dims=-3, keepdim=False
        ).nan_to_num(
            nan=0.0
        )  # (Kbf, F, M, M)



        R_noise = wmean(
            xxH, weights=(1 - gamma_rs_10_11), dims=-3, keepdim=False
        ).nan_to_num(
            nan=0.0
        )  # (Kbf, F, M, M)



        # Gode: calculate RTFs from R_speech
        rtfs = peigvech(R_speech).squeeze(-1).movedim(0, -1)  # (F, M, Kbf)

        # L19: Calculate w by (12) Souden MVDR
        R_noise_inv_R_speech = self._solve(R_noise, R_speech)  # (Kbf, F, M, M)
        w = (R_noise_inv_R_speech / trace(R_noise_inv_R_speech)).nan_to_num(
            nan=0.0
        )  # (Kbf, F, M, M)



        if not self.latency_constraint:
            w_store = w.unsqueeze(-3).expand(
                -1, -1, self.L, -1, -1
            )  # (Kbf, F, L, M, M)
            # L20: Apply beamformer to Xn by (13)
            z = w.mH @ Xn  # (Kbf, F, M, L)

            # L19-L20: Alternatively, we can use torchaudio's SoudenMVDR
            z_t = self._smvdrMIMO(
                Xn=Xn,  # (F, M, L)
                R_speech=R_speech,  # (Kbf, F, M, M)
                R_noise=R_noise,  # (Kbf, F, M, M)
            )  # (Kbf, F, M, L)
        else:
            w_store = torch.cat(
                [
                    self.w_old[Kbf].unsqueeze(-3).expand(-1, -1, self.L - 1, -1, -1),
                    w.unsqueeze(-3),
                ],
                dim=-3,
            )  # (Kbf, F, L, M, M)
            # L20: Apply beamformer to Xn by (13)
            z = torch.cat(
                [
                    self.w_old[Kbf].mH @ Xn[..., :-1],  # (Kbf, F, M, L-1)
                    w.mH @ Xn[..., -1:],  # (Kbf, F, M, 1)
                ],
                dim=-1,
            )  # (Kbf, F, M, L)

            # L19-L20: Alternatively, we can use torchaudio's SoudenMVDR
            z_t_old = self._smvdrMIMO(
                Xn=Xn[..., :-1],  # (F, M, L-1)
                R_speech=R_speech,  # (Kbf, F, M, M)
                R_noise=R_noise,  # (Kbf, F, M, M)
            )  # (Kbf, F, M, L-1)
            z_t_new = self._smvdrMIMO(
                Xn=Xn[..., -1:],  # (F, M, 1)
                R_speech=self.R_speech_old[Kbf],  # (Kbf, F, M, M)
                R_noise=self.R_noise_old[Kbf],  # (Kbf, F, M, M)
            )  # (Kbf, F, M, 1)
            z_t = torch.cat([z_t_old, z_t_new], dim=-1)  # (Kbf, F, M, L)

            # Store old variables for next block
            self._zero_buffers()
            self.w_old[Kbf] = w
            self.R_speech_old[Kbf] = R_speech
            self.R_noise_old[Kbf] = R_noise

        return self._align2activity(torch.stack([z, z_t]), rtfs, dn, Kbf, w_store)

    def _align2activity(
        self,
        z: torch.Tensor,
        rtfs: torch.Tensor,
        dn: torch.Tensor,
        Kbf: torch.Tensor,
        w_store: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[list[int]], list[torch.Tensor]
    ]:
        """Aligns the beamformed outputs and RTFs to the activity of sources.

        Args:
            z (torch.Tensor): Beamformed outputs of shape (B, Kbf, F, M, L)
            rtfs (torch.Tensor): RTFs of shape (F, M, Kbf)
            dn (torch.Tensor): Activity tensor of shape
            Kbf (torch.Tensor): Active source indices (no noise) of shape (Kbf,)
            w_store (torch.Tensor): Beamformer weights of shape (Kbf, F, L, M, M)
        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor], list[list[int]], list[torch.Tensor]]:
                - List of enhanced outputs for each frame in the block.
                - List of RTFs for each frame in the block.
                - List of active source indices for each frame in the block.
                - List of beamformer weights for each frame in the block.
        """
        enhanced_block = []
        block_rtfs = []
        block_Kbf = []
        beamformer_block = []
        F, M, _ = rtfs.shape
        for t in range(self.L):
            active_sources_t = torch.nonzero(dn[:, t] > 0).squeeze(-1)
            active_speech_sources_t = [
                int(idx.item())
                for idx in active_sources_t
                if idx.item() in Kbf.tolist()
            ]
            Kt = len(active_speech_sources_t)
            if Kt == 0:
                enhanced_block.append(
                    torch.zeros(
                        (0, F, M, 1),
                        device=z.device,
                        dtype=z.dtype,
                    )
                )
                block_rtfs.append(
                    torch.zeros(
                        (F, M, 0),
                        device=z.device,
                        dtype=z.dtype,
                    )
                )
                block_Kbf.append([])
                beamformer_block.append(
                    torch.zeros(
                        (0, F, 1, M, M),
                        device=z.device,
                        dtype=z.dtype,
                    )
                )
            else:
                indices_in_Kbf = [
                    Kbf.tolist().index(idx) for idx in active_speech_sources_t
                ]
                enhanced_block.append(
                    z[:, indices_in_Kbf, :, :, t : t + 1]
                )  # List of (B, Kt, F, M, 1), length L
                block_rtfs.append(
                    rtfs[:, :, indices_in_Kbf]
                )  # List of (F, M, Kt), length L
                block_Kbf.append(
                    self._source_idx2id(active_speech_sources_t)
                )  # List of length L of lists of length Kt
                beamformer_block.append(
                    w_store[indices_in_Kbf, :, t : t + 1]
                )  # List of (Kt, F, 1, M, M), length L

        return enhanced_block, block_rtfs, block_Kbf, beamformer_block

    def _source_idx2id(self, source_idxs: list[int] | int) -> list[int] | int:
        """Converts source indices to source IDs.

        Args:
            source_idxs (list[int] | int): Source indices.

        Returns:
            list[int] | int: Source IDs.
        """
        if isinstance(source_idxs, int):
            return self.source_ids[source_idxs]
        else:
            return [self.source_ids[idx] for idx in source_idxs]

    def _num_stability4gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        """Ensures numerical stability of gamma by clipping its values to [0, 1e-6 - 1].
        Args:
            gamma (torch.Tensor): Gamma tensor. (K, F, T)
        Returns:
            torch.Tensor: Numerically stable gamma tensor.
        """
        gamma = torch.clamp(gamma, min=0.0, max=1.0)
        gamma[torch.where(gamma < 1e-6)] = 0.0
        return gamma

    def _smvdrMIMO(
        self, Xn: torch.Tensor, R_speech: torch.Tensor, R_noise: torch.Tensor
    ):
        """Applies Souden MVDR beamforming to the input STFT data for all reference channels

        Args:
            Xn (torch.Tensor): STFT data of shape (F, M, L)
            R_speech (torch.Tensor): Speech covariance matrix of shape (Kbf, F, M, M)
            R_noise (torch.Tensor): Noise covariance matrix of shape (Kbf, F, M, M)
            ref_channel (int): Reference channel index

        Returns:
            torch.Tensor: Beamformed output of shape (Kbf, F, L)
        """
        z_Klist = []
        for k in range(R_speech.shape[0]):
            if (R_speech[k] == 0).all() or (R_noise[k] == 0).all():
                z_Klist.append(torch.zeros_like(Xn))  # (F, M, L)
            else:
                z_Mlist = []
                for ref_channel in range(Xn.shape[-2]):
                    z_Mlist.append(
                        self.SMVDR(
                            specgram=Xn.permute(1, 0, 2),  # (M, F, L)
                            psd_s=R_speech[k],  # (F, M, M)
                            psd_n=R_noise[k],  # (F, M, M)
                            reference_channel=ref_channel,
                        ).unsqueeze(-2)
                    )  # list of (F, 1, L), length M
                z_Klist.append(
                    torch.cat(z_Mlist, dim=-2)
                )  # list of (F, M, L), length Kbf

        return torch.stack(z_Klist)  # (Kbf, F, M, L)

    def _zero_buffers(self) -> None:
        """
        Resets the internal buffer state variables used for enforcing the latency constraint.
        Sets the previous beamformer weights and spatial covariance matrices to zero.
        """
        if self.latency_constraint:
            self.w_old[:, :, :, :] = (
                0  # (Kmax, F, M, M) We need slicing here to avoid changing the tensor to a scalar
            )
            self.R_speech_old[:, :, :, :] = (
                0  # (Kmax, F, M, M)  We need slicing here to avoid changing the tensor to a scalar
            )
            self.R_noise_old[:, :, :, :] = (
                0  # (Kmax, F, M, M)  We need slicing here to avoid changing the tensor to a scalar
            )

    @staticmethod
    def _solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Solves a system of linear equations `AX = B` robustly.

        Regularizes the matrix `A` and falls back to a least squares solution 
        if the exact solver fails.

        Args:
            A (torch.Tensor): The left-hand side matrix of shape `[..., M, M]`.
            B (torch.Tensor): The right-hand side matrix of shape `[..., M, K]`.

        Returns:
            torch.Tensor: The solution matrix `X` of shape `[..., M, K]`.
        """
        A = zero2identity(regularize(A, reg_factor=1e-6))
        try:
            return torch.linalg.solve(A, B)
        except RuntimeError:
            return torch.linalg.lstsq(A, B).solution

    def _segment_into_blocks(self, quantity: torch.Tensor) -> torch.Tensor:
        """Segments the STFT into overlapping blocks.

        Args:
            quantity (torch.Tensor): Quantity of shape (..., T)

        Returns:
            torch.Tensor: Segmented quantity of shape (num_blocks, ..., block_size)
        """
        T = quantity.shape[-1]
        padded_quantity = torch.nn.functional.pad(
            quantity,
            pad=(self.C, self.L - (T % self.L)),
            mode="constant",
            value=0,
        )  # (..., T_padded)

        # I guess the following is not needed since unfold will handle the case where T is not multiple of L ?!
        # padded_quantity = padded_quantity[..., : N * self.L]

        quantity_blocks = padded_quantity.unfold(
            dimension=-1, size=self.C + self.L, step=self.L
        )  # (..., num_blocks, block_size)
        quantity_blocks = quantity_blocks.movedim(
            -2, 0
        )  # (num_blocks, ..., block_size)

        return quantity_blocks
