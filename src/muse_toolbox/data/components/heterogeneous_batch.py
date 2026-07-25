from __future__ import annotations
import torch
from muse_toolbox.data.simulation.scenario_generation import Segment
from muse_toolbox.utils.math.complex_angles import hermitian_angle
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from muse_toolbox.models.source_counting.estimators.base_estimator import BaseSourceCountEstimator
    from muse_toolbox.losses.base_loss import BaseLoss
    from muse_toolbox.models.components.channel_combinator.base_channel_combinator import BaseChannelCombinator
    from muse_toolbox.models.components.feature_extractors.base_feature import BaseFeatureExtractor


class HeterogeneousBatch:
    """
    A smart container for mixed-type, mixed-length audio/feature data.
    It handles device transfer and orchestrates feature extraction + padding.
    """

    def __init__(
        self,
        raw_audio: list[torch.Tensor] = [],
        stft_audio: list[torch.Tensor] = [],
        features: list[torch.Tensor | dict[str, torch.Tensor]] = [],
        meta: dict[str, list[Any]] = {},
        stft_info: dict[str, Any] | None = None,
        feature_info: dict[str, Any] | None = None,
        stft_data_needed: bool = False,
    ):
        self.raw_audio = raw_audio
        self.stft_audio = stft_audio
        self.features = features
        self.meta = meta
        self.stft_info = stft_info
        self.feature_info = feature_info
        self.stft_data_needed = stft_data_needed
        self.device = torch.device("cpu")
        self.batch_size = len(next(iter(meta.values()))) if meta else 0

        # Pipeline State
        self.status = "input"  # 'input' -> 'features' -> 'estimates'
        self.mask: torch.Tensor | None = None  # (B, T_max)
        self.padded_features: torch.Tensor | None = None  # (B, J, T_max)
        self.padded_estimates: torch.Tensor | None = None  # (B, J, T_max)
        self.estimates: list[torch.Tensor] | torch.Tensor | None = (
            None  # Output of estimator
        )

        if "gt_rtf_stream" not in self.meta:
            self.meta["gt_rtf_stream"] = []
            self.meta["gt_ids_stream"] = []
            self.meta["id_map"] = []
        for i in range(self.batch_size):
            sad_frames = self.meta["sad_frames"][i]  # Dict of SAD tensors (T,)
            rtfs = self.meta["rtfs"][i]  # Dict of RTF tensors
            segments = self.meta["segments"][i]  # List[Segment]

            # Compute ground-truth RTF stream for this item
            gt_rtf_stream, gt_ids_stream, id_map = ground_truth_rtf_stream(
                sad_frames, rtfs, segments
            )
            self.meta["gt_rtf_stream"].append(gt_rtf_stream)
            self.meta["gt_ids_stream"].append(gt_ids_stream)
            self.meta["id_map"].append(id_map)

    def to(self, device: torch.device):
        """
        Moves all internal tensors to the specified device.
        Called automatically by Lightning if transfer_batch_to_device is overridden.
        """
        self.device = device

        # Move raw audio
        self.raw_audio = [t.to(device) for t in self.raw_audio]

        # Move STFT audio
        self.stft_audio = [t.to(device) for t in self.stft_audio]

        # Move features
        new_features = []
        for t in self.features:
            if isinstance(t, torch.Tensor):
                new_features.append(t.to(device))
            elif isinstance(t, dict):
                new_features.append({k: v.to(device) for k, v in t.items()})
            else:
                raise TypeError(f"Unexpected type in self.features: {type(t)}")
        self.features = new_features

        # Move tensor targets in meta (e.g., source_count)
        if "source_count" in self.meta:
            sc = self.meta["source_count"]
            # source_count is likely a list of 1D tensors (variable length)
            self.meta["source_count"] = [t.to(device) for t in sc]

        # Move RTFs in meta
        if "rtfs" in self.meta:
            new_rtfs_batch = []
            for item in self.meta["rtfs"]:
                if isinstance(item, dict):
                    new_item = {k: v.to(device) for k, v in item.items()}
                    new_rtfs_batch.append(new_item)
                elif isinstance(item, list):
                    new_item = [t.to(device) for t in item]
                    new_rtfs_batch.append(new_item)
                else:
                    new_rtfs_batch.append(item)
            self.meta["rtfs"] = new_rtfs_batch

        if "gt_rtf_stream" in self.meta:
            new_gt_rtf_batch = []
            for item in self.meta["gt_rtf_stream"]:
                if isinstance(item, dict):
                    new_item = {k: v.to(device) for k, v in item.items()}
                    new_gt_rtf_batch.append(new_item)
                elif isinstance(item, list):
                    new_item = [t.to(device) for t in item]
                    new_gt_rtf_batch.append(new_item)
                else:
                    new_gt_rtf_batch.append(item)
            self.meta["gt_rtf_stream"] = new_gt_rtf_batch

        if "sad_frames" in self.meta:
            new_sad_batch = []
            for item in self.meta["sad_frames"]:
                if isinstance(item, dict):
                    new_item = {k: v.to(device) for k, v in item.items()}
                    new_sad_batch.append(new_item)
                elif isinstance(item, list):
                    new_item = [t.to(device) for t in item]
                    new_sad_batch.append(new_item)
                else:
                    new_sad_batch.append(item)
            self.meta["sad_frames"] = new_sad_batch

        return self

    def randomly_permute_channels(self) -> "HeterogeneousBatch":
        """
        Randomly permutes the channel dimension (M) for raw_audio and stft_audio.
        This acts as a data augmentation to prevent the model from learning
        specific microphone array geometries or channel orderings.
        
        Note: This does not affect pre-computed features if they have already
        condensed the channel dimension (e.g., STFT_Conv). To use this, you must 
        process STFTs or raw audio on the fly (e.g. force_load_stft=True).
        """
        for i in range(len(self.raw_audio)):
            M = self.raw_audio[i].shape[0]
            idx = torch.randperm(M, device=self.device)
            self.raw_audio[i] = self.raw_audio[i][idx]
            
        for i in range(len(self.stft_audio)):
            # Assuming shape is (M, F, T) as passed to extractor
            M = self.stft_audio[i].shape[0]
            idx = torch.randperm(M, device=self.device)
            self.stft_audio[i] = self.stft_audio[i][idx]
            
        # For precomputed features, we attempt to permute if M is preserved as dim 0
        # (e.g. pure_stft or log_mel outputs), but this is brittle for arbitrary J-dim features.
        for i in range(len(self.features)):
            item = self.features[i]
            if isinstance(item, torch.Tensor) and item.ndim == 3: # (M, Feat, T)
                M = item.shape[0]
                idx = torch.randperm(M, device=self.device)
                self.features[i] = item[idx]
                
        return self

    def apply_feature_extractor(
        self, extractor: "BaseFeatureExtractor"
    ) -> "HeterogeneousBatch":
        """
        Runs the feature extractor on all available input types.
        Unifies results into self.features and clears raw inputs.
        """
        if self.status != "input":
            # If already processed, do nothing or raise error
            return self

        self.processed_features = []

        # 1. Process Raw Audio
        for item in self.raw_audio:
            # item: (M, N) -> Unsqueeze to (1, M, N) for batch processing
            feat = extractor.forward_raw_audio(item.unsqueeze(0))
            self.processed_features.append(feat.squeeze(0))  # Store as (J, T)

        # 2. Process STFT Audio
        for item in self.stft_audio:
            # item: (F, M, T) or similar.
            # We assume item has correct dimensions for the extractor except batch.
            feat = extractor.forward_stft(item.unsqueeze(0))
            self.processed_features.append(feat.squeeze(0))

        # 3. Process Precomputed Features
        for item in self.features:
            if isinstance(item, torch.Tensor):
                # item: (J, T) -> Unsqueeze to (1, J, T)
                feat = extractor.forward_precomputed_features(item.unsqueeze(0))
                self.processed_features.append(feat.squeeze(0))
            elif isinstance(item, dict):
                # item is a dict of tensors. We need to unsqueeze each tensor.
                unsqueezed_item = {k: v.unsqueeze(0) for k, v in item.items()}
                feat = extractor.forward_precomputed_features_dict(unsqueezed_item)
                self.processed_features.append(feat.squeeze(0))
            else:
                raise TypeError(f"Unexpected type in self.features: {type(item)}")

        # Update State
        self.raw_audio = []  # Clear to save memory
        self.stft_audio = []
        self.status = "features"

        return self

    def apply_channel_combinator(
        self, combinator: "BaseChannelCombinator"
    ) -> "HeterogeneousBatch":
        """
        Pads features (if not already padded) and applies the channel combinator.
        """
        if self.status != "features":
            raise ValueError(
                f"Cannot apply channel combinator. Current status: {self.status}. Expected: 'features'"
            )

        if not self.processed_features:
            raise ValueError("Batch is empty, no features to process.")

        self._pad_features()
        assert self.padded_features is not None, "Padding failed to produce features"

        # Apply Combinator on the padded tensor
        self.padded_features = combinator.forward(self.padded_features)
        
        return self

    def _pad_features(self):
        """Internal helper to pad processed features to the maximum length."""
        if self.padded_features is not None:
            return
            
        max_len = max([x.shape[-1] for x in self.processed_features])
        feat_shape = self.processed_features[0].shape[:-1]
        B = len(self.processed_features)

        self.padded_features = torch.zeros(
            (B, *feat_shape, max_len), device=self.device, dtype=self.processed_features[0].dtype
        )
        self.mask = torch.zeros((B, max_len), device=self.device, dtype=torch.bool)

        for i, feat in enumerate(self.processed_features):
            length = feat.shape[-1]
            self.padded_features[i, ..., :length] = feat
            self.mask[i, :length] = True

    def apply_source_count_estimator(
        self, estimator: "BaseSourceCountEstimator"
    ) -> "HeterogeneousBatch":
        """
        Pads features (if not already), and runs the estimator.
        """
        if self.status != "features":
            raise ValueError(
                f"Cannot apply estimator. Current status: {self.status}. Expected: 'features'"
            )

        if not self.processed_features:
            raise ValueError("Batch is empty, no features to process.")

        self._pad_features()
        assert self.padded_features is not None, "Padding failed to produce features"

        # Run Estimator
        self.padded_estimates = estimator.forward(self.padded_features)

        # Unpad Estimates into List Form
        self.estimates = [
            pe[self.mask[i], :] for i, pe in enumerate(self.padded_estimates)
        ]

        self.status = "estimates"
        return self

    def compute_loss(self, loss_fn: "BaseLoss") -> dict[str, torch.Tensor]:
        """
        Computes loss using the internal estimates, targets, and mask.
        Handles the masking logic so the Loss function doesn't have to.
        """
        if self.status == "loss":
            return {"loss": self.loss}

        if self.status != "estimates" or self.padded_estimates is None:
            raise ValueError("Cannot compute loss. Estimates not available.")

        if "source_count" not in self.meta:
            raise ValueError(
                "Cannot compute loss. 'source_count' target missing in meta."
            )

        # 1. Get Predictions
        preds = self.padded_estimates  # (B, T_max, C)

        # 2. Prepare Targets (Pad to match T_max)
        targets_list = self.meta["source_count"]  # List of (Ti,)
        B, T_max = preds.shape[0], preds.shape[1]

        padded_targets = torch.zeros((B, T_max), device=self.device, dtype=torch.long)

        # We use the same mask we generated during feature padding
        # But we must ensure targets align with that mask
        for i, target in enumerate(targets_list):
            length = target.shape[0]
            # Check whether target length fits to feature length
            if self.mask is not None and not self.mask[i].sum() == length:
                raise ValueError(
                    f"Length mismatch for sample {i}: "
                    f"feature length={self.mask[i].sum().item()}, "
                    f"target length={length}."
                )
            valid_len = min(length, T_max)
            padded_targets[i, :valid_len] = target[:valid_len]

        # 3. Apply Masking (Flattening)
        # We select only the valid time steps for loss computation
        if self.mask is not None:
            preds_flat = preds.reshape(-1, preds.shape[-1])  # (B*T, C)
            targets_flat = padded_targets.reshape(-1)  # (B*T)
            mask_flat = self.mask.reshape(-1)  # (B*T)

            valid_preds = preds_flat[mask_flat].unsqueeze(0)  # (1, N_valid, C)
            valid_targets = targets_flat[mask_flat].unsqueeze(0)  # (1, N_valid)

            self.loss = loss_fn.compute_loss(valid_preds, valid_targets)
        else:
            self.loss = loss_fn.compute_loss(preds, padded_targets)

        self.status = "loss"
        return {"loss": self.loss}

    def print_summary(self):
        """
        Prints a summary of the batch contents and current status.
        """
        print("HeterogeneousBatch Summary:")
        print(f"  - Device: {self.device}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Status: {self.status}")
        print(f"  - Raw Audio Samples: {len(self.raw_audio)}")
        print(f"  - STFT Audio Samples: {len(self.stft_audio)}")
        print(f"  - Feature Samples: {len(self.features)}")
        print(f"  - Meta Keys: {list(self.meta.keys())}")
        if self.mask is not None:
            print(f"  - Mask Shape: {self.mask.shape}")
        if self.padded_features is not None:
            print(f"  - Padded Features Shape: {self.padded_features.shape}")
        if self.padded_estimates is not None:
            print(f"  - Padded Estimates Shape: {self.padded_estimates.shape}")
        if self.estimates is not None:
            print(f"  - Number of Estimate Samples: {len(self.estimates)}")
        print(50 * "-")
        if self.estimates is not None:
            HAs = self.compute_rtf_error()
            for bidx in range(self.batch_size):
                print(f"Sample {bidx}:")
                [
                    print(
                        f"RTFs: {list(est_rtf.shape)}\t SIDs: {est_sid[-1].tolist()}\t GT-SIDs: {gt_sid.tolist()}\t GT-RTFs: {list(gt_rtf.shape)}\t HA (mean, median, max): {ha}"
                    )
                    for est_rtf, est_sid, gt_rtf, gt_sid, ha in zip(
                        self.estimates[bidx][0],
                        self.estimates[bidx][1],
                        self.meta["gt_rtf_stream"][bidx],
                        self.meta["gt_ids_stream"][bidx],
                        HAs[bidx],
                    )
                ]
        else:
            for bidx in range(self.batch_size):
                [
                    print(f"GT-SIDs: {gt_sid.tolist()}\t GT-RTFs: {list(gt_rtf.shape)}")
                    for gt_rtf, gt_sid in zip(
                        self.meta["gt_rtf_stream"][bidx],
                        self.meta["gt_ids_stream"][bidx],
                    )
                ]

    def compute_rtf_error(self) -> list[list[tuple[float, float, float]]]:
        """
        Computes the mean, median, and max Hermitian angle between estimated and
        ground truth RTFs for each segment, accounting for source IDs.

        Returns:
            list[list[tuple[float, float, float]]]: A list (over samples) of lists (over segments)
            of tuples (mean_angle, median_angle, max_angle).
        """
        results = []
        if self.estimates is None:
            return results

        # Iterate over each sample in the batch
        for bidx in range(self.batch_size):
            sample_results = []

            # Check if we have data for this sample
            if (
                bidx >= len(self.estimates)
                or "gt_rtf_stream" not in self.meta
                or bidx >= len(self.meta["gt_rtf_stream"])
            ):
                results.append([])
                continue

            est_rtf_stream = self.estimates[bidx][0]
            est_sid_stream = self.estimates[bidx][1]
            gt_rtf_stream = self.meta["gt_rtf_stream"][bidx]
            gt_ids_stream = self.meta["gt_ids_stream"][bidx]

            # Iterate over segments within the sample
            for est_rtf, est_sid, gt_rtf, gt_sid in zip(
                est_rtf_stream, est_sid_stream, gt_rtf_stream, gt_ids_stream
            ):
                # Get IDs
                # est_sid is [T, K_est], we take the last time step to determine active IDs
                if isinstance(est_sid, torch.Tensor) and est_sid.ndim >= 2:
                    current_est_ids = est_sid[-1].tolist()
                else:
                    raise ValueError(
                        f"Unexpected shape for est_sid: {est_sid.shape if isinstance(est_sid, torch.Tensor) else 'N/A'}"
                    )
                # elif isinstance(est_sid, torch.Tensor):
                #     current_est_ids = est_sid.tolist()
                # elif isinstance(est_sid, (list, tuple)):
                #     current_est_ids = est_sid
                # else:
                #     current_est_ids = []

                if isinstance(gt_sid, torch.Tensor):
                    current_gt_ids = gt_sid.tolist()
                else:
                    raise ValueError(f"Unexpected type for gt_sid: {type(gt_sid)}")
                #
                # elif isinstance(gt_sid, (list, tuple)):
                #     current_gt_ids = gt_sid
                # else:
                #     current_gt_ids = []

                common_ids = set(current_est_ids) & set(current_gt_ids)

                segment_angles = []

                for uid in common_ids:
                    # Find indices
                    idx_est = current_est_ids.index(uid)
                    idx_gt = current_gt_ids.index(uid)

                    # Extract RTFs: [F, T, M]
                    vec_est = est_rtf[..., idx_est]
                    vec_gt = gt_rtf[..., idx_gt]

                    # Compute Hermitian angles
                    # hermitian_angle computes angle between vectors.
                    # Inputs are [F, T, M]. We want angle along dim M (last dim).
                    angles = hermitian_angle(vec_est, vec_gt, dim=-1)
                    segment_angles.append(angles.flatten())

                if segment_angles:
                    all_angles = torch.cat(segment_angles)

                    mean_val = all_angles.mean().item()
                    median_val = all_angles.median().item()
                    max_val = all_angles.max().item()

                    sample_results.append((mean_val, median_val, max_val))
                else:
                    sample_results.append((float("nan"), float("nan"), float("nan")))

            results.append(sample_results)

        return results


def ground_truth_rtf_stream(
    sad_frames: dict, rtfs: dict, segments: list[Segment]
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[int, str]]:
    """
    Computes the ground truth RTF stream and ID stream based on SAD frames.

    Args:
        sad_frames: A dictionary mapping source IDs to their SAD frame tensors.
        rtfs: A dictionary mapping source IDs to their RTF tensors.
        segments: A list of Segment objects defining the boundaries.

    Returns:
        gt_rtf_stream: A list of tensors (F, T_seg, M, K_seg) per segment.
        gt_ids_stream: A list of tensors (K_seg,) containing integer source IDs.
        id_map: A dictionary mapping integer IDs (0, 1...) to string source IDs.
    """
    gt_rtf_stream = []
    gt_ids_stream = []

    # Mappings for global ID tracking (Source String <-> Integer ID)
    str_to_int_id = {}
    next_global_id = 0

    # Helper to get device/dtype reference
    ref_tensor = next(iter(rtfs.values()))

    for seg in segments:
        start, end = seg.start, seg.end
        T_seg = end - start

        # Identify active sources in this segment
        active_source_ids = []

        # Use sorted keys so that the stacking order is deterministic.
        # This also ensures that if multiple sources appear for the first time
        # in the same segment, they are assigned IDs alphabetically.
        for source_id in sorted(sad_frames.keys()):
            if source_id == "noise":
                continue

            # Check if active in this window
            if sad_frames[source_id][start:end].sum() > 0:
                active_source_ids.append(source_id)

        K_seg = len(active_source_ids)

        # Determine Integer IDs for this segment.
        # Assign new IDs to first-time appearances.
        segment_int_ids = []
        for sid in active_source_ids:
            if sid not in str_to_int_id:
                str_to_int_id[sid] = next_global_id
                next_global_id += 1
            segment_int_ids.append(str_to_int_id[sid])

        if K_seg == 0:
            # No active sources: Shape (F, T_seg, M, 0)
            F = ref_tensor.shape[0]
            M = ref_tensor.shape[-2]  # Assuming (F, 1, M, 1)

            seg_rtfs = torch.zeros(
                (F, T_seg, M, 0), dtype=ref_tensor.dtype, device=ref_tensor.device
            )
            seg_ids = torch.tensor([], dtype=torch.long, device=ref_tensor.device)
        else:
            # Stack active RTFs
            stacked_rtfs_list = []
            for sid in active_source_ids:
                r = rtfs[sid]  # (F, 1, M, 1)

                # Tile over T_seg: (F, 1, M, 1) -> (F, T_seg, M, 1)
                r_expanded = r.expand(-1, T_seg, -1, -1)
                stacked_rtfs_list.append(r_expanded)

            # Concatenate along last dim -> (F, T_seg, M, K_seg)
            seg_rtfs = torch.cat(stacked_rtfs_list, dim=-1)

            # Create ID tensor -> (K_seg,)
            seg_ids = torch.tensor(
                segment_int_ids, dtype=torch.long, device=ref_tensor.device
            )

        gt_rtf_stream.append(seg_rtfs)
        gt_ids_stream.append(seg_ids)

    # Create output dictionary: Top-level Int -> Source String
    id_map = {v: k for k, v in str_to_int_id.items()}

    return gt_rtf_stream, gt_ids_stream, id_map
