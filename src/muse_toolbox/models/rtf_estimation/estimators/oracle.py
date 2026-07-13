import torch
import logging
from typing import Any

from .base_rtf_estimator import BaseRTFestimator
from muse_toolbox.utils import Segment, randdir, hermitian_angle

log = logging.getLogger(__name__)


class Oracle(BaseRTFestimator):
    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration dictionary for the Oracle estimator.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {"type": "Oracle"}

    def forward_(
        self,
        Rn: torch.Tensor,
        Ry: torch.Tensor,
        G: torch.Tensor,
        Rv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Internal forward method. Oracle estimator requires metadata kwargs instead.
        """
        raise NotImplementedError(
            "Oracle estimator requires metadata kwargs (sad_frames, oracle_rtfs, etc.) "
            "and should be called via forward() with these arguments."
        )

    def forward(
        self,
        noise_cov: torch.Tensor,
        noisy_cov: torch.Tensor,
        old_rtfs: torch.Tensor,
        old_noisy_cov: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Oracle implementation for majority vote.
        Handles both batched (B, F, T, M, K) and unbatched (F, M, K) old_rtfs inputs.
        """

        if "segment" in kwargs and "sad_frames" in kwargs and "oracle_rtfs" in kwargs:
            segment: Segment = kwargs.pop("segment")
            sad_frames: dict[str, torch.Tensor] = kwargs.pop("sad_frames")
            oracle_rtfs: dict[str, torch.Tensor] = kwargs.pop("oracle_rtfs")
        else:
            raise ValueError(
                "Missing required keyword arguments 'segment', 'sad_frames', and 'oracle_rtfs' for Oracle."
            )

        is_batched = old_rtfs.ndim == 5

        if is_batched:
            raise NotImplementedError(
                "Batched forward_oracle not fully implemented/tested yet. Use batch_size=1 or RTFModule."
            )

        # Unbatched case:
        # old_rtfs: (F, 1, M, K_old)
        assert (
            old_rtfs.ndim == 4
        ), "Expected old_rtfs to have 4 dimensions (F, 1, M, K_old) in unbatched mode."

        # 1. Compute Activity Counts
        active_sources = []
        for k, sf in sad_frames.items():
            if k == "noise":
                continue
            count = sf[segment.start : segment.end].sum()
            if count > 0:
                active_sources.append((k, count))

        if not active_sources:
            raise ValueError("No active sources found in metadata for this segment.")

        # 2. Unzip into synchronized lists
        # source_ids[i] is guaranteed to correspond to counts[i]
        source_ids = [k for k, v in active_sources]
        counts = torch.stack([v for k, v in active_sources])

        # 3. Perform Top-K selection
        F, _, M, K_old = old_rtfs.shape
        k_target = min(K_old + 1, len(source_ids))

        _, topk_indices = torch.topk(counts, k_target)

        # 4. Map indices back to source IDs safely
        top_k_source_ids = [source_ids[i] for i in topk_indices.tolist()]

        # 5. Extract Candidates
        # Stack candidates: (F, k_target, M)
        seg_oracle_rtfs = torch.cat(
            [oracle_rtfs[sid] for sid in top_k_source_ids], dim=-1
        )

        T_seg = segment.end - segment.start

        return self._select_new_rtf(seg_oracle_rtfs, old_rtfs).repeat(1, T_seg, 1, 1)

    def _select_new_rtf(
        self, seg_oracle_rtfs: torch.Tensor, old_rtfs: torch.Tensor
    ) -> torch.Tensor:
        """
        Selects the best new RTF from oracle candidates to add to the existing set.

        Logic:
           - Candidates = seg_oracle_rtfs [F, 1, M, K_oracle]
           - Existing   = old_rtfs        [F, 1, M, K_old]
           - We want the candidate 'c' that maximizes the minimum distance to any existing 's'.
             i.e. argmax_{c} ( min_{s} (Angle(c, s)) )

        Handles edge cases where K_oracle < K_est (hallucination required) or K_old=0.
        """
        F, _, M, K_oracle = seg_oracle_rtfs.shape
        K_old = old_rtfs.shape[-1]
        dtype = seg_oracle_rtfs.dtype
        device = seg_oracle_rtfs.device

        # --- Edge Case 1: No Oracle Candidates ---
        # The estimator thinks there is a new source, but oracle says silence.
        # Must return a valid shaped tensor (random direction).
        if K_oracle == 0:
            return randdir(F, 1, M, 1, device=device, dtype=dtype)

        # --- Edge Case 2: First Source Activation ---
        # No existing sources to compare. Pick the most active one (index 0).
        if K_old == 0:
            # Return the first candidate
            return seg_oracle_rtfs[..., 0:1]

        # --- Standard Case: Max-Min Distance Selection ---

        # 1. Prepare for broadcasting
        # We want to compare every Candidate (C) vs every Reference (R)
        # Vector dimension is M (last dimension after permute/squeeze)

        # Candidate: [F, 1, M, K_oracle] -> [F, K_oracle, M] -> [F, K_oracle, 1, M]
        c = seg_oracle_rtfs.squeeze(1).permute(0, 2, 1).unsqueeze(2)

        # Reference: [F, 1, M, K_old]    -> [F, K_old, M]    -> [F, 1, K_old, M]
        r = old_rtfs.squeeze(1).permute(0, 2, 1).unsqueeze(1)

        # 2. Compute Angles [F, K_oracle, K_old]
        # Measures similarity between every candidate and every existing source
        angles = hermitian_angle(c, r, dim=-1)

        # 3. Aggregation
        # robustness: Average angle over Frequencies to get a single scalar score
        # [K_oracle, K_old]
        mean_angles = angles.mean(dim=-4).squeeze(-1)

        # 4. Novelty Score
        # For each candidate, find how close it is to its *closest* neighbor in old_rtfs.
        # [K_oracle]
        min_dist_to_existing, _ = mean_angles.min(dim=-1)

        # 5. Selection
        # Pick the candidate that is furthest away from the set of existing sources.
        best_candidate_idx = min_dist_to_existing.argmax()
        max_novelty = min_dist_to_existing[best_candidate_idx]

        # --- Edge Case 3: Over-Estimation / Duplication ---
        # If the comprehensive Oracle set is smaller than K_est, or if K_est overestimates,
        # the "best" candidate might actually be a duplicate of an existing source.
        # If the novelty is near zero, we shouldn't add a duplicate (causes singularity).
        # Instead, we add a random vector to fill the slot harmlessly.
        NOVELTY_THRESHOLD = 1e-2  # approx 0.5 degrees

        if max_novelty < NOVELTY_THRESHOLD:
            return randdir(F, 1, M, 1, device=device, dtype=dtype)

        # Return the selected candidate [F, 1, M, 1]
        return seg_oracle_rtfs[..., best_candidate_idx : best_candidate_idx + 1]
