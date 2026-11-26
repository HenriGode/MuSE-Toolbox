import logging
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_average_precision

from muse_toolbox.metrics.base_metric import BaseMetric

log = logging.getLogger(__name__)


class AveragePrecision(BaseMetric):
    """Corpus-pooled multi-class Average Precision (AP) metric for source counting.

    Computes one-vs-rest precision-recall curve area (AP) for each discrete source-count
    class ``k in {0, ..., K_max}`` directly from frame-level softmax posterior
    probabilities matching Cornell et al. (2022) Eq. (4), along with macro mean ``mAP``.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    requires_reference = True

    def __init__(self, max_sources: int = 4, *args: Any, **kwargs: Any) -> None:
        """Initializes the AveragePrecision metric.

        Args:
            max_sources (int, optional): Expected maximum simultaneous sources
                (determines base class count ``max_sources + 1``). Defaults to ``4``.
            *args (Any): Variable length arguments passed to ``BaseMetric``.
            **kwargs (Any): Arbitrary keyword arguments passed to ``BaseMetric``.
        """
        super().__init__(*args, requires_numpy=False, **kwargs)
        self.max_sources = int(max_sources)
        self.num_classes = self.max_sources + 1
        self.prob_chunks: list[torch.Tensor] = []
        self.target_chunks: list[torch.Tensor] = []
        self.num_mics_chunks: list[int | None] = []

    @staticmethod
    def _extract_chunk_num_mics(meta: dict[str, Any], idx: int) -> int | None:
        """Extracts integer ``num_mics`` for item ``idx`` from batch metadata."""
        if not isinstance(meta, dict):
            return None
        val: Any = None
        top_mics = meta.get("num_mics")
        if isinstance(top_mics, list) and idx < len(top_mics):
            val = top_mics[idx]
        elif isinstance(top_mics, torch.Tensor) and idx < top_mics.numel():
            val = int(top_mics.reshape(-1)[idx].item())
        else:
            sp_list = meta.get("scenario_params")
            if isinstance(sp_list, list) and idx < len(sp_list) and isinstance(sp_list[idx], dict):
                val = sp_list[idx].get("num_mics")
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def update(
        self,
        preds: list[torch.Tensor] | torch.Tensor,
        targets: list[torch.Tensor] | torch.Tensor,
        meta: dict[str, Any],
        dataloader_idx: int,
    ) -> None:
        """Accumulates frame-level softmax posterior probabilities and target counts.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): Unnormalized logit tensors of
                shape ``(T_i, C)``.
            targets (list[torch.Tensor] | torch.Tensor): Integer ground-truth count
                tensors of shape ``(T_i,)``.
            meta (dict[str, Any]): Batch metadata dictionary.
            dataloader_idx (int): Active dataloader index (unused).
        """
        _ = dataloader_idx
        if self.requires_reference and targets is None:
            return

        pred_list = [preds] if isinstance(preds, torch.Tensor) else list(preds)
        target_list = [targets] if isinstance(targets, torch.Tensor) else list(targets)

        for idx, (p, t) in enumerate(zip(pred_list, target_list)):
            if p.numel() == 0 or t.numel() == 0:
                continue
            probs = torch.softmax(p.detach().float().cpu(), dim=-1)
            t_flat = t.detach().long().cpu().reshape(-1)
            min_len = min(int(probs.shape[0]), int(t_flat.numel()))
            if min_len <= 0:
                continue
            self.prob_chunks.append(probs[:min_len])
            self.target_chunks.append(t_flat[:min_len])
            self.num_mics_chunks.append(self._extract_chunk_num_mics(meta, idx))

    def _compute_ap_from_chunks(
        self,
        prob_chunks: list[torch.Tensor],
        target_chunks: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Computes per-class AP, ``mAP``, ``AP_VAD``, and ``AP_OSD`` from chunk lists.

        Args:
            prob_chunks (list[torch.Tensor]): Softmax probability tensors ``(T_i, C)``.
            target_chunks (list[torch.Tensor]): Integer count tensors ``(T_i,)``.

        Returns:
            dict[str, torch.Tensor]: Scalar float32 tensors for ``AP_{k}spk``,
                ``mAP``, ``AP_VAD``, and ``AP_OSD``.
        """
        if not target_chunks:
            results = {
                f"AP_{k}spk": torch.tensor(0.0, dtype=torch.float32)
                for k in range(self.num_classes)
            }
            results["mAP"] = torch.tensor(0.0, dtype=torch.float32)
            results["AP_VAD"] = torch.tensor(0.0, dtype=torch.float32)
            results["AP_OSD"] = torch.tensor(0.0, dtype=torch.float32)
            return results

        all_targets = torch.cat(target_chunks, dim=0)
        if all_targets.numel() == 0:
            results = {
                f"AP_{k}spk": torch.tensor(0.0, dtype=torch.float32)
                for k in range(self.num_classes)
            }
            results["mAP"] = torch.tensor(0.0, dtype=torch.float32)
            results["AP_VAD"] = torch.tensor(0.0, dtype=torch.float32)
            results["AP_OSD"] = torch.tensor(0.0, dtype=torch.float32)
            return results

        max_target = int(all_targets.max().item())
        max_pred_cols = max(int(p.shape[-1]) for p in prob_chunks)
        effective_num_classes = max(self.num_classes, max_target + 1, max_pred_cols)

        padded_probs: list[torch.Tensor] = []
        for p in prob_chunks:
            pad_cols = effective_num_classes - int(p.shape[-1])
            if pad_cols > 0:
                padded_probs.append(F.pad(p, (0, pad_cols), value=0.0))
            else:
                padded_probs.append(p)
        all_probs = torch.cat(padded_probs, dim=0)

        results: dict[str, torch.Tensor] = {}
        present_aps: list[torch.Tensor] = []

        for k in range(effective_num_classes):
            binary_target = (all_targets == k).to(torch.int64)
            if int(binary_target.sum().item()) == 0:
                ap_val = torch.tensor(0.0, dtype=torch.float32)
            else:
                ap_raw = binary_average_precision(
                    all_probs[:, k], binary_target, thresholds=None
                )
                ap_val = torch.nan_to_num(ap_raw, nan=0.0).to(torch.float32)
                present_aps.append(ap_val)
            results[f"AP_{k}spk"] = ap_val

        results["mAP"] = (
            torch.stack(present_aps).mean()
            if present_aps
            else torch.tensor(0.0, dtype=torch.float32)
        )

        # Derived binary VAD (k >= 1) and OSD (k >= 2) Average Precision
        vad_target = (all_targets >= 1).to(torch.int64)
        if int(vad_target.sum().item()) > 0 and all_probs.shape[1] > 1:
            vad_prob = all_probs[:, 1:].sum(dim=-1).clamp(0.0, 1.0)
            vad_raw = binary_average_precision(vad_prob, vad_target, thresholds=None)
            results["AP_VAD"] = torch.nan_to_num(vad_raw, nan=0.0).to(torch.float32)
        else:
            results["AP_VAD"] = torch.tensor(0.0, dtype=torch.float32)

        osd_target = (all_targets >= 2).to(torch.int64)
        if int(osd_target.sum().item()) > 0 and all_probs.shape[1] > 2:
            osd_prob = all_probs[:, 2:].sum(dim=-1).clamp(0.0, 1.0)
            osd_raw = binary_average_precision(osd_prob, osd_target, thresholds=None)
            results["AP_OSD"] = torch.nan_to_num(osd_raw, nan=0.0).to(torch.float32)
        else:
            results["AP_OSD"] = torch.tensor(0.0, dtype=torch.float32)

        return results

    def compute(self) -> dict[str, torch.Tensor]:
        """Computes per-class AP scores, ``mAP``, ``AP_VAD``, and ``AP_OSD``.

        Returns:
            dict[str, torch.Tensor]: Mapping containing ``AP_{k}spk``, ``mAP``,
                ``AP_VAD``, and ``AP_OSD`` across accumulated frames.
        """
        return self._compute_ap_from_chunks(self.prob_chunks, self.target_chunks)

    def compute_by_num_mics(self) -> dict[int, dict[str, float]]:
        """Computes pooled AP metrics stratified by integer ``num_mics`` metadata.

        Returns:
            dict[int, dict[str, float]]: Sorted mapping ``{num_mics: metrics_dict}``.
        """
        grouped_probs: dict[int, list[torch.Tensor]] = {}
        grouped_targets: dict[int, list[torch.Tensor]] = {}
        for probs, targets, nm in zip(
            self.prob_chunks, self.target_chunks, self.num_mics_chunks
        ):
            if nm is None:
                continue
            grouped_probs.setdefault(nm, []).append(probs)
            grouped_targets.setdefault(nm, []).append(targets)

        out: dict[int, dict[str, float]] = {}
        for nm in sorted(grouped_probs.keys()):
            res = self._compute_ap_from_chunks(
                grouped_probs[nm], grouped_targets[nm]
            )
            out[nm] = {k: float(v.item()) for k, v in res.items()}
        return out

    def get_dataframe(self) -> pd.DataFrame | None:
        """Corpus-level AP does not produce a per-scenario row table.

        Returns:
            pd.DataFrame | None: Always ``None``.
        """
        return None

    def reset(self) -> None:
        """Resets accumulated probability, target, and microphone-count buffers."""
        super().reset()
        self.prob_chunks = []
        self.target_chunks = []
        self.num_mics_chunks = []
