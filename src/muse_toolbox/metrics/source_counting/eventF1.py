import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from muse_toolbox.metrics.base_metric import BaseMetric
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class TolerantEventF1(BaseMetric):
    """Full-recording Tolerant Event-Based F1 and Detection Delay metric.

    Accumulates stitched frame sequences per ``recording_id``, extracts
    multiplicity-aware elementary source activation and deactivation transitions,
    solves optimal 1D minimum-distance bipartite matching within
    ``tolerance_time``, computes precision/recall/F1 alongside detection delay
    statistics, and exposes transition-stratified per-event records via
    ``get_events_dataframe()``.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    requires_reference = True

    def __init__(
        self,
        transform: STFTtransform,
        tolerance_time: float = 1.0,
        past_tolerance_time: float = 0.0,
        future_tolerance_time: float | None = None,
        causal_mode: bool | str = "auto",
        estimator_causal: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the TolerantEventF1 metric.

        Supports symmetric matching for non-causal models
        (``[-tolerance_time, +tolerance_time]``) and asymmetric causal matching
        for strictly causal models (``[-past_tolerance_time, +future_tolerance_time]``,
        defaulting to ``[0.0, +1.0] s`` so premature transitions before true onsets
        cannot be paired with future ground-truth events).

        Args:
            transform (STFTtransform): STFT transform used for frame/time conversions.
            tolerance_time (float, optional): Symmetric / default future tolerance
                window in seconds. Defaults to ``1.0``.
            past_tolerance_time (float, optional): Allowed look-ahead tolerance before
                a true event when ``is_causal_eval=True`` (in seconds). Defaults to
                ``0.0``.
            future_tolerance_time (float | None, optional): Optional explicit future
                tolerance window in seconds overriding ``tolerance_time``. Defaults to
                ``None``.
            causal_mode (bool | str, optional): Evaluation matching mode
                (``"auto"``, ``"causal"``/``True``, or ``"non_causal"``/``False``).
                When ``"auto"``, resolves from ``estimator_causal``. Defaults to
                ``"auto"``.
            estimator_causal (bool, optional): Underlying estimator causality flag
                forwarded by ``OSCmodule`` when ``causal_mode="auto"``. Defaults to
                ``False``.
            *args (Any): Variable length arguments passed to ``BaseMetric``.
            **kwargs (Any): Arbitrary keyword arguments passed to ``BaseMetric``.

        Raises:
            ValueError: If ``causal_mode`` is invalid or any tolerance is negative.
        """
        super().__init__(*args, requires_numpy=False, **kwargs)
        if isinstance(causal_mode, bool):
            self.is_causal_eval = causal_mode
        elif isinstance(causal_mode, str):
            mode_norm = causal_mode.strip().lower()
            if mode_norm == "auto":
                self.is_causal_eval = bool(estimator_causal)
            elif mode_norm in {"causal", "true"}:
                self.is_causal_eval = True
            elif mode_norm in {"non_causal", "noncausal", "false"}:
                self.is_causal_eval = False
            else:
                raise ValueError(
                    f"Unsupported causal_mode='{causal_mode}'. "
                    "Expected bool, 'auto', 'causal', or 'non_causal'."
                )
        else:
            raise ValueError(
                f"Unsupported causal_mode='{causal_mode}'. "
                "Expected bool, 'auto', 'causal', or 'non_causal'."
            )

        self.causal_mode = causal_mode
        self.transform = transform
        self.future_tolerance_time = (
            float(future_tolerance_time)
            if future_tolerance_time is not None
            else float(tolerance_time)
        )
        self.tolerance_time = self.future_tolerance_time
        self.past_tolerance_time = (
            float(past_tolerance_time)
            if self.is_causal_eval
            else self.future_tolerance_time
        )
        if self.future_tolerance_time < 0.0 or self.past_tolerance_time < 0.0:
            raise ValueError(
                "TolerantEventF1 tolerance windows must be non-negative, got "
                f"past_tolerance_time={self.past_tolerance_time}, "
                f"future_tolerance_time={self.future_tolerance_time}."
            )

        self.future_tolerance = int(
            self.transform.times2frames(self.future_tolerance_time).item()
        )
        self.past_tolerance = int(
            self.transform.times2frames(self.past_tolerance_time).item()
        )
        self.tolerance = self.future_tolerance
        self.frame_shift = float(self.transform.frame_shift)

        self._recording_preds: dict[str, list[torch.Tensor]] = {}
        self._recording_targets: dict[str, list[torch.Tensor]] = {}
        self._recording_num_mics: dict[str, int | None] = {}
        self._recording_order: list[str] = []

        self._cache_dirty: bool = True
        self._cached_results: dict[str, torch.Tensor] | None = None
        self._cached_events_df: pd.DataFrame | None = None

    @staticmethod
    def _extract_elementary_events(
        count_sequence: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extracts multiplicity-aware unit activation and deactivation events.

        A transition ``a -> b`` at frame ``t`` of step magnitude ``|b - a|`` expands
        into ``|b - a|`` elementary unit transitions sharing timestamp ``t`` and
        preserving both their unit transition label (``k -> k+1`` or ``k -> k-1``)
        and parent raw jump label (``a -> b``). Frame ``t = 0`` is never counted as a
        transition because differences use ``prepend=count_sequence[:1]``.

        Args:
            count_sequence (torch.Tensor): 1D integer source-count sequence ``(T,)``.

        Returns:
            tuple[list[dict[str, Any]], list[dict[str, Any]]]: Tuple of
                ``(activation_events, deactivation_events)``.
        """
        act_events: list[dict[str, Any]] = []
        deact_events: list[dict[str, Any]] = []
        if count_sequence.numel() <= 1:
            return act_events, deact_events

        diffs = torch.diff(count_sequence, prepend=count_sequence[:1])
        change_indices = (diffs != 0).nonzero(as_tuple=False).squeeze(1)

        for idx_tensor in change_indices:
            frame_idx = int(idx_tensor.item())
            k_prev = int(count_sequence[frame_idx - 1].item())
            k_next = int(count_sequence[frame_idx].item())
            raw_jump = f"{k_prev}->{k_next}"
            if k_next > k_prev:
                for k in range(k_prev, k_next):
                    act_events.append(
                        {
                            "frame": frame_idx,
                            "k_from": k,
                            "step": f"{k}->{k + 1}",
                            "raw_jump": raw_jump,
                        }
                    )
            elif k_next < k_prev:
                for k in range(k_prev, k_next, -1):
                    deact_events.append(
                        {
                            "frame": frame_idx,
                            "k_from": k,
                            "step": f"{k}->{k - 1}",
                            "raw_jump": raw_jump,
                        }
                    )

        return act_events, deact_events

    def _match_events_optimal(
        self,
        recording_id: str,
        event_type: str,
        pred_events: list[dict[str, Any]],
        true_events: list[dict[str, Any]],
    ) -> tuple[int, int, int, list[float], list[dict[str, Any]]]:
        """Solves minimum-distance 1D bipartite matching within ``self.tolerance``.

        Args:
            recording_id (str): Identifier of the continuous recording.
            event_type (str): Event category (``"act"`` or ``"deact"``).
            pred_events (list[dict[str, Any]]): Extracted predicted events.
            true_events (list[dict[str, Any]]): Extracted ground-truth events.

        Returns:
            tuple[int, int, int, list[float], list[dict[str, Any]]]:
                ``(tp, fp, fn, matched_delays_s, event_rows)``.
        """
        event_rows: list[dict[str, Any]] = []
        delays_s: list[float] = []

        n_pred = len(pred_events)
        n_true = len(true_events)
        matched_pred: set[int] = set()
        matched_true: set[int] = set()

        if n_pred > 0 and n_true > 0:
            forbidden_cost = 1e9
            cost = np.full((n_pred, n_true), forbidden_cost, dtype=np.float64)
            for i, pe in enumerate(pred_events):
                p_frame = int(pe["frame"])
                p_k = int(pe["k_from"])
                for j, te in enumerate(true_events):
                    t_frame = int(te["frame"])
                    delta_frames = p_frame - t_frame
                    if -self.past_tolerance <= delta_frames <= self.future_tolerance:
                        t_k = int(te["k_from"])
                        cost[i, j] = float(abs(delta_frames)) + 1e-4 * float(
                            abs(p_k - t_k)
                        )

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                pe = pred_events[r]
                te = true_events[c]
                delta_frames = int(pe["frame"]) - int(te["frame"])
                if -self.past_tolerance <= delta_frames <= self.future_tolerance:
                    matched_pred.add(r)
                    matched_true.add(c)
                    t_frame = int(te["frame"])
                    p_frame = int(pe["frame"])
                    t_time = t_frame * self.frame_shift
                    p_time = p_frame * self.frame_shift
                    delay = p_time - t_time
                    delays_s.append(delay)
                    event_rows.append(
                        {
                            "recording_id": recording_id,
                            "event_type": event_type,
                            "status": "TP",
                            "true_frame": t_frame,
                            "pred_frame": p_frame,
                            "true_time_s": t_time,
                            "pred_time_s": p_time,
                            "delay_s": delay,
                            "true_step": te["step"],
                            "true_raw_jump": te["raw_jump"],
                            "pred_step": pe["step"],
                            "pred_raw_jump": pe["raw_jump"],
                        }
                    )

        for j, te in enumerate(true_events):
            if j not in matched_true:
                t_frame = int(te["frame"])
                event_rows.append(
                    {
                        "recording_id": recording_id,
                        "event_type": event_type,
                        "status": "FN",
                        "true_frame": t_frame,
                        "pred_frame": None,
                        "true_time_s": t_frame * self.frame_shift,
                        "pred_time_s": np.nan,
                        "delay_s": np.nan,
                        "true_step": te["step"],
                        "true_raw_jump": te["raw_jump"],
                        "pred_step": None,
                        "pred_raw_jump": None,
                    }
                )

        for i, pe in enumerate(pred_events):
            if i not in matched_pred:
                p_frame = int(pe["frame"])
                event_rows.append(
                    {
                        "recording_id": recording_id,
                        "event_type": event_type,
                        "status": "FP",
                        "true_frame": None,
                        "pred_frame": p_frame,
                        "true_time_s": np.nan,
                        "pred_time_s": p_frame * self.frame_shift,
                        "delay_s": np.nan,
                        "true_step": None,
                        "true_raw_jump": None,
                        "pred_step": pe["step"],
                        "pred_raw_jump": pe["raw_jump"],
                    }
                )

        tp = len(matched_pred)
        fp = n_pred - tp
        fn = n_true - tp
        return tp, fp, fn, delays_s, event_rows

    @staticmethod
    def _summarize_delays(
        delays: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes mean, median, IQR, and mean-absolute detection delay in seconds.

        Args:
            delays (list[float]): Signed detection delays ``pred_time_s - true_time_s``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Scalar float32
                tensors ``(mean, median, iqr, abs_mean)``.
        """
        if not delays:
            zero = torch.tensor(0.0, dtype=torch.float32)
            return zero, zero, zero, zero
        arr = torch.tensor(delays, dtype=torch.float32)
        mean_val = arr.mean()
        median_val = arr.median()
        q75 = torch.quantile(arr, 0.75)
        q25 = torch.quantile(arr, 0.25)
        iqr_val = q75 - q25
        abs_mean_val = arr.abs().mean()
        return mean_val, median_val, iqr_val, abs_mean_val

    def update(
        self,
        preds: list[torch.Tensor] | torch.Tensor,
        targets: list[torch.Tensor] | torch.Tensor,
        meta: dict[str, Any],
        dataloader_idx: int,
    ) -> None:
        """Accumulates stitched frame predictions and ground-truth counts by ``recording_id``.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): Prediction logit or class tensors.
            targets (list[torch.Tensor] | torch.Tensor): Ground-truth count tensors.
            meta (dict[str, Any]): Metadata dictionary containing ``recording_id`` or
                ``scenario_id``.
            dataloader_idx (int): Active dataloader index.
        """
        if self.requires_reference and targets is None:
            return

        pred_list = [preds] if isinstance(preds, torch.Tensor) else list(preds)
        target_list = [targets] if isinstance(targets, torch.Tensor) else list(targets)
        if not pred_list:
            return

        raw_rec_ids = meta.get("recording_id") or meta.get("scenario_id")
        if isinstance(raw_rec_ids, list) and len(raw_rec_ids) == len(pred_list):
            rec_ids = [str(rid) for rid in raw_rec_ids]
        else:
            rec_ids = [f"rec_{dataloader_idx}_{i}" for i in range(len(pred_list))]

        for idx, (p, t, rec_id) in enumerate(zip(pred_list, target_list, rec_ids)):
            pred_count = torch.argmax(p, dim=-1) if p.ndim >= 2 else p
            p_cpu = pred_count.detach().reshape(-1).to(dtype=torch.int64, device="cpu")
            t_cpu = t.detach().reshape(-1).to(dtype=torch.int64, device="cpu")
            min_len = min(int(p_cpu.numel()), int(t_cpu.numel()))
            if min_len <= 0:
                continue

            if rec_id not in self._recording_preds:
                self._recording_preds[rec_id] = []
                self._recording_targets[rec_id] = []
                self._recording_order.append(rec_id)
                self._recording_num_mics[rec_id] = self._extract_item_num_mics(
                    meta, idx
                )

            self._recording_preds[rec_id].append(p_cpu[:min_len])
            self._recording_targets[rec_id].append(t_cpu[:min_len])

        self._cache_dirty = True

    @staticmethod
    def _extract_item_num_mics(meta: dict[str, Any], idx: int) -> int | None:
        """Extracts integer ``num_mics`` for batch item ``idx`` if present."""
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

    def _compute_for_recording_ids(
        self, recording_ids: list[str]
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        """Computes aggregated F1, Delay, and per-event rows for a list of recordings."""
        tp_act = fp_act = fn_act = 0
        tp_deact = fp_deact = fn_deact = 0
        delays_act: list[float] = []
        delays_deact: list[float] = []
        all_rows: list[dict[str, Any]] = []

        for rec_id in recording_ids:
            full_pred = torch.cat(self._recording_preds[rec_id], dim=0)
            full_true = torch.cat(self._recording_targets[rec_id], dim=0)
            nm = self._recording_num_mics.get(rec_id)

            pred_acts, pred_deacts = self._extract_elementary_events(full_pred)
            true_acts, true_deacts = self._extract_elementary_events(full_true)

            a_tp, a_fp, a_fn, a_del, a_rows = self._match_events_optimal(
                rec_id, "act", pred_acts, true_acts
            )
            d_tp, d_fp, d_fn, d_del, d_rows = self._match_events_optimal(
                rec_id, "deact", pred_deacts, true_deacts
            )
            for row in a_rows:
                row["num_mics"] = nm
            for row in d_rows:
                row["num_mics"] = nm

            tp_act += a_tp
            fp_act += a_fp
            fn_act += a_fn
            tp_deact += d_tp
            fp_deact += d_fp
            fn_deact += d_fn
            delays_act.extend(a_del)
            delays_deact.extend(d_del)
            all_rows.extend(a_rows)
            all_rows.extend(d_rows)

        results: dict[str, torch.Tensor] = {}
        for name, tp_i, fp_i, fn_i in (
            ("act", tp_act, fp_act, fn_act),
            ("deact", tp_deact, fp_deact, fn_deact),
            (
                "overall",
                tp_act + tp_deact,
                fp_act + fp_deact,
                fn_act + fn_deact,
            ),
        ):
            tp_t = torch.tensor(float(tp_i), dtype=torch.float32)
            fp_t = torch.tensor(float(fp_i), dtype=torch.float32)
            fn_t = torch.tensor(float(fn_i), dtype=torch.float32)
            prec = tp_t / (tp_t + fp_t + 1e-6)
            rec = tp_t / (tp_t + fn_t + 1e-6)
            f1 = 2.0 * (prec * rec) / (prec + rec + 1e-6)
            results[f"precision_{name}"] = prec
            results[f"recall_{name}"] = rec
            results[f"F1score_{name}"] = f1

        for name, d_list in (
            ("act", delays_act),
            ("deact", delays_deact),
            ("overall", delays_act + delays_deact),
        ):
            d_mean, d_med, d_iqr, d_abs = self._summarize_delays(d_list)
            results[f"delay_mean_{name}"] = d_mean
            results[f"delay_median_{name}"] = d_med
            results[f"delay_iqr_{name}"] = d_iqr
            results[f"delay_abs_mean_{name}"] = d_abs

        return results, all_rows

    def _evaluate_all(self) -> None:
        """Concatenates recordings and computes idempotent F1, Delay, and event table."""
        if not self._cache_dirty and self._cached_results is not None:
            return

        results, all_rows = self._compute_for_recording_ids(self._recording_order)
        self._cached_results = results
        self._cached_events_df = pd.DataFrame(all_rows) if all_rows else None
        self._cache_dirty = False

    def compute_by_num_mics(self) -> dict[int, dict[str, float]]:
        """Computes event F1 and detection delay metrics stratified by ``num_mics``.

        Returns:
            dict[int, dict[str, float]]: Sorted mapping ``{num_mics: metrics_dict}``.
        """
        grouped: dict[int, list[str]] = {}
        for rec_id in self._recording_order:
            nm = self._recording_num_mics.get(rec_id)
            if nm is None:
                continue
            grouped.setdefault(nm, []).append(rec_id)

        out: dict[int, dict[str, float]] = {}
        for nm in sorted(grouped.keys()):
            res, _ = self._compute_for_recording_ids(grouped[nm])
            out[nm] = {k: float(v.item()) for k, v in res.items()}
        return out

    def compute(self) -> dict[str, torch.Tensor]:
        """Computes aggregated precision, recall, F1, and detection delay metrics.

        Returns:
            dict[str, torch.Tensor]: Event F1 and delay summary scalar tensors.
        """
        self._evaluate_all()
        assert self._cached_results is not None
        return dict(self._cached_results)

    def get_dataframe(self) -> pd.DataFrame | None:
        """Returns ``None`` so chunk-level ``test_metrics.csv`` aligns with frame metrics.

        Per-event and transition-stratified delay records are exposed separately via
        ``get_events_dataframe()``.

        Returns:
            pd.DataFrame | None: Always ``None``.
        """
        return None

    def get_events_dataframe(self) -> pd.DataFrame | None:
        """Returns the transition-stratified per-event log and detection delays.

        Returns:
            pd.DataFrame | None: DataFrame with columns ``recording_id``, ``event_type``,
                ``status``, ``true_frame``, ``pred_frame``, ``true_time_s``,
                ``pred_time_s``, ``delay_s``, ``true_step``, ``true_raw_jump``,
                ``pred_step``, and ``pred_raw_jump``, or ``None`` if no events occurred.
        """
        self._evaluate_all()
        if self._cached_events_df is None or self._cached_events_df.empty:
            return None
        return self._cached_events_df.copy()

    def reset(self) -> None:
        """Resets accumulated recording buffers and cached event evaluations."""
        super().reset()
        self._recording_preds = {}
        self._recording_targets = {}
        self._recording_num_mics = {}
        self._recording_order = []
        self._cache_dirty = True
        self._cached_results = None
        self._cached_events_df = None
