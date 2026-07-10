import torch
import pandas as pd
from muse_toolbox.metrics.common.base_metric import BaseMetric
from muse_toolbox.utils import STFTtransform
from typing import Optional, List


class TolerantEventF1(BaseMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False  # Important for list states
    requires_reference = True

    # States for aggregated metrics
    tp_act: torch.Tensor
    fp_act: torch.Tensor
    fn_act: torch.Tensor
    tp_deact: torch.Tensor
    fp_deact: torch.Tensor
    fn_deact: torch.Tensor

    # States for per-sample dataframe (must be lists of tensors)
    per_sample_tp_act: List[torch.Tensor]
    per_sample_fp_act: List[torch.Tensor]
    per_sample_fn_act: List[torch.Tensor]
    per_sample_tp_deact: List[torch.Tensor]
    per_sample_fp_deact: List[torch.Tensor]
    per_sample_fn_deact: List[torch.Tensor]

    def __init__(
        self, tolerance_time: float, transform: STFTtransform, *args, **kwargs
    ):
        super().__init__(*args, requires_numpy=False, **kwargs)
        self.tolerance_time = tolerance_time
        self.transform = transform
        self.tolerance = self.transform.times2frames(self.tolerance_time)

        # States for aggregated metrics
        for name in ["tp_act", "fp_act", "fn_act", "tp_deact", "fp_deact", "fn_deact"]:
            self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")

        # States for per-sample results
        for name in [
            "per_sample_tp_act",
            "per_sample_fp_act",
            "per_sample_fn_act",
            "per_sample_tp_deact",
            "per_sample_fp_deact",
            "per_sample_fn_deact",
        ]:
            self.add_state(name, default=[], dist_reduce_fx="cat")

        # Store scenario_ids as a regular attribute
        self.scenario_ids = []

    def _find_events(self, count_sequence: torch.Tensor):
        # ... (implementation unchanged)
        diffs = torch.diff(count_sequence, prepend=count_sequence.new_zeros(1))
        activations = (diffs > 0).nonzero(as_tuple=False).squeeze(1)
        deactivations = (diffs < 0).nonzero(as_tuple=False).squeeze(1)
        return activations, deactivations

    def _match_events(self, pred_events: torch.Tensor, true_events: torch.Tensor):
        # ... (implementation unchanged)
        tp = 0
        matches = torch.zeros_like(true_events, dtype=torch.bool)
        if pred_events.numel() == 0 or true_events.numel() == 0:
            return tp, matches

        for pred_event in pred_events:
            distances = torch.abs(true_events - pred_event)
            best_match_idx = torch.argmin(distances)
            if (
                distances[best_match_idx] <= self.tolerance
                and not matches[best_match_idx]
            ):
                tp += 1
                matches[best_match_idx] = True
        return tp, matches

    def update(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        meta: dict,
        dataloader_idx: int,
    ):
        pred_counts = [torch.argmax(p, dim=-1) for p in preds]

        for pred_count, target in zip(pred_counts, targets):
            pred_acts, pred_deacts = self._find_events(pred_count)
            true_acts, true_deacts = self._find_events(target)

            tp_act, _ = self._match_events(pred_acts, true_acts)
            fp_act = pred_acts.numel() - tp_act
            fn_act = true_acts.numel() - tp_act

            tp_deact, _ = self._match_events(pred_deacts, true_deacts)
            fp_deact = pred_deacts.numel() - tp_deact
            fn_deact = true_deacts.numel() - tp_deact

            # Update aggregate states
            self.tp_act += tp_act
            self.fp_act += fp_act
            self.fn_act += fn_act
            self.tp_deact += tp_deact
            self.fp_deact += fp_deact
            self.fn_deact += fn_deact

            # Append to per-sample states as tensors
            self.per_sample_tp_act.append(torch.tensor(tp_act))
            self.per_sample_fp_act.append(torch.tensor(fp_act))
            self.per_sample_fn_act.append(torch.tensor(fn_act))
            self.per_sample_tp_deact.append(torch.tensor(tp_deact))
            self.per_sample_fp_deact.append(torch.tensor(fp_deact))
            self.per_sample_fn_deact.append(torch.tensor(fn_deact))

        self.scenario_ids.extend(meta["scenario_id"])

    def compute(self) -> dict:
        # ... (implementation unchanged)
        results = {}
        for event_type in ["act", "deact"]:
            tp = getattr(self, f"tp_{event_type}").float()
            fp = getattr(self, f"fp_{event_type}").float()
            fn = getattr(self, f"fn_{event_type}").float()
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            results[f"precision_{event_type}"] = precision
            results[f"recall_{event_type}"] = recall
            results[f"F1score_{event_type}"] = f1

        total_tp = self.tp_act.float() + self.tp_deact.float()
        total_fp = self.fp_act.float() + self.fp_deact.float()
        total_fn = self.fn_act.float() + self.fn_deact.float()
        overall_precision = total_tp / (total_tp + total_fp + 1e-6)
        overall_recall = total_tp / (total_tp + total_fn + 1e-6)
        overall_f1 = (
            2
            * (overall_precision * overall_recall)
            / (overall_precision + overall_recall + 1e-6)
        )
        results["precision_overall"] = overall_precision
        results["recall_overall"] = overall_recall
        results["F1score_overall"] = overall_f1
        return results

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        if not self.scenario_ids:
            return None

        results_dict = {
            "tp_act": [x.item() for x in self.per_sample_tp_act],
            "fp_act": [x.item() for x in self.per_sample_fp_act],
            "fn_act": [x.item() for x in self.per_sample_fn_act],
            "tp_deact": [x.item() for x in self.per_sample_tp_deact],
            "fp_deact": [x.item() for x in self.per_sample_fp_deact],
            "fn_deact": [x.item() for x in self.per_sample_fn_deact],
        }

        df = pd.DataFrame(results_dict, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df

    def reset(self) -> None:
        """Resets metric states and the scenario_ids list."""
        super().reset()
        self.scenario_ids = []
