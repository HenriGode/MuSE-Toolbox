import torch
import pandas as pd
from muse_toolbox.metrics.base_metric import BaseMetric
from muse_toolbox.utils import STFTtransform
from muse_toolbox.data.simulation.scenario_generation import activity_dict2tensor


class RefMetric(BaseMetric):
    """Reference Metric class for segment-based evaluation in MuSE-Toolbox.

    Inherits from `BaseMetric` and provides common orchestration logic for 
    metrics like SDR, STOI, PESQ, and SiSDR. Automatically handles segmenting 
    the audio based on Voice Activity Detection (VAD) ground truth and grouping 
    the results by the number of active speakers (e.g., A1, A2, A3).
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    requires_reference = True

    total_angle: torch.Tensor
    total_samples: torch.Tensor
    per_sample_results: list[torch.Tensor]
    scenario_ids: list[str]

    def __init__(
        self,
        metric_name: str,
        transform: STFTtransform,
        ref_channels: list[int],
        *args,
        **kwargs,
    ):
        """Initializes the RefMetric.

        Args:
            metric_name (str): The prefix for this metric in logs (e.g., 'SDR').
            transform (STFTtransform): Transformer to convert STFT inputs back to time-domain if necessary.
            ref_channels (list[int]): Indices of channels to use as reference signals.
            *args: Variable length argument list passed to `BaseMetric`.
            **kwargs: Arbitrary keyword arguments passed to `BaseMetric`.
        """
        super().__init__(*args, requires_numpy=False, **kwargs)

        self.transform = transform
        self.metric_name = metric_name
        self.ref_channels = ref_channels  # Assuming the first channel is the reference TODO: make this configurable

        self.one_sample_results = {}
        self.Segnames = ["A1", "A2", "A3", "D1", "D2"]
        self.Allnames = [""] + [f"_{name}" for name in self.Segnames]
        self.Metnames = [self.metric_name]

        for name in self.Allnames:
            for agg in self.Metnames:
                self.add_state(
                    f"{agg}{name}", default=torch.tensor(0.0), dist_reduce_fx="sum"
                )
                self.add_state(
                    f"{agg}{name}_samples",
                    default=torch.tensor(0),
                    dist_reduce_fx="sum",
                )
                self.one_sample_results[f"{agg}{name}"] = torch.tensor(0.0)
                self.one_sample_results[f"{agg}{name}_samples"] = torch.tensor(0)

        self.add_state("per_sample_results", default=[], dist_reduce_fx="cat")
        self.add_state("scenario_ids", default=[], dist_reduce_fx="cat")

        self.scenario_ids: list[str] = []

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Evaluates the core metric logic for a degraded signal against a reference.
        
        Must be implemented by the child class. Usually contains a try-except block 
        returning NaN on failure to avoid crashing long evaluations on bad segments.

        Args:
            deg (torch.Tensor): The degraded or enhanced audio signal tensor.
            ref (torch.Tensor): The ground truth reference audio signal tensor.

        Returns:
            torch.Tensor: The computed metric score.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError

    def update(
        self,
        preds: list[
            tuple[
                dict[str, torch.Tensor],
                list[torch.Tensor],
                list[torch.Tensor],
                dict[str, torch.Tensor],
            ]
        ],
        targets: tuple[dict, torch.Tensor],
        meta: dict,
        dataloader_idx: int,
    ) -> None:
        """Updates the metric state with new batch data and calculates per-segment scores.

        Args:
            preds: Nested list containing the predicted model outputs and beamformer dicts.
            targets: A tuple representing ground truth targets.
            meta (dict): A dictionary containing 'sad_samples', 'id_map', and 'references'.
            dataloader_idx (int): Index of the current dataloader.
        """
        for bidx in range(len(preds)):
            pred = preds[bidx]
            sad_samples = meta["sad_samples"][bidx]
            id_map = meta["id_map"][bidx]
            refs = meta["references"][bidx]

            _, sad_samples_tensor, target_id_stream, seg_borders = activity_dict2tensor(
                sad_samples, id_map
            )

            input_target, _ = self._construct_gt_target(refs, target_id_stream, id_map)
            input_target = input_target[self.ref_channels, :]

            input_noisy = torch.stack([v for v in refs.values()]).sum(dim=0)[
                self.ref_channels, :
            ]

            Nsig = input_target.shape[-1]

            pred_targets = {
                bf_type: self.transform.decode(
                    pt[:, self.ref_channels], num_samples=Nsig
                ).cpu()
                for bf_type, pt in pred[0].items()
            }

            sig_results = {}
            sig_results["all"] = self._compute_ref_metric(
                input_target, input_noisy, pred_targets
            )

            Kseg_old = 0
            seg_ids = []

            for name in self.Segnames:
                sig_results[name] = []

            for start, end in zip(seg_borders[:-1], seg_borders[1:]):
                Nseg = end - start

                Kseg = sad_samples_tensor[:-1, start:end].any(dim=-1).sum(dim=0)
                if Kseg > Kseg_old:
                    segid = f"A{Kseg}"
                elif Kseg < Kseg_old:
                    segid = f"D{Kseg}"
                else:
                    segid = f"S{Kseg}"
                seg_ids.append(segid)
                Kseg_old = Kseg

                if Nseg < 1 or target_id_stream[start] == -3:
                    continue

                if segid in self.Segnames:
                    seg_pred_targets = {
                        bf_type: pt[:, start:end]
                        for bf_type, pt in pred_targets.items()
                    }
                    sig_results[segid].append(
                        self._compute_ref_metric(
                            input_target[:, start:end],
                            input_noisy[:, start:end],
                            seg_pred_targets,
                        )
                    )

            if not hasattr(self, "bf_types"):
                self.bf_types = set()

            # Update the set of seen bf_types with any new ones from this batch
            self.bf_types.update(pred_targets.keys())

            # Create a sorted list for consistent iteration
            sorted_bf_types = sorted(list(self.bf_types))

            # Update states
            per_seg_tensors = []
            for name in self.Allnames:
                if name == "":
                    res = [sig_results["all"]]
                elif sig_results[name.removeprefix("_")] == []:
                    length = 1 + 2 * len(sorted_bf_types)
                    per_seg_tensors.append(
                        torch.tensor(float("nan"), device=input_target.device)
                        .unsqueeze(0)
                        .expand(length)
                    )
                    continue
                else:
                    res = sig_results[name.removeprefix("_")]

                metname = f"{self.metric_name}{name}"

                inp = torch.stack([r[f"{self.metric_name}i"] for r in res])
                vals = [inp.mean()]

                for bf_type in sorted_bf_types:
                    # Some bf_types might not exist in this specific prediction, pad with NaN if missing
                    try:
                        out = torch.stack(
                            [r[f"{self.metric_name}o_{bf_type}"] for r in res]
                        )
                        delta = out - inp
                    except KeyError:
                        out = torch.tensor(float("nan"), device=input_target.device)
                        delta = torch.tensor(float("nan"), device=input_target.device)

                    if bf_type == sorted_bf_types[0] and not torch.isnan(delta).all():
                        getattr(self, metname).add_(delta.sum())
                        getattr(self, f"{metname}_samples").add_(
                            torch.tensor(delta.numel())
                        )

                    vals.append(out.mean() if out.dim() > 0 else out)
                    vals.append(delta.mean() if delta.dim() > 0 else delta)

                per_seg_tensors.append(torch.stack(vals))

            per_sample_tensor = torch.stack(per_seg_tensors)
            self.per_sample_results.append(per_sample_tensor)

            self.scenario_ids.append(meta["scenario_id"][bidx])

    def _compute_ref_metric(
        self,
        input_target: torch.Tensor,
        input_noisy: torch.Tensor,
        output_targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Computes the metric for both the input (noisy) and output (enhanced) signals.

        Args:
            input_target (torch.Tensor): Ground truth target tensor.
            input_noisy (torch.Tensor): The noisy/degraded input tensor.
            output_targets (dict[str, torch.Tensor]): Dictionary of beamformer outputs.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the evaluated metric results.
        """
        sig_results = {}

        metric_input = self.evaluate_metric(
            input_noisy,
            input_target,
        )
        sig_results[f"{self.metric_name}i"] = metric_input

        for bf_type, output_target in output_targets.items():
            metric_output = self.evaluate_metric(
                output_target,
                input_target,
            )
            sig_results[f"{self.metric_name}o_{bf_type}"] = metric_output

        return sig_results

    def _construct_gt_target(
        self,
        refs: dict[str, torch.Tensor],
        target_id_stream: torch.Tensor,
        id_map: dict[int, str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Constructs the ground truth target and interferer signals from references.

        Args:
            refs (dict[str, torch.Tensor]): Dictionary of reference signals.
            target_id_stream (torch.Tensor): Stream mapping time frames to speaker IDs.
            id_map (dict[int, str]): Map from integer ID to speaker string key.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Concatenated target and interferer signals.
        """
        gt_target = []
        gt_interferer = []
        unique_ids, counts = torch.unique_consecutive(
            target_id_stream, return_counts=True
        )
        t_start = 0
        for uid, count in zip(unique_ids, counts):
            t_end = t_start + count.item()

            if uid.item() != -3:
                target_id = id_map[uid.item()]
                ref_signal = refs[target_id][:, t_start:t_end]
                interferer_signals = torch.zeros_like(ref_signal)
                for key, ref in refs.items():
                    if key != target_id and key != "noise":
                        interferer_signals += ref[:, t_start:t_end]
            else:
                ref_signal = torch.zeros(
                    (refs[next(iter(refs))].shape[0], count.item()),
                    device=refs[next(iter(refs))].device,
                )
                interferer_signals = torch.zeros_like(ref_signal)
            gt_target.append(ref_signal)
            gt_interferer.append(interferer_signals)
            t_start = t_end
        gt_target = torch.cat(gt_target, dim=1)
        gt_interferer = torch.cat(gt_interferer, dim=1)
        return gt_target, gt_interferer

    def compute(self) -> dict:
        """Aggregates all segment statistics and computes the final averaged metric scores.

        Returns:
            dict: A dictionary mapping the metric names and segment conditions (e.g., 'SDRA1') 
            to their final computed floating point values.
        """
        results = {}
        for name in self.Allnames:
            metname = f"{self.metric_name}{name}"
            total = getattr(self, metname)
            samples = getattr(self, f"{metname}_samples")
            if samples > 0:
                results[metname] = (total / samples).item()
            else:
                results[metname] = float("nan")
        return results

    def get_dataframe(self) -> pd.DataFrame | None:
        """Constructs a DataFrame of all evaluated scenario results.

        Returns:
            pd.DataFrame | None: A dataframe with results or None if no scenarios were evaluated.
        """
        if not self.scenario_ids:
            return None

        results_dict = {}
        for n, name in enumerate(self.Allnames):
            # Input metric
            met = f"{self.metric_name}i"
            results_dict[f"{met}{name}"] = [
                x[n, 0].item() for x in self.per_sample_results
            ]

            # Output and Delta metrics for each bf_type
            if hasattr(self, "bf_types"):
                sorted_bf_types = sorted(list(self.bf_types))
                for i, bf_type in enumerate(sorted_bf_types):
                    idx_out = 1 + 2 * i
                    idx_delta = 2 + 2 * i

                    met_o = f"{self.metric_name}o_{bf_type}"
                    met_d = f"D{self.metric_name}_{bf_type}"

                    results_dict[f"{met_o}{name}"] = [
                        x[n, idx_out].item() for x in self.per_sample_results
                    ]
                    results_dict[f"{met_d}{name}"] = [
                        x[n, idx_delta].item() for x in self.per_sample_results
                    ]

        df = pd.DataFrame(results_dict, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df
