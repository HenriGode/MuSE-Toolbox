from unicodedata import name
import torch
import pandas as pd
from muse_toolbox.metrics.common.base_metric import BaseMetric
from typing import Optional, List
from muse_toolbox.utils import activity_dict2tensor, STFTtransform, computeSNR, move2device


class SINR(BaseMetric):
    is_differentiable = False
    higher_is_better = True  # Higher SINR is better
    full_state_update = False
    requires_reference = True

    total_angle: torch.Tensor
    total_samples: torch.Tensor
    per_sample_results: List[torch.Tensor]

    def __init__(self, transform: STFTtransform, ref_channels, *args, **kwargs):
        super().__init__(*args, requires_numpy=False, **kwargs)

        self.transform = transform
        self.ref_channels = ref_channels  # Assuming the first channel is the reference TODO: make this configurable

        self.one_sample_results = {}
        self.Segnames = ["A1", "A2", "A3", "D1", "D2"]
        self.Allnames = [""] + [f"_{name}" for name in self.Segnames]
        self.Metnames = ["SINR", "SIR", "SNR", "HGSDR"]
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

        self.scenario_ids: List[str] = []

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
    ):
        for bidx in range(len(preds)):
            pred = preds[bidx]
            # gt_ids = meta["gt_ids_stream"][bidx]
            sad_samples = meta["sad_samples"][bidx]
            id_map = meta["id_map"][bidx]
            refs = meta["references"][bidx]

            pred_bfs = {
                bf_type: pred[3][bf_type][..., self.ref_channels]
                for bf_type in pred[3].keys()
            }  # dict of (F, T, M, 1)

            _, sad_samples_tensor, target_id_stream, seg_borders = activity_dict2tensor(
                sad_samples, id_map
            )  # (N,) whereby N is the number of samples in the stream

            first_pred_bf = next(iter(pred_bfs.values()))
            device = first_pred_bf.device

            input_target, input_interferer = self._construct_gt_target(
                move2device(refs, device), target_id_stream, id_map
            )  # (M, N)
            input_target = input_target[self.ref_channels, :]  # (1, N)
            input_interferer = input_interferer[self.ref_channels, :]  # (1, N)
            input_noise = refs["noise"][self.ref_channels, :].to(
                device=device
            )  # (1, N)
            tad = (target_id_stream != -3)[None, ...].to(device=device)  # (1, N)

            sig_results = {"all": {}}

            if not hasattr(self, "bf_types"):
                self.bf_types = set()
            self.bf_types.update(pred_bfs.keys())
            sorted_bf_types = sorted(list(self.bf_types))

            # Store the computed overall metrics for each bf_type
            for bf_type, pred_bf in pred_bfs.items():
                shadow_sigs = self._shadow_filtering(
                    refs, pred_bf
                )  # dict[str, torch.Tensor] (1, N)

                output_target, output_interferer = self._construct_gt_target(
                    shadow_sigs, target_id_stream, id_map
                )  # (1, N)
                output_noise = shadow_sigs["noise"][self.ref_channels, :]  # (1, N)

                # Compute overall SNR metrics for this bf_type
                bftype_metrics = self._computeSNR_metrics(
                    (input_target, input_interferer, input_noise),
                    (output_target, output_interferer, output_noise),
                    tad,
                )

                # Combine input metrics directly (only once mapped) and output metrics with _{bf_type}
                for m_key, m_val in bftype_metrics.items():
                    if m_key.endswith("i") and m_key not in ["Gi", "Di"]:
                        sig_results["all"][m_key] = m_val
                    else:
                        sig_results["all"][f"{m_key}_{bf_type}"] = m_val

            Kseg_old = 0
            seg_ids = []

            seg_type_inds = {
                segid: torch.tensor([], dtype=torch.long) for segid in self.Segnames
            }

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
                    seg_type_inds[segid] = torch.cat(
                        (
                            seg_type_inds[segid],
                            torch.arange(int(start), int(end)),
                        )
                    )

            for segid in self.Segnames:
                if seg_type_inds[segid].numel() > 0:
                    sig_results[segid] = {}
                    for bf_type, pred_bf in pred_bfs.items():
                        # We need to re-compute shadow signals segments per bf_type for proper indexing
                        shadow_sigs = self._shadow_filtering(refs, pred_bf)
                        output_target, output_interferer = self._construct_gt_target(
                            shadow_sigs, target_id_stream, id_map
                        )
                        output_noise = shadow_sigs["noise"][self.ref_channels, :]

                        bftype_metrics = self._computeSNR_metrics(
                            (
                                input_target[:, seg_type_inds[segid]],
                                input_interferer[:, seg_type_inds[segid]],
                                input_noise[:, seg_type_inds[segid]],
                            ),
                            (
                                output_target[:, seg_type_inds[segid]],
                                output_interferer[:, seg_type_inds[segid]],
                                output_noise[:, seg_type_inds[segid]],
                            ),
                            tad[:, seg_type_inds[segid]],
                        )

                        for m_key, m_val in bftype_metrics.items():
                            if m_key.endswith("i") and m_key not in ["Gi", "Di"]:
                                sig_results[segid][m_key] = m_val
                            else:
                                sig_results[segid][f"{m_key}_{bf_type}"] = m_val

            # Update states to store delta values of SINR, SIR, and SNR (Delta is equal to ouput - input)
            per_seg_tensors = []
            num_base_metrics = len(self.Metnames)
            # Calculate length per segment metrics list. For each Metname: 1 input, and for each bftype: 2 (output, delta).
            # For each single metrics (Gt, Gi, etc.), 6 base * len(bftypes).
            metrics_length = num_base_metrics * (
                1 + 2 * len(sorted_bf_types)
            ) + 6 * len(sorted_bf_types)

            for name in self.Allnames:
                if name == "":
                    res = sig_results["all"]
                elif name.removeprefix("_") not in sig_results:
                    per_seg_tensors.append(
                        torch.tensor(float("nan")).unsqueeze(0).expand(metrics_length)
                    )
                    continue
                else:
                    res = sig_results[name.removeprefix("_")]

                per_met_tensors = []
                for met in self.Metnames:
                    metname = f"{met}{name}"

                    inp = res[f"{met}i"]
                    per_met_tensors.append(inp.unsqueeze(0))

                    for bf_type in sorted_bf_types:
                        try:
                            out = res[f"{met}o_{bf_type}"]
                            delta = out - inp
                        except KeyError:
                            out = torch.tensor(float("nan"), device=device)
                            delta = torch.tensor(float("nan"), device=device)

                        if (
                            bf_type == sorted_bf_types[0]
                            and not torch.isnan(delta).all()
                        ):
                            getattr(self, metname).add_(delta)
                            getattr(self, f"{metname}_samples").add_(torch.tensor(1))

                        per_met_tensors.append(out.unsqueeze(0))
                        per_met_tensors.append(delta.unsqueeze(0))

                for g in ["Gt", "Gi", "Gn", "Dt", "Di", "Dn"]:
                    for bf_type in sorted_bf_types:
                        try:
                            val = res[f"{g}_{bf_type}"]
                        except KeyError:
                            val = torch.tensor(float("nan"), device=device)
                        per_met_tensors.append(val.unsqueeze(0))

                per_seg_tensors.append(torch.cat(per_met_tensors, dim=0))

            per_sample_tensor = torch.stack(per_seg_tensors)
            self.per_sample_results.append(per_sample_tensor)

            self.scenario_ids.append(meta["scenario_id"][bidx])

    def _computeSNR_metrics(
        self,
        input_sigs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output_sigs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tad: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_target, input_interferer, input_noise = input_sigs
        output_target, output_interferer, output_noise = output_sigs
        result = {}
        result["SINRi"] = (
            computeSNR(input_target, input_interferer + input_noise, tad)
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["SINRo"] = (
            computeSNR(output_target, output_interferer + output_noise, tad)
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["SIRi"] = (
            computeSNR(input_target, input_interferer, tad)
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["SIRo"] = (
            computeSNR(output_target, output_interferer, tad)
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["SNRi"] = (
            computeSNR(input_target, input_noise, tad).cpu().squeeze(-1).squeeze(-1)
        )
        result["SNRo"] = (
            computeSNR(output_target, output_noise, tad).cpu().squeeze(-1).squeeze(-1)
        )
        result["HGSDRi"] = (
            computeSNR(
                input_target,
                input_target + input_interferer + input_noise - input_target,
                tad,
            )
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["HGSDRo"] = (
            computeSNR(
                output_target,
                output_target + output_interferer + output_noise - input_target,
                tad,
            )
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["Gt"] = (
            computeSNR(output_target, input_target, tad).cpu().squeeze(-1).squeeze(-1)
        )
        result["Gi"] = (
            computeSNR(output_interferer, input_interferer, tad)
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["Gn"] = (
            computeSNR(output_noise, input_noise, tad).cpu().squeeze(-1).squeeze(-1)
        )
        result["Dt"] = (
            computeSNR(input_target, output_target - input_target, tad)
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["Di"] = (
            computeSNR(input_interferer, output_interferer - input_interferer, tad)
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        result["Dn"] = (
            computeSNR(input_noise, output_noise - input_noise, tad)
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
        )
        return result

    def _shadow_filtering(
        self, refs: dict[str, torch.Tensor], pred_bf: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Apply shadow filtering to the reference signals using the predicted beamforming weights.

        Args:
            refs (dict[str, torch.Tensor]): Dictionary of reference signals.
            pred_bf (torch.Tensor): Predicted beamforming weights.

        Returns:
            dict[str, torch.Tensor]: Shadow filtered signals.
        """
        shadow_sigs = {}
        for key, ref_signal in refs.items():
            Nsig = ref_signal.shape[-1]
            # Transform reference signal to frequency domain
            ref_stft = (
                self.transform.encode(ref_signal).transpose(-1, -2).unsqueeze(-1)
            )  # (F, T, M, 1)
            # Apply beamforming weights
            bf_applied = pred_bf.mH @ ref_stft.to(pred_bf.device)  # (F, T, 1, 1)
            # Inverse transform to time domain
            shadow_signal = self.transform.decode(
                bf_applied.squeeze(-1).transpose(-1, -2), num_samples=Nsig
            )  # (1, N)
            shadow_sigs[key] = shadow_signal  # (N,)
        return shadow_sigs

    def _construct_gt_target(
        self,
        refs: dict[str, torch.Tensor],
        target_id_stream: torch.Tensor,
        id_map: dict[int, str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construct the ground truth target signal based on the reference signals and the target ID stream.

        Args:
            refs (dict[str, torch.Tensor]): Dictionary of reference signals.
            target_id_stream (torch.Tensor): Tensor of target IDs.
            id_map (dict[int, str]): Mapping from integer IDs to string labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Ground truth target signal, interference signal.
        """
        gt_target = []
        gt_interferer = []
        # loop over segments containing multiple frames which all have the same target ID
        unique_ids, counts = torch.unique_consecutive(
            target_id_stream, return_counts=True
        )
        t_start = 0
        for uid, count in zip(unique_ids, counts):
            t_end = t_start + count.item()

            if uid.item() != -3:
                target_id = id_map[uid.item()]
                ref_signal = refs[target_id][:, t_start:t_end]  # (M, count)
                # interferer signal is sum of all other active sources except target and exept noise (key: "noise")
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
        gt_target = torch.cat(gt_target, dim=1)  # (M, T)
        gt_interferer = torch.cat(gt_interferer, dim=1)  # (M, T)
        return gt_target, gt_interferer  # (M, T)

    def compute(self) -> dict:
        results = {}
        for name in self.Allnames:
            for met in self.Metnames:
                metname = f"{met}{name}"
                total = getattr(self, metname)
                samples = getattr(self, f"{metname}_samples")
                if samples > 0:
                    results[metname] = (total / samples).item()
                else:
                    results[metname] = float("nan")
        return results

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        if not self.scenario_ids:
            return None

        results_dict = {}
        for n, name in enumerate(self.Allnames):
            if not hasattr(self, "bf_types"):
                continue

            sorted_bf_types = sorted(list(self.bf_types))
            idx = 0

            # For each primary metric (SINR, SIR, SNR, SDR)
            for m, met in enumerate(self.Metnames):
                # 1. Input metric
                results_dict[f"{met}i{name}"] = [
                    x[n, idx].item() for x in self.per_sample_results
                ]
                idx += 1

                # 2. Output and delta metrics for each bf_type
                for bf_type in sorted_bf_types:
                    results_dict[f"{met}o_{bf_type}{name}"] = [
                        x[n, idx].item() for x in self.per_sample_results
                    ]
                    idx += 1

                    results_dict[f"D{met}_{bf_type}{name}"] = [
                        x[n, idx].item() for x in self.per_sample_results
                    ]
                    idx += 1

            # For singular gain metrics
            for g in ["Gt", "Gi", "Gn", "Dt", "Di", "Dn"]:
                for bf_type in sorted_bf_types:
                    results_dict[f"{g}_{bf_type}{name}"] = [
                        x[n, idx].item() for x in self.per_sample_results
                    ]
                    idx += 1

        df = pd.DataFrame(results_dict, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df
