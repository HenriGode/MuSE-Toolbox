import torch
import pandas as pd
from muse_toolbox.metrics.common.base_metric import BaseMetric
from typing import Optional, List
from muse_toolbox.utils import activity_dict2tensor, STFTtransform
from torchmetrics.audio import SignalDistortionRatio


class SDR(BaseMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    requires_reference = True

    total_angle: torch.Tensor
    total_samples: torch.Tensor
    per_sample_results: List[torch.Tensor]
    scenario_ids: List[str]

    def __init__(
        self,
        transform: STFTtransform,
        use_cg_iter: Optional[int] = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        load_diag: Optional[float] = None,
        ref_channels: list[int] = [0],
        *args,
        **kwargs,
    ):
        super().__init__(
            *args, requires_numpy=False, ref_channels=ref_channels, **kwargs
        )

        self.transform = transform

        self.SDR_fun = SignalDistortionRatio(
            use_cg_iter=use_cg_iter,
            filter_length=filter_length,
            zero_mean=zero_mean,
            load_diag=load_diag,
        )
        self.ref_channel = 0  # Assuming the first channel is the reference TODO: make this configurable

        self.one_sample_results = {}
        self.Segnames = ["A1", "A2", "A3", "D1", "D2"]
        self.Allnames = [""] + [f"_{name}" for name in self.Segnames]
        self.Metnames = ["SDR"]
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

    def compute_sdr(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Compute SDR.
        """
        # Ensure correct shapes (torchmetrics usually expects (..., time))
        try:
            return self.SDR_fun(deg, ref)
        except Exception:
            # Return NaN if computation fails (e.g. extremely short segments)
            return torch.tensor(float("nan"), device=ref.device)

    def update(
        self,
        preds: list[
            tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]
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

            _, sad_samples_tensor, target_id_stream, seg_borders = activity_dict2tensor(
                sad_samples, id_map
            )  # (N,) whereby N is the number of samples in the stream

            input_target, _ = self._construct_gt_target(
                refs, target_id_stream, id_map
            )  # (M, N)
            input_target = input_target[
                self.ref_channel : self.ref_channel + 1, :
            ]  # (1, N)

            input_noisy = torch.stack([v for v in refs.values()]).sum(dim=0)[
                self.ref_channel : self.ref_channel + 1, :
            ]  # (1, N)

            Nsig = input_target.shape[-1]

            pred_target = self.transform.decode(
                pred[0], num_samples=Nsig
            ).cpu()  # (F, 1, T)

            sig_results = {}
            sig_results["all"] = self._computeSDRmetric(
                input_target, input_noisy, pred_target
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
                    sig_results[segid].append(
                        self._computeSDRmetric(
                            input_target[:, start:end],
                            input_noisy[:, start:end],
                            pred_target[:, start:end],
                        )
                    )

            # Update states to store delta values
            per_seg_tensors = []
            for name in self.Allnames:
                if name == "":
                    res = sig_results["all"]
                elif sig_results[name.removeprefix("_")] == []:
                    # append nan if no segments of this type were present
                    per_seg_tensors.append(
                        torch.tensor(float("nan")).unsqueeze(0).expand(3)
                    )
                    continue
                else:
                    res = sig_results[name.removeprefix("_")]

                metname = f"SDR{name}"

                out = (
                    torch.stack([r[f"SDRo"] for r in res])
                    if isinstance(res, list)
                    else res[f"SDRo"]
                )
                inp = (
                    torch.stack([r[f"SDRi"] for r in res])
                    if isinstance(res, list)
                    else res[f"SDRi"]
                )

                delta = out - inp
                if delta.ndim == 0:
                    getattr(self, metname).add_(delta)
                    getattr(self, f"{metname}_samples").add_(
                        torch.tensor(1)
                    )  # Count one sample per update call
                    per_seg_tensors.append(torch.stack([inp, out, delta]))  # (3,)
                elif delta.ndim == 1:
                    getattr(self, metname).add_(delta.sum())
                    getattr(self, f"{metname}_samples").add_(
                        torch.tensor(delta.numel())
                    )  # Count one sample per segment
                    per_seg_tensors.append(
                        torch.stack(
                            [
                                inp.mean(),
                                out.mean(),
                                delta.mean(),
                            ]
                        )
                    )  # (3,)
                else:
                    raise ValueError("Delta SDR has more than 1 dimension.")

            per_sample_tensor = torch.stack(per_seg_tensors)  # (6, 3)
            self.per_sample_results.append(per_sample_tensor)

            self.scenario_ids.append(meta["scenario_id"][bidx])

    def _computeSDRmetric(
        self,
        input_target: torch.Tensor,
        input_noisy: torch.Tensor,
        output_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the SDR metric for the reference signals.

        Args:
            input_target (torch.Tensor): Ground truth target signal. Shape: (1, N)
            input_noisy (torch.Tensor): Noisy input signal. Shape: (1, N)
            output_target (torch.Tensor): Predicted target signal. Shape: (1, N)
        Returns:
            dict[str, torch.Tensor]: Dictionary containing SDR metrics.
        """
        sig_results = {}

        sdr_input = self.compute_sdr(
            input_noisy,
            input_target,
        )  # scalar
        sdr_output = self.compute_sdr(
            output_target,
            input_target,
        )  # scalar

        sig_results["SDRi"] = sdr_input
        sig_results["SDRo"] = sdr_output

        return sig_results

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
            metname = f"SDR{name}"
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
            for m, met in enumerate(
                [
                    "SDRi",
                    "SDRo",
                    "DSDR",
                ]
            ):
                results_dict[f"{met}{name}"] = [
                    x[n, m].item() for x in self.per_sample_results
                ]

        df = pd.DataFrame(results_dict, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df
