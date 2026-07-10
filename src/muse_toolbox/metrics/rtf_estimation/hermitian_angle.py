import torch
import pandas as pd
from muse_toolbox.metrics.common.base_metric import BaseMetric
from typing import Optional
from muse_toolbox.utils import (
    hermitian_angle,
    activity_dict2tensor,
    STFTtransform,
    wmean,
    get_real_dtype,
)


class HermitianAngle(BaseMetric):
    is_differentiable = False
    higher_is_better = False  # Assuming smaller angle is better
    full_state_update = False
    requires_reference = True

    MHA: torch.Tensor
    MHA_A1: torch.Tensor
    MHA_A2: torch.Tensor
    MHA_A3: torch.Tensor
    MHA_D1: torch.Tensor
    MHA_D2: torch.Tensor

    WMHA: torch.Tensor  # Weighted Mean Hermitian Angle
    WMHA_A1: torch.Tensor
    WMHA_A2: torch.Tensor
    WMHA_A3: torch.Tensor
    WMHA_D1: torch.Tensor
    WMHA_D2: torch.Tensor

    MHA_samples: torch.Tensor
    MHA_A1_samples: torch.Tensor
    MHA_A2_samples: torch.Tensor
    MHA_A3_samples: torch.Tensor
    MHA_D1_samples: torch.Tensor
    MHA_D2_samples: torch.Tensor

    WMHA_samples: torch.Tensor
    WMHA_A1_samples: torch.Tensor
    WMHA_A2_samples: torch.Tensor
    WMHA_A3_samples: torch.Tensor
    WMHA_D1_samples: torch.Tensor
    WMHA_D2_samples: torch.Tensor

    per_sample_results: list[torch.Tensor]
    scenario_ids: list[str]

    def __init__(self, transform: STFTtransform, *args, **kwargs):
        super().__init__(*args, requires_numpy=False, **kwargs)
        self.transform = transform
        self.ref_weights = {}
        # States for aggregated metrics
        self.one_sample_results = {}
        self.HAnames = ["HA", "HA_A1", "HA_A2", "HA_A3", "HA_D1", "HA_D2"]
        self.AGGnames = ["M", "WM"]
        for name in self.HAnames:
            for agg in self.AGGnames:
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
            gt_rtfs = meta["gt_rtf_stream"][bidx]
            gt_ids = meta["gt_ids_stream"][bidx]
            sad_frames = meta["sad_frames"][bidx]
            id_map = meta["id_map"][bidx]
            refs = meta["references"][bidx]

            pred_rtfs = pred[1]  # list of (F, T_seg, M, Kt_pred), length N_segs
            # pred_ids = pred[2]  # list of (T_seg, Kt_pred), length N_segs

            pred_rtfs_framewise = [
                row for tensor in pred_rtfs for row in tensor.permute(1, 0, 2, 3)
            ]
            # pred_ids_framewise = [row for tensor in pred_ids for row in tensor]

            _, _, target_id_stream, _ = activity_dict2tensor(
                sad_frames, id_map
            )  # (T, K_gt)

            # HA_all_seg = []  # (2, T_seg), length N_segs
            # HA_last_seg = []  # list of (2, T_seg), length N_segs

            t = 0
            Kseg_old = 0
            seg_ids = []

            for gt_seg, gt_id_seg in zip(gt_rtfs, gt_ids):
                T_seg = gt_seg.shape[-3]
                Kseg = gt_id_seg.shape[0]
                if Kseg > Kseg_old:
                    segid = f"A{Kseg}"
                elif Kseg < Kseg_old:
                    segid = f"D{Kseg}"
                else:
                    segid = f"S{Kseg}"
                seg_ids.append(segid)
                Kseg_old = Kseg

                pred_rtf_seg = pred_rtfs_framewise[t : t + T_seg]
                # pred_id_seg = pred_ids_framewise[t : t + T_seg]
                ha_last = self._compute_hermitian_angle_last(
                    pred_rtf_seg,
                    gt_seg,
                    gt_id_seg,
                    target_id_stream[t : t + T_seg],
                    refs,
                    id_map,
                )  # (2, T_seg)

                # ha_all = self._compute_hermitian_angle_all(
                #     pred_rtf_seg, gt_seg, pred_id_seg, gt_id_seg
                # )
                self.one_sample_results["MHA"] += torch.sum(ha_last[0]).to(self.device)
                self.one_sample_results["MHA_samples"] += ha_last[0].numel()
                self.one_sample_results["WMHA"] += torch.sum(ha_last[1]).to(self.device)
                self.one_sample_results["WMHA_samples"] += ha_last[1].numel()
                if segid in ["A1", "A2", "A3", "D1", "D2"]:
                    self.one_sample_results[f"MHA_{segid}"] += torch.sum(ha_last[0]).to(
                        self.device
                    )
                    self.one_sample_results[f"MHA_{segid}_samples"] += ha_last[
                        0
                    ].numel()
                    self.one_sample_results[f"WMHA_{segid}"] += torch.sum(
                        ha_last[1]
                    ).to(self.device)
                    self.one_sample_results[f"WMHA_{segid}_samples"] += ha_last[
                        1
                    ].numel()

                # match seg_ids[-1]:
                #     case "A1":
                #         self.MHA_A1 += torch.sum(ha_last[0])
                #         self.MHA_A1_samples += ha_last[0].numel()
                #         self.WMHA_A1 += torch.sum(ha_last[1])
                #         self.WMHA_A1_samples += ha_last[1].numel()
                #     case "A2":
                #         self.MHA_A2 += torch.sum(ha_last[0])
                #         self.MHA_A2_samples += ha_last[0].numel()
                #         self.WMHA_A2 += torch.sum(ha_last[1])
                #         self.WMHA_A2_samples += ha_last[1].numel()
                #     case "A3":
                #         self.MHA_A3 += torch.sum(ha_last[0])
                #         self.MHA_A3_samples += ha_last[0].numel()
                #         self.WMHA_A3 += torch.sum(ha_last[1])
                #         self.WMHA_A3_samples += ha_last[1].numel()
                #     case "D1":
                #         self.MHA_D1 += torch.sum(ha_last[0])
                #         self.MHA_D1_samples += ha_last[0].numel()
                #         self.WMHA_D1 += torch.sum(ha_last[1])
                #         self.WMHA_D1_samples += ha_last[1].numel()
                #     case "D2":
                #         self.MHA_D2 += torch.sum(ha_last[0])
                #         self.MHA_D2_samples += ha_last[0].numel()
                #         self.WMHA_D2 += torch.sum(ha_last[1])
                #         self.WMHA_D2_samples += ha_last[1].numel()
                #     case _:
                #         pass

                # HA_all_seg.append(ha_all)
                # HA_last_seg.append(ha_last)
                t += T_seg

            # HA_last_stream = torch.cat(HA_last_seg, dim=-1)  # (2, T)
            # # HA_all_stream = torch.cat(HA_all_seg, dim=-1)  # (2, T)
            # self.MHA += torch.sum(HA_last_stream[0])
            # self.WMHA += torch.sum(HA_last_stream[1])
            # self.MHA_samples += HA_last_stream[0].numel()
            # self.WMHA_samples += HA_last_stream[1].numel()
            per_sample_tensor = []
            for name in self.HAnames:
                for agg in self.AGGnames:
                    total = self.one_sample_results[f"{agg}{name}"]
                    samples = self.one_sample_results[f"{agg}{name}_samples"]
                    if samples > 0:
                        avg = total / samples
                    else:
                        avg = torch.tensor(float("nan"), device=total.device)
                    per_sample_tensor.append(avg.unsqueeze(0))
                    # Update global states
                    getattr(self, f"{agg}{name}").add_(total)
                    getattr(self, f"{agg}{name}_samples").add_(samples)
                    # Reset one sample results
                    self.one_sample_results[f"{agg}{name}"] = torch.tensor(0.0)
                    self.one_sample_results[f"{agg}{name}_samples"] = torch.tensor(0)

            self.per_sample_results.append(
                torch.cat(per_sample_tensor, dim=0).reshape(6, 2)
            )
            self.scenario_ids.append(meta["scenario_id"][bidx])

            self.ref_weights = {}  # Clear cached weights after each sample

    def _compute_hermitian_angle_last(
        self,
        pred_rtf_seg: list[torch.Tensor],
        gt_seg: torch.Tensor,
        gt_ids: torch.Tensor,
        target_id: torch.Tensor,
        refs: dict[str, torch.Tensor],
        id_map: dict[int, str],
    ) -> torch.Tensor:
        """Computes the hermitian angle between the prediction
        and the ground truth per segment considering only
        the latest activated source.

        Args:
            pred_rtf_seg (list[torch.Tensor]): List of tensors each of shape (F, T_seg, M, Kt_pred)
            gt_seg (torch.Tensor): (F, T_seg, M, Kt_gt)
            gt_ids (torch.Tensor): (Kt_gt,)
            target_id (torch.Tensor): (T_seg,)
            refs (dict[str, torch.Tensor]): Reference signals
            id_map (dict[int, str]): Mapping from target IDs to reference keys

        Returns:
            torch.Tensor: Hermitian angle for the segment
        """
        HAlist = []
        for t, (target_id, pred_rtf) in enumerate(zip(target_id, pred_rtf_seg)):

            if target_id == -3:  # Silent segment / No active speaker / No target
                if pred_rtf.shape[-1] == 0:
                    HAlist.append(
                        torch.zeros(
                            2, dtype=get_real_dtype(gt_seg), device=gt_seg.device
                        )
                    )  # No prediction
                else:
                    HAlist.append(
                        torch.ones(
                            2, dtype=get_real_dtype(gt_seg), device=gt_seg.device
                        )
                        * torch.pi
                        / 2
                    )  # Wrong prediction
            else:
                if pred_rtf.shape[-1] == 0:
                    HAlist.append(
                        torch.ones(
                            2, dtype=get_real_dtype(gt_seg), device=gt_seg.device
                        )
                        * torch.pi
                        / 2
                    )  # No prediction
                else:
                    ha_last = (
                        hermitian_angle(
                            pred_rtf[..., -1:],
                            gt_seg[:, t, :, (gt_ids == target_id)],
                            dim=-2,
                        )
                        .squeeze(-2)
                        .squeeze(-1)
                    )  # (F,)
                    mha = torch.mean(ha_last, dim=0)  # (,)
                    wmha = self._ha2wmha(ha_last, refs, id_map[int(target_id.item())])
                    HAlist.append(torch.stack([mha, wmha], dim=0))
        return torch.stack(HAlist, dim=-1)

    def _ha2wmha(
        self, ha: torch.Tensor, refs: dict[str, torch.Tensor], ref_key: str
    ) -> torch.Tensor:
        """Convert Hermitian Angle to Weighted Mean Hermitian Angle (WMHA)

        Args:
            ha (torch.Tensor): Hermitian Angle tensor of shape (T_seg,)
            ref (torch.Tensor): Reference signal tensor of shape (F, T)

        Returns:
            torch.Tensor: Weighted Mean Hermitian Angle
        """
        # if self.ref_weights does not exist, create it now

        if ref_key in self.ref_weights:
            weights = self.ref_weights[ref_key]
        else:
            weights = self._compute_ref_weights(refs[ref_key], device=ha.device)
            self.ref_weights[ref_key] = weights
        return wmean(ha, weights=weights, dims=0, keepdim=False)  # Normalize weights

    def _compute_ref_weights(
        self, ref: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Compute weights based on reference signal energy

        Args:
            ref (torch.Tensor): Reference signal tensor of shape (F, T)
            device (torch.device): Device to perform computation on

        Returns:
            torch.Tensor: Weights tensor of shape (F,)
        """
        ref_stft = self.transform.encode(ref).to(device)  # (F, M, T)
        active_frames = torch.where(ref_stft != 0)[2].unique()
        # Compute weights based on reference signal energy
        energy = torch.sum(
            ref_stft[..., active_frames].abs() ** 2, dim=(-1, -2)
        )  # (F,)
        weights = energy / torch.sum(energy)
        return weights

    # def _compute_hermitian_angle_all(
    #     self,
    #     pred_seg: torch.Tensor,
    #     gt_seg: torch.Tensor,
    #     pred_ids: torch.Tensor,
    #     gt_ids: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Computes the hermitian angle between the prediction
    #     and the ground truth per segment considering either
    #     all active sources in the segment or only the latest activated source.

    #     Args:
    #         pred_seg (torch.Tensor): (F, T_seg, M, Kt_pred)
    #         gt_seg (torch.Tensor): (F, T_seg, M, Kt_gt)
    #         pred_ids (List[int]): List of predicted source IDs in the segment
    #         gt_ids (List[int]): List of ground truth source IDs in the segment

    #     Returns:
    #         torch.Tensor: _description_
    #     """
    #     if len(gt_ids) == 0:
    #         return torch.ones_like(pred_seg[0:2, :, 0, :]) * torch.pi / 2
    #     elif len(pred_ids) == 0:
    #         return torch.ones_like(gt_seg[0:2, :, 0, :]) * torch.pi / 2
    #     else:
    #         equal_ids = set(pred_ids).intersection(set(gt_ids))
    #         non_matched_pred_ids = set(pred_ids) - equal_ids
    #         non_matched_gt_ids = set(gt_ids) - equal_ids
    #         num_non_matched = len(non_matched_pred_ids) + len(non_matched_gt_ids)
    #         if len(equal_ids) != 0:
    #             ha_all = []
    #             for eid in equal_ids:
    #                 ha_eid = hermitian_angle(
    #                     pred_seg[..., (pred_ids == eid)],
    #                     gt_seg[..., (gt_ids == eid)],
    #                     dim=-2,
    #                 ).squeeze(-2)
    #                 ha_all.append(ha_eid)
    #             ha_all = torch.cat(ha_all, dim=-1)
    #             mha = torch.mean(ha_all, dim=0)
    #             wmha = self._ha2wmha(ha_all, refs[id_map[int(target_id.item())]])

    #             ha_all = torch.cat(
    #                 [
    #                     ha_all,
    #                     torch.ones_like(pred_seg[0:2, :, 0, :num_non_matched])
    #                     * torch.pi
    #                     / 2,
    #                 ],
    #                 dim=-1,
    #             )
    #         return ha_all
    #         ha = hermitian_angle(pred_seg[..., -1:], gt_seg, dim=-2)
    #     t = 5
    #     # return ha_all, ha_last

    def compute(self) -> dict:
        results = {}
        for name in self.HAnames:
            for agg in self.AGGnames:
                total = getattr(self, f"{agg}{name}")
                samples = getattr(self, f"{agg}{name}_samples")
                if samples > 0:
                    mean_angle = total / samples
                else:
                    mean_angle = torch.tensor(float("nan"), device=total.device)
                results[f"{agg}{name}"] = mean_angle

        return results

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        if not self.scenario_ids:
            return None

        results_dict = {}
        for n, name in enumerate(self.HAnames):
            for a, agg in enumerate(self.AGGnames):
                results_dict[f"{agg}{name}"] = [
                    x[n, a].item() for x in self.per_sample_results
                ]

        df = pd.DataFrame(results_dict, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df
