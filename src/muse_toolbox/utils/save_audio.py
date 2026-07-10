import torch
import pandas as pd
from .base_metric import BaseMetric
from typing import Optional, List
from muse_toolbox.utils import STFTtransform
from torchcodec.encoders import AudioEncoder
import os
from muse_toolbox.utils import activity_dict2tensor


class SAVE_AUDIO(BaseMetric):
    is_differentiable = False
    higher_is_better = True  # Higher STOI is better
    full_state_update = False
    requires_reference = True

    total_angle: torch.Tensor
    total_samples: torch.Tensor
    per_sample_results: List[torch.Tensor]

    def __init__(
        self,
        transform: STFTtransform,
        model_name: str,
        sad_model_name: Optional[str] = None,
        block_size_frames: Optional[int] = None,
        pre_context_frames: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, requires_numpy=False, **kwargs)

        self.transform = transform
        self.model_name = model_name
        self.sad_model_name = sad_model_name
        if self.model_name == "BlockOnlineGSS":
            self.block_size_frames = block_size_frames
            self.pre_context_frames = pre_context_frames
        else:
            self.block_size_frames = None
            self.pre_context_frames = None

        self.ref_channel = 0  # Assuming the first channel is the reference TODO: make this configurable

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

            Nsig = refs[next(iter(refs))].shape[-1]

            dataset_id, split_name, scenario_id = self._split_scenario_id(
                meta["scenario_id"][bidx]
            )
            savedir = f"./output_audio/J2_RUN/{dataset_id}/STFT_{self.transform.sampling_frequency}_{self.transform.nfft}_{self.transform.hop_length}/{self.sad_model_name}/{split_name}"
            if self.model_name == "BlockOnlineGSS":
                save_dir_model = f"{savedir}/{self.model_name}_bl{self.block_size_frames}_pc{self.pre_context_frames}"
            else:
                save_dir_model = f"{savedir}/{self.model_name}"

            for bf_type, pred_target_stft in pred[0].items():
                pred_target = self.transform.decode(
                    pred_target_stft, num_samples=Nsig
                ).cpu()  # (1, N)

                save_dir_bf = f"{save_dir_model}/{bf_type}"
                os.makedirs(save_dir_bf, exist_ok=True)

                AudioEncoder(
                    samples=pred_target.cpu(),
                    sample_rate=int(self.transform.sampling_frequency),
                ).to_file(dest=f"{save_dir_bf}/scenario_{scenario_id}.wav")

            save_dir_noisy = f"{savedir}/noisy"
            os.makedirs(save_dir_noisy, exist_ok=True)
            if not os.path.exists(f"{save_dir_noisy}/scenario_{scenario_id}.wav"):
                noisy = torch.stack([r[self.ref_channel] for r in refs.values()]).sum(
                    dim=0, keepdim=True
                )
                AudioEncoder(
                    samples=noisy.cpu(),
                    sample_rate=int(self.transform.sampling_frequency),
                ).to_file(dest=f"{save_dir_noisy}/scenario_{scenario_id}.wav")

            save_dir_groundtruth = f"{savedir}/groundtruth"
            os.makedirs(save_dir_groundtruth, exist_ok=True)
            if not os.path.exists(f"{save_dir_groundtruth}/scenario_{scenario_id}.wav"):

                _, sad_samples_tensor, target_id_stream, seg_borders = (
                    activity_dict2tensor(sad_samples, id_map)
                )  # (N,) whereby N is the number of samples in the stream

                input_target, _ = self._construct_gt_target(
                    refs, target_id_stream, id_map
                )  # (M, N)

                AudioEncoder(
                    samples=input_target.cpu(),
                    sample_rate=int(self.transform.sampling_frequency),
                ).to_file(dest=f"{save_dir_groundtruth}/scenario_{scenario_id}.wav")

    def compute(self) -> Optional[dict]:
        return {"saved": torch.tensor(1)}

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return None

    def _split_scenario_id(self, scenario_id: str):
        dataset_id_splitname, scenario_id = scenario_id.split("_generator_")
        # The dataset_id still contains the actial dataset_id and the split name
        # the split name is everything after the last underscore
        dataset_id = "_".join(dataset_id_splitname.split("_")[:-1])
        split_name = dataset_id_splitname.split("_")[-1]
        return dataset_id, split_name, scenario_id

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
