import torch
import pandas as pd
from .base_metric import BaseMetric
from typing import Optional, List
from muse_toolbox.utils import STFTtransform
from muse_toolbox.utils import activity_dict2tensor


class META2DF(BaseMetric):
    is_differentiable = False
    higher_is_better = True  # Higher STOI is better
    full_state_update = False
    requires_reference = True

    input_snr: List[float]
    fixed_seg_length: List[float]
    scenario_ids: List[str]

    def __init__(self, transform: STFTtransform, model_name: str, *args, **kwargs):
        super().__init__(*args, requires_numpy=False, **kwargs)

        self.transform = transform
        self.model_name = model_name

        self.ref_channel = 0  # Assuming the first channel is the reference TODO: make this configurable

        # add input snr and fixed seg length so that they are saved to the combined dataframe
        self.add_state("input_snr", default=[], dist_reduce_fx="cat")
        self.add_state("fixed_seg_length", default=[], dist_reduce_fx="cat")
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
            scenario_params = meta["scenario_params"][bidx]

            self.input_snr.append(scenario_params["snr"])
            self.fixed_seg_length.append(scenario_params["fixed_time_between_events"])

            self.scenario_ids.append(meta["scenario_id"][bidx])

    def compute(self) -> Optional[dict]:
        return {"added_meta": torch.tensor(1)}

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        results_dict = {
            "input_snr": self.input_snr,
            "fixed_seg_length": self.fixed_seg_length,
        }
        df = pd.DataFrame(results_dict, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df
