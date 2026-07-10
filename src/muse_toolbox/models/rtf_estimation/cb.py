import torch
import math
import matplotlib

import building_blocks as bb
import muse_toolbox.utils as utilities
from muse_toolbox.models.common.base_model import BaseLitModel

from typing import List, Union

matplotlib.use("agg")

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class CBmodule(BaseLitModel):

    def __init__(
        self,
        transform: utilities.STFTtransform,
        smoothing_time_constant: float,  # [s]
        only_noise_period: float,  # [s]
        fix_old_RTF_method: str,  # ['last', 'half']
        batch_size: int = 1,
        computation_method: str = "closed_form",  # 'closed-form' or 'gradient-descent'
        additional_vectors: Union[
            str, List[str]
        ] = "none",  # ['none', 'orthogonal', 'random']
        noise_handling: Union[
            str, List[str]
        ] = "whitening",  # ['whitening', 'subtraction', 'none']
        source_activity_method: Union[
            None, torch.nn.Module
        ] = None,  # None = "oracle" or source activity estimation method model
        metrics_test: Union[tuple, str] = ("SINR"),
    ):
        super().__init__(
            model_name="RTF_Estimator", batch_size=batch_size, metrics_test=metrics_test
        )

        ## Processing parameters
        self.transform = transform
        self.smoothing_time_constant = smoothing_time_constant
        self.only_noise_period = only_noise_period
        self.fix_old_RTF_method = fix_old_RTF_method

        ## Choice of RTF estimation method(s)
        # Choice of computation method
        # 'closed-form' or 'gradient-descent'
        self.computation_method = computation_method
        # Choice of additional vector(s)
        # 'none', 'orthogonal' or 'random'
        self.additional_vectors = (
            additional_vectors
            if isinstance(additional_vectors, list)
            else [additional_vectors]
        )
        # Choice of noise handling method(s)
        # 'whitening', 'subtraction' or 'none'
        self.noise_handling = (
            noise_handling if isinstance(noise_handling, list) else [noise_handling]
        )
        self.CBwrapper = bb.CBwrapper(
            transform=self.transform,
            smoothing_time_constant=self.smoothing_time_constant,
            only_noise_period=self.only_noise_period,
            fix_old_RTF_method=self.fix_old_RTF_method,
            computation_method=self.computation_method,
            additional_vectors=self.additional_vectors,
            noise_handling=self.noise_handling,
        )
        self.source_activity_method = source_activity_method

        self.num_params = self.count_parameters()
        self.save_hyperparameters(ignore=["source_activity_method"])

    def forward_(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mix = x["input"]

        if self.source_activity_method is None:
            activation_times = x["activation_times"]
        else:
            activation_times = self.source_activity_method(x)[
                "estimated_source_activity"
            ]

        estimated_RTFs = self.CBwrapper(
            {
                "input": mix,
                "activation_times": activation_times,
            }
        )

        return {
            "estimated_RTFs": estimated_RTFs,
        }

    # def test_step(self, batch, batch_idx, dataloader_idx = 0):
    #     return super().test_step(batch, batch_idx, dataloader_idx)
