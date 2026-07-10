import torch
from muse_toolbox.models.common.base_model import BaseLitModel
from typing import Optional
from muse_toolbox.utils import STFTtransform, HeterogeneousBatch
from muse_toolbox.models.building_blocks.cgmm import PriorCGMM
import numpy as np
from torchaudio.transforms import MVDR as TorchMVDR


class OnlineCGMM_MVDR(BaseLitModel):
    def __init__(
        self,
        transform: STFTtransform,
        max_sources: int,
        chunk_size: int,  # number of frames per chunk
        batch_size: int = 1,
        loss_config: dict = {"CrossEntropy": None},
        optimizer_config: Optional[dict] = None,
        lr_scheduler_config: Optional[dict] = None,
        metrics_train: Optional[dict] = None,
        metrics_val: Optional[dict] = None,
        metrics_test: Optional[dict] = None,
        compute_complexity_metrics: bool = False,
        check_causality: bool = False,
    ):
        super().__init__(
            model_name="OnlineCGMM",
            batch_size=batch_size,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            metrics_test=metrics_test,
            compute_complexity_metrics=compute_complexity_metrics,
            check_causality=check_causality,
            transform=transform,
        )

        self.max_sources = max_sources
        self.chunk_size = chunk_size
        self.stft_mat_for_init = None  # To be set before running forward_
        self.beamformer = TorchMVDR(
            ref_channel=0,
            solution="stv_evd",
            multi_mask=False,
            diag_loading=True,
            diag_eps=1e-6,
            online=True,
        )

    def forward_(self, batch: HeterogeneousBatch):

        results = []

        # 2. Level 1 Loop
        # Iterate over each scenario in the batch
        for b_idx in range(batch.batch_size):
            # Extract single scenario data (handling padding if necessary)
            # You might need lengths from batch.meta['seq_len'] to slice correctly

            # Delegate to Level 2
            scenario_result = self._forward_scenario(
                stft=batch.stft_audio[b_idx],
                oracle_rirs=batch.meta["scenario_params"][b_idx]["rirs"],
            )
            results.append(scenario_result)

        batch.estimates = results  # List of list of tensors

        batch.print_summary()

        return batch

    def _forward_scenario(self, stft, oracle_rirs):
        """_summary_

        Args:
            batch (_type_): _description_
            b_idx (_type_): _description_
        """
        stft_mat = stft.permute(1, 0, 2)
        stft_mat_for_init = self._get_prior_from_training_data(oracle_rirs)
        M, valid_n_fft, T = stft_mat.shape

        cgmmEngine = [
            PriorCGMM(stft_mat_for_init[:, i, :], K=self.max_sources + 1)
            for i in range(valid_n_fft)
        ]

        # For each chunk, do MAP estimation to simulate online update
        mask_results = torch.empty((valid_n_fft, self.max_sources + 1, 0))
        stft_out = torch.empty((valid_n_fft, 0))
        chunk_num = int(T / self.chunk_size)
        for c in range(chunk_num):
            # ==== Online CGMM
            offset = self.chunk_size * c
            for i in range(valid_n_fft):
                cgmmEngine[i].run(stft_mat[:, i, offset : offset + self.chunk_size])
            # Get the spatial covariance matrix
            R = torch.tensor(
                [cgmmEngine[i].getR() for i in range(valid_n_fft)]
            )  # (valid_n_fft, K, M, M)
            # Get the posterior results
            postArray = torch.tensor(
                [cgmmEngine[i].getPost() for i in range(valid_n_fft)]
            )  # (valid_n_fft, K, T)
            if c == 0:
                mask_results = postArray
            else:
                mask_results = torch.cat([mask_results, postArray], dim=2)
            # === MVDR
            Rv, Rx = R[:, 0, :, :], R[:, 1, :, :]
            # Do MVDR by using Rv and Rx
            bf_mask = torch.tensor(
                mask_results[:, 1, offset : offset + self.chunk_size]
            )
            stft_out_online = self.beamformer(
                stft_mat[:, :, offset : offset + self.chunk_size],
                bf_mask,
                (1 - bf_mask),
            )  # (valid_n_fft, T)
            if c == 0:
                stft_out = stft_out_online
            else:
                stft_out = torch.cat([stft_out, stft_out_online], dim=1)

        # OLA back to wav form
        wav_out = ...

    def _get_prior_from_training_data(self, oracle_rirs) -> torch.Tensor:
        # Implement a method to extract prior CGMM parameters from training data
        pass
