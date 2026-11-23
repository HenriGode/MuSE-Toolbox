import logging
from typing import Any
import torch
import functools

from muse_toolbox.models.base_model import BaseLitModel
from muse_toolbox.models.components.feature_extractors import BaseFeatureExtractor
from muse_toolbox.models.source_counting.estimators import BaseSourceCountEstimator
from muse_toolbox.models.components.channel_combinator.base_channel_combinator import BaseChannelCombinator
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class COSADmodule(BaseLitModel):
    """
    A PyTorch Lightning module for Continuous Online Source Activity Detection (COSAD).

    This module orchestrates a two-stage pipeline:
    1. Feature Extraction: Processes raw STFT input into discriminative features.
    2. Source Count Estimation: Uses the extracted features to estimate the number of 
       active sources over time.
    """

    def __init__(
        self,
        transform: STFTtransform,
        feature_extractor: BaseFeatureExtractor,
        source_count_estimator: BaseSourceCountEstimator,
        channel_combinator: BaseChannelCombinator | Any | None = None,
        batch_size: int = 1,
        loss_config: dict[str, Any] = {"CrossEntropy": None},
        optimizer_config: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
        metrics_train: dict[str, Any] | None = None,
        metrics_val: dict[str, Any] | None = None,
        metrics_test: dict[str, Any] | None = None,
        compute_complexity_metrics: bool = False,
        check_causality: bool = False,
        permute_channels: bool = True,
        eval_stitch_mode: str = "auto",
    ) -> None:
        """
        Initializes the COSADmodule.

        Args:
            transform (utilities.STFTtransform): The STFT configuration for transforming audio signals.
            feature_extractor (BaseFeatureExtractor): Module responsible for extracting features from the STFT.
            source_count_estimator (BaseSourceCountEstimator): Module responsible for predicting source activity.
            channel_combinator (Optional[BaseChannelCombinator]): Module for condensing channel features.
            batch_size (int): Batch size for processing.
            loss_config (dict[str, Any]): Configuration for the loss function.
            optimizer_config (Optional[dict[str, Any]]): Configuration for the optimizer.
            lr_scheduler_config (Optional[dict[str, Any]]): Configuration for the learning rate scheduler.
            metrics_train (Optional[dict[str, Any]]): Metrics to track during training.
            metrics_val (Optional[dict[str, Any]]): Metrics to track during validation.
            metrics_test (Optional[dict[str, Any]]): Metrics to track during testing.
            compute_complexity_metrics (bool): Whether to profile computational complexity.
            check_causality (bool): Whether to enforce causality checks on the model.
            permute_channels (bool): If True, randomly permutes channels during training.
            eval_stitch_mode (str): Sliding-window evaluation stitching strategy across
                overlapping chunks (`"auto"`, `"overlap_avg"`, `"causal_burnin"`, or `"none"`).
                When set to `"auto"` (default), uses `"causal_burnin"` if
                `source_count_estimator.causal` is `True`, and `"overlap_avg"` (Cornell et al.'s
                overlapping sliding-window logit averaging) when `causal` is `False`.
        """
        if isinstance(feature_extractor, functools.partial):
            feature_extractor = feature_extractor(transform=transform)
            
        if isinstance(channel_combinator, functools.partial):
            channel_combinator = channel_combinator(
                input_feature_dim=feature_extractor.feature_dim
            )
            
        estimator_input_dim = feature_extractor.feature_dim
        # If a channel combinator is present, it handles channel condensation but preserves feature dim
        # (or provides an explicit out_feature_dim if it changes it).
        if channel_combinator is not None:
            if hasattr(channel_combinator, 'out_feature_dim'):
                estimator_input_dim = channel_combinator.out_feature_dim

        if isinstance(source_count_estimator, functools.partial):
            source_count_estimator = source_count_estimator(
                input_dim=estimator_input_dim,
                transform=transform
            )
            
        super().__init__(
            model_name=f"COSAD_{feature_extractor.__class__.__name__}_{source_count_estimator.__class__.__name__}",
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

        self.transform = transform

        # Assign injected dependencies
        self.feature_extractor = feature_extractor
        self.channel_combinator = channel_combinator
        self.source_count_estimator = source_count_estimator
        self.permute_channels = permute_channels
        self.eval_stitch_mode = eval_stitch_mode
        self._eval_overlap_state: dict[str, dict[str, Any] | None] = {
            "val": None,
            "test": None,
        }

        self.num_params = self.count_parameters()

        # Save hyperparameters, but ignore the complex objects (modules)
        # as they are part of the model structure, not just config params.
        self.save_hyperparameters(
            ignore=["feature_extractor", "source_count_estimator", "channel_combinator", "transform"]
        )

        # Manually save the configs of the injected dependencies
        self.hparams["feature_config"] = self.feature_extractor.get_config()
        if self.channel_combinator is not None and hasattr(self.channel_combinator, "get_config"):
            self.hparams["combinator_config"] = self.channel_combinator.get_config()
        self.hparams["estimator_config"] = self.source_count_estimator.get_config()
        self.hparams["transform_config"] = self.transform.get_config()

        self._verbose_parameters()

    def _verbose_parameters(self, indent: str = "") -> None:
        """
        Logs the parameters of the module and its sub-components in a structured format.

        Args:
            indent (str): A string to prepend to each log line for indentation.
        """
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Batch Size: {self.batch_size}")
        log.info(f"{indent}  Test Metrics: {self.metrics_test}")
        
        if hasattr(self.transform, "_verbose_parameters"):
            self.transform._verbose_parameters(indent=indent + "  ")

        if hasattr(self.feature_extractor, "_verbose_parameters"):
            self.feature_extractor._verbose_parameters(indent=indent + "  ")
        else:
            log.info(
                f"{indent}  Feature Extractor: {self.feature_extractor.__class__.__name__}"
            )
            
        if self.channel_combinator is not None:
            if hasattr(self.channel_combinator, "_verbose_parameters"):
                self.channel_combinator._verbose_parameters(indent=indent + "  ")
            else:
                log.info(
                    f"{indent}  Channel Combinator: {self.channel_combinator.__class__.__name__}"
                )

        if hasattr(self.source_count_estimator, "_verbose_parameters"):
            self.source_count_estimator._verbose_parameters(indent=indent + "  ")
        else:
            log.info(
                f"{indent}  Source Count Estimator: {self.source_count_estimator.__class__.__name__}"
            )

    def forward(self, batch: dict) -> dict:
        """
        Executes the forward pass of the COSAD pipeline using raw dictionary batches.
        """
        x = batch["input"]
        valid_mics = batch.get("mic_lengths", None)
        
        # 0. Data Augmentation
        if self.permute_channels and self.training:
            B, M, T = x.shape
            for i in range(B):
                m_len = valid_mics[i].item() if valid_mics is not None else M
                if m_len > 1:
                    perm = torch.randperm(m_len, device=x.device)
                    x[i, :m_len, :] = x[i, perm, :]

        # 1. Feature Extraction
        # Note: the feature extractor is responsible for calling self.transform() internally if it uses STFT.
        features = self.feature_extractor(x, valid_mics=valid_mics)
        
        # 1.5 Generate Valid Feature Mask
        feature_mask = self.feature_extractor.get_valid_feature_mask(valid_mics, max_M=x.shape[1])
        batch["feature_mask"] = feature_mask
        
        # 2. Channel Combination (Optional)
        if self.channel_combinator is not None:
            features = self.channel_combinator(features, feature_mask=feature_mask)

        # 3. Detection
        estimates = self.source_count_estimator(features)
        
        return {
            "estimates": estimates,
            "time_lengths": batch["time_lengths"],
            "meta": batch["meta"]
        }

    def get_active_eval_stitch_mode(self) -> str:
        """Resolves the active evaluation sliding-window stitching mode.

        Returns:
            str: One of `"overlap_avg"`, `"causal_burnin"`, or `"none"`.

        Raises:
            ValueError: If `self.eval_stitch_mode` is not a recognized mode.
        """
        mode = (self.eval_stitch_mode or "auto").lower()
        if mode == "auto":
            is_causal = bool(getattr(self.source_count_estimator, "causal", False))
            return "causal_burnin" if is_causal else "overlap_avg"
        if mode in {"overlap_avg", "causal_burnin", "none"}:
            return mode
        raise ValueError(
            f"Unsupported eval_stitch_mode='{self.eval_stitch_mode}'. "
            "Expected 'auto', 'overlap_avg', 'causal_burnin', or 'none'."
        )

    def on_validation_epoch_start(self) -> None:
        """Resets sliding-window overlap buffer state at the start of validation."""
        self._eval_overlap_state["val"] = None

    def on_test_epoch_start(self) -> None:
        """Resets sliding-window overlap buffer state at the start of testing."""
        self._eval_overlap_state["test"] = None

    def _prepare_eval_estimates_and_targets(
        self, processed_batch: dict[str, Any], step_type: str
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
        """Stitches or trims overlapping sliding-window chunks during evaluation.

        Supports two unified sliding-window schemes on deterministic evaluation grids:
        1. `"overlap_avg"` (Cornell et al., 2022 non-causal baseline):
           Applies a sliding window (default 3.0 s with 50% / 1.5 s overlap) and averages
           the unnormalized class logits across overlapping frame regions of consecutive
           chunks from the same recording before computing frame-level predictions.
        2. `"causal_burnin"` (Strictly causal online chunking):
           Applies a sliding window (e.g., 7.0 s with 2.0 s overlap) where the first chunk
           of a recording is evaluated in full (`[0 : T_win]`), and every subsequent chunk
           discards its initial `overlap_frames` burn-in warmup prefix (`[0 : O]`) and
           evaluates only the newly advancing causal tail (`[O : T_win]`) with zero future
           look-ahead or overlap averaging.

        If chunk grid metadata (`"recording_id"`, `"start_sample"`, `"hop_samples"`) is not
        present in `processed_batch["meta"]` or `stitch_mode == "none"`, falls back to full
        valid-frame slicing via `super()._prepare_eval_estimates_and_targets()`.

        Args:
            processed_batch (dict[str, Any]): Forward pass output dictionary.
            step_type (str): Evaluation stage (`"val"` or `"test"`).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]: Stitched
                per-chunk `(estimates, targets, meta_dict)` lists ready for metric update.
        """
        stitch_mode = self.get_active_eval_stitch_mode()
        meta = processed_batch["meta"]
        has_grid_meta = (
            isinstance(meta, dict)
            and "recording_id" in meta
            and "start_sample" in meta
            and "hop_samples" in meta
        )
        if stitch_mode == "none" or not has_grid_meta:
            return super()._prepare_eval_estimates_and_targets(
                processed_batch, step_type
            )

        meta_dict = meta.copy()
        estimates = processed_batch["estimates"]
        time_lengths = processed_batch["time_lengths"]
        targets = meta_dict["source_count"]
        max_frames = int(estimates.shape[1])
        hop_length = (
            int(self.transform.hop_length)
            if self.transform is not None and hasattr(self.transform, "hop_length")
            else 1
        )

        recording_ids = meta_dict["recording_id"]
        start_samples = meta_dict["start_sample"]
        hop_samples_list = meta_dict["hop_samples"]
        is_last_chunks = meta_dict.get(
            "is_last_chunk", [False] * len(time_lengths)
        )

        valid_estimates: list[torch.Tensor] = []
        valid_targets: list[torch.Tensor] = []

        for i in range(len(time_lengths)):
            sample_len = int(time_lengths[i].item())
            target_i = targets[i].to(estimates.device)
            valid_len = self._sample_length_to_frame_count(
                sample_len=sample_len,
                max_frames=max_frames,
                target_len=int(target_i.shape[0]),
            )
            est_i = estimates[i, :valid_len]
            tgt_i = target_i[:valid_len]

            rec_id = str(recording_ids[i])
            start_sample = int(start_samples[i])
            hop_samples = int(hop_samples_list[i])
            is_last_chunk = bool(is_last_chunks[i])

            hop_frames = max(1, min(valid_len, hop_samples // hop_length))
            overlap_frames = max(0, valid_len - hop_frames)

            if stitch_mode == "causal_burnin":
                if start_sample == 0 or overlap_frames == 0:
                    valid_estimates.append(est_i)
                    valid_targets.append(tgt_i)
                else:
                    valid_estimates.append(est_i[overlap_frames:valid_len])
                    valid_targets.append(tgt_i[overlap_frames:valid_len])

            elif stitch_mode == "overlap_avg":
                ov_frames = min(hop_frames, overlap_frames)
                state = self._eval_overlap_state.get(step_type)
                is_consecutive = (
                    state is not None
                    and state["recording_id"] == rec_id
                    and state["next_start_sample"] == start_sample
                    and state["tail_logits"].shape[0] == ov_frames
                )

                if not is_consecutive or ov_frames == 0:
                    # First chunk of recording (or no overlap): emit non-overlapping head [:hop_frames]
                    emitted_est = est_i[:hop_frames]
                    emitted_tgt = tgt_i[:hop_frames]
                else:
                    # Consecutive overlapping chunk: average previous tail with current head [:ov_frames]
                    prev_tail_logits = state["tail_logits"].to(est_i.device)
                    avg_overlap = 0.5 * (prev_tail_logits + est_i[:ov_frames])
                    if hop_frames > ov_frames:
                        emitted_est = torch.cat(
                            [avg_overlap, est_i[ov_frames:hop_frames]], dim=0
                        )
                    else:
                        emitted_est = avg_overlap
                    emitted_tgt = tgt_i[:hop_frames]

                if is_last_chunk or ov_frames == 0:
                    if valid_len > hop_frames:
                        emitted_est = torch.cat(
                            [emitted_est, est_i[hop_frames:valid_len]], dim=0
                        )
                        emitted_tgt = torch.cat(
                            [emitted_tgt, tgt_i[hop_frames:valid_len]], dim=0
                        )
                    self._eval_overlap_state[step_type] = None
                else:
                    new_tail_logits = (
                        est_i[hop_frames : hop_frames + ov_frames].detach().clone()
                    )
                    new_tail_targets = (
                        tgt_i[hop_frames : hop_frames + ov_frames].detach().clone()
                    )
                    self._eval_overlap_state[step_type] = {
                        "recording_id": rec_id,
                        "next_start_sample": start_sample + hop_samples,
                        "tail_logits": new_tail_logits,
                        "tail_targets": new_tail_targets,
                    }

                valid_estimates.append(emitted_est)
                valid_targets.append(emitted_tgt)

        return valid_estimates, valid_targets, meta_dict

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """
        Executes a prediction step.

        Computes the forward pass and returns the processed batch, ensuring 
        estimates are condensed to binary discrete decisions if configured.

        Args:
            batch: The batch of data to process.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the dataloader.

        Returns:
            processed_batch: The processed batch, ready for evaluation.
        """
        predictions = self(batch)
        est = predictions["estimates"]
        if est is not None:
            predictions["estimates"] = torch.argmax(est, dim=-1, keepdim=True)
        return predictions
