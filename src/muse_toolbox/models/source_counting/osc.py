import functools
import inspect
import logging
from typing import Any

import torch

from muse_toolbox.models.base_model import BaseLitModel
from muse_toolbox.models.components.feature_extractors import BaseFeatureExtractor
from muse_toolbox.models.source_counting.estimators import BaseSourceCountEstimator
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class OSCmodule(BaseLitModel):
    """PyTorch Lightning module for Online Source Counting (OSC).

    Supports single-stream feature extraction (e.g., Log-Mel, flattened CSIPD, wGMSC)
    and dual-stream spectral + spatial conditioning extraction (e.g., Cornell et al.
    Early Fusion and Late FiLM Fusion), sharing a single STFT computation when both
    extractors operate in the STFT domain.
    """

    def __init__(
        self,
        transform: STFTtransform,
        feature_extractor: BaseFeatureExtractor | Any,
        source_count_estimator: BaseSourceCountEstimator | Any,
        cond_feature_extractor: BaseFeatureExtractor | Any | None = None,
        num_mics: int | None = None,
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
        """Initializes the OSCmodule.

        Args:
            transform (STFTtransform): STFT configuration for transforming audio signals.
            feature_extractor (BaseFeatureExtractor | functools.partial): Primary feature
                extractor module (or Hydra partial accepting ``transform``).
            source_count_estimator (BaseSourceCountEstimator | functools.partial): Source
                count estimator module (or Hydra partial).
            cond_feature_extractor (BaseFeatureExtractor | functools.partial | None, optional):
                Optional spatial conditioning feature extractor for dual-stream fusion
                baselines (Early Fusion or Late FiLM). Defaults to ``None``.
            num_mics (int | None, optional): Fixed microphone count ``M`` used by fixed-array
                baseline estimators to resolve ``M``-dependent channel counts (such as
                ``P = M*(M-1)//2`` pairs for CSIPD) at initialization time. Not required
                for ``C = 1`` extractors (e.g. single-channel Log-Mel, GMSC, wGMSC).
                Defaults to ``None``.
            batch_size (int, optional): Batch size for logging. Defaults to ``1``.
            loss_config (dict[str, Any], optional): Loss function configuration.
            optimizer_config (dict[str, Any] | None, optional): Optimizer configuration.
            lr_scheduler_config (dict[str, Any] | None, optional): LR scheduler configuration.
            metrics_train (dict[str, Any] | None, optional): Training metrics config.
            metrics_val (dict[str, Any] | None, optional): Validation metrics config.
            metrics_test (dict[str, Any] | None, optional): Test metrics config.
            compute_complexity_metrics (bool, optional): Enable complexity profiling.
            check_causality (bool, optional): Enable causality verification checks.
            permute_channels (bool, optional): Randomly permute microphone channels
                during training when ``True``. Defaults to ``True``.
            eval_stitch_mode (str, optional): Sliding-window stitching mode
                (``"auto"``, ``"overlap_avg"``, ``"causal_burnin"``, or ``"none"``).
        """
        if isinstance(feature_extractor, functools.partial):
            feature_extractor = feature_extractor(transform=transform)

        if isinstance(cond_feature_extractor, functools.partial):
            cond_feature_extractor = cond_feature_extractor(transform=transform)

        if isinstance(source_count_estimator, functools.partial):
            est_sig = inspect.signature(source_count_estimator.func.__init__)
            est_params = est_sig.parameters
            est_kwargs: dict[str, Any] = {
                "input_dim": feature_extractor.feature_dim,
                "transform": transform,
            }
            if "num_input_channels" in est_params:
                est_kwargs["num_input_channels"] = feature_extractor.output_channels(
                    num_mics
                )
            elif num_mics is not None:
                in_channels = feature_extractor.output_channels(num_mics)
                est_kwargs["input_dim"] = feature_extractor.feature_dim * in_channels

            if cond_feature_extractor is not None:
                if "cond_input_dim" not in est_params:
                    raise ValueError(
                        f"{source_count_estimator.func.__name__} does not accept `cond_input_dim`. "
                        "Use FusionTransformerEstimator when `cond_feature_extractor` is configured."
                    )
                est_kwargs["cond_input_dim"] = cond_feature_extractor.feature_dim
                if "cond_num_channels" in est_params:
                    est_kwargs["cond_num_channels"] = (
                        cond_feature_extractor.output_channels(num_mics)
                    )

            source_count_estimator = source_count_estimator(**est_kwargs)

        model_tag = (
            f"OSC_{feature_extractor.__class__.__name__}"
            + (
                f"+{cond_feature_extractor.__class__.__name__}"
                if cond_feature_extractor is not None
                else ""
            )
            + f"_{source_count_estimator.__class__.__name__}"
        )

        super().__init__(
            model_name=model_tag,
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
            estimator_causal=bool(getattr(source_count_estimator, "causal", False)),
        )

        self.transform = transform
        self.feature_extractor = feature_extractor
        self.cond_feature_extractor = cond_feature_extractor
        self.source_count_estimator = source_count_estimator
        self.num_mics = num_mics
        self.permute_channels = permute_channels
        self.eval_stitch_mode = eval_stitch_mode
        self._eval_overlap_state: dict[str, dict[str, Any] | None] = {
            "val": None,
            "test": None,
            "predict": None,
        }

        self.num_params = self.count_parameters()

        self.save_hyperparameters(
            ignore=[
                "feature_extractor",
                "cond_feature_extractor",
                "source_count_estimator",
                "transform",
            ]
        )

        self.hparams["feature_config"] = self.feature_extractor.get_config()
        if self.cond_feature_extractor is not None:
            self.hparams["cond_feature_config"] = (
                self.cond_feature_extractor.get_config()
            )
        self.hparams["estimator_config"] = self.source_count_estimator.get_config()
        self.hparams["transform_config"] = self.transform.get_config()

        self._verbose_parameters()

    @property
    def is_single_channel_system(self) -> bool:
        """Returns ``True`` if both primary and optional conditioning extractors use 1 channel.

        Returns:
            bool: ``True`` for single-channel reference models (e.g. single-channel Log-Mel),
                ``False`` for multi-channel or spatial models.
        """
        primary_sc = bool(getattr(self.feature_extractor, "is_single_channel_input", False))
        cond_sc = (
            True
            if self.cond_feature_extractor is None
            else bool(getattr(self.cond_feature_extractor, "is_single_channel_input", False))
        )
        return primary_sc and cond_sc

    def _verbose_parameters(self, indent: str = "") -> None:
        """Logs the parameters of the module and its sub-components.

        Args:
            indent (str, optional): Leading indentation string. Defaults to ``""``.
        """
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Batch Size: {self.batch_size}")
        if self.num_mics is not None:
            log.info(f"{indent}  Configured num_mics: {self.num_mics}")
        log.info(f"{indent}  Test Metrics: {self.metrics_test}")

        if hasattr(self.transform, "_verbose_parameters"):
            self.transform._verbose_parameters(indent=indent + "  ")

        if hasattr(self.feature_extractor, "_verbose_parameters"):
            self.feature_extractor._verbose_parameters(indent=indent + "  ")
        else:
            log.info(
                f"{indent}  Feature Extractor: {self.feature_extractor.__class__.__name__}"
            )

        if self.cond_feature_extractor is not None:
            if hasattr(self.cond_feature_extractor, "_verbose_parameters"):
                self.cond_feature_extractor._verbose_parameters(indent=indent + "  ")
            else:
                log.info(
                    f"{indent}  Conditioning Feature Extractor: "
                    f"{self.cond_feature_extractor.__class__.__name__}"
                )

        if hasattr(self.source_count_estimator, "_verbose_parameters"):
            self.source_count_estimator._verbose_parameters(indent=indent + "  ")
        else:
            log.info(
                f"{indent}  Source Count Estimator: "
                f"{self.source_count_estimator.__class__.__name__}"
            )

    def forward(self, batch: dict) -> dict:
        """Executes the forward pass of the OSC pipeline on raw audio dictionary batches."""
        x = batch["input"]
        valid_mics = batch.get("mic_lengths", None)

        # 0. Data Augmentation (shared across primary & conditioning streams)
        if self.permute_channels and self.training:
            x = x.clone()
            B, M, _ = x.shape
            for i in range(B):
                m_len = int(valid_mics[i].item()) if valid_mics is not None else M
                if m_len > 1:
                    perm = torch.randperm(m_len, device=x.device)
                    x[i, :m_len, :] = x[i, perm, :]

        # 1. Feature Extraction (compute shared STFT once when both extractors use STFT)
        cond_features: torch.Tensor | None = None
        if (
            self.cond_feature_extractor is not None
            and self.feature_extractor.uses_stft
            and self.cond_feature_extractor.uses_stft
        ):
            stft = self.transform.encode(x)
            features = self.feature_extractor(
                stft, input_type="stft", valid_mics=valid_mics
            )
            cond_features = self.cond_feature_extractor(
                stft, input_type="stft", valid_mics=valid_mics
            )
        else:
            features = self.feature_extractor(x, valid_mics=valid_mics)
            if self.cond_feature_extractor is not None:
                cond_features = self.cond_feature_extractor(x, valid_mics=valid_mics)

        batch["feature_mask"] = self.feature_extractor.get_valid_feature_mask(
            valid_mics, max_M=x.shape[1]
        )
        if self.cond_feature_extractor is not None:
            batch["cond_feature_mask"] = (
                self.cond_feature_extractor.get_valid_feature_mask(
                    valid_mics, max_M=x.shape[1]
                )
            )

        # 2. Source Count Estimation
        time_lengths = batch.get("time_lengths", None)
        valid_frames = (
            self.transform.samples2frames(time_lengths)
            if time_lengths is not None
            else None
        )
        est_sig = inspect.signature(self.source_count_estimator.forward)
        est_kwargs: dict[str, Any] = {}
        if "valid_frames" in est_sig.parameters:
            est_kwargs["valid_frames"] = valid_frames

        if cond_features is not None:
            estimates = self.source_count_estimator(
                features, cond_features=cond_features, **est_kwargs
            )
        else:
            estimates = self.source_count_estimator(features, **est_kwargs)

        return {
            "estimates": estimates,
            "time_lengths": batch["time_lengths"],
            "meta": batch["meta"],
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

    def on_validation_epoch_end(self) -> None:
        """Clears overlap buffer state and logs epoch validation metrics."""
        self._eval_overlap_state["val"] = None
        super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        """Resets sliding-window overlap buffer state at the start of testing."""
        self._eval_overlap_state["test"] = None

    def on_test_epoch_end(self) -> None:
        """Clears overlap buffer state at the end of testing."""
        self._eval_overlap_state["test"] = None

    def on_predict_epoch_start(self) -> None:
        """Resets sliding-window overlap buffer state at the start of prediction."""
        self._eval_overlap_state["predict"] = None

    def on_predict_epoch_end(self) -> None:
        """Clears sliding-window overlap buffer state at the end of prediction."""
        self._eval_overlap_state["predict"] = None

    def _prepare_eval_estimates_and_targets(
        self,
        processed_batch: dict[str, Any],
        step_type: str,
        dataloader_idx: int = 0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
        """Stitches or trims overlapping sliding-window chunks during evaluation/prediction.

        Supports two unified sliding-window schemes on deterministic evaluation grids,
        including arbitrarily short trailing end-of-recording chunks:
        1. `"overlap_avg"` (Cornell et al., 2022 non-causal baseline):
           Applies a sliding window (default 3.0 s with 50% / 1.5 s overlap) and averages
           the unnormalized class logits across overlapping frame regions of consecutive
           chunks from the same recording before computing frame-level predictions.
        2. `"causal_burnin"` (Strictly causal online chunking):
           Applies a sliding window (e.g., 7.0 s with 2.0 s overlap) where the first chunk
           of a recording is evaluated in full (`[0 : valid_len]`), and every subsequent chunk
           tracks `emitted_end_frame` per recording to discard its exact overlapping burn-in
           prefix (`emitted_end_frame - start_frame`) and emit only newly advancing frames.

        If chunk grid metadata (`"recording_id"`, `"start_sample"`, `"hop_samples"`) is not
        present in `processed_batch["meta"]` or `stitch_mode == "none"`, falls back to full
        valid-frame slicing (when targets are present) or plain length slicing.

        Args:
            processed_batch (dict[str, Any]): Forward pass output dictionary.
            step_type (str): Evaluation stage (`"val"`, `"test"`, or `"predict"`).
            dataloader_idx (int, optional): Active dataloader index. Defaults to `0`.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]: Stitched
                per-chunk `(estimates, targets, meta_dict)` lists.
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
            if isinstance(meta, dict) and meta.get("source_count") is not None:
                return super()._prepare_eval_estimates_and_targets(
                    processed_batch, step_type, dataloader_idx=dataloader_idx
                )
            estimates = processed_batch["estimates"]
            time_lengths = processed_batch["time_lengths"]
            max_frames = int(estimates.shape[1])
            plain_est: list[torch.Tensor] = []
            for i in range(len(time_lengths)):
                s_len = int(time_lengths[i].item())
                v_len = self._sample_length_to_frame_count(
                    sample_len=s_len, max_frames=max_frames, target_len=max_frames
                )
                plain_est.append(estimates[i, :v_len])
            return plain_est, [], meta.copy() if isinstance(meta, dict) else {}

        meta_dict = meta.copy()
        estimates = processed_batch["estimates"]
        time_lengths = processed_batch["time_lengths"]
        targets = meta_dict.get("source_count", None)
        max_frames = int(estimates.shape[1])
        hop_length = (
            int(self.transform.hop_length)
            if self.transform is not None and hasattr(self.transform, "hop_length")
            else 1
        )

        batch_len = len(time_lengths)
        recording_ids = meta_dict["recording_id"]
        start_samples = meta_dict["start_sample"]
        hop_samples_list = meta_dict["hop_samples"]
        is_last_chunks = meta_dict.get("is_last_chunk", [False] * batch_len)

        valid_estimates: list[torch.Tensor] = []
        valid_targets: list[torch.Tensor] = []
        kept_indices: list[int] = []

        for i in range(batch_len):
            sample_len = int(time_lengths[i].item())
            if targets is not None and i < len(targets) and targets[i] is not None:
                target_i: torch.Tensor | None = targets[i].to(estimates.device)
                target_len = int(target_i.shape[0])
            else:
                target_i = None
                target_len = max_frames

            valid_len = self._sample_length_to_frame_count(
                sample_len=sample_len,
                max_frames=max_frames,
                target_len=target_len,
            )
            est_i = estimates[i, :valid_len]
            tgt_i = target_i[:valid_len] if target_i is not None else None

            rec_id = str(recording_ids[i])
            start_sample = int(start_samples[i])
            hop_samples = int(hop_samples_list[i])
            is_last_chunk = bool(is_last_chunks[i])

            configured_hop_frames = max(1, hop_samples // hop_length)
            start_frame = start_sample // hop_length

            if stitch_mode == "causal_burnin":
                state = self._eval_overlap_state.get(step_type)
                is_same_stream = (
                    state is not None
                    and state.get("dataloader_idx", 0) == dataloader_idx
                    and state.get("recording_id") == rec_id
                    and start_sample > 0
                )
                prev_end = int(state["emitted_end_frame"]) if is_same_stream else 0
                burnin_frames = (
                    min(valid_len, max(0, prev_end - start_frame))
                    if is_same_stream
                    else 0
                )
                emitted_est = est_i[burnin_frames:valid_len]
                emitted_tgt = (
                    tgt_i[burnin_frames:valid_len] if tgt_i is not None else None
                )
                new_end = max(prev_end, start_frame + valid_len)
                if is_last_chunk:
                    self._eval_overlap_state[step_type] = None
                else:
                    self._eval_overlap_state[step_type] = {
                        "dataloader_idx": dataloader_idx,
                        "recording_id": rec_id,
                        "emitted_end_frame": new_end,
                    }

            elif stitch_mode == "overlap_avg":
                state = self._eval_overlap_state.get(step_type)
                is_consecutive = (
                    state is not None
                    and state.get("dataloader_idx", 0) == dataloader_idx
                    and state.get("recording_id") == rec_id
                    and state.get("next_start_sample") == start_sample
                )

                if is_consecutive:
                    assert state is not None
                    prev_tail = state["tail_logits"].to(est_i.device)
                    n_ov = min(int(prev_tail.shape[0]), valid_len)
                    avg_ov = 0.5 * (prev_tail[:n_ov] + est_i[:n_ov])
                    curr_est = torch.cat([avg_ov, est_i[n_ov:valid_len]], dim=0)
                else:
                    curr_est = est_i

                if is_last_chunk or valid_len <= configured_hop_frames:
                    emitted_est = curr_est
                    emitted_tgt = tgt_i
                    self._eval_overlap_state[step_type] = None
                else:
                    emitted_est = curr_est[:configured_hop_frames]
                    emitted_tgt = (
                        tgt_i[:configured_hop_frames] if tgt_i is not None else None
                    )
                    new_tail_logits = (
                        curr_est[configured_hop_frames:valid_len].detach().clone()
                    )
                    self._eval_overlap_state[step_type] = {
                        "dataloader_idx": dataloader_idx,
                        "recording_id": rec_id,
                        "next_start_sample": start_sample + hop_samples,
                        "tail_logits": new_tail_logits,
                    }
            else:
                emitted_est = est_i
                emitted_tgt = tgt_i

            if emitted_est.shape[0] > 0:
                valid_estimates.append(emitted_est)
                if emitted_tgt is not None:
                    valid_targets.append(emitted_tgt)
                kept_indices.append(i)

        if len(kept_indices) != batch_len:
            filtered_meta: dict[str, Any] = {}
            for k, v in meta_dict.items():
                if isinstance(v, list) and len(v) == batch_len:
                    filtered_meta[k] = [v[idx] for idx in kept_indices]
                else:
                    filtered_meta[k] = v
            meta_dict = filtered_meta

        return valid_estimates, valid_targets, meta_dict

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Executes a sliding-window stitched prediction step.

        Runs the forward pass, applies the same ``overlap_avg`` / ``causal_burnin``
        stitched emission logic used during ``val``/``test``, and returns CPU-offloaded
        discrete class count predictions per chunk ready for per-recording concatenation.

        Args:
            batch (dict): Input batch dictionary from `raw_audio_collate_fn`.
            batch_idx (int): Current batch index.
            dataloader_idx (int, optional): Active dataloader index. Defaults to `0`.

        Returns:
            dict[str, Any]: Stitched predictions (`"estimates"` list of 1D tensors)
                and aligned chunk metadata (`"recording_id"`, `"scenario_id"`,
                `"is_last_chunk"`).
        """
        _ = batch_idx
        processed_batch = self(batch)
        stitched_logits, _, meta_dict = self._prepare_eval_estimates_and_targets(
            processed_batch, "predict", dataloader_idx=dataloader_idx
        )
        discrete_chunks = [
            torch.argmax(est_i, dim=-1).detach().cpu() for est_i in stitched_logits
        ]
        batch_len = len(discrete_chunks)
        compact_meta = {
            "recording_id": list(
                meta_dict.get("recording_id", [f"rec_{i}" for i in range(batch_len)])
            ),
            "scenario_id": list(
                meta_dict.get("scenario_id", [f"sc_{i}" for i in range(batch_len)])
            ),
            "is_last_chunk": list(
                meta_dict.get("is_last_chunk", [True] * batch_len)
            ),
        }
        return {
            "estimates": discrete_chunks,
            "meta": compact_meta,
        }
