import inspect
import logging
from abc import abstractmethod
from typing import Any, cast

import lightning as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import muse_toolbox.losses as losses
import muse_toolbox.metrics as metrics
import torch
from muse_toolbox.utils import STFTtransform
from torchmetrics import MetricCollection



log = logging.getLogger(__name__)


class BaseLitModel(pl.LightningModule):
    """
    BaseLitModel is an abstract base class for PyTorch Lightning models, designed to handle
    the core training, validation, and testing logic.

    Attributes:
        batch_size (int): The batch size used during training and evaluation.
        model_name (str): The name of the model.
        loss_config (dict | None): Configuration for the loss function.
        optimizer_config (dict | None): Configuration for the optimizer.
        lr_scheduler_config (dict | None): Configuration for the learning rate scheduler.
        criterion (Callable): The loss function initialized based on the provided loss_config.
        metric_collections (dict): A dictionary to store metric collections for test and validation stages.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 1,
        loss_config: dict | None = None,
        optimizer_config: dict | None = None,
        lr_scheduler_config: dict | None = None,
        metrics_train: dict | None = None,
        metrics_val: dict | None = None,
        metrics_test: dict | None = None,
        transform: STFTtransform | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.loss_config = loss_config
        self.model_name = model_name
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.transform = transform
        self.sad_model_name = kwargs.get("sad_model_name", None)

        if self.loss_config is not None:
            loss_name = list(self.loss_config.keys())[0]
            loss_params = self.loss_config[loss_name]
            if loss_params is None:
                loss_params = {}
            if hasattr(losses, loss_name):
                self.criterion = getattr(losses, loss_name)(**loss_params)
            else:
                raise ValueError(f"Loss function {loss_name} not found in losses module.")

        self.metrics_train = metrics_train if metrics_train is not None else {}
        self.metrics_val = metrics_val if metrics_val is not None else {}
        self.metrics_test = metrics_test if metrics_test is not None else {}
        self._extra_metric_kwargs = dict(kwargs)

        self.metric_collections = {
            "train": self._build_metric_collection(self.metrics_train),
            "val": self._build_metric_collection(self.metrics_val),
            "test": self._build_metric_collection(self.metrics_test),
        }
        self.test_dataset_names: list[str] = []
        self.test_dataset_max_sources: dict[str, int | None] = {}
        self.test_metric_collections: dict[str, MetricCollection] = {}

    def _build_metric_collection(
        self,
        metric_config: dict[str, Any],
        stage_override: str | None = None,
        max_sources_override: int | None = None,
        skip_confusion_matrix: bool = False,
    ) -> MetricCollection:
        """Instantiates a `MetricCollection` from a metric configuration dictionary.

        Args:
            metric_config (dict[str, Any]): Mapping of metric class names to parameter dicts.
            stage_override (str | None, optional): Optional stage prefix override
                (e.g., ``"test/ami"``) passed to metrics accepting ``stage``.
            max_sources_override (int | None, optional): Optional dataset-specific
                ``max_sources`` override passed to metrics accepting ``max_sources``.
            skip_confusion_matrix (bool, optional): If True, disables any
                ``ConfusionMatrix`` instance in this collection (used for the pooled
                overall collection when test datasets have heterogeneous ``max_sources``).

        Returns:
            MetricCollection: Configured `torchmetrics.MetricCollection`.
        """
        metric_list = []
        for met_name, met_params in metric_config.items():
            if not met_name:
                continue

            metric_class = getattr(metrics, met_name)
            all_params: dict[str, Any] = {
                **(met_params or {}),
                "transform": self.transform,
                "model_name": self.model_name,
                "sad_model_name": self.sad_model_name,
                **self._extra_metric_kwargs,
            }
            if stage_override is not None and "stage" in all_params:
                all_params["stage"] = stage_override
            if max_sources_override is not None and "max_sources" in all_params:
                all_params["max_sources"] = max_sources_override

            sig = inspect.signature(metric_class.__init__)
            valid_params = {
                k: v for k, v in all_params.items() if k in sig.parameters
            }
            metric_instance = metric_class(**valid_params)
            if skip_confusion_matrix and met_name == "ConfusionMatrix":
                if hasattr(metric_instance, "enabled"):
                    metric_instance.enabled = False
            metric_list.append(metric_instance)

        return MetricCollection(metric_list, compute_groups=False)

    def configure_test_datasets(
        self,
        dataset_names: list[str],
        dataset_max_sources: list[int | None] | dict[str, int | None] | None = None,
    ) -> None:
        """Configures per-dataset and pooled overall test metric collections.

        Args:
            dataset_names (list[str]): Ordered list of test dataset names matching
                the order of test dataloaders passed to ``trainer.test()``.
            dataset_max_sources (list[int | None] | dict[str, int | None] | None, optional):
                Optional per-dataset ``max_sources`` values. If multiple test datasets
                have different ``max_sources``, the pooled overall ``ConfusionMatrix``
                is disabled while each per-dataset ``ConfusionMatrix`` uses its own
                ``max_sources``.
        """
        self.test_dataset_names = list(dataset_names)
        max_sources_map: dict[str, int | None] = {}
        if isinstance(dataset_max_sources, dict):
            for name in self.test_dataset_names:
                max_sources_map[name] = dataset_max_sources.get(name)
        elif isinstance(dataset_max_sources, list):
            for idx, name in enumerate(self.test_dataset_names):
                max_sources_map[name] = (
                    dataset_max_sources[idx]
                    if idx < len(dataset_max_sources)
                    else None
                )
        else:
            for name in self.test_dataset_names:
                max_sources_map[name] = None
        self.test_dataset_max_sources = max_sources_map

        known_max_sources = {
            ms for ms in max_sources_map.values() if ms is not None
        }
        # Also compare against the model/config default max_sources in metrics_test if present
        cfg_cm = (self.metrics_test or {}).get("ConfusionMatrix") or {}
        default_ms = cfg_cm.get("max_sources")
        all_known_ms = set(known_max_sources)
        if default_ms is not None and len(self.test_dataset_names) <= 1 and not all_known_ms:
            all_known_ms.add(int(default_ms))

        heterogeneous_max_sources = len(known_max_sources) > 1
        if heterogeneous_max_sources:
            log.info(
                f"Test datasets have heterogeneous max_sources ({sorted(known_max_sources)}); "
                "disabling combined overall ConfusionMatrix while keeping per-dataset ConfusionMatrix instances."
            )

        single_override_ms = (
            next(iter(known_max_sources))
            if len(known_max_sources) == 1
            else None
        )
        self.metric_collections["test"] = self._build_metric_collection(
            self.metrics_test,
            stage_override="test",
            max_sources_override=single_override_ms,
            skip_confusion_matrix=heterogeneous_max_sources,
        )
        self.test_metric_collections = {
            ds_name: self._build_metric_collection(
                self.metrics_test,
                stage_override=f"test/{ds_name}",
                max_sources_override=max_sources_map.get(ds_name),
                skip_confusion_matrix=False,
            )
            for ds_name in self.test_dataset_names
        }

    def get_test_dataset_name(self, dataloader_idx: int) -> str:
        """Returns the dataset identifier corresponding to a test dataloader index.

        Args:
            dataloader_idx (int): Index of the active test dataloader.

        Returns:
            str: Dataset identifier string.
        """
        if 0 <= dataloader_idx < len(self.test_dataset_names):
            return self.test_dataset_names[dataloader_idx]
        return f"dataset_{dataloader_idx}"

    @abstractmethod
    def forward(self, batch: dict) -> dict:
        """Perform a forward pass through the model."""
        raise NotImplementedError

    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Total number of parameters that require gradients.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configures the optimizer and learning rate scheduler for PyTorch Lightning.

        Returns:
            OptimizerLRScheduler: A dictionary containing the 'optimizer' and optionally the 'lr_scheduler'.

        Raises:
            ValueError: If the model has no trainable parameters, or if the optimizer 
                configuration is missing or invalid.
        """
        if self.count_parameters() == 0:
            raise ValueError(
                "The model has no trainable parameters. No optimizer/training/fitting needed."
            )

        # --- 1. Configure Optimizer ---
        if self.optimizer_config is None:
            raise ValueError("No optimizer configuration provided.")

        opt_name = list(self.optimizer_config.keys())[0]
        opt_params = dict(self.optimizer_config[opt_name] or {})
        weight_decay = float(opt_params.pop("weight_decay", 0.0))

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in trainable_params if p.ndim >= 2]
        no_decay_params = [p for p in trainable_params if p.ndim < 2]
        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

        if hasattr(torch.optim, opt_name):
            optimizer = getattr(torch.optim, opt_name)(param_groups, **opt_params)
        else:
            raise ValueError(f"Optimizer {opt_name} not found in torch.optim")

        # --- 2. Configure Scheduler ---
        if self.lr_scheduler_config is None:
            return cast(OptimizerLRScheduler, {"optimizer": optimizer})

        sched_name = list(self.lr_scheduler_config.keys())[0]
        sched_params = dict(self.lr_scheduler_config[sched_name] or {})

        # Handle special cases for certain schedulers
        # For example, if "total_steps" is specified as "estimated_stepping_batches",
        # replace it with the actual estimated stepping batches from the trainer.
        if sched_name == "OneCycleLR":
            if total_steps := sched_params.pop("total_steps", None):
                if isinstance(total_steps, str) and total_steps == "estimated_stepping_batches":
                    sched_params["total_steps"] = int(self.trainer.estimated_stepping_batches)

        # Extract Lightning-specific scheduler config keys if present
        # These are keys that Lightning expects in the lr_scheduler dict, but the torch scheduler does not.
        lightning_keys = ["monitor", "interval", "frequency", "strict", "name"]
        lightning_config = {k: sched_params.pop(k) for k in lightning_keys if k in sched_params}

        if hasattr(torch.optim.lr_scheduler, sched_name):
            scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, **sched_params)
        else:
            raise ValueError(f"Scheduler {sched_name} not found in torch.optim.lr_scheduler")

        lr_scheduler_dict = {"scheduler": scheduler, **lightning_config}
        return cast(OptimizerLRScheduler, {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict})



    def _flatten_and_mask(self, preds: torch.Tensor, targets: list[torch.Tensor], time_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Flattens predictions and targets, filtering out the padded elements based on time_lengths.
        """
        transform = getattr(self, "transform", None)
        if transform is None:
            raise ValueError("STFTtransform is required to calculate valid frames for loss masking.")
            
        valid_frames = transform.samples2frames(time_lengths)
        B, max_frames = preds.shape[0], preds.shape[1]
        
        padded_targets = torch.zeros((B, max_frames), device=preds.device, dtype=torch.long)
        for i, target in enumerate(targets):
            length = target.shape[0]
            valid_len = min(length, max_frames)
            padded_targets[i, :valid_len] = target.to(preds.device)[:valid_len]
            
        mask = torch.arange(max_frames, device=preds.device).unsqueeze(0) < valid_frames.unsqueeze(1)
        
        valid_preds = preds[mask]
        valid_targets = padded_targets[mask]
        
        return valid_preds.unsqueeze(0), valid_targets.unsqueeze(0)

    def _common_step(
        self, batch: dict, idx: int, step_type: str, dataloader_idx: int = 0
    ) -> tuple[dict, dict]:
        """
        Executes a common forward pass, computes the loss, and logs the results.

        Args:
            batch: The input batch (dict).
            idx (int): The batch index.
            step_type (str): The step type (e.g., 'train', 'val', 'test') used as a prefix for logging.
            dataloader_idx (int, optional): Active dataloader index. Defaults to 0.

        Returns:
            tuple[dict, dict]: A tuple containing the loss dictionary and the processed batch.
        """
        processed_batch = self(batch)
        
        preds = processed_batch["estimates"]
        targets = processed_batch["meta"]["source_count"]
        time_lengths = processed_batch["time_lengths"]
        
        valid_preds, valid_targets = self._flatten_and_mask(preds, targets, time_lengths)
        loss_targets = valid_targets
        if (
            step_type != "train"
            and valid_preds.ndim >= 2
            and valid_preds.shape[-1] > 1
            and loss_targets.numel() > 0
        ):
            # Guard evaluation loss if a zero-shot test dataset has higher max_sources than the trained classifier head.
            loss_targets = loss_targets.clamp(min=0, max=valid_preds.shape[-1] - 1)
        loss_dict = {"loss": self.criterion.compute_loss(valid_preds, loss_targets)}

        if step_type == "test" and len(self.test_dataset_names) > 1:
            ds_name = self.get_test_dataset_name(dataloader_idx)
            log_payload = {
                f"test/{ds_name}/{x}": y for x, y in loss_dict.items()
            }
            self.log_dict(
                log_payload,
                on_step=True,
                on_epoch=True,
                reduce_fx="mean",
                batch_size=self.batch_size,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )
        else:
            self.log_dict(
                {f"{step_type}/{x}": y for x, y in loss_dict.items()},
                on_step=True,
                on_epoch=True,
                reduce_fx="mean",
                batch_size=self.batch_size,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        return loss_dict, processed_batch

    def _sample_length_to_frame_count(
        self, sample_len: int, max_frames: int, target_len: int
    ) -> int:
        """Converts a waveform sample length to the valid STFT frame count.

        Args:
            sample_len (int): Waveform duration in time samples.
            max_frames (int): Maximum available frame dimension in the model output.
            target_len (int): Frame length of the ground-truth target sequence.

        Returns:
            int: Clamped number of valid STFT frames.
        """
        transform = getattr(self, "transform", None)
        if transform is not None and hasattr(transform, "samples2frames"):
            frame_count = int(transform.samples2frames(sample_len).item())
        elif transform is not None and hasattr(transform, "hop_length"):
            frame_count = sample_len // int(transform.hop_length)
        else:
            frame_count = sample_len
        return max(1, min(frame_count, max_frames, target_len))

    def _prepare_eval_estimates_and_targets(
        self,
        processed_batch: dict[str, Any],
        step_type: str,
        dataloader_idx: int = 0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
        """Slices batch estimates and targets to valid STFT frame lengths for metric evaluation.

        Subclasses (such as `OSCmodule`) may override this hook to apply sliding-window
        overlap logit averaging or causal burn-in frame trimming before updating metrics.

        Args:
            processed_batch (dict[str, Any]): Output dictionary from `forward()`.
            step_type (str): Evaluation stage identifier (`"val"` or `"test"`).
            dataloader_idx (int, optional): Active dataloader index. Defaults to 0.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]: Tuple of
                `(valid_estimates, valid_targets, meta_dict)`.
        """
        meta_dict = processed_batch["meta"].copy()
        estimates = processed_batch["estimates"]
        time_lengths = processed_batch["time_lengths"]
        targets = meta_dict["source_count"]

        valid_estimates: list[torch.Tensor] = []
        valid_targets: list[torch.Tensor] = []
        max_frames = int(estimates.shape[1])

        for i in range(len(time_lengths)):
            sample_len = int(time_lengths[i].item())
            target_i = targets[i].to(estimates.device)
            valid_len = self._sample_length_to_frame_count(
                sample_len=sample_len,
                max_frames=max_frames,
                target_len=int(target_i.shape[0]),
            )
            valid_estimates.append(estimates[i, :valid_len])
            valid_targets.append(target_i[:valid_len])

        if "recording_id" not in meta_dict:
            scenario_ids = meta_dict.get("scenario_id")
            if isinstance(scenario_ids, list) and len(scenario_ids) == len(valid_estimates):
                meta_dict["recording_id"] = list(scenario_ids)
            else:
                meta_dict["recording_id"] = [
                    f"rec_{dataloader_idx}_{i}" for i in range(len(valid_estimates))
                ]

        return valid_estimates, valid_targets, meta_dict

    def _compute_and_log_stitched_eval_loss(
        self,
        estimates: list[torch.Tensor],
        targets: list[torch.Tensor],
        step_type: str,
        dataloader_idx: int = 0,
    ) -> None:
        """Computes and logs frame-weighted cross-entropy loss on stitched eval frames.

        Args:
            estimates (list[torch.Tensor]): Stitched per-chunk logit tensors ``(T_i, C)``.
            targets (list[torch.Tensor]): Stitched per-chunk integer count targets ``(T_i,)``.
            step_type (str): Evaluation stage (``"val"`` or ``"test"``).
            dataloader_idx (int, optional): Active dataloader index. Defaults to ``0``.
        """
        if not estimates or not hasattr(self, "criterion"):
            return

        cat_preds = torch.cat(estimates, dim=0).unsqueeze(0)
        cat_targets = torch.cat(targets, dim=0).unsqueeze(0).long()
        num_frames = int(cat_targets.numel())
        if num_frames == 0:
            return

        if cat_preds.shape[-1] > 1:
            cat_targets = cat_targets.clamp(min=0, max=cat_preds.shape[-1] - 1)

        loss_val = self.criterion.compute_loss(cat_preds, cat_targets)
        if step_type == "test" and len(self.test_dataset_names) > 1:
            ds_name = self.get_test_dataset_name(dataloader_idx)
            log_payload = {f"test/{ds_name}/loss": loss_val}
        else:
            log_payload = {f"{step_type}/loss": loss_val}

        self.log_dict(
            log_payload,
            on_step=False,
            on_epoch=True,
            reduce_fx="mean",
            batch_size=num_frames,
            prog_bar=False,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def _metric_step(
        self,
        processed_batch: dict,
        dataloader_idx: int,
        step_type: str,
        prepared: tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]] | None = None,
    ) -> None:
        """
        Updates the metric collections based on the estimates from the forward pass.

        Args:
            processed_batch: The processed batch containing estimates and metadata.
            dataloader_idx (int): The index of the dataloader.
            step_type (str): The step type ('val', 'test').
            prepared (tuple | None, optional): Pre-stitched ``(estimates, targets, meta_dict)``
                tuple so stateful chunk stitching is not executed twice.
        """
        if prepared is None:
            estimates, targets, meta_dict = self._prepare_eval_estimates_and_targets(
                processed_batch, step_type, dataloader_idx=dataloader_idx
            )
        else:
            estimates, targets, meta_dict = prepared
        if not estimates:
            return

        meta_dict["dataloader_idx"] = len(estimates) * [dataloader_idx]

        if step_type == "test":
            ds_name = self.get_test_dataset_name(dataloader_idx)
            if ds_name not in self.test_metric_collections:
                self.test_metric_collections[ds_name] = self._build_metric_collection(
                    self.metrics_test,
                    stage_override=f"test/{ds_name}",
                )
            ds_meta = meta_dict.copy()
            ds_meta["dataset"] = len(estimates) * [ds_name]
            self.test_metric_collections[ds_name].update(
                estimates, targets, ds_meta, dataloader_idx
            )

            overall_meta = meta_dict.copy()
            overall_meta["dataset"] = len(estimates) * [ds_name]
            if len(self.test_dataset_names) > 1 and "scenario_id" in overall_meta:
                overall_meta["scenario_id"] = [
                    f"{ds_name}/{sid}" for sid in overall_meta["scenario_id"]
                ]
            self.metric_collections[step_type].update(
                estimates, targets, overall_meta, dataloader_idx
            )
        else:
            self.metric_collections[step_type].update(
                estimates, targets, meta_dict, dataloader_idx
            )

    def compute_test_metrics_summary(self) -> dict[str, Any]:
        """Computes per-dataset, overall pooled (micro), and overall macro test metrics.

        Returns:
            dict[str, Any]: Dictionary with keys ``"per_dataset"``
                (``dict[str, dict[str, Any]]``), ``"overall_pooled"``
                (``dict[str, Any]``), and ``"overall_macro"`` (``dict[str, Any]``).
        """
        overall_pooled = self.metric_collections["test"].compute()
        per_dataset: dict[str, dict[str, Any]] = {}
        for ds_name, collection in self.test_metric_collections.items():
            per_dataset[ds_name] = collection.compute()

        overall_macro: dict[str, Any] = {}
        if per_dataset:
            all_keys: list[str] = []
            for ds_metrics in per_dataset.values():
                for k in ds_metrics.keys():
                    if k not in all_keys:
                        all_keys.append(k)

            for k in all_keys:
                vals: list[float] = []
                ref_tensor: torch.Tensor | None = None
                for ds_metrics in per_dataset.values():
                    if k not in ds_metrics:
                        continue
                    v = ds_metrics[k]
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            vals.append(float(v.detach().cpu().item()))
                            ref_tensor = v
                    elif isinstance(v, (int, float)):
                        vals.append(float(v))
                if vals:
                    mean_val = sum(vals) / len(vals)
                    overall_macro[k] = (
                        torch.tensor(mean_val, dtype=torch.float32, device=ref_tensor.device)
                        if ref_tensor is not None
                        else mean_val
                    )

        return {
            "per_dataset": per_dataset,
            "overall_pooled": overall_pooled,
            "overall_macro": overall_macro,
        }

    def reset_test_metrics(self) -> None:
        """Resets both the pooled overall and per-dataset test metric collections."""
        self.metric_collections["test"].reset()
        for collection in self.test_metric_collections.values():
            collection.reset()

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        """
        Defines the training step.

        Args:
            batch: The training batch.
            idx (int): The batch index.

        Returns:
            torch.Tensor: The computed scalar loss for backpropagation.
        """
        return self._common_step(batch, idx, "train")[0]["loss"]

    def validation_step(self, batch: dict, idx: int, dataloader_idx: int = 0) -> None:
        """
        Defines the validation step using continuous stitched outputs for loss and metrics.

        Args:
            batch: The validation batch.
            idx (int): The batch index.
            dataloader_idx (int, optional): The index of the dataloader.
        """
        _ = idx
        processed_batch = self(batch)
        prepared = self._prepare_eval_estimates_and_targets(
            processed_batch, "val", dataloader_idx=dataloader_idx
        )
        estimates, targets, _ = prepared
        self._compute_and_log_stitched_eval_loss(
            estimates, targets, step_type="val", dataloader_idx=dataloader_idx
        )
        self._metric_step(processed_batch, dataloader_idx, "val", prepared=prepared)

    def test_step(self, batch: dict, idx: int, dataloader_idx: int = 0) -> None:
        """
        Defines the test step using continuous stitched outputs for loss and metrics.

        Args:
            batch: The test batch.
            idx (int): The batch index.
            dataloader_idx (int, optional): The index of the dataloader.
        """
        _ = idx
        processed_batch = self(batch)
        prepared = self._prepare_eval_estimates_and_targets(
            processed_batch, "test", dataloader_idx=dataloader_idx
        )
        estimates, targets, _ = prepared
        self._compute_and_log_stitched_eval_loss(
            estimates, targets, step_type="test", dataloader_idx=dataloader_idx
        )
        self._metric_step(processed_batch, dataloader_idx, "test", prepared=prepared)

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """
        Defines the prediction step.

        Args:
            batch: The prediction batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.

        Returns:
            processed_batch: The processed batch.
        """
        return self(batch)

    def on_validation_epoch_end(self) -> None:
        """Computes and logs validation metrics at the end of the validation epoch."""
        metrics_output_dict = self.metric_collections["val"].compute()

        # 2. Create a new dictionary with the 'val/' prefix for logging
        prefixed_metrics = {f"val/{k}": v for k, v in metrics_output_dict.items()}

        # 3. Log the prefixed dictionary
        self.log_dict(prefixed_metrics, sync_dist=True)

        # 4. Reset the metrics for the next epoch
        self.metric_collections["val"].reset()
