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

from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch

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

        # Helper function to initialize metrics
        def _create_metric_list(metric_config: dict) -> list:
            metric_list = []
            for met_name, met_params in metric_config.items():
                if not met_name:
                    continue

                metric_class = getattr(metrics, met_name)

                # Combine model-level kwargs with metric-specific params from config
                all_params = {
                    **(met_params or {}),
                    **{
                        "transform": self.transform,
                        "model_name": self.model_name,
                        "sad_model_name": self.sad_model_name,
                        **kwargs,
                    },
                }

                # Inspect the metric's __init__ signature
                sig = inspect.signature(metric_class.__init__)

                # Filter the combined params to only include what the metric accepts
                valid_params = {
                    k: v for k, v in all_params.items() if k in sig.parameters
                }

                metric_list.append(metric_class(**valid_params))
            return metric_list

        self.metric_collections = {
            "train": MetricCollection(
                _create_metric_list(self.metrics_train), compute_groups=False
            ),
            "val": MetricCollection(
                _create_metric_list(self.metrics_val), compute_groups=False
            ),
            "test": MetricCollection(
                _create_metric_list(self.metrics_test), compute_groups=False
            ),
        }

    @abstractmethod
    def forward_(self, batch: HeterogeneousBatch) -> HeterogeneousBatch:
        """
        Abstract method to define the forward pass of the model.

        Must be implemented in subclasses to define the core computation.

        Args:
            batch (HeterogeneousBatch): The input batch data.

        Returns:
            HeterogeneousBatch: The processed batch.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_dict(self, batch: dict) -> dict:
        """
        Abstract method to define the forward pass for dict-based batches.
        """
        raise NotImplementedError

    def forward(self, batch: dict | HeterogeneousBatch) -> dict | HeterogeneousBatch:
        """Perform a forward pass through the model."""
        if isinstance(batch, HeterogeneousBatch):
            return self.forward_(batch)
        elif isinstance(batch, dict):
            return self.forward_dict(batch)
        else:
            raise NotImplementedError(f"Unsupported batch type: {type(batch)}")

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
        opt_params = self.optimizer_config[opt_name] or {}

        if hasattr(torch.optim, opt_name):
            optimizer = getattr(torch.optim, opt_name)(self.parameters(), **opt_params)
        else:
            raise ValueError(f"Optimizer {opt_name} not found in torch.optim")

        # --- 2. Configure Scheduler ---
        if self.lr_scheduler_config is None:
            return cast(OptimizerLRScheduler, {"optimizer": optimizer})

        sched_name = list(self.lr_scheduler_config.keys())[0]
        sched_params = self.lr_scheduler_config[sched_name] or {}

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

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        """
        Overrides PyTorch Lightning's default batch transfer logic to seamlessly handle 
        custom `HeterogeneousBatch` objects.

        Args:
            batch (Any): The input batch, typically a HeterogeneousBatch.
            device (torch.device): The target device (e.g., 'cuda:0', 'cpu').
            dataloader_idx (int): The index of the dataloader providing the batch.

        Returns:
            Any: The batch moved to the target device.
        """
        if isinstance(batch, HeterogeneousBatch):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _flatten_and_mask(self, preds: torch.Tensor, targets: list[torch.Tensor], time_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Flattens predictions and targets, filtering out the padded elements based on time_lengths.
        """
        if not hasattr(self, "transform") or getattr(self, "transform") is None:
            raise ValueError("STFTtransform is required to calculate valid frames for loss masking.")
            
        valid_frames = self.transform.samples2frames(time_lengths)
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

    def _common_step(self, batch: dict | HeterogeneousBatch, idx: int, step_type: str) -> tuple[dict, dict | HeterogeneousBatch]:
        """
        Executes a common forward pass, computes the loss, and logs the results.

        Args:
            batch: The input batch (dict or HeterogeneousBatch).
            idx (int): The batch index.
            step_type (str): The step type (e.g., 'train', 'val', 'test') used as a prefix for logging.

        Returns:
            tuple[dict, dict | HeterogeneousBatch]: A tuple containing the loss dictionary and the processed batch.
        """
        processed_batch = self(batch)
        
        if isinstance(processed_batch, dict):
            preds = processed_batch["estimates"]
            targets = processed_batch["meta"]["source_count"]
            time_lengths = processed_batch["time_lengths"]
            
            valid_preds, valid_targets = self._flatten_and_mask(preds, targets, time_lengths)
            loss_dict = {"loss": self.criterion.compute_loss(valid_preds, valid_targets)}
        else:
            loss_dict = processed_batch.compute_loss(self.criterion)
            
        self.log_dict(
            {f"{step_type}/{x}": y for x, y in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            reduce_fx="mean",
            batch_size=self.batch_size,
            prog_bar=False,
            sync_dist=True,
        )

        return loss_dict, processed_batch

    def _metric_step(
        self, processed_batch: dict | HeterogeneousBatch, dataloader_idx: int, step_type: str
    ) -> None:
        """
        Updates the metric collections based on the estimates from the forward pass.

        Args:
            processed_batch: The processed batch containing estimates and metadata.
            dataloader_idx (int): The index of the dataloader.
            step_type (str): The step type ('val', 'test').
        """
        if isinstance(processed_batch, dict):
            meta_dict = processed_batch["meta"].copy()
            estimates = processed_batch["estimates"]
            time_lengths = processed_batch["time_lengths"]
            targets = meta_dict["source_count"]
            
            # Slice the estimates and targets to their valid lengths to form lists
            valid_estimates = []
            valid_targets = []
            for i in range(len(time_lengths)):
                valid_len = time_lengths[i]
                valid_estimates.append(estimates[i, :valid_len])
                valid_targets.append(targets[i][:valid_len].to(estimates.device))
                
            estimates = valid_estimates
            targets = valid_targets
        else:
            meta_dict = processed_batch.meta.copy()
            estimates = processed_batch.estimates
            targets = meta_dict["source_count"]
            
        meta_dict["dataloader_idx"] = self.batch_size * [dataloader_idx]
        self.metric_collections[step_type].update(
            estimates, targets, meta_dict, dataloader_idx
        )

    def training_step(self, batch: dict | HeterogeneousBatch, idx: int) -> torch.Tensor:
        """
        Defines the training step.

        Args:
            batch: The training batch.
            idx (int): The batch index.

        Returns:
            torch.Tensor: The computed scalar loss for backpropagation.
        """
        return self._common_step(batch, idx, "train")[0]["loss"]

    def validation_step(self, batch: dict | HeterogeneousBatch, idx: int, dataloader_idx: int = 0) -> None:
        """
        Defines the validation step.

        Args:
            batch: The validation batch.
            idx (int): The batch index.
            dataloader_idx (int, optional): The index of the dataloader.
        """
        _, processed_batch = self._common_step(batch, idx, "val")
        self._metric_step(processed_batch, dataloader_idx, "val")

    def test_step(self, batch: dict | HeterogeneousBatch, idx: int, dataloader_idx: int = 0) -> None:
        """
        Defines the test step.

        Args:
            batch: The test batch.
            idx (int): The batch index.
            dataloader_idx (int, optional): The index of the dataloader.
        """
        _, processed_batch = self._common_step(batch, idx, "test")
        self._metric_step(processed_batch, dataloader_idx, "test")

    def predict_step(
        self, batch: dict | HeterogeneousBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> dict | HeterogeneousBatch:
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
