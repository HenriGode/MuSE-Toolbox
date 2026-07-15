import inspect
import logging
import os
from abc import abstractmethod
from typing import Any

import lightning as pl
import losses
import metrics
import pandas as pd
import torch
import utilities
from muse_toolbox.utils import STFTtransform
import wandb
from torchmetrics import MetricCollection

from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch

log = logging.getLogger(__name__)


class BaseLitModel(pl.LightningModule):
    """
    BaseLitModel is an abstract base class for PyTorch Lightning models, designed to handle various functionalities
    such as training, validation, testing, and model complexity evaluation. It also supports processing long utterances
    and includes mechanisms for handling NaN gradients and logging metrics.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size used during training and evaluation.
        loss (str): The name of the loss function to use.
        model_name (str): The name of the model.
        my_optimizer (str): The optimizer to use, default is "AdamW".
        my_lr_scheduler (str): The learning rate scheduler to use, default is "ReduceLROnPlateau".
        compute_complexity_metrics (bool): Whether to compute complexity metrics for the model.
        check_causality (bool): Whether to check the causality of the model.
        save_target (bool): Whether to save the target signals during testing.
        process_long_utterance (bool): Whether to process long utterances in chunks.
        nan_batch_counter (float): Counter for batches with NaN gradients.
        criterion (Callable): The loss function initialized based on the provided loss name.
        metrics_test (Union[tuple, str]): Metrics to evaluate during testing.
        metrics_val (Union[tuple, str]): Metrics to evaluate during validation.
        metric_collections (dict): A dictionary to store metric collections for test and validation stages.
        complexity_metrics (dict): Stores computed complexity metrics.
        test_outputs (list): A list to store test outputs.

    Methods:
        forward_(x): Abstract method to define the forward pass of the model. Must be implemented in subclasses.
        forward(x): Handles the forward pass, optionally processing long utterances in chunks.
        count_parameters(): Counts the number of trainable parameters in the model.
        configure_optimizers(): Configures the optimizer and learning rate scheduler.
        on_after_backward(): Handles NaN gradients by setting them to zero and logging the NaN batch counter.
        training_step(batch, idx): Defines the training step, computes the loss, and logs training metrics.
        validation_step(batch, idx, dataloader_idx): Defines the validation step, computes the loss, and logs validation metrics.
        test_step(batch, batch_idx, dataloader_idx): Defines the test step, computes metrics, and saves processed audio files.
        on_test_epoch_end(): Collects test results and saves them to a CSV file.
        train_dataloader(): Returns the training dataloader from the datamodule.
        save_individual_wave(dataloader_idx, wave_tensor, filename): Saves individual waveforms to disk.
        on_fit_start(): Executes tasks at the start of training, such as computing complexity metrics or checking causality.
        on_test_start(): Executes tasks at the start of testing, such as computing complexity metrics or checking causality.
        compute_complexity_metrics_fn(): Computes and logs complexity metrics such as inference time, memory usage, and FLOPS.
        check_causality_fn(allowed_latency_s): Checks whether the model satisfies the causality requirement within the allowed latency.
        log_wave(batch, idx, output, stage): Logs audio waveforms (noisy and enhanced) to WandB.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 1,
        loss_config: dict | None = None,
        optimizer_config: dict | None = None,
        lr_scheduler_config: dict | None = None,
        compute_complexity_metrics: bool = False,
        check_causality: bool = False,
        metrics_train: dict | None = None,
        metrics_val: dict | None = None,
        metrics_test: dict | None = None,
        transform: STFTtransform | None = None,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.loss_config = loss_config
        self.model_name = model_name
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.compute_complexity_metrics = compute_complexity_metrics
        self.check_causality = check_causality
        self.transform = transform

        self.nan_batch_counter = 0.0

        if self.loss_config is not None:
            loss_name = list(self.loss_config.keys())[0]
            loss_params = self.loss_config[loss_name]
            if loss_params is None:
                loss_params = {}
            if hasattr(losses, loss_name):
                self.criterion = getattr(losses, loss_name)(**loss_params)
            else:
                # raise an error that the loss function is not found in the losses module
                raise ValueError(
                    f"Loss function {loss_name} not found in losses module."
                )

        self.metrics_train = metrics_train if metrics_train is not None else {}
        self.metrics_val = metrics_val if metrics_val is not None else {}
        self.metrics_test = metrics_test if metrics_test is not None else {}

        self.sad_model_name = kwargs.get("sad_model_name", None)

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
                _create_metric_list(self.metrics_train),
                compute_groups=False,
            ),
            "val": MetricCollection(
                _create_metric_list(self.metrics_val),
                compute_groups=False,
            ),
            "test": MetricCollection(
                _create_metric_list(self.metrics_test),
                compute_groups=False,
            ),
        }

        self.complexity_metrics = None

        self.test_outputs = []

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

    def forward(self, batch: dict) -> dict:
        """
        Perform a forward pass through the model.
        """

        return self.forward_(batch)

    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Total number of parameters that require gradients.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configures the optimizer and learning rate scheduler for PyTorch Lightning.

        Returns:
            dict[str, Any]: A dictionary containing the 'optimizer' and optionally the 'lr_scheduler'.

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
        opt_params = self.optimizer_config[opt_name]
        if opt_params is None:
            opt_params = {}

        if hasattr(torch.optim, opt_name):
            optimizer = getattr(torch.optim, opt_name)(self.parameters(), **opt_params)
        else:
            raise ValueError(f"Optimizer {opt_name} not found in torch.optim")

        # --- 2. Configure Scheduler ---
        if self.lr_scheduler_config is None:
            return {"optimizer": optimizer}

        sched_name = list(self.lr_scheduler_config.keys())[0]
        sched_params = self.lr_scheduler_config[sched_name]
        if sched_params is None:
            sched_params = {}

        # Handle special cases for certain schedulers
        # For example, if "total_steps" is specified as "estimated_stepping_batches",
        # replace it with the actual estimated stepping batches from the trainer.
        if sched_name == "OneCycleLR":
            if total_steps := sched_params.pop("total_steps", None):
                if (
                    isinstance(total_steps, str)
                    and total_steps == "estimated_stepping_batches"
                ):
                    sched_params["total_steps"] = int(
                        self.trainer.estimated_stepping_batches
                    )

        # Extract Lightning-specific scheduler config keys if present
        # These are keys that Lightning expects in the lr_scheduler dict, but the torch scheduler does not.
        lightning_keys = ["monitor", "interval", "frequency", "strict", "name"]
        lightning_config = {
            k: sched_params.pop(k) for k in lightning_keys if k in sched_params
        }

        if hasattr(torch.optim.lr_scheduler, sched_name):
            scheduler = getattr(torch.optim.lr_scheduler, sched_name)(
                optimizer, **sched_params
            )
        else:
            raise ValueError(
                f"Scheduler {sched_name} not found in torch.optim.lr_scheduler"
            )

        # Construct the final configuration dictionary
        lr_scheduler_dict = {"scheduler": scheduler}
        lr_scheduler_dict.update(lightning_config)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}



    def on_after_backward(self) -> None:
        """
        Handles NaN gradients after the backward pass by setting them to zero.
        Increments and logs the `nan_batch_counter` if any NaN gradients are detected.
        """
        increase_nan_batch_counter = False
        for param in self.parameters():
            if param.grad is not None:
                nan_grads = torch.isnan(param.grad)
                if torch.any(nan_grads):
                    param.grad[nan_grads] = 0.0
                    increase_nan_batch_counter = True
        if increase_nan_batch_counter:
            self.nan_batch_counter += 1

        self.log(
            "ptl/nan_batch_counter",
            self.nan_batch_counter,
            batch_size=self.batch_size,
        )
        return super().on_after_backward()

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

    def _get_targets_from_meta(
        self, batch: dict | list[dict]
    ) -> list[torch.Tensor] | torch.Tensor:
        """
        Extracts source count targets from the batch metadata.

        Args:
            batch (dict | list[dict]): The input batch or list of sub-batches.

        Returns:
            list[torch.Tensor] | torch.Tensor: The extracted source count targets.

        Raises:
            ValueError: If the batch format is unsupported.
        """
        if isinstance(batch, list):

            targets = []
            for b in batch:
                subbatch_targets = b["meta"]["source_count"]
                try:
                    subbatch_targets = torch.stack(subbatch_targets, dim=0)
                    targets.append(subbatch_targets)
                except Exception:
                    targets.extend([sbt.unsqueeze(0) for sbt in subbatch_targets])

            try:
                targets = torch.cat(targets, dim=0)
            except Exception:
                pass
            return targets

        elif isinstance(batch, dict):
            targets = batch["meta"]["source_count"]
            try:
                return torch.stack(targets, dim=0)
            except Exception:
                pass
            return [sc.unsqueeze(0) for sc in targets]

        else:
            raise ValueError("Batch must be a dict or a list of dicts.")

    def _common_step(self, batch: HeterogeneousBatch, idx: int, step_type: str) -> tuple[dict, HeterogeneousBatch]:
        """
        Executes a common forward pass, computes the loss, and logs the results.

        Args:
            batch (HeterogeneousBatch): The input batch.
            idx (int): The batch index.
            step_type (str): The step type (e.g., 'train', 'val', 'test') used as a prefix for logging.

        Returns:
            tuple[dict, HeterogeneousBatch]: A tuple containing the loss dictionary and the processed batch.
        """
        processed_batch = self(batch)
        loss = processed_batch.compute_loss(self.criterion)
        self.log_dict(
            {f"{step_type}/{x}": y for x, y in loss.items()},
            on_step=True,
            on_epoch=True,
            reduce_fx="mean",
            batch_size=self.batch_size,
            prog_bar=False,
            sync_dist=True,
        )

        return loss, processed_batch

    def _metric_step(
        self, processed_batch: HeterogeneousBatch, dataloader_idx: int, step_type: str
    ) -> None:
        """
        Updates the metric collections based on the estimates from the forward pass.

        Args:
            processed_batch (HeterogeneousBatch): The processed batch containing estimates and metadata.
            dataloader_idx (int): The index of the dataloader.
            step_type (str): The step type ('val', 'test').
        """
        meta_dict = processed_batch.meta.copy()
        meta_dict["dataloader_idx"] = self.batch_size * [dataloader_idx]
        targets = meta_dict["source_count"]
        self.metric_collections[step_type].update(
            processed_batch.estimates, targets, meta_dict, dataloader_idx
        )

    def training_step(self, batch: HeterogeneousBatch, idx: int) -> torch.Tensor:
        """
        Defines the training step.

        Args:
            batch (HeterogeneousBatch): The training batch.
            idx (int): The batch index.

        Returns:
            torch.Tensor: The computed scalar loss for backpropagation.
        """
        return self._common_step(batch, idx, "train")[0]["loss"]

    def validation_step(self, batch: HeterogeneousBatch, idx: int, dataloader_idx: int = 0) -> None:
        """
        Defines the validation step.

        Args:
            batch (HeterogeneousBatch): The validation batch.
            idx (int): The batch index.
            dataloader_idx (int): The index of the dataloader.
        """
        processed_batch = self._common_step(batch, idx, "val")[1]
        self._metric_step(processed_batch, dataloader_idx, "val")

    def test_step(self, batch: HeterogeneousBatch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Defines the test step.

        Args:
            batch (HeterogeneousBatch): The test batch.
            batch_idx (int): The batch index.
            dataloader_idx (int): The index of the dataloader.
        """
        processed_batch = self._common_step(batch, batch_idx, "test")[1]
        self._metric_step(processed_batch, dataloader_idx, "test")

    def on_validation_epoch_end(self) -> None:
        """
        Computes and logs validation metrics at the end of the validation epoch.
        """
        # 1. Compute the aggregated metrics from the 'val' collection
        metrics_output_dict = self.metric_collections["val"].compute()

        # 2. Create a new dictionary with the 'val/' prefix for logging
        prefixed_metrics = {f"val/{k}": v for k, v in metrics_output_dict.items()}

        # 3. Log the prefixed dictionary
        self.log_dict(prefixed_metrics, sync_dist=True)

        # 4. Reset the metrics for the next epoch
        self.metric_collections["val"].reset()

    def on_test_epoch_end(self) -> None:
        # --- Part 1: Log aggregated metrics to W&B ---

        # 1. Compute the aggregated metrics from the 'test' collection
        metrics_output_dict = self.metric_collections["test"].compute()

        # 2. Create a new dictionary with the 'test/' prefix for logging
        prefixed_metrics = {f"test/{k}": v for k, v in metrics_output_dict.items()}

        # 3. Log the prefixed dictionary
        self.log_dict(prefixed_metrics, sync_dist=True)

        # --- Part 2: Collect detailed results and save to CSV ---

        # 4. Collect individual DataFrames from each metric
        dataframes = []
        for metric in self.metric_collections["test"].values():
            if hasattr(metric, "get_dataframe"):
                df = metric.get_dataframe()
                if df is not None and not df.empty:
                    dataframes.append(df)

        if not dataframes:
            log.warning("No dataframes to save from muse_toolbox.metrics.")
            self.metric_collections["test"].reset()
            return

        # 5. Merge all dataframes into one
        combined_df = pd.concat(dataframes, axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        # 6. Add complexity metrics if they exist
        if self.complexity_metrics is not None:
            for key, value in self.complexity_metrics.items():
                combined_df[key] = value

        # 7. Save the combined dataframe to a CSV file
        save_dir = f"./results/J2_RUN/{self.trainer.datamodule.id}/STFT_{self.transform.sampling_frequency}_{self.transform.nfft}_{self.transform.hop_length}/{self.sad_model_name}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.model_name == "BlockOnlineGSS":
            filename = f"{self.model_name}_bl{self.block_size_frames}_pc{self.pre_context_frames}_results.csv"
        else:
            filename = f"{self.model_name}_results.csv"
        filepath = os.path.join(save_dir, filename)

        combined_df.to_csv(filepath, index=True)

        # 8. Reset the metrics
        self.metric_collections["test"].reset()



    def on_fit_start(self) -> None:
        """
        Hook executed at the start of the `fit` stage (training/validation).
        Triggers complexity and causality checks if configured.
        """
        if self.compute_complexity_metrics:
            self.compute_complexity_metrics_fn()
        if self.check_causality:
            self.check_causality_fn()

    def on_test_start(self) -> None:
        """
        Hook executed at the start of the `test` stage.
        Triggers complexity and causality checks if configured.
        """
        if self.compute_complexity_metrics:
            self.compute_complexity_metrics_fn()
        if self.check_causality:
            self.check_causality_fn()

    def compute_complexity_metrics_fn(self):
        """
        Computes and logs computational complexity for the speaker counting model.

        This function measures and logs:
        - Total trainable parameters.
        - Inference time on both CPU and GPU.
        - Memory usage during inference.
        - Total GFLOPs and GMACs.

        The results are printed to the console and logged to the active wandb run.
        It uses the PyTorch profiler with warmup steps for accurate measurements.
        """
        if not torch.cuda.is_available():
            log.info("CUDA not available, skipping GPU profiling.")

        log.info("Computing complexity metrics...")

        # --- 1. Basic Model & Input Setup ---
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Assume 1-channel audio. Your models seem to handle this internally.
        num_channels = 1

        # Use a standard 1-second audio clip for profiling.
        fs = getattr(self, "fs", 16000)
        input_len = fs

        # Profiling parameters
        warmup_steps = 2
        active_steps = 10
        schedule = torch.profiler.schedule(
            wait=1, warmup=warmup_steps, active=active_steps, repeat=1
        )
        self.eval()

        # --- 2. CPU Profiling ---
        original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.to("cpu")

        # Create a dummy input dictionary, as expected by your model's forward pass
        example_input = {"input": torch.randn(1, num_channels, input_len, device="cpu")}

        with torch.no_grad():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=schedule,
                with_flops=True,
                profile_memory=True,
                record_shapes=False,
            ) as prof_cpu:
                for _ in range(1 + warmup_steps + active_steps):
                    _ = self.forward(example_input)
                    prof_cpu.step()

        prof_result_cpu = prof_cpu.key_averages()
        torch.set_num_threads(original_num_threads)

        # --- 3. GPU Profiling ---
        prof_result_gpu = None
        if torch.cuda.is_available():
            self.to("cuda")
            example_input_gpu = {k: v.to("cuda") for k, v in example_input.items()}
            with torch.no_grad():
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=schedule,
                ) as prof_gpu:
                    for _ in range(1 + warmup_steps + active_steps):
                        _ = self.forward(example_input_gpu)
                        prof_gpu.step()
            prof_result_gpu = prof_gpu.key_averages()

        # --- 4. Aggregate and Format Results ---
        # CPU metrics
        total_cpu_time_us = sum(item.cpu_time_total for item in prof_result_cpu)
        inference_time_cpu_ms = (total_cpu_time_us / active_steps) / 1000.0

        # Use FLOPs from CPU profiler as it's more reliable
        total_flops = (
            sum(item.flops for item in prof_result_cpu) if prof_result_cpu else 0
        )
        total_macs = total_flops / 2

        memory_used_bytes = sum(item.cpu_memory_usage for item in prof_result_cpu)

        # GPU metrics
        inference_time_gpu_ms = 0.0
        if prof_result_gpu:
            total_gpu_time_us = sum(
                item.self_cuda_time_total for item in prof_result_gpu
            )
            inference_time_gpu_ms = (total_gpu_time_us / active_steps) / 1000.0

        # --- 5. Print and Log ---
        # Using your existing utilities. If names differ, adjust them here.
        try:
            param_str = utilities.format_parameters(total_params)
            mem_str = utilities.format_memory(memory_used_bytes)
            flops_str = utilities.format_flops(total_flops)
            macs_str = utilities.format_flops(total_macs)
        except (ImportError, AttributeError):
            log.warning(
                "Could not import formatting utilities. Printing raw values."
            )
            param_str = str(total_params)
            mem_str = f"{memory_used_bytes} B"
            flops_str = str(total_flops)
            macs_str = str(total_macs)

        table = f"""
        +--------------------------------+-------------------------+
        |    Computational Complexity    |          Value          |
        +--------------------------------+-------------------------+
        | Trainable Parameters           | {param_str:<23} |
        | Inference Time (CPU, ms)       | {inference_time_cpu_ms:<23.2f} |
        | Inference Time (GPU, ms)       | {inference_time_gpu_ms:<23.2f} |
        | Memory Usage (CPU)             | {mem_str:<23} |
        | GFLOPs                         | {flops_str:<23} |
        | GMACs                          | {macs_str:<23} |
        +--------------------------------+-------------------------+
        """
        log.info("\n" + table)

        if wandb.run:
            wandb.run.summary["complexity/total_params"] = total_params
            wandb.run.summary["complexity/inference_time_cpu_ms"] = (
                inference_time_cpu_ms
            )
            wandb.run.summary["complexity/inference_time_gpu_ms"] = (
                inference_time_gpu_ms
            )
            wandb.run.summary["complexity/memory_used_bytes"] = memory_used_bytes
            wandb.run.summary["complexity/gflops"] = total_flops / 1e9
            wandb.run.summary["complexity/gmacs"] = total_macs / 1e9

        self.complexity_metrics = {
            "total_params": total_params,
            "inference_time_cpu_ms": inference_time_cpu_ms,
            "inference_time_gpu_ms": inference_time_gpu_ms,
            "memory_used_bytes": memory_used_bytes,
            "gflops": total_flops / 1e9,
            "gmacs": total_macs / 1e9,
        }

        # Return model to its original device if it was changed
        self.to(self.device)

    def check_causality_fn(self, allowed_latency_s: float = 0.010) -> None:
        """
        Checks whether the model satisfies the algorithmic causality requirement within 
        the allowed latency.

        Injects NaNs into a synthetic audio signal after a random point and ensures 
        the model's output does not propagate those NaNs backward beyond the allowed 
        latency window.

        Args:
            allowed_latency_s (float): The maximum allowed algorithmic latency in seconds.
        """
        self.eval()

        num_channels = getattr(self, "num_channels", 1)
        binaural = getattr(self, "binaural", False)
        if "Bilat" in self.model_name or self.model_name == "BiConvTasNet":
            binaural = True
        if binaural:
            num_channels *= 2

        allowed_latency = int(allowed_latency_s * self.fs)
        sig_len_range = [2.0, 8.0]
        num_repetitions = 10
        for _ in range(num_repetitions):
            length = (
                int(
                    torch.rand(1).item() * (sig_len_range[1] - sig_len_range[0])
                    + sig_len_range[0]
                )
                * self.fs
            )
            sig = torch.randn(length)
            sig = sig / sig.abs().max() * 0.9
            p = torch.randint(0, length, (1,)).item()
            sig[p:] = float("nan")
            sig = sig.unsqueeze(0).unsqueeze(0)
            sig = sig.repeat(1, num_channels, 1)

            inp = {"input": sig}
            with torch.no_grad():
                est_sig = self.forward(inp)["input_proc"]

            if p - allowed_latency + 1 >= 1 and torch.any(
                torch.isnan(est_sig[0, 0, : p - allowed_latency + 1])
            ):
                log.error(
                    f"Your model does NOT satisfy the algorithmic latency requirement of {allowed_latency_s} s!"
                )
                return

        log.info(
            f"Your model satisfies the algorithmic latency requirement of {allowed_latency_s} s!"
        )
