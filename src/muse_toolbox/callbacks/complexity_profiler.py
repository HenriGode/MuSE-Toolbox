"""Callback for computing model complexity metrics via PyTorch Profiler."""

import logging
import torch
import wandb
from lightning.pytorch import Callback, Trainer, LightningModule
from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch
from muse_toolbox.utils import format_flops, format_memory, format_parameters, format_time

log = logging.getLogger(__name__)


class ComplexityProfilerCallback(Callback):
    """Computes and logs computational complexity (FLOPs, MACs, timings) for a model."""

    def __init__(self, warmup_steps: int = 2, active_steps: int = 10) -> None:
        """Initializes the complexity profiler.

        Args:
            warmup_steps (int): Number of warmup steps for the profiler. Default is 2.
            active_steps (int): Number of active recording steps. Default is 10.
        """
        super().__init__()
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Triggers the complexity check before training starts."""
        self._compute_complexity_metrics(pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Triggers the complexity check before testing starts."""
        self._compute_complexity_metrics(pl_module)

    def _compute_complexity_metrics(self, pl_module: LightningModule) -> None:
        """
        Computes and logs computational complexity for the model.

        Args:
            pl_module (LightningModule): The PyTorch Lightning module to profile.
        """
        if not torch.cuda.is_available():
            log.info("CUDA not available, skipping GPU profiling.")

        # Safely check if complexity metrics are disabled via model hyperparameters
        compute_complexity = getattr(pl_module, "compute_complexity_metrics", None)
        if compute_complexity is None and hasattr(pl_module, "hparams"):
            compute_complexity = pl_module.hparams.get("compute_complexity_metrics", True)
        if compute_complexity is None:
            compute_complexity = True

        if not compute_complexity:
            log.info("Complexity profiling disabled for this model (compute_complexity_metrics=False).")
            return

        log.info("Computing complexity metrics...")

        # Basic Model & Input Setup
        total_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        num_channels = 2
        fs = getattr(pl_module, "fs", 16000)
        input_len = 1 * fs # 1 second signal

        schedule = torch.profiler.schedule(
            wait=1, warmup=self.warmup_steps, active=self.active_steps, repeat=1
        )
        original_device = pl_module.device
        original_mode = pl_module.training
        pl_module.eval()

        # --- CPU Profiling ---
        original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        pl_module.to("cpu")

        # Pre-allocate random tensor to avoid profiling randn overhead
        cpu_audio = torch.randn(num_channels, input_len, device="cpu")

        with torch.no_grad():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=schedule,
                with_flops=True,
                profile_memory=True,
                record_shapes=False,
            ) as prof_cpu:
                for _ in range(1 + self.warmup_steps + self.active_steps):
                    batch = HeterogeneousBatch(raw_audio=[cpu_audio])
                    batch.to(pl_module.device)
                    _ = pl_module(batch)
                    prof_cpu.step()

        prof_result_cpu = prof_cpu.key_averages()
        torch.set_num_threads(original_num_threads)

        # --- GPU Profiling ---
        prof_result_gpu = None
        if torch.cuda.is_available():
            pl_module.to("cuda")
            gpu_audio = cpu_audio.to("cuda")
            with torch.no_grad():
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=schedule,
                ) as prof_gpu:
                    for _ in range(1 + self.warmup_steps + self.active_steps):
                        batch = HeterogeneousBatch(raw_audio=[gpu_audio])
                        batch.to(pl_module.device)
                        _ = pl_module(batch)
                        prof_gpu.step()
            prof_result_gpu = prof_gpu.key_averages()

        # --- Aggregate Results ---
        total_cpu_time_us = sum(item.cpu_time_total for item in prof_result_cpu)
        inference_time_cpu_ms = (total_cpu_time_us / self.active_steps) / 1000.0

        total_flops = sum(item.flops for item in prof_result_cpu) if prof_result_cpu else 0
        total_macs = total_flops / 2
        memory_used_bytes = sum(item.cpu_memory_usage for item in prof_result_cpu)

        audio_len_s = input_len / fs
        rtf_cpu = (inference_time_cpu_ms / 1000.0) / audio_len_s

        inference_time_gpu_ms = 0.0
        gpu_memory_used_bytes = 0
        rtf_gpu = 0.0
        if prof_result_gpu:
            total_gpu_time_us = sum(item.self_device_time_total for item in prof_result_gpu)
            inference_time_gpu_ms = (total_gpu_time_us / self.active_steps) / 1000.0
            gpu_memory_used_bytes = sum(getattr(item, "device_memory_usage", 0) for item in prof_result_gpu)
            rtf_gpu = (inference_time_gpu_ms / 1000.0) / audio_len_s

        # --- Format and Print ---
        try:
            param_str = format_parameters(total_params)
            mem_str = format_memory(memory_used_bytes)
            gpu_mem_str = format_memory(gpu_memory_used_bytes)
            flops_str = format_flops(total_flops)
            macs_str = format_flops(total_macs)
            cpu_time_str = format_time(inference_time_cpu_ms / 1000.0, detailed=True)
            gpu_time_str = format_time(inference_time_gpu_ms / 1000.0, detailed=True)
            audio_len_str = format_time(audio_len_s, detailed=True)
        except (ImportError, AttributeError):
            log.warning("Could not import formatting utilities. Printing raw values.")
            param_str = str(total_params)
            mem_str = f"{memory_used_bytes} B"
            gpu_mem_str = f"{gpu_memory_used_bytes} B"
            flops_str = str(total_flops)
            macs_str = str(total_macs)
            cpu_time_str = f"{inference_time_cpu_ms:.2f} ms"
            gpu_time_str = f"{inference_time_gpu_ms:.2f} ms"
            audio_len_str = f"{audio_len_s:.2f} s"

        table = f"""
        +--------------------------------+-------------------------+
        |    Computational Complexity    |          Value          |
        +--------------------------------+-------------------------+
        | Trainable Parameters           | {param_str:<23} |
        | Audio Duration                 | {audio_len_str:<23} |
        | Inference Time (CPU)           | {cpu_time_str:<23} |
        | Inference Time (GPU)           | {gpu_time_str:<23} |
        | Real-Time Factor (CPU)         | {rtf_cpu:<23.4f} |
        | Real-Time Factor (GPU)         | {rtf_gpu:<23.4f} |
        | Memory Usage (CPU)             | {mem_str:<23} |
        | Memory Usage (GPU)             | {gpu_mem_str:<23} |
        | GFLOPs                         | {flops_str:<23} |
        | GMACs                          | {macs_str:<23} |
        +--------------------------------+-------------------------+
        """
        log.info("\n" + table)

        # Store in model for CSV saving compatibility, if needed
        setattr(pl_module, "complexity_metrics", {
            "total_params": total_params,
            "inference_time_cpu_ms": inference_time_cpu_ms,
            "inference_time_gpu_ms": inference_time_gpu_ms,
            "rtf_cpu": rtf_cpu,
            "rtf_gpu": rtf_gpu,
            "memory_used_bytes": memory_used_bytes,
            "gpu_memory_used_bytes": gpu_memory_used_bytes,
            "gflops": total_flops / 1e9,
            "gmacs": total_macs / 1e9,
        })

        if wandb.run:
            wandb.run.summary["complexity/total_params"] = total_params
            wandb.run.summary["complexity/inference_time_cpu_ms"] = inference_time_cpu_ms
            wandb.run.summary["complexity/inference_time_gpu_ms"] = inference_time_gpu_ms
            wandb.run.summary["complexity/rtf_cpu"] = rtf_cpu
            wandb.run.summary["complexity/rtf_gpu"] = rtf_gpu
            wandb.run.summary["complexity/memory_used_bytes"] = memory_used_bytes
            wandb.run.summary["complexity/gpu_memory_used_bytes"] = gpu_memory_used_bytes
            wandb.run.summary["complexity/gflops"] = total_flops / 1e9
            wandb.run.summary["complexity/gmacs"] = total_macs / 1e9

        pl_module.to(original_device)
        
        # Restore original training mode
        pl_module.train(original_mode)
