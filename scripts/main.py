"""Main entry point for MuSE-Toolbox experiments.

This script acts as the Hydra execution entry point, handling global setup
and dispatching to the correct pipeline based on the provided configuration.
"""

import os
from pathlib import Path

# Resolve absolute path to project root and set environment variable for Hydra
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import hydra
import torch
from omegaconf import DictConfig

from muse_toolbox.pipelines.source_counting_pipeline import run_source_counting_pipeline
from muse_toolbox.pipelines.rtf_estimation_pipeline import run_rtf_estimation_pipeline

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main execution script that dispatches to the correct pipeline based on config.

    Example HPC execution:
        python scripts/main.py task=rtf_estimation model.learning_rate=0.001

    Tasks supported:
        - source_counting: Runs only the source counting (SAD) training/eval.
        - rtf_estimation: Runs only the RTF training/eval (using oracle or precomputed SAD).
        - joint: Runs source counting (predicting to disk), then passes those
                 predictions directly into the RTF estimation pipeline.

    Args:
        cfg (DictConfig): The hierarchical Hydra configuration object.

    Raises:
        ValueError: If the requested task is unknown.
    """
    # 1. Global Setup
    matmul_precision = cfg.get("matmul_precision", "highest")
    torch.set_float32_matmul_precision(matmul_precision)
    log.info(f"PyTorch matmul precision set to: {matmul_precision}")
    
    task = cfg.get("task", "source_counting")
    log.info(f"Starting MuSE-Toolbox Experiment with task: {task}")
    
    if task == "source_counting":
        run_source_counting_pipeline(cfg)
        
    elif task == "rtf_estimation":
        run_rtf_estimation_pipeline(cfg)
        
    elif task == "joint":
        log.info("=== Running Joint Pipeline ===")
        # 1. Run source counting and get the directory where predictions were saved
        # We enforce prediction saving in joint mode
        from omegaconf import OmegaConf
        
        sc_cfg = OmegaConf.create(cfg) # duplicate to avoid mutating global
        sc_cfg.predict = True
        if "predictions_dir" not in sc_cfg:
            sc_cfg.predictions_dir = "predictions/joint_run"
            
        predictions_dir = run_source_counting_pipeline(sc_cfg)
        
        # 2. Run RTF estimation, dynamically pointing it to the predictions we just made
        log.info(f"=== Moving to RTF Estimation with predictions from {predictions_dir} ===")
        rtf_cfg = OmegaConf.create(cfg)
        rtf_cfg.use_oracle_activations = False
        
        run_rtf_estimation_pipeline(rtf_cfg, custom_predictions_dir=predictions_dir)
        
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'source_counting', 'rtf_estimation', or 'joint'.")

if __name__ == "__main__":
    main()
