"""Source counting pipeline for MuSE-Toolbox.

This module handles the training, testing, and prediction phases for
source counting models, dynamically instantiated via Hydra.
"""

import gc
import logging
import os
from pathlib import Path

import torch
import hydra
import lightning as pl
from omegaconf import DictConfig

# Required for HeterogeneousBatch handling when saving predictions
from muse_toolbox.utils import HeterogeneousBatch

log = logging.getLogger(__name__)


def run_source_counting_pipeline(cfg: DictConfig) -> str:
    """Executes the source counting pipeline based on the configuration.
    
    This general pipeline dynamically instantiates any compatible DataModule (e.g. Brudex, PraAnf)
    and Model (e.g. COSADmodule) using Hydra. It trains, tests, and optionally predicts 
    and saves the estimated source activities to disk for downstream use by the RTF pipeline.
    
    Args:
        cfg (DictConfig): The hierarchical Hydra configuration object.
        
    Returns:
        str: The path to the directory where predictions were saved (if applicable).
    """
    log.info("Starting source counting pipeline...")
    
    # 1. Instantiate DataModule
    log.info("Instantiating DataModule...")
    datamodule = hydra.utils.instantiate(cfg.dataset)
    
    # 2. Instantiate Model
    log.info("Instantiating Model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # 3. Setup Logger
    logger = None
    if "logger" in cfg:
        logger = hydra.utils.instantiate(cfg.logger)
        if hasattr(logger, "log_hyperparams"):
            # safely log the config dictionary
            try:
                from omegaconf import OmegaConf
                logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
            except Exception:
                pass
            
    # 4. Setup Callbacks
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))
            
    # 5. Setup Trainer
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    
    # 6. Train and Test
    if cfg.get("train", True):
        log.info("Starting training...")
        trainer.fit(model, datamodule=datamodule)
        
    if cfg.get("test", True):
        log.info("Starting testing...")
        trainer.test(model, datamodule=datamodule)
        
    # 7. Predict and Save (Optional)
    predictions_dir = cfg.get("predictions_dir", None)
    
    if cfg.get("predict", False) and predictions_dir is not None:
        log.info(f"Running prediction and saving results to {predictions_dir}...")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Determine checkpoint path (use best if available)
        ckpt_path = "best" if cfg.get("train", True) else None
        
        # We can predict on the test_dataloader
        if not hasattr(datamodule, "test_ds") or datamodule.test_ds is None:
            datamodule.setup("test")
            
        # Split test dataset into two halves to avoid OOM (Restored from old J1_RUN)
        len_test_ds = len(datamodule.test_ds)
        split_idx = len_test_ds // 2
        indices_parts = [
            list(range(0, split_idx)),
            list(range(split_idx, len_test_ds)),
        ]

        for i, indices in enumerate(indices_parts):
            if not indices:
                continue

            log.info(f"Predicting part {i+1}/2 ({len(indices)} samples)...")
            subset_ds = torch.utils.data.Subset(datamodule.test_ds, indices)
            subset_dl = torch.utils.data.DataLoader(
                subset_ds,
                batch_size=datamodule.batch_size,
                num_workers=datamodule.num_workers,
                shuffle=False,
                collate_fn=datamodule.test_ds.collate_fn,
            )
            
            predictions = trainer.predict(model, dataloaders=subset_dl, ckpt_path=ckpt_path)
            
            # Save predictions using the logic from J1_RUN
            if isinstance(predictions, list):
                for batch in predictions:
                    if isinstance(batch, HeterogeneousBatch):
                        est_activity = batch.estimates
                        if not isinstance(est_activity, list):
                            raise ValueError("Expected batch.estimates to be a list of tensors.")
                            
                        scenario_ids = batch.meta["scenario_id"]
                        
                        for idx, sid in enumerate(scenario_ids):
                            save_path = os.path.join(predictions_dir, f"{sid}.pt")
                            torch.save(est_activity[idx].cpu(), save_path)
                    else:
                        log.warning("Batch is not a HeterogeneousBatch. Custom save logic may be needed.")
            
            # Cleanup memory after each chunk
            del predictions
            gc.collect()
            torch.cuda.empty_cache()
                    
        log.info(f"Predictions successfully saved to {predictions_dir}.")
        return predictions_dir
    
    log.info("Source counting pipeline completed.")
    return predictions_dir
