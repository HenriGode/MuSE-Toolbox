"""Source counting pipeline for MuSE-Toolbox."""

import os
import torch
import hydra
import lightning as pl
from pathlib import Path

# Required for HeterogeneousBatch handling when saving predictions
from muse_toolbox.utils import HeterogeneousBatch


def run_source_counting_pipeline(cfg) -> str:
    """Executes the source counting pipeline based on the provided configuration.
    
    This general pipeline dynamically instantiates any compatible DataModule (e.g. Brudex, PraAnf)
    and Model (e.g. COSADmodule) using Hydra. It trains, tests, and optionally predicts 
    and saves the estimated source activities to disk for downstream use by the RTF pipeline.
    
    Args:
        cfg: The configuration object (from Hydra).
        
    Returns:
        str: The path to the directory where predictions were saved (if applicable).
    """
    print(f"Starting source counting pipeline...")
    
    # 1. Instantiate DataModule
    print("Instantiating DataModule...")
    datamodule = hydra.utils.instantiate(cfg.dataset)
    
    # 2. Instantiate Model
    print("Instantiating Model...")
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
        print("Starting training...")
        trainer.fit(model, datamodule=datamodule)
        
    if cfg.get("test", True):
        print("Starting testing...")
        trainer.test(model, datamodule=datamodule)
        
    # 7. Predict and Save (Optional)
    predictions_dir = cfg.get("predictions_dir", None)
    
    if cfg.get("predict", False) and predictions_dir is not None:
        print(f"Running prediction and saving results to {predictions_dir}...")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Determine checkpoint path (use best if available)
        ckpt_path = "best" if cfg.get("train", True) else None
        
        # We can predict on the test_dataloader
        if not hasattr(datamodule, "test_ds") or datamodule.test_ds is None:
            datamodule.setup("test")
            
        predictions = trainer.predict(model, dataloaders=datamodule.test_dataloader(), ckpt_path=ckpt_path)
        
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
                    print("Warning: Batch is not a HeterogeneousBatch. Custom save logic may be needed.")
                    
        print(f"Predictions successfully saved to {predictions_dir}.")
        return predictions_dir
    
    print("Source counting pipeline completed.")
    return predictions_dir
