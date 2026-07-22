# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Source counting pipeline for MuSE-Toolbox.

This module handles the training, testing, and prediction phases for
source counting models, dynamically instantiated via Hydra.
"""

import gc
import logging
import os
from typing import Sized, cast

import hydra
import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Required for HeterogeneousBatch handling when saving predictions
from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch

log = logging.getLogger(__name__)


def run_source_counting_pipeline(cfg: DictConfig) -> str | None:
    """Executes the source counting pipeline based on the configuration.
    
    This general pipeline dynamically instantiates any compatible DataModule (e.g. Brudex, PraAnf)
    and Model (e.g. COSADmodule) using Hydra. It trains, tests, and optionally predicts 
    and saves the estimated source activities to disk for downstream use by the RTF pipeline.
    
    Args:
        cfg (DictConfig): The hierarchical Hydra configuration object.
        
    Returns:
        str | None: The path to the directory where predictions were saved (if applicable).
    """
    log.info("Starting source counting pipeline...")
    
    # 1. Instantiate DataModule
    log.info("Instantiating DataModule...")
    dataset_cfg = cast(DictConfig, cfg.get("dataset"))
    datamodule: LightningDataModule = cast(LightningDataModule, hydra.utils.instantiate(dataset_cfg))
    
    # 2. Instantiate Model
    log.info("Instantiating Model...")
    model_cfg = cast(DictConfig, cfg.get("model"))
    model: LightningModule = cast(LightningModule, hydra.utils.instantiate(model_cfg))
    
    # 3. Setup Logger
    logger: Logger | None = None
    if "logger" in cfg:
        logger_cfg = cast(DictConfig, cfg.get("logger"))
        logger = cast(Logger | None, hydra.utils.instantiate(logger_cfg))
        if logger is not None and hasattr(logger, "log_hyperparams"):
            # safely log the config dictionary
            try:
                from omegaconf import OmegaConf
                
                params = OmegaConf.to_container(cfg, resolve=True)
                if isinstance(params, dict):
                    logger.log_hyperparams(cast(dict[str, object], params))
            except Exception:
                pass
            
    # 4. Setup Callbacks
    callbacks: list[Callback] = []
    if "callbacks" in cfg:
        callbacks_cfg = cast(DictConfig, cfg.get("callbacks"))
        for _, cb_conf in cast(dict[str, DictConfig], cast(object, callbacks_cfg)).items():
            callbacks.append(cast(Callback, hydra.utils.instantiate(cb_conf)))
            
    # 5. Setup Trainer
    trainer_cfg = cast(DictConfig, cfg.get("trainer"))
    trainer: Trainer = cast(Trainer, hydra.utils.instantiate(trainer_cfg, logger=logger, callbacks=callbacks))
    
    # 6. Train and Test
    if cfg.get("train", True):
        log.info("Starting training...")
        trainer.fit(model, datamodule=datamodule)
        
    if cfg.get("test", True):
        log.info("Starting testing...")
        _ = trainer.test(model, datamodule=datamodule)
        
    # 7. Predict and Save (Optional)
    predict_flag = cfg.get("predict", False)
    predictions_dir = cast(str | None, cfg.get("predictions_dir", None))
    
    if predict_flag:
        predict_splits = cfg.get("predict_splits", ["test"])
        
        # Determine base_dir for dynamic resolution
        try:
            from hydra.core.hydra_config import HydraConfig
            base_dir = str(HydraConfig.get().runtime.output_dir)
        except Exception:
            base_dir = str(trainer.default_root_dir)

        # Determine checkpoint path (use best if available)
        ckpt_path = cast(str | None, "best" if cfg.get("train", True) else None)
        
        for split in predict_splits:
            if predictions_dir is None:
                split_predictions_dir = os.path.join(base_dir, split, "predictions")
            else:
                split_predictions_dir = predictions_dir
                
            log.info(f"Running prediction for split '{split}' and saving results to {split_predictions_dir}...")
            os.makedirs(split_predictions_dir, exist_ok=True)
            
            # Setup datamodule for the split
            ds = getattr(datamodule, f"{split}_ds", None)
            if ds is None:
                datamodule.setup(split)
                ds = getattr(datamodule, f"{split}_ds", None)
                
            if ds is None:
                raise ValueError(f"DataModule did not create a '{split}_ds' attribute during setup('{split}').")
                
            # Split dataset into two halves to avoid OOM
            len_ds = len(cast(Sized, ds))
            split_idx = len_ds // 2
            indices_parts = [
                list(range(0, split_idx)),
                list(range(split_idx, len_ds)),
            ]

            for i, indices in enumerate(indices_parts):
                if not indices:
                    continue

                log.info(f"Predicting part {i+1}/2 ({len(indices)} samples) for '{split}'...")
                subset_ds = torch.utils.data.Subset(cast("torch.utils.data.Dataset[object]", ds), indices)
                subset_dl = torch.utils.data.DataLoader(
                    subset_ds,
                    batch_size=getattr(datamodule, "batch_size", 1),
                    num_workers=getattr(datamodule, "num_workers", 0),
                    shuffle=False,
                    collate_fn=getattr(cast(object, ds), "collate_fn", None),
                )
                
                predictions = trainer.predict(model, dataloaders=subset_dl, ckpt_path=ckpt_path)
                
                if isinstance(predictions, list):
                    for batch in predictions:
                        if isinstance(batch, HeterogeneousBatch):
                            est_activity = batch.estimates
                            if not isinstance(est_activity, list):
                                raise ValueError("Expected batch.estimates to be a list of tensors.")
                                
                            scenario_ids = cast(list[str], batch.meta["scenario_id"])
                            
                            for idx, sid in enumerate(scenario_ids):
                                save_path = os.path.join(split_predictions_dir, f"{sid}.pt")
                                torch.save(est_activity[idx].cpu(), save_path)
                        else:
                            log.warning("Batch is not a HeterogeneousBatch. Custom save logic may be needed.")
                
                # Cleanup memory after each chunk
                del predictions
                _ = gc.collect()
                torch.cuda.empty_cache()
                        
            log.info(f"Predictions for '{split}' successfully saved to {split_predictions_dir}.")
            
        if predictions_dir is not None:
            return predictions_dir
        else:
            # Fallback return for downstream tasks (like joint pipeline) to know where to find test predictions
            return os.path.join(base_dir, "test", "predictions")
    
    log.info("Source counting pipeline completed.")
    return predictions_dir
