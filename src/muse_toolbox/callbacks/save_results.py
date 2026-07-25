"""Callback for saving aggregated test metrics to CSV."""

import logging
import os
import pandas as pd
from typing import cast
from lightning.pytorch import Callback, Trainer, LightningModule
from muse_toolbox.models.base_model import BaseLitModel
from muse_toolbox.metrics.base_metric import BaseMetric

log = logging.getLogger(__name__)


class SaveTestResultsCallback(Callback):
    """Extracts test metrics and saves them to a CSV file dynamically."""

    def __init__(self, save_dir: str | None = None) -> None:
        """Initializes the callback.

        Args:
            save_dir (str | None): Optional explicit path. If None, it dynamically 
                                   resolves using Hydra's runtime output dir.
        """
        super().__init__()
        self.save_dir = save_dir

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Aggregates metrics and writes them to a CSV.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): The active PyTorch Lightning module.
        """
        # 1. Compute and log W&B metrics
        model = cast(BaseLitModel, pl_module)
        metrics_output_dict = model.metric_collections["test"].compute()
        prefixed_metrics = {f"test/{k}": v for k, v in metrics_output_dict.items()}
        pl_module.log_dict(prefixed_metrics, sync_dist=True)

        # 2. Extract DataFrames
        dataframes = []
        for metric in model.metric_collections["test"].values():
            if isinstance(metric, BaseMetric):
                df = metric.get_dataframe()
                if df is not None and not df.empty:
                    dataframes.append(df)

        if not dataframes:
            log.warning("No dataframes to save from muse_toolbox.metrics.")
            model.metric_collections["test"].reset()
            return

        # 3. Merge DataFrames
        combined_df = pd.concat(dataframes, axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        # 4. Attach complexity metrics if they exist
        complexity_metrics = getattr(pl_module, "complexity_metrics", None)
        if isinstance(complexity_metrics, dict):
            for key, value in complexity_metrics.items():
                combined_df[key] = value

        # 3. Determine dynamic output directory
        if self.save_dir is None:
            # Fallback to Hydra's output directory, or the trainer's root dir
            try:
                from hydra.core.hydra_config import HydraConfig
                base_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                base_dir = trainer.default_root_dir
                
            # Use PyTorch Lightning state flags to determine split
            split_folder = "test"
            if trainer.validating:
                split_folder = "val"
            elif trainer.training:
                split_folder = "train"
                
            out_dir = os.path.join(str(base_dir), split_folder, "results")
        else:
            out_dir = str(self.save_dir)
            
        os.makedirs(out_dir, exist_ok=True)
        
        # 4. Save to CSV
        save_path = os.path.join(out_dir, "test_metrics.csv")
        combined_df.to_csv(save_path, index=False)
        log.info(f"Saved metric results to {save_path}")

        # 5. Reset metrics
        model.metric_collections["test"].reset()
