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

    def __init__(self, save_dir: str) -> None:
        """Initializes the callback.

        Args:
            save_dir (str): The dynamic path provided by Hydra where CSVs should be saved.
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

        # 5. Build dynamic filename
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_name = getattr(pl_module, "model_name", "model")
        block_size = getattr(pl_module, "block_size_frames", "")
        pre_context = getattr(pl_module, "pre_context_frames", "")

        if model_name == "BlockOnlineGSS":
            filename = f"{model_name}_bl{block_size}_pc{pre_context}_results.csv"
        else:
            filename = f"{model_name}_results.csv"
            
        filepath = os.path.join(self.save_dir, filename)
        
        log.info(f"Saving test results to {filepath}")
        combined_df.to_csv(filepath, index=True)

        # 6. Reset
        model.metric_collections["test"].reset()
