"""Callback for saving aggregated and per-dataset test metrics to CSV."""

import logging
import os
from typing import Any, cast

import pandas as pd
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torchmetrics import MetricCollection

from muse_toolbox.metrics.base_metric import BaseMetric
from muse_toolbox.models.base_model import BaseLitModel

log = logging.getLogger(__name__)


def _extract_collection_dataframe(
    collection: MetricCollection, dataset_name: str | None = None
) -> pd.DataFrame | None:
    """Extracts and merges per-scenario DataFrames from all metrics in a collection.

    Args:
        collection (MetricCollection): Metric collection to extract DataFrames from.
        dataset_name (str | None, optional): Optional dataset identifier added as
            the leading ``"dataset"`` column.

    Returns:
        pd.DataFrame | None: Combined per-scenario DataFrame, or ``None`` if empty.
    """
    dataframes: list[pd.DataFrame] = []
    for metric in collection.values():
        if isinstance(metric, BaseMetric):
            df = metric.get_dataframe()
            if df is not None and not df.empty:
                dataframes.append(df.reset_index())

    if not dataframes:
        return None

    combined_df = pd.concat(dataframes, axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()].copy()
    if dataset_name is not None and "dataset" not in combined_df.columns:
        combined_df.insert(0, "dataset", dataset_name)
    return combined_df


def _metrics_dict_to_scalar_row(
    dataset_label: str, metrics_dict: dict[str, Any]
) -> dict[str, Any]:
    """Converts a computed metric dictionary into a scalar summary row.

    Args:
        dataset_label (str): Dataset name or aggregate label (e.g. ``"overall_pooled"``).
        metrics_dict (dict[str, Any]): Output dictionary from ``MetricCollection.compute()``.

    Returns:
        dict[str, Any]: Row dictionary suitable for ``pd.DataFrame``.
    """
    row: dict[str, Any] = {"dataset": dataset_label}
    for key, val in metrics_dict.items():
        if key == "added_meta":
            continue
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                row[key] = float(val.detach().cpu().item())
        elif isinstance(val, (int, float)):
            row[key] = float(val)
    return row


class SaveTestResultsCallback(Callback):
    """Extracts per-dataset and combined test metrics and saves them to CSV files."""

    def __init__(self, save_dir: str | None = None) -> None:
        """Initializes the callback.

        Args:
            save_dir (str | None): Optional explicit path. If None, it dynamically
                resolves using Hydra's runtime output dir.
        """
        super().__init__()
        self.save_dir = save_dir

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Aggregates per-dataset and combined test metrics, logs them, and writes CSVs.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): The active PyTorch Lightning module.
        """
        model = cast(BaseLitModel, pl_module)
        summary = model.compute_test_metrics_summary()
        per_dataset: dict[str, dict[str, Any]] = summary["per_dataset"]
        overall_pooled: dict[str, Any] = summary["overall_pooled"]
        overall_macro: dict[str, Any] = summary["overall_macro"]

        # 1. Build logger payload: backward-compatible `test/<metric>`, per-dataset
        # `test/<ds_name>/<metric>`, and `test/overall_pooled/<metric>` + `test/overall_macro/<metric>`.
        prefixed_metrics: dict[str, Any] = {
            f"test/{k}": v for k, v in overall_pooled.items()
        }
        for ds_name, ds_metrics in per_dataset.items():
            for k, v in ds_metrics.items():
                prefixed_metrics[f"test/{ds_name}/{k}"] = v

        if len(per_dataset) > 1:
            for k, v in overall_pooled.items():
                prefixed_metrics[f"test/overall_pooled/{k}"] = v
            for k, v in overall_macro.items():
                prefixed_metrics[f"test/overall_macro/{k}"] = v

        pl_module.log_dict(
            prefixed_metrics, sync_dist=True, add_dataloader_idx=False
        )

        # 2. Resolve output directory
        if self.save_dir is None:
            try:
                from hydra.core.hydra_config import HydraConfig

                base_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                base_dir = trainer.default_root_dir

            split_folder = "test"
            if trainer.validating:
                split_folder = "val"
            elif trainer.training:
                split_folder = "train"

            out_dir = os.path.join(str(base_dir), split_folder, "results")
        else:
            out_dir = str(self.save_dir)

        os.makedirs(out_dir, exist_ok=True)

        # 3. Extract and save per-dataset and combined per-scenario DataFrames
        per_dataset_dfs: list[pd.DataFrame] = []
        for ds_name, collection in model.test_metric_collections.items():
            ds_df = _extract_collection_dataframe(
                collection, dataset_name=ds_name
            )
            if ds_df is not None and not ds_df.empty:
                per_dataset_dfs.append(ds_df)
                if len(model.test_metric_collections) > 1:
                    ds_csv_path = os.path.join(
                        out_dir, f"test_metrics_{ds_name}.csv"
                    )
                    ds_df.to_csv(ds_csv_path, index=False)
                    log.info(
                        f"Saved per-dataset metric results for '{ds_name}' to {ds_csv_path}"
                    )

        if per_dataset_dfs:
            combined_df: pd.DataFrame | None = pd.concat(
                per_dataset_dfs, axis=0, ignore_index=True
            )
        else:
            combined_df = _extract_collection_dataframe(
                model.metric_collections["test"]
            )

        # 4. Save summary table CSV (per-dataset rows + overall_pooled + overall_macro)
        summary_rows: list[dict[str, Any]] = []
        for ds_name, ds_metrics in per_dataset.items():
            summary_rows.append(_metrics_dict_to_scalar_row(ds_name, ds_metrics))
        if overall_pooled:
            summary_rows.append(
                _metrics_dict_to_scalar_row("overall_pooled", overall_pooled)
            )
        if len(per_dataset) > 1 and overall_macro:
            summary_rows.append(
                _metrics_dict_to_scalar_row("overall_macro", overall_macro)
            )
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(out_dir, "test_metrics_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            log.info(f"Saved summary test metrics to {summary_path}")

        if combined_df is None or combined_df.empty:
            log.warning("No per-scenario dataframes to save from muse_toolbox.metrics.")
            model.reset_test_metrics()
            return

        # 5. Attach complexity metrics if they exist
        complexity_metrics = getattr(pl_module, "complexity_metrics", None)
        if isinstance(complexity_metrics, dict):
            for key, value in complexity_metrics.items():
                combined_df[key] = value

        # 6. Save combined per-scenario CSV
        save_path = os.path.join(out_dir, "test_metrics.csv")
        combined_df.to_csv(save_path, index=False)
        log.info(f"Saved metric results to {save_path}")

        # 7. Reset metrics
        model.reset_test_metrics()
