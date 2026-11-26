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


def _extract_collection_events_dataframe(
    collection: MetricCollection, dataset_name: str | None = None
) -> pd.DataFrame | None:
    """Extracts event-level and detection-delay DataFrames from metrics supporting them.

    Args:
        collection (MetricCollection): Metric collection to inspect.
        dataset_name (str | None, optional): Optional dataset identifier added as
            the leading ``"dataset"`` column.

    Returns:
        pd.DataFrame | None: Event-level DataFrame, or ``None`` if no events exist.
    """
    event_dfs: list[pd.DataFrame] = []
    for metric in collection.values():
        if hasattr(metric, "get_events_dataframe"):
            df = metric.get_events_dataframe()
            if df is not None and not df.empty:
                event_dfs.append(df.reset_index(drop=True))

    if not event_dfs:
        return None

    combined = pd.concat(event_dfs, axis=0, ignore_index=True)
    if dataset_name is not None and "dataset" not in combined.columns:
        combined.insert(0, "dataset", dataset_name)
    return combined


def _build_collection_num_mics_rows(
    collection: MetricCollection,
    ds_df: pd.DataFrame | None,
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Computes per-``num_mics`` stratified metric summary rows for one dataset collection.

    Args:
        collection (MetricCollection): Evaluated metrics for one dataset.
        ds_df (pd.DataFrame | None): Per-chunk DataFrame containing ``num_mics`` and
            ``num_frames`` alongside frame metrics (Accuracy, MAE, MSE).
        dataset_name (str): Dataset label for the ``dataset`` column.

    Returns:
        list[dict[str, Any]]: Sorted summary rows keyed by ``(dataset, num_mics)``.
    """
    ap_by_nm: dict[int, dict[str, float]] = {}
    ef1_by_nm: dict[int, dict[str, float]] = {}
    for metric in collection.values():
        if hasattr(metric, "compute_by_num_mics"):
            res = metric.compute_by_num_mics()
            if isinstance(res, dict):
                if any("AP_" in k or k == "mAP" for v in res.values() for k in v):
                    ap_by_nm.update(res)
                if any("F1score_" in k for v in res.values() for k in v):
                    ef1_by_nm.update(res)

    df_by_nm: dict[int, dict[str, Any]] = {}
    if ds_df is not None and not ds_df.empty and "num_mics" in ds_df.columns:
        valid_df = ds_df.dropna(subset=["num_mics"]).copy()
        if not valid_df.empty:
            valid_df["num_mics_int"] = valid_df["num_mics"].astype(int)
            for nm, group in valid_df.groupby("num_mics_int"):
                nm_int = int(nm)
                row_stats: dict[str, Any] = {}
                if "recording_id" in group.columns:
                    row_stats["num_recordings"] = int(
                        group["recording_id"].dropna().nunique()
                    )
                if "num_frames" in group.columns:
                    weights = group["num_frames"].fillna(0.0).astype(float)
                    total_frames = float(weights.sum())
                    row_stats["num_frames"] = int(total_frames)
                    for col in ("Accuracy", "MAE", "MSE"):
                        if col in group.columns:
                            vals = group[col].astype(float)
                            if total_frames > 0.0:
                                row_stats[col] = float(
                                    (vals * weights).sum() / total_frames
                                )
                            else:
                                row_stats[col] = float(vals.mean())
                else:
                    for col in ("Accuracy", "MAE", "MSE"):
                        if col in group.columns:
                            row_stats[col] = float(group[col].astype(float).mean())
                df_by_nm[nm_int] = row_stats

    all_m_values = sorted(
        set(ap_by_nm.keys()) | set(ef1_by_nm.keys()) | set(df_by_nm.keys())
    )
    rows: list[dict[str, Any]] = []
    for nm in all_m_values:
        row: dict[str, Any] = {"dataset": dataset_name, "num_mics": int(nm)}
        if nm in df_by_nm:
            row.update(df_by_nm[nm])
        if nm in ap_by_nm:
            row.update(ap_by_nm[nm])
        if nm in ef1_by_nm:
            row.update(ef1_by_nm[nm])
        rows.append(row)
    return rows


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

        # 3b. Extract and save transition-stratified per-event & detection-delay CSVs
        per_dataset_event_dfs: list[pd.DataFrame] = []
        for ds_name, collection in model.test_metric_collections.items():
            ev_df = _extract_collection_events_dataframe(
                collection, dataset_name=ds_name
            )
            if ev_df is not None and not ev_df.empty:
                per_dataset_event_dfs.append(ev_df)
                if len(model.test_metric_collections) > 1:
                    ev_ds_path = os.path.join(out_dir, f"test_events_{ds_name}.csv")
                    ev_df.to_csv(ev_ds_path, index=False)
                    log.info(
                        f"Saved per-dataset event results for '{ds_name}' to {ev_ds_path}"
                    )

        if per_dataset_event_dfs:
            combined_events_df: pd.DataFrame | None = pd.concat(
                per_dataset_event_dfs, axis=0, ignore_index=True
            )
        else:
            combined_events_df = _extract_collection_events_dataframe(
                model.metric_collections["test"]
            )

        if combined_events_df is not None and not combined_events_df.empty:
            events_path = os.path.join(out_dir, "test_events.csv")
            combined_events_df.to_csv(events_path, index=False)
            log.info(f"Saved event and detection-delay results to {events_path}")

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

        # 5b. Save stratified per-num_mics summary CSV for multi-mic / subsampled datasets
        num_mics_rows: list[dict[str, Any]] = []
        if model.test_metric_collections:
            for ds_name, collection in model.test_metric_collections.items():
                ds_sub_df = (
                    combined_df[combined_df["dataset"] == ds_name]
                    if "dataset" in combined_df.columns
                    else combined_df
                )
                num_mics_rows.extend(
                    _build_collection_num_mics_rows(collection, ds_sub_df, ds_name)
                )
        else:
            num_mics_rows.extend(
                _build_collection_num_mics_rows(
                    model.metric_collections["test"], combined_df, "default"
                )
            )
        if num_mics_rows:
            nm_df = pd.DataFrame(num_mics_rows)
            nm_path = os.path.join(out_dir, "test_metrics_by_num_mics.csv")
            nm_df.to_csv(nm_path, index=False)
            log.info(f"Saved num_mics-stratified test metrics to {nm_path}")

        # 6. Save combined per-scenario CSV
        save_path = os.path.join(out_dir, "test_metrics.csv")
        combined_df.to_csv(save_path, index=False)
        log.info(f"Saved metric results to {save_path}")

        # 7. Reset metrics
        model.reset_test_metrics()
