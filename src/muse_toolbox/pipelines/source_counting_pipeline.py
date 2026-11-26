# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Source counting pipeline for MuSE-Toolbox.

This module handles the training, testing, and prediction phases for
source counting models, dynamically instantiated via Hydra.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Any, Sized, cast

import hydra
import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from muse_toolbox.data.components.collate import raw_audio_collate_fn

log = logging.getLogger(__name__)

_CONFIGS_DATASET_DIR = (
    Path(__file__).resolve().parents[3] / "configs" / "dataset"
)


def _load_dataset_yaml_with_defaults(
    dataset_name: str, dataset_dir: Path = _CONFIGS_DATASET_DIR
) -> DictConfig:
    """Recursively loads a dataset YAML config and merges its `defaults` chain.

    Args:
        dataset_name (str): Dataset config stem (e.g., ``"ami"``, ``"brudex"``).
        dataset_dir (Path, optional): Directory containing dataset YAML files.

    Returns:
        DictConfig: Merged dataset configuration without ``defaults``.

    Raises:
        FileNotFoundError: If the requested dataset YAML does not exist.
    """
    yaml_name = (
        dataset_name if dataset_name.endswith(".yaml") else f"{dataset_name}.yaml"
    )
    yaml_path = dataset_dir / yaml_name
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset config '{dataset_name}' not found at {yaml_path}."
        )

    raw_cfg = cast(DictConfig, OmegaConf.load(yaml_path))
    defaults_list = raw_cfg.get("defaults", None)
    merged_base = OmegaConf.create({})
    if defaults_list:
        for entry in defaults_list:
            if isinstance(entry, str) and entry != "_self_":
                parent_cfg = _load_dataset_yaml_with_defaults(
                    entry, dataset_dir=dataset_dir
                )
                merged_base = cast(
                    DictConfig, OmegaConf.merge(merged_base, parent_cfg)
                )

    child_cfg = OmegaConf.to_container(raw_cfg, resolve=False)
    if isinstance(child_cfg, dict):
        child_cfg.pop("defaults", None)
    return cast(DictConfig, OmegaConf.merge(merged_base, OmegaConf.create(child_cfg)))


def _extract_max_sources(ds_cfg: DictConfig | dict[str, Any] | None) -> int | None:
    """Extracts `generation_config.max_sources` from a dataset configuration if present.

    Args:
        ds_cfg (DictConfig | dict[str, Any] | None): Dataset configuration mapping.

    Returns:
        int | None: Configured maximum source count, or ``None``.
    """
    if ds_cfg is None:
        return None
    gen_cfg = ds_cfg.get("generation_config", None)
    if gen_cfg is not None and "max_sources" in gen_cfg and gen_cfg["max_sources"] is not None:
        return int(gen_cfg["max_sources"])
    return None


def _resolve_eval_ckpt_path(cfg: DictConfig, trainer: Trainer) -> str | None:
    """Resolves the checkpoint path used for testing and prediction.

    Prefers an explicit ``cfg.ckpt_path`` if configured; otherwise returns
    ``"best"`` when training ran and a ``ModelCheckpoint`` callback recorded a
    non-empty ``best_model_path``, falling back to ``None`` for checkpointless
    or ``fast_dev_run`` executions.

    Args:
        cfg (DictConfig): Root Hydra configuration.
        trainer (Trainer): Configured PyTorch Lightning Trainer instance.

    Returns:
        str | None: Checkpoint specifier passed to ``trainer.test`` / ``trainer.predict``.
    """
    explicit_ckpt = cfg.get("ckpt_path", None)
    if explicit_ckpt:
        return str(explicit_ckpt)
    if cfg.get("train", True):
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        best_model_path = (
            getattr(ckpt_cb, "best_model_path", "") if ckpt_cb is not None else ""
        )
        if best_model_path:
            return "best"
    return None


def _partition_indices_at_recording_boundaries(ds: Any, num_parts: int = 2) -> list[list[int]]:
    """Splits dataset evaluation indices without cutting inside a continuous recording.

    Args:
        ds (Any): Dataset instance exposing optional ``grid`` tuples
            ``(key, start_idx, is_last_chunk)``.
        num_parts (int, optional): Target number of partitions. Defaults to ``2``.

    Returns:
        list[list[int]]: Non-empty lists of contiguous dataset indices.
    """
    len_ds = len(cast(Sized, ds))
    if len_ds == 0:
        return []
    grid = getattr(ds, "grid", None)
    if not isinstance(grid, list) or len(grid) != len_ds or num_parts <= 1:
        return [list(range(len_ds))]

    end_indices = [idx for idx, entry in enumerate(grid) if bool(entry[2])]
    if len(end_indices) <= 1:
        return [list(range(len_ds))]

    target_split = len_ds // num_parts
    split_end = min(end_indices[:-1], key=lambda idx: abs(idx + 1 - target_split))
    cut = split_end + 1
    parts = [list(range(0, cut)), list(range(cut, len_ds))]
    return [part for part in parts if part]


def resolve_test_datamodules(
    cfg: DictConfig,
    primary_datamodule: LightningDataModule | None = None,
    single_channel_expansion: bool | None = None,
) -> list[tuple[str, LightningDataModule, int | None]]:
    """Resolves one or more test DataModules from `cfg.test_datasets` or `cfg.dataset`.

    Supports:
    - ``test_datasets: null`` (default): evaluates on the primary ``cfg.dataset``.
    - ``test_datasets: "brudex"``: evaluates on a single separate named dataset.
    - ``test_datasets: ["ami", "brudex", "pra_anf_circ_8ch"]``: evaluates on multiple
      named datasets from ``configs/dataset/``.
    - ``test_datasets: [{"dataset": "ami", "id": "sanity_ami"}, ...]`` or a dict
      mapping custom labels to dataset names/override dicts.

    Args:
        cfg (DictConfig): Root Hydra configuration.
        primary_datamodule (LightningDataModule | None, optional): Already-instantiated
            training/primary DataModule to reuse when a test dataset matches ``cfg.dataset``.

    Returns:
        list[tuple[str, LightningDataModule, int | None]]: Ordered list of
            ``(dataset_name, datamodule, max_sources)`` tuples.
    """
    primary_ds_cfg = cast(DictConfig, cfg.get("dataset"))
    primary_id = str(primary_ds_cfg.get("id", "default"))
    primary_target = str(primary_ds_cfg.get("_target_", ""))
    primary_max_sources = _extract_max_sources(primary_ds_cfg)

    raw_test_specs = cfg.get("test_datasets", None)
    if raw_test_specs is None or (
        isinstance(raw_test_specs, (list, ListConfig, dict, DictConfig))
        and len(raw_test_specs) == 0
    ):
        dm = (
            primary_datamodule
            if primary_datamodule is not None
            else cast(LightningDataModule, hydra.utils.instantiate(primary_ds_cfg))
        )
        return [(primary_id, dm, primary_max_sources)]

    # Normalize specs into a list of (optional_alias, spec) pairs
    normalized_specs: list[tuple[str | None, Any]] = []
    if isinstance(raw_test_specs, str):
        normalized_specs.append((None, raw_test_specs))
    elif isinstance(raw_test_specs, (list, ListConfig)):
        for item in raw_test_specs:
            normalized_specs.append((None, item))
    elif isinstance(raw_test_specs, (dict, DictConfig)):
        for alias, item in raw_test_specs.items():
            normalized_specs.append((str(alias), item))
    else:
        raise ValueError(
            f"Unsupported type for test_datasets: {type(raw_test_specs)}"
        )

    resolved: list[tuple[str, LightningDataModule, int | None]] = []
    used_names: set[str] = set()

    for alias, spec in normalized_specs:
        overrides: dict[str, Any] = {}
        config_name: str | None = None
        inline_cfg: DictConfig | None = None

        if isinstance(spec, str):
            config_name = spec
        elif isinstance(spec, (dict, DictConfig)):
            spec_dict = dict(
                cast(
                    dict[str, Any],
                    OmegaConf.to_container(spec, resolve=False)
                    if isinstance(spec, DictConfig)
                    else spec,
                )
            )
            if "_target_" in spec_dict and "dataset" not in spec_dict:
                inline_cfg = OmegaConf.create(spec_dict)
            else:
                config_name = str(
                    spec_dict.pop("dataset", None)
                    or spec_dict.pop("config", None)
                    or alias
                )
                alias = str(spec_dict.pop("name", alias)) if ("name" in spec_dict or alias) else None
                overrides = spec_dict
        else:
            raise ValueError(f"Invalid test_datasets entry: {spec}")

        if inline_cfg is not None:
            ds_cfg = inline_cfg
        else:
            assert config_name is not None
            ds_cfg = _load_dataset_yaml_with_defaults(config_name)
            # Inherit shared evaluation window and worker settings from primary cfg.dataset unless overridden
            for shared_key in (
                "eval_chunk_length_s",
                "eval_min_context_s",
                "num_workers",
                "acc_device",
                "single_channel_expansion",
            ):
                if (
                    shared_key in primary_ds_cfg
                    and shared_key not in overrides
                ):
                    ds_cfg[shared_key] = primary_ds_cfg[shared_key]
            if overrides:
                ds_cfg = cast(
                    DictConfig,
                    OmegaConf.merge(ds_cfg, OmegaConf.create(overrides)),
                )

        # Wrap in a root context with `paths` and `transform` so `${dataset.*}`,
        # `${paths.*}`, and `${transform.*}` interpolations resolve identically to main config.
        root_ctx = OmegaConf.create(
            {
                "paths": cfg.get("paths", {}),
                "transform": cfg.get("transform", {}),
                "dataset": ds_cfg,
            }
        )
        resolved_ds_cfg = cast(DictConfig, root_ctx.dataset)
        ds_id = str(resolved_ds_cfg.get("id", config_name or "test"))
        ds_target = str(resolved_ds_cfg.get("_target_", ""))
        ds_max_sources = _extract_max_sources(resolved_ds_cfg)

        # Reuse primary_datamodule if it represents the exact same dataset without conflicting overrides
        can_reuse_primary = (
            primary_datamodule is not None
            and not overrides
            and ds_id == primary_id
            and ds_target == primary_target
        )
        if can_reuse_primary:
            assert primary_datamodule is not None
            dm = primary_datamodule
        else:
            dm = cast(
                LightningDataModule,
                hydra.utils.instantiate(resolved_ds_cfg),
            )
            if (
                single_channel_expansion is not None
                and hasattr(dm, "set_single_channel_expansion")
            ):
                cast(Any, dm).set_single_channel_expansion(
                    bool(single_channel_expansion)
                )

        base_label = alias or config_name or ds_id
        label = base_label
        suffix = 2
        while label in used_names:
            label = f"{base_label}_{suffix}"
            suffix += 1
        used_names.add(label)

        resolved.append((label, dm, ds_max_sources))

    return resolved


def run_source_counting_pipeline(cfg: DictConfig) -> str | None:
    """Executes the source counting pipeline based on the configuration.
    
    This general pipeline dynamically instantiates any compatible DataModule (e.g. Brudex, PraAnf)
    and Model (e.g. OSCmodule) using Hydra. It trains, tests, and optionally predicts 
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

    model_is_single_channel: bool | None = (
        bool(cast(Any, model).is_single_channel_system)
        if hasattr(model, "is_single_channel_system")
        else None
    )
    if (
        model_is_single_channel is not None
        and hasattr(datamodule, "set_single_channel_expansion")
    ):
        cast(Any, datamodule).set_single_channel_expansion(model_is_single_channel)
    
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
        test_entries = resolve_test_datamodules(
            cfg,
            primary_datamodule=datamodule,
            single_channel_expansion=model_is_single_channel,
        )
        test_names: list[str] = []
        test_max_sources: list[int | None] = []
        test_dataloaders = []

        for ds_name, test_dm, ds_ms in test_entries:
            log.info(
                f"Preparing test DataModule '{ds_name}' (id={getattr(test_dm, 'id', ds_name)}, max_sources={ds_ms})..."
            )
            if getattr(test_dm, "test_ds", None) is None:
                precomp_dir = getattr(test_dm, "precomputed_dir", None)
                test_split_ready = (
                    precomp_dir is not None
                    and (precomp_dir / "test").exists()
                    and any((precomp_dir / "test").glob("*_audio.npy"))
                    and not bool(getattr(test_dm, "reset", False))
                )
                if not test_split_ready:
                    test_dm.prepare_data()
                test_dm.setup("test")
            test_names.append(ds_name)
            test_max_sources.append(ds_ms)
            test_dataloaders.append(test_dm.test_dataloader())

        if hasattr(model, "configure_test_datasets"):
            cast(Any, model).configure_test_datasets(
                dataset_names=test_names,
                dataset_max_sources=test_max_sources,
            )

        test_ckpt_path = _resolve_eval_ckpt_path(cfg, trainer)
        _ = trainer.test(
            model, dataloaders=test_dataloaders, ckpt_path=test_ckpt_path
        )

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

        ckpt_path = _resolve_eval_ckpt_path(cfg, trainer)

        for split in predict_splits:
            if predictions_dir is None:
                split_predictions_dir = os.path.join(base_dir, split, "predictions")
            else:
                split_predictions_dir = predictions_dir

            log.info(
                f"Running stitched prediction for split '{split}' and saving results to {split_predictions_dir}..."
            )
            os.makedirs(split_predictions_dir, exist_ok=True)

            # Setup datamodule for the split
            ds = getattr(datamodule, f"{split}_ds", None)
            if ds is None:
                datamodule.setup(split)
                ds = getattr(datamodule, f"{split}_ds", None)

            if ds is None:
                raise ValueError(
                    f"DataModule did not create a '{split}_ds' attribute during setup('{split}')."
                )

            indices_parts = _partition_indices_at_recording_boundaries(ds, num_parts=2)
            rec_buffers: dict[str, list[torch.Tensor]] = {}

            for i, indices in enumerate(indices_parts):
                if not indices:
                    continue

                log.info(
                    f"Predicting part {i + 1}/{len(indices_parts)} ({len(indices)} chunks) for '{split}'..."
                )
                subset_ds = torch.utils.data.Subset(
                    cast("torch.utils.data.Dataset[object]", ds), indices
                )
                subset_dl = torch.utils.data.DataLoader(
                    subset_ds,
                    batch_size=1,
                    num_workers=getattr(datamodule, "num_workers", 0),
                    shuffle=False,
                    collate_fn=raw_audio_collate_fn,
                )

                predictions = trainer.predict(
                    model, dataloaders=subset_dl, ckpt_path=ckpt_path
                )

                if isinstance(predictions, list):
                    for batch in predictions:
                        if isinstance(batch, dict) and "estimates" in batch:
                            est_chunks = batch["estimates"]
                            rec_ids = cast(
                                list[str],
                                batch.get("meta", {}).get("recording_id", []),
                            )
                            is_last_flags = cast(
                                list[bool],
                                batch.get("meta", {}).get(
                                    "is_last_chunk", [True] * len(est_chunks)
                                ),
                            )
                            for est_c, rec_id, is_last in zip(
                                est_chunks, rec_ids, is_last_flags
                            ):
                                rec_buffers.setdefault(str(rec_id), []).append(
                                    est_c.cpu()
                                )
                                if is_last:
                                    full_pred = torch.cat(
                                        rec_buffers.pop(str(rec_id)), dim=0
                                    )
                                    save_path = os.path.join(
                                        split_predictions_dir, f"{rec_id}.pt"
                                    )
                                    torch.save(full_pred, save_path)
                        else:
                            log.warning(
                                "Prediction batch does not match expected dict format with 'estimates'."
                            )

                del predictions
                _ = gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Flush any remaining recording buffers if is_last_chunk was absent
            for rec_id, chunks in rec_buffers.items():
                full_pred = torch.cat(chunks, dim=0)
                save_path = os.path.join(split_predictions_dir, f"{rec_id}.pt")
                torch.save(full_pred, save_path)

            log.info(
                f"Stitched predictions for '{split}' successfully saved to {split_predictions_dir}."
            )
            
        if predictions_dir is not None:
            return predictions_dir
        else:
            # Fallback return for downstream tasks (like joint pipeline) to know where to find test predictions
            return os.path.join(base_dir, "test", "predictions")
    
    log.info("Source counting pipeline completed.")
    return predictions_dir
