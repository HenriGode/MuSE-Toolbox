import os
import tempfile
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

from muse_toolbox.callbacks.save_results import SaveTestResultsCallback
from muse_toolbox.metrics.source_counting.confusionMatrix import ConfusionMatrix
from muse_toolbox.models.components.feature_extractors import (
    LogMel_Feature_Extractor,
)
from muse_toolbox.pipelines.source_counting_pipeline import (
    resolve_test_datamodules,
    run_source_counting_pipeline,
)
from muse_toolbox.utils import STFTtransform


def test_default_configs_and_log_mel_no_deprecation() -> None:
    """Verifies AMI + LogMel (80 bands) defaults and absence of false DeprecationWarning."""
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="default")

    assert cfg.dataset.id == "ami"
    assert cfg.dataset._target_ == "muse_toolbox.data.datamodules.ami.AMIDataModule"
    assert cfg.test_datasets is None
    assert (
        cfg.model.feature_extractor._target_
        == "muse_toolbox.models.components.feature_extractors.LogMel_Feature_Extractor"
    )
    assert cfg.model.feature_extractor.n_mels == 80
    assert cfg.model.feature_extractor.mode == "ref"

    transform = STFTtransform(
        sampling_frequency=16000,
        frame_length=0.025,
        frame_shift=0.010,
        remove_DC=True,
        remove_Nyquist=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        fe = LogMel_Feature_Extractor(transform=transform, mode="ref", n_mels=80)
        dep_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 0, f"Unexpected DeprecationWarning: {dep_warnings}"

    mask = fe.get_valid_feature_mask(torch.tensor([3, 6]), max_M=6)
    assert mask is not None and mask.shape == (2, 1) and mask.all()


def test_resolve_test_datamodules_single_and_multi() -> None:
    """Verifies resolve_test_datamodules for default, string list, and override dicts."""
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default",
            overrides=[
                "dataset=ami",
                "dataset.id=sanity_ami",
                "dataset.reset=false",
            ],
        )

    primary_dm = instantiate(cfg.dataset)
    # 1. Default test_datasets=None reuses primary_datamodule
    entries_default = resolve_test_datamodules(cfg, primary_datamodule=primary_dm)
    assert len(entries_default) == 1
    assert entries_default[0][0] == "sanity_ami"
    assert entries_default[0][1] is primary_dm
    assert entries_default[0][2] == 4

    # 2. Multi-dataset list with overrides pointing to sanity datasets
    cfg_multi = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg_multi.test_datasets = [
        {"dataset": "ami", "id": "sanity_ami", "reset": False},
        {"dataset": "brudex", "id": "sanity_brudex", "reset": False},
        {
            "dataset": "pra_anf_circ_8ch",
            "id": "sanity_pra_anf",
            "reset": False,
        },
    ]
    entries_multi = resolve_test_datamodules(cfg_multi, primary_datamodule=primary_dm)
    assert [name for name, _, _ in entries_multi] == [
        "ami",
        "brudex",
        "pra_anf_circ_8ch",
    ]
    assert [ms for _, _, ms in entries_multi] == [4, 4, 4]


def test_multi_dataset_evaluation_metrics_and_heterogeneous_max_sources() -> None:
    """Verifies per-dataset, pooled, and macro metrics, CSV output, and variable max_sources."""
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default",
            overrides=[
                "dataset=ami",
                "dataset.id=sanity_ami",
                "dataset.reset=false",
                "model=osc",
                "model/feature_extractor=log_mel",
            ],
        )

    model = instantiate(cfg.model)
    model.eval()

    # Configure two test datasets with DIFFERENT max_sources (4 vs 6)
    model.configure_test_datasets(
        dataset_names=["ami_test", "wide_6spk_test"],
        dataset_max_sources=[4, 6],
    )

    # Combined overall ConfusionMatrix must be disabled when max_sources differ,
    # while per-dataset ConfusionMatrix instances remain enabled with 5 and 7 classes.
    assert model.metric_collections["test"]["ConfusionMatrix"].enabled is False
    assert (
        model.test_metric_collections["ami_test"]["ConfusionMatrix"].enabled
        is True
    )
    assert (
        model.test_metric_collections["ami_test"]["ConfusionMatrix"].num_classes
        == 5
    )
    assert (
        model.test_metric_collections["wide_6spk_test"][
            "ConfusionMatrix"
        ].enabled
        is True
    )
    assert (
        model.test_metric_collections["wide_6spk_test"][
            "ConfusionMatrix"
        ].num_classes
        == 7
    )

    # Build two synthetic test batches:
    # Batch 0 (dataloader_idx=0, "ami_test"): targets in [0..4]
    # Batch 1 (dataloader_idx=1, "wide_6spk_test"): includes target source count = 6 (> model's 4)!
    num_samples = 16000  # 1.0 s -> 100 frames at 10 ms hop
    num_frames = int(model.transform.samples2frames(num_samples).item())

    batch_0: dict[str, Any] = {
        "input": torch.randn(1, 4, num_samples),
        "mic_lengths": torch.tensor([4]),
        "time_lengths": torch.tensor([num_samples]),
        "meta": {
            "source_count": [torch.full((num_frames,), 2, dtype=torch.long)],
            "scenario_id": ["scenario_0000_0.00"],
            "scenario_params": [{"num_sources": 2}],
        },
    }
    target_1 = torch.full((num_frames,), 6, dtype=torch.long)
    target_1[: num_frames // 2] = 1
    batch_1: dict[str, Any] = {
        "input": torch.randn(1, 4, num_samples),
        "mic_lengths": torch.tensor([4]),
        "time_lengths": torch.tensor([num_samples]),
        "meta": {
            # Use identical scenario_id to verify no duplicate-index crash across datasets
            "source_count": [target_1],
            "scenario_id": ["scenario_0000_0.00"],
            "scenario_params": [{"num_sources": 6}],
        },
    }

    with torch.no_grad():
        model.test_step(batch_0, idx=0, dataloader_idx=0)
        model.test_step(batch_1, idx=0, dataloader_idx=1)

    summary = model.compute_test_metrics_summary()
    assert set(summary["per_dataset"].keys()) == {"ami_test", "wide_6spk_test"}

    for metric_key in ("Accuracy", "MAE", "MSE", "F1score_overall"):
        v_0 = float(summary["per_dataset"]["ami_test"][metric_key].item())
        v_1 = float(summary["per_dataset"]["wide_6spk_test"][metric_key].item())
        v_macro = float(summary["overall_macro"][metric_key].item())
        assert abs(v_macro - 0.5 * (v_0 + v_1)) < 1e-5

    # Test SaveTestResultsCallback CSV generation and logger call
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = SaveTestResultsCallback(save_dir=tmpdir)
        dummy_trainer = MagicMock()
        dummy_trainer.validating = False
        dummy_trainer.training = False
        model.log_dict = MagicMock()

        cb.on_test_epoch_end(dummy_trainer, model)

        logged_dict = model.log_dict.call_args[0][0]
        assert "test/Accuracy" in logged_dict
        assert "test/ami_test/Accuracy" in logged_dict
        assert "test/wide_6spk_test/Accuracy" in logged_dict
        assert "test/overall_pooled/Accuracy" in logged_dict
        assert "test/overall_macro/Accuracy" in logged_dict

        summary_csv = Path(tmpdir) / "test_metrics_summary.csv"
        combined_csv = Path(tmpdir) / "test_metrics.csv"
        ds0_csv = Path(tmpdir) / "test_metrics_ami_test.csv"
        ds1_csv = Path(tmpdir) / "test_metrics_wide_6spk_test.csv"

        assert summary_csv.exists()
        assert combined_csv.exists()
        assert ds0_csv.exists()
        assert ds1_csv.exists()

        df_summary = pd.read_csv(summary_csv)
        assert list(df_summary["dataset"]) == [
            "ami_test",
            "wide_6spk_test",
            "overall_pooled",
            "overall_macro",
        ]
        df_combined = pd.read_csv(combined_csv)
        assert "dataset" in df_combined.columns
        assert "scenario_id" in df_combined.columns
        assert len(df_combined) == 2

    # Also verify standalone ConfusionMatrix dynamic class expansion when unexpected count > max_sources arrives
    cm = ConfusionMatrix(max_sources=4, stage="test")
    assert cm.num_classes == 5
    cm.update(
        preds=[torch.randn(10, 5)],
        targets=[torch.tensor([0, 1, 2, 3, 4, 5, 6, 2, 1, 0])],
        meta={},
        dataloader_idx=0,
    )
    assert cm.num_classes == 7
    assert cm.conf_matrix_metric.confmat.shape == (7, 7)


def test_run_source_counting_pipeline_multi_dataset_end_to_end() -> None:
    """Runs run_source_counting_pipeline with multi-dataset testing on real sanity datasets."""
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default",
            overrides=[
                "dataset=ami",
                "dataset.id=sanity_ami",
                "dataset.reset=false",
                "train=false",
                "test=true",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "+trainer.limit_test_batches=2",
                "~logger",
            ],
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_run = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg_run.test_datasets = [
            {"dataset": "ami", "id": "sanity_ami", "reset": False},
            {"dataset": "brudex", "id": "sanity_brudex", "reset": False},
            {
                "dataset": "pra_anf_circ_8ch",
                "id": "sanity_pra_anf",
                "reset": False,
            },
        ]
        cfg_run.callbacks = {
            "save_results": {
                "_target_": "muse_toolbox.callbacks.save_results.SaveTestResultsCallback",
                "save_dir": tmpdir,
            }
        }

        run_source_counting_pipeline(cfg_run)

        summary_path = Path(tmpdir) / "test_metrics_summary.csv"
        combined_path = Path(tmpdir) / "test_metrics.csv"
        assert summary_path.exists()
        assert combined_path.exists()

        for ds_name in ("ami", "brudex", "pra_anf_circ_8ch"):
            assert (Path(tmpdir) / f"test_metrics_{ds_name}.csv").exists()

        summary_df = pd.read_csv(summary_path)
        assert list(summary_df["dataset"]) == [
            "ami",
            "brudex",
            "pra_anf_circ_8ch",
            "overall_pooled",
            "overall_macro",
        ]
        for col in ("Accuracy", "MAE", "MSE", "F1score_overall"):
            assert col in summary_df.columns
            assert summary_df[col].notna().all()

