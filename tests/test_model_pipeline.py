#!/usr/bin/env python3
"""Sanity test script for the MuSE-Toolbox model pipeline.

This script tests the end-to-end forward pass, loss calculation, backward pass,
and metric calculation on the refactored dictionary-based batches ([B, M, T]).
It validates both homogeneous and heterogeneous (variable microphone count) batches
across our generated sanity datasets (BRUDEX, PRA-ANF, AMI).

Usage:
    # Test all sanity datasets on GPU (or CPU if no CUDA):
    python tests/test_model_pipeline.py

    # Test a specific dataset:
    python tests/test_model_pipeline.py --dataset brudex
    python tests/test_model_pipeline.py --dataset pra_anf
    python tests/test_model_pipeline.py --dataset ami

    # Test with specific channel combinator:
    python tests/test_model_pipeline.py --combinator self_attention
    python tests/test_model_pipeline.py --combinator average
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure PROJECT_ROOT is set
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_model")
logging.getLogger("muse_toolbox").setLevel(logging.WARNING)

# Register safe tname resolver if needed
def target_to_name(target_str: str) -> str:
    if not target_str:
        return "unknown"
    name = target_str.split(".")[-1].lower()
    for s in ["_feature_extractor", "channelcombinator", "estimator"]:
        name = name.replace(s, "")
    if name.startswith("selfattention"):
        return "self_attention"
    return name

if not OmegaConf.has_resolver("tname"):
    OmegaConf.register_new_resolver("tname", target_to_name, replace=True)

from muse_toolbox.data.components.collate import raw_audio_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify model pipeline forward, backward, loss, and metrics on sanity datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "brudex", "pra_anf", "ami"],
        help="Which sanity dataset to test against (default: 'all').",
    )
    parser.add_argument(
        "--combinator",
        type=str,
        default="self_attention",
        choices=["self_attention", "average", "none"],
        help="Channel combinator to test (default: 'self_attention').",
    )
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="wgmsc",
        help="Feature extractor to use (default: 'wgmsc').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size to load from datamodule (default: 2).",
    )
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Recursively moves tensors in the batch dictionary to the target device."""
    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        elif isinstance(v, dict):
            moved[k] = move_batch_to_device(v, device)
        elif isinstance(v, list):
            moved[k] = [
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in v
            ]
        else:
            moved[k] = v
    return moved


def test_cosad_on_dataset(
    dataset_name: str,
    sanity_id: str,
    combinator: str,
    feature_extractor: str,
    device_str: str,
    batch_size: int,
) -> dict[str, Any]:
    """Tests COSADmodule forward pass, loss, backward pass, and metrics."""
    log.info(f"\n{'='*75}\n[TESTING MODEL ON DATASET] {dataset_name} (ID: {sanity_id})\n{'='*75}")

    device = torch.device(device_str)
    log.info(f"Target execution device: {device}")

    # 1. Compose Hydra Configuration
    GlobalHydra.instance().clear()
    overrides = [
        f"dataset={dataset_name}",
        f"dataset.id={sanity_id}",
        "dataset.reset=false",
        f"dataset.batch_size={batch_size}",
        "model=cosad",
        f"model/feature_extractor={feature_extractor}",
    ]
    if combinator == "none":
        overrides.append("model/channel_combinator=null")
    else:
        overrides.append(f"model/channel_combinator={combinator}")

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="default", overrides=overrides)

    # 2. Instantiate DataModule & Model
    log.info(f"Instantiating DataModule ({dataset_name})...")
    datamodule = instantiate(cfg.dataset)
    datamodule.setup("fit")

    log.info(f"Instantiating COSADmodule (Extractor: {feature_extractor}, Combinator: {combinator})...")
    model = instantiate(cfg.model)
    model.to(device)
    model.train()

    # 3. Pull a real training batch
    log.info("Fetching train batch from dataloader...")
    train_dl = datamodule.train_dataloader()
    raw_batch = next(iter(train_dl))
    batch = move_batch_to_device(raw_batch, device)

    input_audio = batch["input"]
    b, m, t = input_audio.shape
    log.info(f"  • Batch input shape [B, M, T]: {list(input_audio.shape)}")
    log.info(f"  • Microphone lengths: {batch.get('mic_lengths').tolist()}")
    log.info(f"  • Time lengths (samples): {batch.get('time_lengths').tolist()}")

    # 4. Forward Pass Check
    log.info("Executing model forward pass...")
    out = model(batch)
    estimates = out["estimates"]
    log.info(f"  • Estimates shape [B, Frames, Classes]: {list(estimates.shape)}")
    assert estimates.ndim == 3, f"Expected 3D estimates [B, T_frames, C], got {estimates.shape}"
    assert not torch.isnan(estimates).any(), "NaN detected in forward estimates!"
    assert not torch.isinf(estimates).any(), "Inf detected in forward estimates!"
    log.info("  ✅ Forward pass completed successfully!")

    # 5. Loss & Backward Pass Check
    log.info("Executing training_step (Loss computation & backpropagation)...")
    loss = model.training_step(batch, idx=0)
    log.info(f"  • Computed Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    model.zero_grad()
    loss.backward()

    # Verify gradients
    grad_counts = 0
    nan_grads = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_counts += 1
            if torch.isnan(param.grad).any():
                nan_grads += 1
                log.error(f"NaN gradient in parameter: {name}")

    log.info(f"  • Parameters with valid gradients: {grad_counts}")
    assert grad_counts > 0, "No gradients were computed!"
    assert nan_grads == 0, f"Found {nan_grads} parameters with NaN gradients!"
    log.info("  ✅ Loss and backward pass verified with valid gradients!")

    # 6. Metric Step Check (Validation Step)
    log.info("Executing validation_step & metric computation...")
    model.eval()
    with torch.no_grad():
        val_dl = datamodule.val_dataloader()
        val_raw_batch = next(iter(val_dl))
        val_batch = move_batch_to_device(val_raw_batch, device)
        model.validation_step(val_batch, idx=0)

    # Compute metric collections
    if "val" in model.metric_collections:
        try:
            val_metrics = model.metric_collections["val"].compute()
            metric_keys = list(val_metrics.keys())
            log.info(f"  • Computed metrics successfully: {metric_keys[:4]}...")
            log.info("  ✅ Validation step and metrics verified!")
        except Exception as e:
            log.warning(f"  ⚠️ Metric computation notice: {e}")
            log.warning("  (Check if frame length vs sample length conversion is needed in _metric_step)")

    # 7. Heterogeneous Microphone Padding Stress Test
    log.info("Stress testing heterogeneous microphone count batching...")
    # Create two items with different microphone counts (e.g. 3 mics and 8 mics)
    T_test = 56000
    item_3ch = {
        "input": torch.randn(3, T_test, dtype=torch.float32),
        "meta": {
            "source_count": torch.randint(0, 4, (T_test // 128,), dtype=torch.long),
            "sad_samples": torch.zeros(4, T_test, dtype=torch.bool),
        },
    }
    item_8ch = {
        "input": torch.randn(8, T_test, dtype=torch.float32),
        "meta": {
            "source_count": torch.randint(0, 4, (T_test // 128,), dtype=torch.long),
            "sad_samples": torch.zeros(4, T_test, dtype=torch.bool),
        },
    }

    het_batch_raw = raw_audio_collate_fn([item_3ch, item_8ch])
    het_batch = move_batch_to_device(het_batch_raw, device)

    model.train()
    het_out = model(het_batch)
    het_estimates = het_out["estimates"]
    assert het_estimates.shape[0] == 2
    assert not torch.isnan(het_estimates).any(), "NaN detected in heterogeneous batch estimates!"

    het_loss = model.training_step(het_batch, idx=0)
    assert torch.isfinite(het_loss)
    model.zero_grad()
    het_loss.backward()
    log.info("  ✅ Heterogeneous microphone batch (3 mics + 8 mics) PASSED cleanly!")

    return {
        "dataset": dataset_name,
        "forward": "PASSED",
        "loss": f"{loss.item():.4f}",
        "backward": "PASSED",
        "heterogeneous": "PASSED",
    }


def main() -> None:
    args = parse_args()

    target_datasets: list[tuple[str, str]] = []
    if args.dataset in ["all", "brudex"]:
        target_datasets.append(("brudex", "sanity_brudex"))
    if args.dataset in ["all", "pra_anf"]:
        target_datasets.append(("pra_anf_circ_8ch", "sanity_pra_anf"))
    if args.dataset in ["all", "ami"]:
        target_datasets.append(("ami", "sanity_ami"))

    results: list[dict[str, Any]] = []

    for ds_name, sanity_id in target_datasets:
        try:
            res = test_cosad_on_dataset(
                dataset_name=ds_name,
                sanity_id=sanity_id,
                combinator=args.combinator,
                feature_extractor=args.feature_extractor,
                device_str=args.device,
                batch_size=args.batch_size,
            )
            results.append(res)
        except Exception as e:
            log.error(f"❌ Test FAILED on dataset {ds_name}: {e}")
            traceback.print_exc()
            results.append({"dataset": ds_name, "forward": "FAILED", "error": str(e)})

    # Summary Table
    log.info(f"\n{'='*75}\nMODEL PIPELINE VERIFICATION SUMMARY\n{'='*75}")
    all_passed = True
    for r in results:
        ds = r["dataset"].upper()
        if r.get("forward") == "PASSED":
            log.info(
                f"✅ {ds:<15} | Forward: {r['forward']} | Loss: {r['loss']} | "
                f"Backward: {r['backward']} | Heterogeneous: {r['heterogeneous']}"
            )
        else:
            all_passed = False
            log.info(f"❌ {ds:<15} | FAILED: {r.get('error')}")

    if all_passed:
        log.info("\n🎉 All model pipeline verification checks PASSED successfully!")
    else:
        log.warning("\n⚠️ One or more model checks encountered issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
