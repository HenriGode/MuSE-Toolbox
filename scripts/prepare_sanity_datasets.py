#!/usr/bin/env python3
"""Script to prepare and validate sanity-check datasets for MuSE-Toolbox.

This script instantiates the refactored DataModules (BRUDEX, PRA-ANF, and AMI)
with lightweight scenario and meeting configurations, generates clean memory-mapped
scenario files on disk, and verifies that the resulting batches conform to the
new [B, M, T] dictionary-based data interface.

Usage:
    # Prepare all three sanity datasets:
    python scripts/prepare_sanity_datasets.py --dataset all

    # Prepare only a specific dataset:
    python scripts/prepare_sanity_datasets.py --dataset brudex
    python scripts/prepare_sanity_datasets.py --dataset pra_anf
    python scripts/prepare_sanity_datasets.py --dataset ami

    # Skip reset (keep existing files if already prepared):
    python scripts/prepare_sanity_datasets.py --no-reset
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure PROJECT_ROOT is set before importing Hydra / configs
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
log = logging.getLogger("prepare_sanity")
logging.getLogger("muse_toolbox").setLevel(logging.INFO)

# Register safe tname resolver if not already present
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and validate lightweight sanity datasets under the unified DynamicChunkDataset architecture."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "brudex", "pra_anf", "ami"],
        help="Which sanity dataset to prepare (default: 'all').",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=True,
        help="Force deletion and clean regeneration of existing sanity datasets (default: True).",
    )
    parser.add_argument(
        "--no-reset",
        action="store_false",
        dest="reset",
        help="Do not delete existing precomputed directories if already generated.",
    )
    parser.add_argument(
        "--num-scenarios-train",
        type=int,
        default=4,
        help="Number of training scenarios for synthetic datasets (default: 4).",
    )
    parser.add_argument(
        "--num-scenarios-val",
        type=int,
        default=2,
        help="Number of validation scenarios for synthetic datasets (default: 2).",
    )
    parser.add_argument(
        "--num-scenarios-test",
        type=int,
        default=2,
        help="Number of test scenarios for synthetic datasets (default: 2).",
    )
    parser.add_argument(
        "--ami-train-meeting",
        type=str,
        default="EN2001a",
        help="AMI meeting ID for training split (default: 'EN2001a').",
    )
    parser.add_argument(
        "--ami-val-meeting",
        type=str,
        default="ES2011a",
        help="AMI meeting ID for validation split (default: 'ES2011a').",
    )
    parser.add_argument(
        "--ami-test-meeting",
        type=str,
        default="EN2002a",
        help="AMI meeting ID for test split (default: 'EN2002a').",
    )
    parser.add_argument(
        "--ami-array",
        type=str,
        default="Array1",
        help="AMI microphone array identifier (default: 'Array1').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for checking dataloaders (default: 2).",
    )
    return parser.parse_args()


def prepare_and_verify(
    dataset_name: str,
    hydra_overrides: list[str],
    expected_id: str,
) -> dict[str, Any]:
    """Prepares and validates a single dataset module."""
    log.info(f"\n{'='*70}\n[PREPARING SANITY DATASET] {dataset_name} (ID: {expected_id})\n{'='*70}")

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="default", overrides=[f"dataset={dataset_name}"] + hydra_overrides)

    log.info(f"Instantiating DataModule for {dataset_name}...")
    datamodule = instantiate(cfg.dataset)

    # 1. Run prepare_data()
    log.info("Running datamodule.prepare_data()...")
    datamodule.prepare_data()

    # 2. Check disk artifacts
    precomputed_dir: Path = Path(datamodule.precomputed_dir)
    log.info(f"Verifying disk artifacts at: {precomputed_dir}")

    if not precomputed_dir.exists():
        raise FileNotFoundError(f"Expected directory {precomputed_dir} was not created.")

    split_counts: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        split_dir = precomputed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        audio_files = list(split_dir.glob("*_audio.npy"))
        sad_files = list(split_dir.glob("*_sad.npy"))
        meta_files = list(split_dir.glob("*_meta.pt"))

        log.info(f"  • Split '{split}': {len(audio_files)} audio (.npy), {len(sad_files)} SAD (.npy), {len(meta_files)} meta (.pt)")
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found in {split_dir}")
        if len(audio_files) != len(sad_files) or len(audio_files) != len(meta_files):
            raise ValueError(f"Mismatched file count in {split_dir}: {len(audio_files)} audio vs {len(sad_files)} sad vs {len(meta_files)} meta")

        split_counts[split] = len(audio_files)

    # Check AMI single-speaker index if applicable
    single_spk_index_file = list(precomputed_dir.glob("single_speaker_index_*.npy"))
    if single_spk_index_file:
        index_arr = np.load(single_spk_index_file[0])
        log.info(f"  • Single-speaker augmentation index found: {single_spk_index_file[0].name} ({len(index_arr)} chunks indexed)")

    # 3. Setup datamodule and verify batch shapes
    log.info("Setting up DataModule ('fit' & 'test')...")
    datamodule.setup("fit")
    datamodule.setup("test")

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    train_batch = next(iter(train_dl))
    val_batch = next(iter(val_dl))

    # Validate structure
    assert "input" in train_batch, "Missing 'input' key in batch dictionary"
    assert "meta" in train_batch, "Missing 'meta' key in batch dictionary"

    input_tensor = train_batch["input"]
    meta = train_batch["meta"]

    # Check input shape [B, M, T]
    assert input_tensor.ndim == 3, f"Expected 3D input tensor [B, M, T], got {input_tensor.shape}"
    b, m, t = input_tensor.shape
    fs = getattr(datamodule, "sampling_frequency", 8000)
    duration_s = t / fs

    sad_list = meta.get("sad_samples", [])
    assert isinstance(sad_list, list), f"Expected meta['sad_samples'] to be a list of tensors, got {type(sad_list)}"
    assert len(sad_list) == b, f"Expected len(sad_samples) == batch_size ({b}), got {len(sad_list)}"

    scenario_params = meta.get("scenario_params", [{}])
    aug_applied = scenario_params[0].get("augmentation_applied", False) if scenario_params else False

    log.info(f"Batch Verification PASSED:")
    log.info(f"  • Input Shape [B, M, T]: {input_tensor.shape} (dtype: {input_tensor.dtype})")
    log.info(f"  • Microphones (channels): {m}")
    log.info(f"  • Chunk Length: {t} samples ({duration_s:.1f}s at {fs} Hz)")
    log.info(f"  • SAD Sample Shapes: {[s.shape for s in sad_list]}")
    log.info(f"  • Augmentation applied in item 0: {aug_applied}")

    return {
        "status": "PASSED",
        "dir": str(precomputed_dir),
        "split_counts": split_counts,
        "input_shape": list(input_tensor.shape),
        "num_mics": m,
    }


def main() -> None:
    args = parse_args()

    targets: list[str] = []
    if args.dataset in ["all", "brudex"]:
        targets.append("brudex")
    if args.dataset in ["all", "pra_anf"]:
        targets.append("pra_anf")
    if args.dataset in ["all", "ami"]:
        targets.append("ami")

    reset_str = "true" if args.reset else "false"
    num_scenarios_str = f"[{args.num_scenarios_train},{args.num_scenarios_val},{args.num_scenarios_test}]"

    configs_to_run: dict[str, dict[str, Any]] = {
        "brudex": {
            "dataset_name": "brudex",
            "expected_id": "sanity_brudex",
            "overrides": [
                "dataset.id=sanity_brudex",
                f"dataset.num_scenarios={num_scenarios_str}",
                "dataset.train_items_per_epoch=10",
                f"dataset.batch_size={args.batch_size}",
                f"dataset.reset={reset_str}",
            ],
        },
        "pra_anf": {
            "dataset_name": "pra_anf_circ_8ch",
            "expected_id": "sanity_pra_anf",
            "overrides": [
                "dataset.id=sanity_pra_anf",
                f"dataset.num_scenarios={num_scenarios_str}",
                "dataset.train_items_per_epoch=10",
                f"dataset.batch_size={args.batch_size}",
                f"dataset.reset={reset_str}",
            ],
        },
        "ami": {
            "dataset_name": "ami",
            "expected_id": "sanity_ami",
            "overrides": [
                "dataset.id=sanity_ami",
                f"dataset.train_meetings=['{args.ami_train_meeting}']",
                f"dataset.val_meetings=['{args.ami_val_meeting}']",
                f"dataset.test_meetings=['{args.ami_test_meeting}']",
                f"dataset.arrays=['{args.ami_array}']",
                "dataset.train_items_per_epoch=10",
                f"dataset.batch_size={args.batch_size}",
                f"dataset.reset={reset_str}",
                "dataset.augmentation.enabled=true",
            ],
        },
    }

    results: dict[str, dict[str, Any]] = {}

    for name in targets:
        info = configs_to_run[name]
        try:
            res = prepare_and_verify(
                dataset_name=str(info["dataset_name"]),
                hydra_overrides=list(info["overrides"]),
                expected_id=str(info["expected_id"]),
            )
            results[name] = res
        except Exception as e:
            log.error(f"Failed while preparing dataset '{name}': {e}")
            traceback.print_exc()
            results[name] = {"status": "FAILED", "error": str(e)}

    # Final Summary Table
    log.info(f"\n{'='*70}\nSANITY DATASETS GENERATION SUMMARY\n{'='*70}")
    all_passed = True
    for name, res in results.items():
        status = res.get("status", "UNKNOWN")
        if status != "PASSED":
            all_passed = False
            log.info(f"❌ {name.upper()}: FAILED - {res.get('error')}")
        else:
            log.info(
                f"✅ {name.upper()}: PASSED | Path: {res['dir']} | "
                f"Splits: {res['split_counts']} | Batch: {res['input_shape']}"
            )

    if all_passed:
        log.info("\n🎉 All requested sanity datasets were generated and verified successfully!")
    else:
        log.warning("\n⚠️ One or more sanity datasets encountered issues during generation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
