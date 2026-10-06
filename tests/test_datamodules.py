import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
import logging

logging.getLogger("muse_toolbox").setLevel(logging.WARNING)

def test_dataset(dataset_name, overrides):
    print(f"\n{'='*50}")
    print(f"Testing Dataset: {dataset_name}")
    print(f"{'='*50}")
    
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="default", overrides=[f"dataset={dataset_name}"] + overrides)
        
    print(f"Instantiating {dataset_name} DataModule...")
    datamodule = instantiate(cfg.dataset)
    
    print(f"Running prepare_data()...")
    datamodule.prepare_data()
    
    print(f"Running setup('fit')...")
    datamodule.setup('fit')
    
    print(f"Fetching train_dataloader...")
    train_dl = datamodule.train_dataloader()
    
    print(f"\n--- Checking Train Batch ---")
    batch = next(iter(train_dl))
    print(f"Batch Keys: {list(batch.keys())}")
    print(f"Input Shape: {batch['input'].shape}")
    print(f"Input Type: {batch['input'].dtype}")
    
    meta = batch['meta']
    print(f"Meta Keys: {list(meta.keys())}")
    print(f"SAD Samples Shape: {meta['sad_samples'][0].shape}")
    print(f"SAD Samples Type: {meta['sad_samples'][0].dtype}")
    print(f"Augmentation Applied: {meta['scenario_params'][0].get('augmentation_applied')}")
    if 'single_source_components' in meta and meta['single_source_components'][0] is not None:
         print(f"Has single source components: {meta['single_source_components'][0].shape}")
         
    print(f"\n{dataset_name} PASSED!\n")

if __name__ == "__main__":
    # Test Brudex
    try:
        test_dataset("brudex", overrides=[
            "dataset.num_scenarios=[2,1,1]",
            "dataset.train_items_per_epoch=10",
            "dataset.batch_size=2",
            "dataset.reset=true",
        ])
    except Exception as e:
        print(f"BRUDEX FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test PRA_ANF
    try:
        test_dataset("pra_anf_circ_8ch", overrides=[
            "dataset.num_scenarios=[2,1,1]",
            "dataset.train_items_per_epoch=10",
            "dataset.batch_size=2",
            "dataset.reset=true",
        ])
    except Exception as e:
        print(f"PRA_ANF FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test AMI
    try:
        test_dataset("ami", overrides=[
            "dataset.train_meetings=['EN2002a']",
            "dataset.val_meetings=['EN2002a']",
            "dataset.test_meetings=['EN2002a']",
            "dataset.train_items_per_epoch=10",
            "dataset.batch_size=2",
            "dataset.arrays=['Array1']",
        ])
    except Exception as e:
        print(f"AMI FAILED: {e}")
        import traceback
        traceback.print_exc()
