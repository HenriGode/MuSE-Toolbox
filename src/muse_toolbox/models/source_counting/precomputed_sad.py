"""Precomputed Source Activity Detection (SAD) model.

This module provides a dummy PyTorch model that loads precomputed SAD 
predictions from disk instead of calculating them on the fly.
"""

import logging
import os

import torch

from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch

log = logging.getLogger(__name__)

class PrecomputedSAD(torch.nn.Module):
    """
    A wrapper module to load precomputed SAD (Source Activity Detection) 
    predictions from disk. This allows the RTF module to use cached 
    results instead of running the SAD model on the fly.
    """

    def __init__(self, predictions_dir: str):
        """
        Args:
            predictions_dir (str): Path to the directory containing precomputed 
                .pt files named by scenario ID.
        """
        super().__init__()
        self.predictions_dir = predictions_dir

    def forward(self, x: HeterogeneousBatch) -> list[torch.Tensor]:
        """
        Loads precomputed Source Activity Detection (SAD) results from disk.

        Args:
            x (HeterogeneousBatch): A batch object that contains a 'meta' 
                dictionary with a 'scenario_id' key.

        Returns:
            list[torch.Tensor]: A list of estimated source activity tensors, loaded from disk.

        Raises:
            ValueError: If the input batch lacks the required 'meta' dictionary with 'scenario_id'.
            FileNotFoundError: If the precomputed SAD prediction file is not found on disk.
        """
        if not hasattr(x, "meta") or "scenario_id" not in x.meta:
            raise ValueError("Input batch lacks required 'meta' dictionary with 'scenario_id'.")
            
        scenario_ids = x.meta["scenario_id"]
        estimated_source_activities = []
        device = x.device if hasattr(x, "device") else torch.device("cpu")

        for sid in scenario_ids:
            filepath = os.path.join(self.predictions_dir, f"{sid}.pt")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"SAD prediction not found for {sid} at {filepath}")

            pred = torch.load(filepath, map_location=device)
            estimated_source_activities.append(pred)

        return estimated_source_activities
