import torch
from typing import Any, Dict, List

def raw_audio_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a list of dictionaries containing raw audio and metadata into a batched dictionary.
    
    Handles heterogeneous lengths in both time (T) and channels (M).
    Pads the 'input' tensors to [B, max_M, max_T].
    Extracts lengths to be used as masks in the model pipeline.
    
    Args:
        batch (List[Dict[str, Any]]): A list where each element is a dict with:
            - 'input': torch.Tensor of shape [M_i, T_i]
            - 'meta': Dict containing metadata.
            
    Returns:
        Dict[str, Any]: A batched dictionary:
            - 'input': torch.Tensor [B, max_M, max_T]
            - 'time_lengths': torch.Tensor [B]
            - 'mic_lengths': torch.Tensor [B]
            - 'meta': Collated metadata dictionary
    """
    B = len(batch)
    
    # 1. Find max_M and max_T
    max_M = 0
    max_T = 0
    time_lengths = torch.zeros(B, dtype=torch.long)
    mic_lengths = torch.zeros(B, dtype=torch.long)
    
    for i, item in enumerate(batch):
        audio = item["input"]
        M, T = audio.shape
        time_lengths[i] = T
        mic_lengths[i] = M
        if M > max_M:
            max_M = M
        if T > max_T:
            max_T = T
            
    # 2. Pad the inputs
    padded_inputs = torch.zeros((B, max_M, max_T), dtype=torch.float32)
    for i, item in enumerate(batch):
        audio = item["input"]
        M, T = audio.shape
        padded_inputs[i, :M, :T] = audio
        
    # 3. Collate meta
    collated_meta = {}
    if "meta" in batch[0]:
        for key in batch[0]["meta"].keys():
            collated_meta[key] = [item["meta"][key] for item in batch]
            
    return {
        "input": padded_inputs,
        "time_lengths": time_lengths,
        "mic_lengths": mic_lengths,
        "meta": collated_meta
    }
