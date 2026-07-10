import torch
from typing import List


def slice2segments(
    signal: torch.Tensor, segment_borders: torch.Tensor
) -> List[torch.Tensor]:
    """
    Splits a signal tensor into segments based on provided segment borders.

    Args:
        signal (torch.Tensor): The input signal tensor to be split. This can have any number of dimensions, but
                               the last dimension is assumed to represent time or the sequence to be split.
        segment_borders (torch.Tensor): A 1D tensor containing the segment start and end indices. The borders
                                        define the points in time (or along the sequence) where the segments begin and end.
                                        The length of this tensor should be at least 2 (start and end).

    Returns:
        list[torch.Tensor]: A list of tensor segments, where each segment is sliced from `signal` according to the
                            indices in `segment_borders`. Each segment corresponds to a slice from `seg_start` to `seg_end`
                            along the last dimension of `signal`.
    """

    # Use list comprehension to iterate over pairs of consecutive segment borders (start, end).
    # Each segment is sliced from the `signal` tensor using the start and end indices, slicing along the last dimension.
    # The signal[..., seg_start:seg_end] syntax ensures that slicing occurs on the last dimension, regardless of
    # how many other dimensions the tensor has.
    return [signal[..., segment_borders[0] : segment_borders[1] + 1]] + [
        signal[..., seg_start + 1 : seg_end + 1]
        for seg_start, seg_end in zip(segment_borders[1:-1], segment_borders[2:])
    ]
