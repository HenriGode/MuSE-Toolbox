import torch
from typing import Any, Callable
def dac4torch_fun(
    fun: Callable, tensor: torch.Tensor, broadcast_threshold: int, mode: str = "normal"
) -> Any:
    """Recursively executes a function on a tensor using a divide-and-conquer approach.

    Args:
        fun (Callable): The function to apply.
        tensor (torch.Tensor): The input tensor.
        broadcast_threshold (int): Maximum number of elements before splitting.
        mode (str, optional): Execution mode. Defaults to "normal".

    Returns:
        Any: The computed result.
    """
    try:
        return run_torch_function_with_settings(
            fun,
            tensor,
            mode=mode,
            dac=True,
            broadcast_threshold=broadcast_threshold,
        )
    except RuntimeError:
        return dac4torch_fun(fun, tensor, broadcast_threshold // 2, mode=mode)


def run_torch_function_with_settings(
    fun: Callable,
    tensor: torch.Tensor,
    mode: str = "normal",
    dac: bool = False,
    loop: bool = False,
    **kwargs,
) -> Any:
    """A wrapper to run a torch function with specific settings.
    
    Args:
        fun (Callable): The torch function to be executed.
        tensor (torch.Tensor): The input tensor.
        mode (str): The mode of execution. Options are "normal", "magma", "cuda", or "cpu".
        dac (bool): Whether to use divide-and-conquer.
        loop (bool): Whether to loop over chunks.
        **kwargs: Additional arguments such as broadcast_threshold.
        
    Returns:
        Any: The result of the torch function.
    """
    if dac:
        assert (
            "broadcast_threshold" in kwargs
        ), "Please provide 'broadcast_threshold' in kwargs for 'dac' mode."
        broadcast_threshold = kwargs["broadcast_threshold"]
        broadcast_shape = tensor.shape[:-2]
        num_matrices = tensor.numel() // (tensor.shape[-1] * tensor.shape[-2])
        if num_matrices > broadcast_threshold:
            flat_tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
            mid = flat_tensor.shape[0] // 2
            result1 = run_torch_function_with_settings(
                fun, flat_tensor[:mid], mode=mode, dac=dac, **kwargs
            )
            result2 = run_torch_function_with_settings(
                fun, flat_tensor[mid:], mode=mode, dac=dac, **kwargs
            )
            if isinstance(result1, tuple):
                flat_result = tuple(
                    torch.cat([r1, r2], dim=0) for r1, r2 in zip(result1, result2)
                )
                return tuple(
                    r.reshape(*broadcast_shape, *r.shape[1:]) for r in flat_result
                )
            else:
                flat_result = torch.cat([result1, result2], dim=0)
                return flat_result.reshape(*broadcast_shape, *flat_result.shape[1:])

    if loop:
        assert (
            "broadcast_threshold" in kwargs
        ), "Please provide 'broadcast_threshold' in kwargs for 'dac' mode."
        broadcast_threshold = kwargs["broadcast_threshold"]
        broadcast_shape = tensor.shape[:-2]
        num_matrices = tensor.numel() // (tensor.shape[-1] * tensor.shape[-2])
        if num_matrices > broadcast_threshold:
            flat_tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
            numblocks = flat_tensor.shape[0] // broadcast_threshold + 1

            results = []
            for i in range(numblocks):
                results.append(
                    run_torch_function_with_settings(
                        fun,
                        flat_tensor[
                            i * broadcast_threshold : (i + 1) * broadcast_threshold
                        ],
                        mode=mode,
                        loop=loop,
                        **kwargs,
                    )
                )
            if isinstance(results[0], tuple):
                flat_result = tuple(
                    torch.cat([res[i] for res in results], dim=0)
                    for i in range(len(results[0]))
                )
                return tuple(
                    r.reshape(*broadcast_shape, *r.shape[1:]) for r in flat_result
                )
            else:
                flat_result = torch.cat(results, dim=0)
                return flat_result.reshape(*broadcast_shape, *flat_result.shape[1:])

    if mode == "normal":
        return fun(tensor)
    elif mode == "cpu":
        original_device = tensor.device
        result = fun(tensor.cpu())
        if isinstance(result, tuple):
            return tuple(res.to(original_device) for res in result)
        return result.to(original_device)
    elif mode == "cuda":
        original_device = tensor.device
        result = fun(tensor.cuda())
        if isinstance(result, tuple):
            return tuple(res.to(original_device) for res in result)
        return result.to(original_device)
    elif mode == "magma":
        original_device = tensor.device
        torch.backends.cuda.preferred_linalg_library(backend="magma")
        result = fun(tensor.cuda())
        torch.backends.cuda.preferred_linalg_library(backend="default")
        if isinstance(result, tuple):
            return tuple(res.to(original_device) for res in result)
        return result.to(original_device)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'normal', 'cpu', 'cuda', or 'magma'."
        )


def memory(tensor: torch.Tensor) -> int:
    """Calculates the total memory in bytes occupied by a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        int: The memory size in bytes.
    """
    return tensor.numel() * tensor.element_size()

def move2device(obj: Any, device: torch.device | str) -> Any:
    """
    Recursively moves all tensors within a nested structure to the specified device.

    - Handles nested dicts, lists, tuples, sets.
    - Moves tensor attributes within custom objects (modifies in-place).
    - Respects existing .to() methods (e.g., for nn.Module).
    - Preserves namedtuples.

    Args:
        obj: The object to move.
        device: The target device (e.g., 'cuda', 'cpu').

    Returns:
        The moved object. Standard collections are returned as new objects;
        custom objects are often modified in-place.
    """
    # 1. Base Case: Tensor
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    # 2. Handle Custom Objects that already know how to move (e.g., nn.Module)
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)

    # 3. Recursion for Standard Containers
    if isinstance(obj, dict):
        return {k: move2device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move2device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        # Special handling for namedtuples
        if hasattr(obj, "_fields"):
            return type(obj)(*(move2device(x, device) for x in obj))
        return tuple(move2device(v, device) for v in obj)
    elif isinstance(obj, set):
        return {move2device(v, device) for v in obj}

    # 4. Custom Data Objects (e.g., dataclasses, simple classes)
    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in vars(obj).items():
            # Skip private attributes and callables (methods/functions attached to instance)
            if not attr_name.startswith("_") and not callable(attr_value):
                setattr(obj, attr_name, move2device(attr_value, device))
        return obj

    # 5. Fallback for primitives or unsupported types
    return obj
