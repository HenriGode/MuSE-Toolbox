def dac4torch_fun(
    fun: Callable, tensor: torch.Tensor, broadcast_threshold, mode: str = "normal"
) -> Any:
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
    """
    A wrapper to run a torch function with specific settings.
    Parameters:
        fun (Callable): The torch function to be executed.
        mode (str): The mode of execution. Options are "normal", "magma", or "cpu".
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
    return tensor.numel() * tensor.element_size()