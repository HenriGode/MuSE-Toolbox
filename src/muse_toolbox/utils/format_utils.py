

def format_memory(memory_bytes: int) -> str:
    """Formats a memory size in bytes into a human-readable string.

    Args:
        memory_bytes (int): The memory size in bytes.

    Returns:
        str: Formatted string (e.g., '1.50 MB').
    """
    if memory_bytes < 2**10:
        return f"{memory_bytes} B"
    elif memory_bytes < 2**20:
        return f"{memory_bytes / 2**10:.2f} KB"
    elif memory_bytes < 2**30:
        return f"{memory_bytes / 2**20:.2f} MB"
    else:
        return f"{memory_bytes / 2**30:.2f} GB"


def format_flops(flops: float) -> str:
    """Formats a FLOPS count into a human-readable string.

    Args:
        flops (float): The number of floating point operations.

    Returns:
        str: Formatted string (e.g., '2.50 MFLOPS').
    """
    if flops < 1e3:
        return f"{flops:.2f} FLOPS"
    if flops < 1e6:
        return f"{flops / 1e3:.2f} KFLOPS"
    elif flops < 1e9:
        return f"{flops / 1e6:.2f} MFLOPS"
    else:
        return f"{flops / 1e9:.2f} GFLOPS"


def format_parameters(params: int | float) -> str:
    """Formats a parameter count into a human-readable string.

    Args:
        params (int | float): The number of parameters.

    Returns:
        str: Formatted string (e.g., '10.50 M').
    """
    if params < 1e3:
        return f"{params:.2f}"
    if params < 1e6:
        return f"{params / 1e3:.2f} K"
    elif params < 1e9:
        return f"{params / 1e6:.2f} M"
    else:
        return f"{params / 1e9:.2f} B"
