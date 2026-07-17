""""""

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


def format_time(seconds: float, detailed: bool = False, levels: int = 3) -> str:
    """Formats a time duration into a human-readable string (ns to weeks).

    Args:
        seconds (float): Time duration in seconds.
        detailed (bool): If True, formats using up to `levels` tiers of units for times >= 1s (e.g. '3 weeks 2 days').
        levels (int): Maximum number of unit levels to show (1 to 6). e.g. levels=3 shows (week, day, hour).

    Returns:
        str: Formatted string (e.g., '2.50 ms', '1.50 h', '3 weeks 2 days').
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} \u03bcs"  # microseconds (μs)
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
        
    if not detailed:
        if seconds < 60.0:
            return f"{seconds:.2f} s"
        if seconds < 3600.0:
            return f"{seconds / 60.0:.2f} min"
        if seconds < 86400.0:
            return f"{seconds / 3600.0:.2f} h"
        if seconds < 604800.0:
            return f"{seconds / 86400.0:.2f} days"
        return f"{seconds / 604800.0:.2f} weeks"

    units = [
        ("week", 604800.0),
        ("day", 86400.0),
        ("hour", 3600.0),
        ("min", 60.0),
        ("s", 1.0)
    ]
    
    parts = []
    rem_seconds = seconds
    first_idx = -1
    
    for i, (name, factor) in enumerate(units):
        if first_idx == -1:
            if rem_seconds >= factor or (name == "s" and rem_seconds > 0):
                first_idx = i
            else:
                continue
                
        # Stop if we've consumed the allowed number of levels
        if i >= first_idx + levels:
            break
            
        if name == "s":
            # The 6th level effectively means "allow decimals on seconds"
            # If our permitted levels extend beyond index 4 (s), we can show decimals.
            allow_decimals = (first_idx + levels > 5)
            val = rem_seconds
            if not allow_decimals:
                val = int(val)
            if val > 0:
                parts.append((val, name))
        else:
            val = int(rem_seconds // factor)
            if val > 0:
                parts.append((val, name))
            rem_seconds -= val * factor

    formatted_parts = []
    for val, name in parts:
        if name == "s" and isinstance(val, float):
            val_str = f"{val:.3f}".rstrip('0').rstrip('.') if '.' in f"{val:.3f}" else f"{val}"
            name_str = "s"
        else:
            val_str = str(val)
            name_str = name if val == 1 or name == "s" else name + "s"
        formatted_parts.append(f"{val_str} {name_str}")
        
    return " ".join(formatted_parts) if formatted_parts else "0 s"
