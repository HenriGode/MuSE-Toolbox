import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from typing import Any, Callable
from dataclasses import dataclass
import sys
from line_profiler import LineProfiler


@dataclass
class Segment:
    start: int
    end: int
    num_sources: int
    event_type: str  # 'activation', 'deactivation', 'init', 'constant'


class LineTimer:
    """
    Context manager to profile code line-by-line.
    Requires 'line_profiler' package: `pip install line_profiler`

    Usage:
        def my_func():
            # ... code ...

        with LineTimer(my_func):
            my_func()
    """

    def __init__(self, *functions_to_profile):
        self.profiler = LineProfiler()
        for func in functions_to_profile:
            self.profiler.add_function(func)

    def __enter__(self):
        if self.profiler:
            self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.profiler:
            self.profiler.disable()
            self.profiler.print_stats(stream=sys.stdout)


def print_gpu_tensors():
    """
    Prints a summary of GPU memory usage using PyTorch's memory management API.
    Faster and safer than iterating gc.get_objects().
    """
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    print("\n--- GPU Memory Summary (torch.cuda) ---")

    # 1. Overall stats
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved:  {reserved:.2f} MB")
    print(f"Max Alloc: {max_allocated:.2f} MB")

    # 2. Detailed Memory Map (Snapshot)
    # This captures the native allocator state without freezing the Python interpreter
    try:
        snapshot = torch.cuda.memory_snapshot()
        print(f"Recorded memory snapshot entries: {len(snapshot)}")
        # You can inspect 'snapshot' entries if deeper debugging is needed,
        # but just seeing the totals is usually enough to confirm a leak
        # (if 'Allocated' keeps rising).
    except Exception as e:
        print(f"Could not take memory snapshot: {e}")

    print("---------------------------------------\n")


class CodeTimer:
    """
    Context manager to measure execution time of a code block.

    Usage:
        with CodeTimer("My processing block"):
            # paste your code here
            x = some_function()
            y = x + 1
    """

    def __init__(self, label: str = "Block"):
        self.label = label
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_time = time.perf_counter()

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        print(f"Execution time for '{self.label}': {elapsed_ms:.4f} ms")


def time_string(code_str: str, globals_=None, locals_=None):
    """
    Executes a string of Python code and measures its execution time.
    Supports CUDA synchronization.

    If globals_ and locals_ are not provided, it attempts to use the caller's context.

    Args:
        code_str: The code string to execute.
        globals_: Optional globals dictionary.
        locals_: Optional locals dictionary.
    """
    if globals_ is None:
        try:
            # Inspect stack to grab caller's frame for variable context
            frame = sys._getframe(1)
            globals_ = frame.f_globals
            locals_ = frame.f_locals
        except Exception:
            globals_ = globals()
            locals_ = locals()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        exec(code_str, globals_, locals_)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        start_time = time.perf_counter()
        exec(code_str, globals_, locals_)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

    print(f"Code string execution time: {elapsed_ms:.4f} ms")


def dcn(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _get_obj_str(obj) -> str:
    """Helper function to get a string representation of an object's type and shape."""
    if isinstance(obj, torch.Tensor):
        if obj.device != torch.device("cpu"):
            return f"{type(obj).__name__} on {obj.device} {tuple(obj.shape)}"
        return f"{type(obj).__name__} {tuple(obj.shape)}"
    elif isinstance(obj, np.ndarray):
        return f"{type(obj).__name__} {obj.shape}"
    elif isinstance(obj, dict):
        return f"dict with keys: {list(obj.keys())}"
    else:
        return f"{type(obj).__name__}"


def print_structure(
    obj, indent_size: int = 2, condition: Callable[[torch.Tensor], bool] | None = None
):
    """
    Recursively prints the structure of a Python object in a tree-like format.

    - Handles nested dicts, lists, and tuples.
    - Shows type and shape for torch.Tensors and numpy.ndarrays.
    - Summarizes uniform lists/tuples (e.g., "list of 10 x <class 'int'>").
    - Inspects attributes of custom objects, ignoring methods and private attributes.
    - Displays recursive memory consumption for each node.
    - Optional: Accepts a condition callable for tensors. If provided, checks if any
      tensor satisfies the condition. If none do, prints nothing and returns False.
      If at least one does, prints structure with condition status and returns True.

    Args:
        obj: The Python object to inspect.
        indent_size (int): The number of spaces to use for each indentation level.
        condition (Callable[[torch.Tensor], bool], optional): A function that takes a
            torch.Tensor and returns a bool.

    Returns:
        bool | None: True if condition met (and printed), False if condition not met (not printed),
                     None if condition not provided (standard printing).
    """
    # 0. Pre-check condition if supplied
    if condition is not None:
        has_true_tensor = _check_condition_recursive(obj, condition, set())
        if not has_true_tensor:
            return False

    # 1. First pass: Calculate sizes recursively and cache them
    size_cache = {}

    def calculate_size(o, seen):
        # Handle circular references or already visited objects
        obj_id = id(o)
        if obj_id in seen:
            return 0

        # Calculate shallow size
        if isinstance(o, (torch.Tensor, np.ndarray)):
            s = o.nbytes
        else:
            s = sys.getsizeof(o)

        seen.add(obj_id)

        # Recurse for children
        child_sum = 0
        if isinstance(o, dict):
            for key, value in o.items():
                child_sum += calculate_size(key, seen)
                child_sum += calculate_size(value, seen)
        elif isinstance(o, (list, tuple, set)):
            for item in o:
                child_sum += calculate_size(item, seen)
        elif hasattr(o, "__dict__"):
            for attr_name, attr_value in vars(o).items():
                if not attr_name.startswith("_") and not callable(attr_value):
                    child_sum += calculate_size(attr_value, seen)

        total = s + child_sum
        size_cache[obj_id] = total
        return total

    calculate_size(obj, set())

    # 2. Second pass: Print using cached sizes
    _print_recursive_with_memory(obj, 0, indent_size, size_cache, condition)

    if condition is not None:
        return True
    return None


def _check_condition_recursive(obj, condition, seen):
    """
    Recursively checks if any tensor in the object satisfies the condition.
    """
    obj_id = id(obj)
    if obj_id in seen:
        return False
    seen.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return condition(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            if _check_condition_recursive(
                k, condition, seen
            ) or _check_condition_recursive(v, condition, seen):
                return True
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            if _check_condition_recursive(item, condition, seen):
                return True
    elif hasattr(obj, "__dict__"):
        for attr_name, attr_value in vars(obj).items():
            if not attr_name.startswith("_") and not callable(attr_value):
                if _check_condition_recursive(attr_value, condition, seen):
                    return True
    return False


def _print_recursive_with_memory(
    obj, level: int, indent_size: int, size_cache: dict, condition=None
):
    """The internal recursive worker for print_structure."""
    indent = " " * (level * indent_size)

    # Retrieve pre-calculated size
    total_bytes = size_cache.get(id(obj), 0)
    mem_str = format_memory(total_bytes)

    # --- Basic Types ---
    if obj is None or isinstance(obj, (int, float, str, bool)):
        print(f"{indent}{_get_obj_str(obj)} | Memory: {mem_str}")
        return

    # --- Tensors and Numpy Arrays ---
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        extra = ""
        if condition is not None and isinstance(obj, torch.Tensor):
            extra = f" | condition: {condition(obj)}"
        print(f"{indent}{_get_obj_str(obj)} | Memory: {mem_str}{extra}")
        return

    # --- Dictionaries ---
    if isinstance(obj, dict):
        print(f"{indent}dict with {len(obj)} keys | Memory: {mem_str}")
        for key, value in obj.items():
            print(f"{indent}{' ' * indent_size}key: '{key}'")
            _print_recursive_with_memory(
                value, level + 2, indent_size, size_cache, condition
            )
        return

    # --- Lists and Tuples ---
    if isinstance(obj, (list, tuple)):
        container_type = type(obj).__name__
        if not obj:
            print(f"{indent}{container_type} (empty) | Memory: {mem_str}")
            return

        # Check if all elements are of the same type and shape
        first_item_str = _get_obj_str(obj[0])
        is_uniform = all(_get_obj_str(item) == first_item_str for item in obj)

        if is_uniform:
            print(
                f"{indent}{container_type} of {len(obj)} x {first_item_str} | Memory: {mem_str}"
            )
            i = 0
            item = obj[0]
            # print(f"{indent}{' ' * indent_size}[{i}]:")
            _print_recursive_with_memory(
                item, level, indent_size, size_cache, condition
            )
        else:
            print(
                f"{indent}{container_type} of {len(obj)} items (non-uniform) | Memory: {mem_str}:"
            )
            for i, item in enumerate(obj):
                print(f"{indent}{' ' * indent_size}[{i}]:")
                _print_recursive_with_memory(
                    item, level + 2, indent_size, size_cache, condition
                )
        return

    # --- Custom Objects ---
    if hasattr(obj, "__dict__"):
        print(f"{indent}{type(obj).__name__} | Memory: {mem_str}:")
        # Use vars() to get the __dict__ of attributes
        attributes = vars(obj)
        for attr_name, attr_value in attributes.items():
            # Skip private/magic attributes and methods/callables
            if attr_name.startswith("_") or callable(attr_value):
                continue

            print(f"{indent}{' ' * indent_size}attr: '{attr_name}'")
            _print_recursive_with_memory(
                attr_value, level + 2, indent_size, size_cache, condition
            )
        return

    # --- Fallback for any other type ---
    print(f"{indent}{_get_obj_str(obj)} | Memory: {mem_str}")


def to_one_hot(input_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts a tensor of integer class labels to a one-hot encoded tensor.

    This function validates that all class labels in the input tensor are within
    the valid range [0, num_classes - 1] before performing the conversion.

    Args:
        input_tensor (torch.Tensor): A tensor of integer class labels.
            Shape: (..., T), where ... represents any number of leading dimensions.
        num_classes (int): The total number of classes for one-hot encoding.
            This will be the size of the last dimension in the output tensor.

    Returns:
        torch.Tensor: The one-hot encoded tensor.
            Shape: (..., T, num_classes), with a float data type.

    Raises:
        ValueError: If `input_tensor` contains values less than 0 or
            greater than or equal to `num_classes`.
    """
    # --- Validation Step ---
    # Ensure the input tensor does not contain invalid class indices.
    # Class indices must be in the range [0, num_classes - 1].
    min_val = torch.min(input_tensor)
    max_val = torch.max(input_tensor)

    if min_val < 0 or max_val >= num_classes:
        raise ValueError(
            f"Input tensor contains invalid class indices. "
            f"Values must be in the range [0, {num_classes - 1}], but found "
            f"min value: {min_val} and max value: {max_val}."
        )

    # --- Conversion Step ---
    # Convert the input tensor to Long type, as required by one_hot.
    long_tensor = input_tensor.long()

    # Perform the one-hot encoding.
    one_hot_tensor = F.one_hot(long_tensor, num_classes=num_classes)

    # Convert to float, which is standard for model outputs and loss calculations.
    return one_hot_tensor.float()


def format_memory(memory_bytes):
    if memory_bytes < 2**10:
        return f"{memory_bytes} B"
    elif memory_bytes < 2**20:
        return f"{memory_bytes / 2**10:.2f} KB"
    elif memory_bytes < 2**30:
        return f"{memory_bytes / 2**20:.2f} MB"
    else:
        return f"{memory_bytes / 2**30:.2f} GB"


def format_flops(flops):
    if flops < 1e3:
        return f"{flops:.2f} FLOPS"
    if flops < 1e6:
        return f"{flops / 1e3:.2f} KFLOPS"
    elif flops < 1e9:
        return f"{flops / 1e6:.2f} MFLOPS"
    else:
        return f"{flops / 1e9:.2f} GFLOPS"


def format_parameters(params):
    if params < 1e3:
        return f"{params:.2f}"
    if params < 1e6:
        return f"{params / 1e3:.2f} K"
    elif params < 1e9:
        return f"{params / 1e6:.2f} M"
    else:
        return f"{params / 1e9:.2f} B"


import torch
import numpy as np


def deep_equal(a, b):
    """
    Recursively compares two variables for deep equality.

    Supports nested structures of lists, tuples, sets, dicts,
    torch.tensors, numpy.arrays, and basic types.

    For custom classes, it relies on their `__eq__` method. If `__eq__` is not
    implemented, it will fall back to object identity comparison.

    Args:
        a: The first variable.
        b: The second variable.

    Returns:
        True if the variables are deeply equal, False otherwise.
    """
    # Check if types are the same
    if type(a) is not type(b):
        return False

    # Compare torch tensors
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)

    # Compare numpy arrays
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    # Compare dictionaries
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    # Compare lists and tuples
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))

    # Compare sets
    if isinstance(a, set):
        # Note: This works for sets of hashable types. Sets cannot contain
        # unhashable items like lists or dicts.
        if len(a) != len(b):
            return False
        # For unordered sets, we check if every element in 'a' can be found in 'b'
        # by deep equality comparison. This is O(n^2) but robust.
        b_list = list(b)
        for item_a in a:
            found_match = False
            for i, item_b in enumerate(b_list):
                if deep_equal(item_a, item_b):
                    found_match = True
                    b_list.pop(i)
                    break
            if not found_match:
                return False
        return True

    # For all other types (int, float, str, complex, custom classes),
    # use the standard equality operator.
    return a == b


def plot_vad_debug(audio_signal: torch.Tensor, vad_mask: torch.Tensor, save_path: str):
    """
    Plots audio channels and the VAD mask to verify alignment.

    Args:
        audio_signal: [M, N] tensor (M channels, N samples)
        vad_mask: [N] tensor (N samples), binary (0 or 1) or boolean
        save_path: File path to save the plot (include extension like .png)
    """
    # 1. Data Preparation
    # Ensure CPU and Numpy
    if audio_signal.ndim == 1:
        audio_signal = audio_signal.unsqueeze(0)

    M, N = audio_signal.shape
    audio_np = audio_signal.detach().cpu().numpy()
    mask_np = vad_mask.detach().cpu().float().numpy()

    # Normalize audio to [-1, 1] for visualization scaling
    max_val = np.abs(audio_np).max()
    if max_val > 0:
        audio_np = audio_np / max_val

    time_axis = np.arange(N)

    # 2. Plotting
    plt.figure(figsize=(12, 6))

    # Plot Audio Channels
    # If many channels, plot first one bold, others faint
    if M > 1:
        for m in range(1, M):
            plt.plot(time_axis, audio_np[m], color="gray", alpha=0.3, linewidth=0.5)
        plt.plot(
            time_axis,
            audio_np[0],
            color="black",
            alpha=0.8,
            linewidth=1.0,
            label="Ch 0 Audio",
        )
    else:
        plt.plot(
            time_axis,
            audio_np[0],
            color="black",
            alpha=0.8,
            linewidth=1.0,
            label="Audio",
        )

    # Plot Mask Overlay (Green Shading)
    # where=mask_np>0.5 handles both boolean and float masks
    plt.fill_between(
        time_axis,
        -1,
        1,
        where=(mask_np > 0.5),  # type: ignore
        color="green",
        alpha=0.2,
        label="VAD Active (Region)",
    )

    # Plot Mask Line (Red Step)
    plt.plot(
        time_axis,
        mask_np * 0.9,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="VAD Mask (Signal)",
    )

    # 3. Styling
    plt.ylim(-1.1, 1.1)
    plt.title(f"Audio Signal (Normalized) vs VAD Mask\nChannels: {M}, Samples: {N}")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # 4. Save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Free memory
    print(f"VAD Plot saved to: {save_path}")


def time_function(fun: Callable, *args, **kwargs) -> Any:
    """
    Executes a function and prints its execution time with high precision.
    synchronizes CUDA before and after if available to capture GPU time accurately.

    Args:
        fun (Callable): The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: The output of the function.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = fun(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)  # Returns milliseconds
        print(f"Execution time for '{fun.__name__}': {elapsed_time_ms:.4f} ms")

    else:
        start_time = time.perf_counter()
        result = fun(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Execution time for '{fun.__name__}': {elapsed_time_ms:.4f} ms")

    return result


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
