import torch
import time
from typing import Any, Callable
import sys
from line_profiler import LineProfiler
import logging

logger = logging.getLogger(__name__)


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
        logger.info("CUDA not available.")
        return

    logger.info("\n--- GPU Memory Summary (torch.cuda) ---")

    # 1. Overall stats
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2

    logger.info(f"Allocated: {allocated:.2f} MB")
    logger.info(f"Reserved:  {reserved:.2f} MB")
    logger.info(f"Max Alloc: {max_allocated:.2f} MB")

    # 2. Detailed Memory Map (Snapshot)
    # This captures the native allocator state without freezing the Python interpreter
    try:
        snapshot = torch.cuda.memory_snapshot()
        logger.info(f"Recorded memory snapshot entries: {len(snapshot)}")
        # You can inspect 'snapshot' entries if deeper debugging is needed,
        # but just seeing the totals is usually enough to confirm a leak
        # (if 'Allocated' keeps rising).
    except Exception as e:
        logger.error(f"Could not take memory snapshot: {e}")

    logger.info("---------------------------------------\n")


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

        logger.info(f"Execution time for '{self.label}': {elapsed_ms:.4f} ms")


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

    logger.info(f"Code string execution time: {elapsed_ms:.4f} ms")


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
        logger.info(f"Execution time for '{fun.__name__}': {elapsed_time_ms:.4f} ms")

    else:
        start_time = time.perf_counter()
        result = fun(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed_time_ms = (end_time - start_time) * 1000
        logger.info(f"Execution time for '{fun.__name__}': {elapsed_time_ms:.4f} ms")

    return result
