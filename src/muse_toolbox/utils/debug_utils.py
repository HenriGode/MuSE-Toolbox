from muse_toolbox.utils.format_utils import format_memory
import torch
import numpy as np
from typing import Callable
import logging
import sys

logger = logging.getLogger(__name__)


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
        logger.info(f"{indent}{_get_obj_str(obj)} | Memory: {mem_str}")
        return

    # --- Tensors and Numpy Arrays ---
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        extra = ""
        if condition is not None and isinstance(obj, torch.Tensor):
            extra = f" | condition: {condition(obj)}"
        logger.info(f"{indent}{_get_obj_str(obj)} | Memory: {mem_str}{extra}")
        return

    # --- Dictionaries ---
    if isinstance(obj, dict):
        logger.info(f"{indent}dict with {len(obj)} keys | Memory: {mem_str}")
        for key, value in obj.items():
            logger.info(f"{indent}{' ' * indent_size}key: '{key}'")
            _print_recursive_with_memory(
                value, level + 2, indent_size, size_cache, condition
            )
        return

    # --- Lists and Tuples ---
    if isinstance(obj, (list, tuple)):
        container_type = type(obj).__name__
        if not obj:
            logger.info(f"{indent}{container_type} (empty) | Memory: {mem_str}")
            return

        # Check if all elements are of the same type and shape
        first_item_str = _get_obj_str(obj[0])
        is_uniform = all(_get_obj_str(item) == first_item_str for item in obj)

        if is_uniform:
            logger.info(
                f"{indent}{container_type} of {len(obj)} x {first_item_str} | Memory: {mem_str}"
            )
            i = 0
            item = obj[0]
            _print_recursive_with_memory(
                item, level, indent_size, size_cache, condition
            )
        else:
            logger.info(
                f"{indent}{container_type} of {len(obj)} items (non-uniform) | Memory: {mem_str}:"
            )
            for i, item in enumerate(obj):
                logger.info(f"{indent}{' ' * indent_size}[{i}]:")
                _print_recursive_with_memory(
                    item, level + 2, indent_size, size_cache, condition
                )
        return

    # --- Custom Objects ---
    if hasattr(obj, "__dict__"):
        logger.info(f"{indent}{type(obj).__name__} | Memory: {mem_str}:")
        # Use vars() to get the __dict__ of attributes
        attributes = vars(obj)
        for attr_name, attr_value in attributes.items():
            # Skip private/magic attributes and methods/callables
            if attr_name.startswith("_") or callable(attr_value):
                continue

            logger.info(f"{indent}{' ' * indent_size}attr: '{attr_name}'")
            _print_recursive_with_memory(
                attr_value, level + 2, indent_size, size_cache, condition
            )
        return

    # --- Fallback for any other type ---
    logger.info(f"{indent}{_get_obj_str(obj)} | Memory: {mem_str}")


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
