import random
from typing import TypeVar, Any, overload
from omegaconf import OmegaConf, ListConfig, DictConfig

intfloatstr = TypeVar('intfloatstr', int, float, str)

@overload
def sample_parameter(param: intfloatstr, num: None = None) -> intfloatstr: ...

@overload
def sample_parameter(param: intfloatstr, num: int) -> list[intfloatstr]: ...

@overload
def sample_parameter(param: list[Any], num: None = None) -> Any: ...

@overload
def sample_parameter(param: list[Any], num: int) -> list[Any]: ...

@overload
def sample_parameter(
    param: tuple[intfloatstr, ...], num: None = None
) -> intfloatstr: ...

@overload
def sample_parameter(param: tuple[intfloatstr, ...], num: int) -> list[intfloatstr]: ...

@overload
def sample_parameter(param: dict, num: None = None) -> dict: ...

@overload
def sample_parameter(param: dict, num: int) -> dict: ...


def sample_parameter(
    param: int | float | str | list[Any] | tuple[int | float | str, ...] | dict,
    num: int | None = None,
) -> Any:
    """
    Samples a value or a list of values from a parameter configuration.

    - If param is a single value (int, float, str), it returns that value.
    - If param is a list or a tuple of choices, it returns a random choice.
    - If param is a tuple representing a numeric range, it samples from that range.
    - If param is a dictionary, it samples key-value pairs.

    If the 'num' argument is provided, it returns a list/dict of 'num' unique samples.

    Args:
        param: The parameter configuration (single value, list, tuple, or dict).
        num (int, optional): The number of samples to return. If None, returns a
                                single value. Defaults to None.

    Returns:
        A single sampled value or a list/dict of sampled values.
    """
    if isinstance(param, (ListConfig, DictConfig)):
        param = OmegaConf.to_container(param, resolve=True)

    def _get_single_sample() -> Any:
        if isinstance(param, (int, float, str)):
            return param
        elif isinstance(param, list):
            return random.choice(param)
        elif isinstance(param, tuple):
            if (
                len(param) == 2
                and isinstance(param[0], (int, float))
                and isinstance(param[1], (int, float))
            ):
                low, high = param
                if isinstance(low, int) and isinstance(high, int):
                    return random.randint(low, high)
                else:
                    return random.uniform(float(low), float(high))
            else:  # Assumes a tuple of choices
                return random.choice(param)
        # Handle any dictionary by sampling a single key-value pair
        elif isinstance(param, dict):
            key = random.choice(list(param.keys()))
            return {key: param[key]}
        else:
            raise TypeError(f"Unsupported type for parameter: {param}")

    if num is None:
        return _get_single_sample()
    else:
        # Handle multi-sampling
        if isinstance(param, (list, tuple)):
            # If it's a numeric range, draw 'num' independent samples
            if (
                isinstance(param, tuple)
                and len(param) == 2
                and isinstance(param[0], (int, float))
                and isinstance(param[1], (int, float))
            ):
                return [_get_single_sample() for _ in range(num)]
            # Otherwise, perform unique sampling (without replacement)
            else:
                if num > len(param):
                    raise ValueError(
                        f"Cannot sample {num} unique items from a list of size {len(param)}."
                    )
                return random.sample(param, k=num)
        # Handle any dictionary by sampling 'num' key-value pairs
        elif isinstance(param, dict):
            if num > len(param):
                raise ValueError(
                    f"Cannot sample {num} unique items from a dictionary of size {len(param)}."
                )
            keys = random.sample(list(param.keys()), k=num)
            return {key: param[key] for key in keys}
        # If param is a single value, return a list of that value repeated num times
        elif isinstance(param, (int, float, str)):
            if num > 1:
                raise ValueError(
                    f"Cannot sample {num} unique items from a single value."
                )
            return [param] * num
        else:
            raise TypeError(f"Unsupported type for multi-sampling: {param}")