def get_real_dtype(input_tensor):
    """
    Returns the real dtype corresponding to the input tensor's dtype.
    If the input tensor is complex, it returns the corresponding real dtype.
    Otherwise, it returns the dtype of the input tensor.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor whose dtype is to be checked.

    Returns:
    - torch.dtype: The real dtype corresponding to the input tensor's dtype.
    """
    if input_tensor.is_complex():
        # If the input tensor is complex, return the corresponding real dtype
        if input_tensor.dtype == torch.complex128:
            return torch.float64
        elif input_tensor.dtype == torch.complex64:
            return torch.float32
    else:
        # If the input tensor is already real, return its dtype
        return input_tensor.dtype


def zeropad2fitdims(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    # Get the maximum size along each dimension
    max_sizes = [
        max(tensor.size(dim) for tensor in tensors) for dim in range(tensors[0].dim())
    ]

    padded_tensors = []

    for tensor in tensors:
        # Get the padding needed for each dimension
        padding = []
        for dim, max_size in enumerate(max_sizes):
            padding = [0, max_size - tensor.size(dim)] + padding

        # Apply padding to the tensor and append it to the list of padded tensors
        padded_tensors.append(torch.nn.functional.pad(tensor, padding))

    return padded_tensors


def check_all_elements_equal(tensor: torch.Tensor) -> bool:
    unique_elements = torch.unique(tensor)
    return len(unique_elements) == 1


def nanappend(tensor: torch.Tensor, dim: int, final_length) -> torch.Tensor:
    size = list(tensor.shape)
    size[dim] = int(final_length - size[dim])
    return torch.cat(
        [
            tensor,
            torch.full(size, float("nan"), device=tensor.device, dtype=tensor.dtype),
        ],
        dim=dim,
    )


def check_broadcastable(*shape_list: torch.Size) -> Union[tuple, bool]:
    """
    Check if a list of shapes are broadcastable and return the broadcasted shape.

    Args:
        shape_list (list[tuple]): List of shapes (tuples) representing the dimensions of tensors.

    Returns:
        tuple or bool: Returns the resulting broadcasted shape if broadcastable, else False.
    """
    # Initialize the result with an empty shape
    result_shape = []

    # Find the maximum number of dimensions in the shapes
    max_dims = max(len(shape) for shape in shape_list)

    # Iterate through each dimension from last to first (rightmost to leftmost)
    for dim in range(max_dims):
        current_dim = None  # Track the dimension being checked

        # Iterate through each shape
        for shape in shape_list:
            # Access the current dimension from the right (dim=-1 means last, dim=-2 means second last, etc.)
            shape_dim = shape[-(dim + 1)] if dim < len(shape) else 1

            # If this is the first dimension being checked, set it as the current dimension
            if current_dim is None:
                current_dim = shape_dim
            else:
                # Check if the current dimension is broadcastable
                if shape_dim != 1 and current_dim != 1 and shape_dim != current_dim:
                    return False  # Not broadcastable if neither is 1 and they are not equal

            # Update the current dimension to the larger of the two
            current_dim = max(current_dim, shape_dim)

        # Add the current dimension to the result shape (prepend since we're building the shape backwards)
        result_shape.insert(0, current_dim)

    return tuple(result_shape)


def inv_perm_indices(perm_indices: Union[list, tuple]) -> list:
    """
    Computes the inverse permutation indices such that
    tensor.permute(perm_indices).permute(inv_perm_indices(perm_indices))
    restores the original tensor.

    Parameters:
    - perm_indices (list or tuple): List of permutation indices.

    Returns:
    - inv_perm (list): Inverse permutation indices.
    """
    inv_perm = [0] * len(
        perm_indices
    )  # Initialize a list of zeros with the same length

    for i, p in enumerate(perm_indices):
        inv_perm[p] = i  # Assign the position to the inverse index

    return inv_perm


def generalized_cat(tensor_list, dim):
    """
    Concatenates a list of tensors along the specified dimension,
    broadcasting the tensors to compatible shapes.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to concatenate.
        dim (int): The dimension along which to concatenate.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    if not tensor_list:
        raise ValueError("tensor_list must contain at least one tensor.")

    # Find the maximum number of dimensions
    max_dims = max(tensor.dim() for tensor in tensor_list)

    # Convert negative index to positive
    if dim < 0:
        dim += max_dims

    # Ensure all tensors have the same number of dimensions
    expanded_tensors = []
    shape_list = []
    for tensor in tensor_list:
        shape = list(tensor.shape)
        # Add leading singleton dimensions to match max_dims
        tensor = tensor.view(*([1] * (max_dims - len(shape))), *shape)
        expanded_tensors.append(tensor)
        shape_list.append(tensor.shape)

    # Determine the broadcast shape
    broadcast_shape = check_broadcastable(
        *[shape[:dim] + shape[dim + 1 :] for shape in shape_list]
    )
    assert broadcast_shape, "Shapes of tensors are not broadcastable!"

    # Concatenate along the specified dimension
    broadcast_shape = (
        broadcast_shape[:dim] + (-1,) + broadcast_shape[dim:]
        if not isinstance(broadcast_shape, bool)
        else None
    )
    # Expand all tensors to the broadcast shape
    broadcasted_tensors = [
        tensor.expand(broadcast_shape) for tensor in expanded_tensors
    ]

    # Concatenate along the specified dimension
    return torch.cat(broadcasted_tensors, dim=dim)


def match_dims_to(tensor, target_tensor):
    """
    Increase the dimensions of `tensor` to match the number of dimensions of `target_tensor`
    by adding dimensions of size 1 at the beginning.

    Parameters:
        tensor (torch.Tensor): The tensor to be expanded.
        target_tensor (torch.Tensor): The tensor whose dimensions we want to match.

    Returns:
        torch.Tensor: The expanded tensor.
    """
    # Get the number of dimensions
    tensor_dims = tensor.dim()
    target_dims = target_tensor.dim()

    # Calculate the number of dimensions to add
    dims_to_add = target_dims - tensor_dims

    # Add dimensions of size 1 at the beginning
    if dims_to_add > 0:
        tensor = tensor.view((1,) * dims_to_add + tensor.shape)

    return tensor
