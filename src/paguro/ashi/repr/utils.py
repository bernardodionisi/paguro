from __future__ import annotations


def find_min_dict_depth_recursive(
    d: dict,
    threshold: int | None = None,
    current_depth: int = 0,
) -> int:
    """
    Find the minimum depth of nestedness in a dictionary.

    Nestedness that is less than the given threshold.

    Parameters
    ----------
    d : dict
        The dictionary to check for nestedness.
    current_depth : int, optional
        The current depth level, starts at 0. Default is 0.
    threshold : int, optional
        The depth threshold. The search stops if a depth less than or
        equal to this value is found. Default is None, which means no threshold.

    Returns
    -------
    The minimum depth of nestedness found in the dictionary that is less
    than or equal to the threshold. If no threshold is specified, returns
    the overall minimum depth.
    """
    if not isinstance(d, dict) or not d:
        return current_depth

    if threshold is not None and current_depth >= threshold:
        return current_depth

    min_depths = [
        find_min_dict_depth_recursive(
            v, threshold=threshold, current_depth=current_depth + 1
        )
        for k, v in d.items()
    ]

    return min(min_depths) if min_depths else current_depth


def find_min_depth_iterative(d: dict, threshold: int | None = None):
    """
    Find the minimum depth of nestedness in a dictionary.

    Stops the search if it reaches the specified threshold.

    Parameters
    ----------
    d : dict
        The dictionary to check for nestedness.
    threshold : int, optional
        The depth threshold. The function stops the search when it reaches this depth.

    Returns
    -------
    The minimum depth of nestedness found in the dictionary that is
    less than or equal to the threshold.
    If the threshold is reached, returns the threshold value. If no
    threshold is specified, returns the overall minimum depth.
    """
    if threshold is not None:
        if threshold < 0:
            return 0

    if not isinstance(d, dict) or not d:
        return 0

    stack = [(d, 0)]  # Stack elements are (dict, current_depth)
    min_depth = None  # Initialize as None to handle non-updating cases

    while stack:
        current_dict, depth = stack.pop()

        if not isinstance(current_dict, dict) or not current_dict:
            min_depth = (
                depth if min_depth is None else min(min_depth, depth)
            )
            continue

        if threshold is not None and depth >= threshold:
            return threshold

        for value in current_dict.values():
            stack.append((value, depth + 1))

    return min_depth if min_depth is not None else 0
