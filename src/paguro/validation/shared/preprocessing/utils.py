from __future__ import annotations

from typing import Any, Iterable, Mapping


def _parse_validators_as_iterable(
        inputs: tuple[Any, ...] | tuple[Iterable[Any]],
) -> Iterable[Any]:  # type int validators not Any
    if not inputs:
        return []

    # Treat elements of a single iterable as separate inputs
    if len(inputs) == 1 and _is_iterable(inputs[0]):
        out: Iterable = inputs[0]
        if isinstance(out, Mapping):
            from paguro.validation.shared.preprocessing.preprocess_validators import \
                _preprocess_validator_mapping
            # mapping must not be inside another iterable
            return _preprocess_validator_mapping(mapping=out)
        return out

    return inputs


def _is_iterable(input: Any | Iterable[Any]) -> bool:
    return isinstance(input, Iterable) and not isinstance(
        input, (str, bytes)
    )
