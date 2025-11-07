from __future__ import annotations

import json
from typing import TYPE_CHECKING

from paguro.shared.serialize import CustomJSONDecoder, CustomJSONEncoder

if TYPE_CHECKING:
    from json import JSONDecoder, JSONEncoder
    from typing import Any


def serialize_dict_to_bytes(
        data: dict[str, Any],
        *,
        json_encoder: type[JSONEncoder] | None = None,
) -> dict[bytes, bytes]:
    """
    Serialize both keys and values of a dictionary into UTF-8 encoded bytes.

    Examples
    --------
    >>> serialize_dict_to_bytes({"key1": "value1", "key2": 123})
    {b'key1': b'"value1"', b'key2': b'123'}
    """
    if json_encoder is None:
        json_encoder = CustomJSONEncoder
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary.")

    out: dict[bytes, bytes] = {}
    for key, value in data.items():
        try:
            out[key.encode("utf-8")] = json.dumps(value, cls=json_encoder).encode(
                "utf-8")
        except (TypeError, ValueError) as e:
            raise TypeError(f"Error serializing dictionary: {e}") from e
    return out


def deserialize_dict_from_bytes(
        data: dict[bytes, bytes],
        *,
        json_decoder: type[JSONDecoder] | None = None,
) -> dict[str, Any]:
    """
    Deserialize a dictionary whose keys and values are UTF-8 encoded bytes
    back into a Python dictionary.

    Examples
    --------
    >>> deserialize_dict_from_bytes({b"k1": b'"v1"', b"k2": b'123'})
    {'k1': 'v1', 'k2': 123}
    """
    if json_decoder is None:
        json_decoder = CustomJSONDecoder
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary.")

    out: dict[str, Any] = {}
    for key, val in data.items():
        try:
            out[key.decode("utf-8")] = json.loads(
                val.decode("utf-8"), cls=json_decoder
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as e:
            raise TypeError(f"Error deserializing dictionary: {e}") from e
    return out


def serialize_dict_to_json_str(
        data: dict[str, Any],
        *,
        json_encoder: type[JSONEncoder] | None = None,
) -> str:
    """
    Serialize a Python dictionary into a JSON string.

    Examples
    --------
    >>> serialize_dict_to_json_str({"key1": "value1", "key2": 123})
    '{"key1": "value1", "key2": 123}'
    """
    if json_encoder is None:
        json_encoder = CustomJSONEncoder
    return json.dumps(data, cls=json_encoder, )


def deserialize_dict_from_json_str(
        data: str | bytes | bytearray,
        *,
        json_decoder: type[JSONDecoder] | None = None,
) -> dict[str, Any]:
    """
    Deserialize a JSON string (or bytes) into a Python dictionary.

    Examples
    --------
    >>> deserialize_dict_from_json_str('{"k1": "v1", "k2": 123}')
    {'k1': 'v1', 'k2': 123}
    """
    if json_decoder is None:
        json_decoder = CustomJSONDecoder
    return json.loads(data, cls=json_decoder)


def serialize_dict_values_as_json(
        data: dict[str, Any],
        *,
        json_encoder: type[JSONEncoder] | None = None,
) -> dict[str, str]:
    """
    Serialize each value in a dictionary into its own compact JSON string.
    Keys remain strings.

    Examples
    --------
    >>> serialize_dict_values_as_json({"k1": [1, 2], "k2": {"a": 1}})
    {'k1': '[1,2]', 'k2': '{"a":1}'}
    """
    if json_encoder is None:
        json_encoder = CustomJSONEncoder
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary.")

    out: dict[str, str] = {}
    for key, val in data.items():
        try:
            out[key] = json.dumps(
                val,
                cls=json_encoder,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Unable to serialize value for key '{key}': {e}") from e
    return out


def deserialize_dict_values_from_json(
        data: dict[str, str],
        *,
        json_decoder: type[JSONDecoder] | None = None,
) -> dict[str, Any]:
    """
    Deserialize each JSON-encoded string value in a dict[str, str]
    back into its corresponding Python object.

    Examples
    --------
    >>> deserialize_dict_values_from_json({'a': '"foo"', 'b': '123'})
    {'a': 'foo', 'b': 123}
    """
    if json_decoder is None:
        json_decoder = CustomJSONDecoder
    if not isinstance(data, dict):
        raise TypeError("Input must be a dict of JSON strings.")

    out: dict[str, Any] = {}
    for key, json_val in data.items():
        if not isinstance(json_val, str):
            raise TypeError(f"Value for key '{key}' must be a JSON string.")
        try:
            out[key] = json.loads(json_val, cls=json_decoder)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string for key '{key}': {e.msg}") from e
    return out


# # --------------------------------------------------------------------

def encode_dict_to_json_bytes(
        data: dict[str, Any],
        *,
        json_encoder: type[JSONEncoder] | None = None,
) -> bytes:
    if json_encoder is None:
        json_encoder = CustomJSONEncoder
    """Encode an entire dictionary into a JSON UTF-8 byte string."""
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary.")
    try:
        return json.dumps(data, cls=json_encoder).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Error encoding dictionary to JSON bytes: {e}") from e


def decode_json_bytes_to_dict(
        json_bytes: bytes,
        *,
        json_decoder: type[JSONDecoder] | None = None,
) -> dict[str, Any]:
    if json_decoder is None:
        json_decoder = CustomJSONDecoder
    """Decode UTF-8 JSON bytes into a Python dictionary."""
    if not isinstance(json_bytes, (bytes, bytearray)):
        raise TypeError("Input must be bytes or bytearray.")
    try:
        return json.loads(json_bytes.decode("utf-8"), cls=json_decoder)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise TypeError(f"Error decoding JSON bytes: {e}") from e
