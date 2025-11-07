from __future__ import annotations

from paguro.utils.dependencies import copy, json


# def deserialize(source: str | dict, cls):
#     if isinstance(source, dict):
#         source = copy.deepcopy(source)
#     else:
#         source = json.loads(source, cls=cls)
#     return source


def set_json_type_key(key: str) -> None:
    """
    Set the metadata key for both encoder and decoder.

    Args:
        key: The key to use (e.g., "__type__", "__mylib_type__", "__t__")

    Example:
        set_json_type_key("__mylib_type__")  # Safe, collision-resistant
        set_json_type_key("__t__")           # Compact for storage-critical apps
    """
    from paguro.shared.serialize.decoder import CustomJSONDecoder
    from paguro.shared.serialize.encoder import CustomJSONEncoder

    CustomJSONEncoder.set_type_key(key)
    CustomJSONDecoder.set_type_key(key)
