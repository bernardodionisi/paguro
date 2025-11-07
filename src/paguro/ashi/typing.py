from __future__ import annotations

from typing import Any, Literal, TypeAlias, Union, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from paguro.ashi.info.info import Info

# Selectors: depth (int), exact key (str), or path (tuple[str, ...])
StyleSelector: TypeAlias = Union[int, str, tuple[str, ...]]

# New top-level sections
KeyValues: TypeAlias = Literal["key", "value"]
DefaultKeyValues: TypeAlias = Literal["default_key", "default_value"]
KeyAffixes: TypeAlias = Literal[
    "key"
]  # affixes only support "key" section

# Payload field names
Style: TypeAlias = Literal[
    "color",
    "background",
    "bold",
    "dim",
    "italic",
    "underline",
    "blink",
    "inverted",
    "hidden",
    "strikethrough",
]
Affixes: TypeAlias = Literal[
    "prefix", "suffix", "start_level", "apply_to_deeper_levels"
]

# Sections
StylesSection: TypeAlias = dict[StyleSelector, dict[Style, Any]]
DefaultsSection: TypeAlias = dict[Style, Any]
AffixesSection: TypeAlias = dict[StyleSelector, dict[Affixes, Any]]

# User-facing mappings (partial-friendly)
StyleMapsLike: TypeAlias = dict[
    Union[KeyValues, DefaultKeyValues],
    Union[StylesSection, DefaultsSection]
]
AffixMapsLike: TypeAlias = dict[KeyAffixes, AffixesSection]

# Unified union if you want to accept either shape in one parameter
StyleMapping: TypeAlias = Union[StyleMapsLike, AffixMapsLike]

# Convenience
SimpleStyle: TypeAlias = dict[Style, Any]

# ------------------- INFO

InfoMappingTypes = dict[str, Union[dict, float, int, str, list, tuple]]
InfoTypes = Union["Info", Iterable["Info"], InfoMappingTypes]
