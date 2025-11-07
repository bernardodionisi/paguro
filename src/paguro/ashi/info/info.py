from __future__ import annotations

import warnings
from collections.abc import Iterable, MutableMapping
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from paguro.ashi.info.utils import (
    is_json_serializable,
    make_nested_dict_json_serializable,
    rename_dict_keys,
)
from paguro.ashi.repr.string.box.box import Box
from paguro.utils.dependencies import copy

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
SchemaMode = Literal["off", "infer", "enforced"]


class Info:
    """
    A named, dictionary-like container with an optional schema policy.

    Schema policy (“columns”) is controlled by:
      - _schema_mode: "off" | "infer" | "enforced"
      - _schema_keys: allowed keys when enforced
      - _is_schema_level: True if this Info is *designated* as schema-level.
        (This designation is set by the collection at creation time and persists.)
    """

    __slots__ = (
        "_info",
        "_name",
        "_is_serializable",
        "_allow_non_serializable",
        "_schema_keys",
        "_schema_mode",
        "_is_schema_level",
        "__weakref__",
    )

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._info: dict[str, object] = {}

        # Serialization policy
        self._is_serializable: bool = True
        self._allow_non_serializable: bool = False

        # Schema policy
        self._schema_mode: SchemaMode = "infer"
        self._schema_keys: set[str] = set()

        # Set by InfoList when an Info is first created/added
        # (and never flipped automatically)
        self._is_schema_level: bool = False

    def keys(self):
        return self._info.keys()

    def items(self):
        return self._info.items()

    def values(self):
        return self._info.values()

    def get(self, key, default: object | None = None) -> object | None:
        return self._info.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._info

    # ----------------- representation ---------------------------------

    def __repr__(self) -> str:  # pragma: no cover (human-facing)
        return (
            f"<{self.__class__.__qualname__}("
            f"name='{self._name}', "
            f"is_schema_level={self._is_schema_level}, "
            f"schema_mode={self._schema_mode}, "
            f"schema_keys={len(self._schema_keys)}) "
            f"at {hex(id(self))}>"
        )

    def __str__(self) -> str:
        return str(
            self._get_box(set_content=False)
            .to_string(content=self._info, width_chars=80)
        )

    def _get_box(self, set_content: bool) -> Box:
        if set_content:
            return Box().set_content(content=self._info).set_top_name(
                top_name=self._name)
        return Box().set_top_name(top_name=self._name)

    # ----------------- mapping protocol -----------------

    def __getitem__(self, key: str) -> object:
        return self._info[key]

    def __setitem__(self, key: str, value: object) -> None:
        value, became_serializable = self._coerce_if_needed(key, value)

        if self._schema_mode == "enforced" and key not in self._schema_keys:
            warnings.warn(
                f"\nSkipping {key}.\n"
                f"Schema is enforced for Info {self._name!r} "
                f"'{key}' is not among allowed keys: {self._schema_keys}",
                category=UserWarning,
                stacklevel=2,
            )
            return

        self._info[key] = value
        self._is_serializable = self._is_serializable and became_serializable

    def __delitem__(self, key: str) -> None:
        del self._info[key]

    def __iter__(self):
        return iter(self._info)

    def __len__(self) -> int:
        return len(self._info)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Info):
            return NotImplemented
        return (
                self._name == other._name
                and self._info == other._info
                and self._schema_mode == other._schema_mode
                and self._schema_keys == other._schema_keys
                and self._is_schema_level == other._is_schema_level
        )

    # ----------------- properties -----------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def info(self) -> MappingProxyType[str, object]:
        """Read-only view to protect invariants."""
        return MappingProxyType(self._info)

    # ----------------- configuration -----------------

    def set_allow_non_serializable(self, *, active: bool = False) -> Self:
        self._allow_non_serializable = active
        return self

    def set_schema(
            self,
            *,
            mode: SchemaMode,
            keys: Iterable[str] | str | None = None,
            normalize_existing: bool = True,
    ) -> Self:
        """
        Configure the schema policy.

        Semantics for empty set:
          - mode == "enforced" + keys == empty ⇒ keep only allowed keys (drops all).
          - mode == "infer"     + keys == empty ⇒ just set mode/keys; no drop.

        Other explainers

        mode="enforced" + normalize_existing=True
        → keep only key ∈ schema_keys; everything else is removed right away.

        mode="enforced" + normalize_existing=False
        → record the schema but don’t delete anything now;
        future update/__setitem__ will block out-of-schema keys,
        but existing extras stay until you explicitly rename/sync again.

        mode="infer" or mode="off"
        → normalize_existing has no effect (nothing is dropped)
        """

        if keys is None:
            keys_set: set[str] = set()
        elif isinstance(keys, str):
            keys_set = {keys}
        else:
            keys_set = set(keys)

        self._schema_mode = mode
        # Record explicitly, even if empty
        self._schema_keys = keys_set

        if self._schema_mode == "enforced" and normalize_existing:
            self._info = {
                k: deepcopy(v)
                for k, v in self._info.items()
                if k in keys_set
            }

        return self

    def sync_schema(self, keys: Iterable[str] | str | None) -> None:
        """
        Synchronize with an external schema view.

        - "off": no-op.
        - "enforced": replace allowed set and normalize.
        - "infer": if *all current keys* ⊆ provided set, flip to enforced.
        """
        if keys is None or self._schema_mode == "off":
            return

        keys_set = {keys} if isinstance(keys, str) else set(keys)

        info_keys = set(self._info.keys())
        redundant = info_keys - keys_set

        if self._schema_mode == "enforced":
            self._schema_keys = keys_set
            if redundant or (not keys_set and info_keys):
                self._info = {k: deepcopy(v) for k, v in self._info.items() if
                              k in keys_set}

        elif self._schema_mode == "infer":
            if info_keys.issubset(keys_set):
                self._schema_mode = "enforced"
                self._schema_keys = keys_set

    # ----------------- core ops -----------------

    def update(self, **mapping: object) -> Self:
        """
        In-place update.

        - Enforced schema: out-of-schema keys are warned and skipped.
        - Values are deep-copied where relevant to avoid external mutation.
        """
        for key, raw_value in mapping.items():
            value, became_serializable = self._coerce_if_needed(key, raw_value)

            if self._schema_mode == "enforced" and key not in self._schema_keys:
                warnings.warn(
                    f"\nSkipping {key}.\n"
                    f"Schema is enforced and '{key}' is not among allowed keys.",
                    category=UserWarning,
                    stacklevel=2,
                )
                continue

            self._info[key] = deepcopy(value)
            self._is_serializable = self._is_serializable and became_serializable

        return self

    def rename_keys(self, mapping: dict[str, str]) -> None:
        self._info = rename_dict_keys(original_dict=self._info, mapping=mapping)

    def join(self, *, other: Info, suffix: str | None = None) -> Self:
        """
        Join two `Info` objects by merging keys.
        If a conflicting key has a different value and `suffix` is provided,
        the right value is stored under a unique suffixed key.
        """
        if not isinstance(other, Info):
            raise ValueError("The 'other' argument must be an Info instance.")

        new = copy.deepcopy(self)

        if self._name != other._name:
            warnings.warn(
                f"\nYou are joining two Info with names: "
                f"'{self._name}' and '{other._name}'.\n"
                f"The resulting Info will retain the name: '{self._name}'",
                category=UserWarning,
                stacklevel=2,
            )

        for key, value in other._info.items():
            if key in new._info:
                try:
                    different_value = new._info[key] != value
                except TypeError:
                    different_value = True

                if different_value and suffix is not None:
                    new_key = new._generate_unique_key(key, suffix)
                    new._info[new_key] = value
            else:
                new._info[key] = value

        return new

    # ----------------- helpers -----------------

    def _generate_unique_key(self, key: str, suffix: str) -> str:
        new_key = f"{key}{suffix}"
        while new_key in self._info:
            new_key += suffix
        return new_key

    def to_dict(self, *, include_name: bool = True) -> dict:
        """Materialize to a plain dict."""
        if include_name:
            return {self._name: self._info}
        return dict(self._info)

    # ----------------- internals -----------------

    def _coerce_if_needed(self, key: str, value: object) -> tuple[object, bool]:
        """
        Returns (value_to_store, is_serializable_now).

        - If serializable: pass through.
        - If dict and allow_non_serializable=False: coerce (warn) to serializable dict.
        - If allow_non_serializable=True: accept as-is and mark non-serializable.
        - Else: raise ValueError.
        """
        if is_json_serializable(value):
            return value, True

        if isinstance(value, dict) and not self._allow_non_serializable:
            warnings.warn(
                f"\nValue for key '{key}' is not JSON serializable."
                f"\nMaking it JSON serializable by iteratively taking the objects' string representation.",
                category=UserWarning,
                stacklevel=2,
            )
            coerced = make_nested_dict_json_serializable(obj=value)
            return coerced, True

        if self._allow_non_serializable:
            return value, False

        raise ValueError(f"Value for key '{key}' is not JSON serializable")
