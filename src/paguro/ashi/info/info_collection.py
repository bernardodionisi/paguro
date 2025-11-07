from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Literal

from paguro.ashi.info.info import Info, SchemaMode
from paguro.ashi.repr.string.utils import join_ststr
from paguro.utils.dependencies import copy, json

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
    from collections.abc import Mapping
    from paguro.ashi.typing import InfoTypes

__all__ = ["InfoCollection", "MissingInfoNameException", "callable_to_column_mapping"]


class InfoCollection:
    """
    Container for multiple `Info` objects with an optional *collection* schema view.

    Mutation model
    --------------
    In-place (setters):
      - set_schema(...)
      - sync_schema(...)
    Immutable (return a new collection):
      - with_schema(...), with_synced_schema(...)
      - append(...), update(...)
      - rename_schema_keys(...)
      - drop(...), drop_where(...)

    Notes
    -----
    - Schema-level designation (`Info._is_schema_level`) is decided **once** at creation
      time by the collection (based on the collection's current schema view), and then
      preserved. `sync_schema` only aligns data/layout; it does not flip designation.
    """

    __slots__ = (
        "_info",
        "_schema_keys",
        "_schema_mode",
        "__weakref__",
    )

    def __init__(self, info: InfoTypes | None = None) -> None:
        self._info: list[Info] = _pre_process_info(info)
        self._schema_mode: SchemaMode = "infer"
        self._schema_keys: set[str] = set()

    # ----------------- representation -----------------

    def __repr__(self) -> str:  # pragma: no cover
        off = sum(1 for i in self._info if i._schema_mode == "off")
        inf = sum(1 for i in self._info if i._schema_mode == "infer")
        enf = sum(1 for i in self._info if i._schema_mode == "enforced")
        return (
            f"{self.__class__.__qualname__}("
            f"infos={len(self._info)}, "
            f"collection_schema_mode={self._schema_mode}, "
            f"collection_schema_keys={len(self._schema_keys)}, "
            f"members[off={off}, infer={inf}, enforced={enf}]"
            f")"
        )

    def __str__(self) -> str:
        return str(join_ststr([str(i) for i in self._info], separator="\n"))

    # ----------------- sequence-ish protocol -----------------

    def __bool__(self) -> bool:
        return bool(self._info)

    def __len__(self) -> int:
        return len(self._info)

    def __iter__(self):
        return iter(self._info)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __getitem__(self, key: int | str) -> Info:
        if isinstance(key, str):
            try:
                idx = self.names.index(key)
            except ValueError as e:
                raise KeyError(f"No Info named '{key}'") from e
            return self._info[idx]
        return self._info[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfoCollection):
            return NotImplemented  # type: ignore[return-value]
        return (
                self._info == other._info
                and self._schema_mode == other._schema_mode
                and self._schema_keys == other._schema_keys
        )

    # ----------------- properties -----------------

    @property
    def names(self) -> list[str]:
        return [i._name for i in self._info]

    # ----------------- schema management -----------------
    # In-place setters

    def set_schema(
            self,
            *,
            mode: SchemaMode,
            keys: Iterable[str] | str | None = None,
            normalize_existing: bool = True,
    ) -> Self:
        """
        In-place: set the *collection* schema view and propagate it to members.

        Empty set behavior:
          - mode == "enforced" + empty keys ⇒ members normalize to empty sets.
          - mode == "infer"    + empty keys ⇒ just update modes/keys; no drops.
        """
        if keys is None:
            keys_set: set[str] = set()
        elif isinstance(keys, str):
            keys_set = {keys}
        else:
            keys_set = set(keys)

        self._schema_mode = mode
        self._schema_keys = keys_set

        for info in self._info:
            info.set_schema(mode=mode, keys=keys_set,
                            normalize_existing=normalize_existing)

        return self

    def sync_schema(
            self,
            keys: Iterable[str] | str | None,
            *,
            drop_empty: Literal["never", "schema", "all"] = "never",
    ) -> None:
        """
        In-place: align *schema-level* members with the given record schema.

        - Updates the collection view (`_schema_keys`).
        - Enforces (mode='enforced', normalize_existing=True) **only** on members
          where `info._is_schema_level` is True.
        - Non-schema members are left untouched unless `drop_empty="all"`.
        """
        if keys is None:
            return

        keys_set = {keys} if isinstance(keys, str) else set(keys)
        self._schema_keys = keys_set

        for info in self._info:
            if info._is_schema_level:
                info.set_schema(mode="enforced", keys=keys_set, normalize_existing=True)

        if drop_empty != "never":
            if drop_empty == "schema":
                self._info = [i for i in self._info if
                              not (i._is_schema_level and len(i.info) == 0)]
            else:  # "all"
                self._info = [i for i in self._info if len(i.info) > 0]

    # Immutable counterparts

    def with_schema(
            self,
            *,
            mode: SchemaMode,
            keys: Iterable[str] | str | None = None,
            normalize_existing: bool = True,
    ) -> Self:
        """Return a copy with `set_schema(...)` applied."""
        new = copy.deepcopy(self)
        return new.set_schema(mode=mode, keys=keys,
                              normalize_existing=normalize_existing)

    def with_synced_schema(
            self,
            keys: Iterable[str] | str | None,
            *,
            drop_empty: Literal["never", "schema", "all"] = "never",
    ) -> Self:
        """Return a copy with `sync_schema(...)` applied."""
        if keys is None:
            return self
        new = copy.deepcopy(self)
        new.sync_schema(keys, drop_empty=drop_empty)
        return new

    # ----------------- immutable mutators -----------------

    def rename_schema_keys(
            self,
            mapping: Mapping[str, str],
            *,
            apply_to_members: Literal["enforced", "all", "none"] = "enforced",
    ) -> Self:
        """
        Return a new collection with schema keys (and optionally member data) renamed.

        Data renaming rules:
          - "none": never rename data keys.
          - "all": rename all members' data keys.
          - "enforced": rename data keys only for members with enforced schema and only
                        when their *effective* schema is member-specific (i.e., not identical
                        to the collection schema before rename).
        """
        if not mapping:
            return self

        # Validate against current collection schema
        old_coll_schema = set(self._schema_keys)
        for old, n in mapping.items():
            if n in old_coll_schema and n not in mapping:
                raise ValueError(
                    f"Schema rename '{old}' -> '{n}' collides with an existing key "
                    f"that is not being renamed"
                )

        new = copy.deepcopy(self)

        # Update collection schema
        new._schema_keys = {
            mapping.get(k, k)
            for k in self._schema_keys
        }

        # Update members
        for info in new._info:
            member_mode = info._schema_mode
            member_schema = set(info._schema_keys)
            data_keys = set(info.info.keys())

            eff_member_schema = (member_schema & data_keys) if member_schema else set()

            # Rename the *effective* member schema (preserving mode)
            if eff_member_schema:
                renamed_eff = {mapping.get(k, k) for k in eff_member_schema}
                new_member_schema = (member_schema - eff_member_schema) | renamed_eff
                info.set_schema(
                    mode=member_mode,
                    keys=new_member_schema,
                    normalize_existing=False,  # do not drop values here
                )

            # Decide whether to also rename DATA keys
            if apply_to_members == "all":
                info.rename_keys(mapping=dict(mapping))
            elif apply_to_members == "enforced":
                if (
                        member_mode == "enforced" and
                        eff_member_schema and
                        eff_member_schema != old_coll_schema
                ):
                    info.rename_keys(mapping=dict(mapping))
            # "none": leave data as-is

        return new

    def update(self, info: str | Info | None = None, **mapping: Any) -> Self:
        """
        Immutable update.

        - If `info` is provided, updates/creates that Info (by name or instance).
        - If `info` is omitted, try to route to a *single* schema-level target:
            * if mapping keys ⊆ collection schema and exactly one schema-level member exists; or
            * if exactly one schema-level member exists and its schema covers these keys;
            * else raise `MissingInfoNameException`.
        """
        if info is None and not mapping:
            return self

        new = copy.deepcopy(self)
        if info is not None:
            return new._update_named(info=info, **mapping)
        return new._update_from_mapping(**mapping)

    def append(self, info: str | Info, **mapping: Any) -> Self:
        """
        Immutable append.

        If `mapping` is provided, it is applied to the new Info before adding, then
        schema-level designation is decided once based on the *current* collection schema.
        """
        new = copy.deepcopy(self)

        obj = Info(name=info) if isinstance(info, str) else info
        if mapping:
            obj = obj.update(**mapping)

        # Decide schema-level once, based on current collection schema
        if not obj._is_schema_level and mapping and new._schema_keys:
            keys = set(mapping.keys())
            if keys and keys.issubset(new._schema_keys):
                obj._is_schema_level = True
                obj.set_schema(mode="enforced", keys=set(new._schema_keys),
                               normalize_existing=False)

        if obj._name in new:
            raise ValueError(f"An Info named '{obj._name}' already exists")

        new._info.append(copy.deepcopy(obj))
        return new

    # ----------------- internals for update routing -----------------

    def _update_named(self, info: str | Info, **mapping: Any) -> Self:
        """Create-or-update a specific target by name/instance (immutable on `self`)."""
        name = info if isinstance(info, str) else info._name

        if name in self:
            idx = self.names.index(name)
            self._info[idx].update(**mapping)
        else:
            obj = Info(name=name) if isinstance(info, str) else info
            obj = obj.update(**mapping)

            # Decide schema-level once at creation time
            if not obj._is_schema_level and mapping and self._schema_keys:
                keys = set(mapping.keys())
                if keys and keys.issubset(self._schema_keys):
                    obj._is_schema_level = True
                    obj.set_schema(mode="enforced", keys=set(self._schema_keys),
                                   normalize_existing=False)

            self._info.append(copy.deepcopy(obj))

        return self

    def _update_from_mapping(self, **mapping: Any) -> Self:
        """
        Bare mapping update (no target name). Heuristic:
        - If mapping keys ⊆ collection schema and exactly one schema-level member: update it.
        - Else if exactly one schema-level (or enforced) member exists and its schema covers these keys: update it.
        - Else raise `MissingInfoNameException`.
        """
        keys = set(mapping.keys())

        # Helper: treat as schema-level if explicitly marked OR currently enforced
        def _is_schema_like(i: Info) -> bool:
            return i._is_schema_level or i._schema_mode == "enforced"

        # Case 1: keys ⊆ collection schema → need exactly one schema-like member
        is_schema_mapping = not (keys - self._schema_keys)
        if is_schema_mapping:
            names = [i._name for i in self._info if _is_schema_like(i)]
            if len(names) == 1:
                self._get_by_name(names[0]).update(**mapping)
                return self
            raise MissingInfoNameException(
                is_columns_info=True, names=names if names else None, mapping=mapping
            )

        # Case 2: collection schema not decisive → fallback to single schema-like member
        schema_like_members = [i for i in self._info if _is_schema_like(i)]
        if len(schema_like_members) == 1:
            target = schema_like_members[0]
            if keys.issubset(target._schema_keys):
                target.update(**mapping)
                return self

        # Else: we cannot decide
        raise MissingInfoNameException(is_columns_info=False, names=None,
                                       mapping=mapping)

    def _get_by_name(self, name: str) -> Info:
        try:
            idx = self.names.index(name)
        except ValueError as e:
            raise KeyError(f"No Info named '{name}'") from e
        return self._info[idx]

    # ----------------- immutable removals -----------------

    def drop(
            self,
            names: str | Iterable[str],
            *,
            missing: Literal["error", "ignore"] = "error",
    ) -> Self:
        """Return a copy without the specified Info(s) by name."""
        to_remove = {names} if isinstance(names, str) else set(names)
        existing = set(self.names)
        missing_names = to_remove - existing
        if missing == "error" and missing_names:
            raise KeyError(f"No Info named: {sorted(missing_names)}")

        new = copy.deepcopy(self)
        new._info = [i for i in new._info if i._name not in to_remove]
        return new

    def drop_where(self, predicate: Callable[[Info], bool]) -> Self:
        """Return a copy with all Info objects for which `predicate(info)` is True removed."""
        new = copy.deepcopy(self)
        new._info = [i for i in new._info if not predicate(i)]
        return new

    # ----------------- (de)serialization (no attributes) -----------------

    def to_dict(self, *, keep_empty: bool = False) -> dict[str, dict[str, Any]]:
        """Plain dict materialization: {name: { ... }} (no attributes)."""
        out: dict[str, dict[str, Any]] = {}
        for i in self._info:
            body = i.to_dict(include_name=False)
            if body or keep_empty:
                out[i._name] = dict(body)
        return out

    def serialize(self) -> str:
        """JSON serialization of `to_dict` (no attributes)."""
        return json.dumps(self.to_dict(keep_empty=False))

    @classmethod
    def deserialize(cls, source: str) -> InfoCollection:
        """Create from JSON produced by `serialize` (non-attribute variant)."""
        content = json.loads(source)
        items: list[Info] = []
        for name, mapping in content.items():
            items.append(Info(name=name).update(**mapping))
        return cls(info=items)

    # --- snapshot helpers (place inside InfoCollection) ---

    def _to_info_collection_snapshot(self) -> dict[str, dict]:
        """
        Materialize a snapshot that includes each Info's data and attributes.
        Does NOT include collection-level attrs.
        Shape:
          {
            "<info-name>": {
              "info": {...},           # plain dict data
              "attrs": {
                "_is_schema_level": bool,
                "_schema_mode": "off" | "infer" | "enforced",
                "_schema_keys": [ ... ]  # list for JSON
              }
            },
            ...
          }
        """
        out: dict[str, dict] = {}
        for i in self._info:
            out[i._name] = {
                "info": i.to_dict(include_name=False),
                "attrs": {
                    "_is_schema_level": i._is_schema_level,
                    "_schema_mode": i._schema_mode,
                    "_schema_keys": sorted(i._schema_keys),
                },
            }
        return out

    @classmethod
    def _from_info_collection_snapshot(
            cls,
            snapshot: dict[str, dict]) -> InfoCollection:
        """
        Inverse of `_to_info_collection_snapshot`.
        Rebuilds Info objects and restores their attributes as saved.
        Does not set collection-level attrs; callers can set/ sync afterwards.
        """
        items: list[Info] = []
        for name, content in snapshot.items():
            info_map = dict(content.get("info", {}))
            attrs = dict(content.get("attrs", {}))

            is_schema_level = bool(attrs.get("_is_schema_level", False))
            schema_mode = attrs.get("_schema_mode", "infer")
            schema_keys = set(attrs.get("_schema_keys", []))

            info = Info(name=name).update(**info_map)
            info._is_schema_level = is_schema_level
            info._schema_mode = schema_mode  # type: ignore[assignment]
            info._schema_keys = set(schema_keys)

            items.append(info)

        return cls(items)

    def _serialize_info_collection_snapshot(self) -> str:
        """
        JSON string of `_to_info_collection_snapshot()` with deterministic ordering.
        """
        snap = self._to_info_collection_snapshot()
        return json.dumps(snap, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _deserialize_info_collection_snapshot(cls, source: str) -> "InfoCollection":
        """
        Inverse of `_serialize_info_collection_snapshot`.
        """
        snap = json.loads(source)
        if not isinstance(snap, dict):
            raise ValueError("Invalid snapshot: expected a top-level dict.")
        return cls._from_info_collection_snapshot(snap)


# ----------------- exceptions & helpers -----------------


class InfoException(Exception):
    """Base class for Info/InfoCollection exceptions."""


class MissingInfoNameException(InfoException):
    """Raised when a bare mapping cannot be routed to a unique target."""

    def __init__(self, *, is_columns_info: bool, names: list[str] | None,
                 mapping: dict):
        if is_columns_info:
            if names:
                names_str = "\n\t- ".join(names)
                msg = (
                    f"\n\nYou are trying to update schema-level information with {mapping}.\n"
                    f"There are {len(names)} different schema-level targets, with names:\n\t- {names_str}\n"
                    f"Please specify a name where the information will be stored: "
                    f".update('name', **{mapping})"
                )
            else:
                msg = (
                    f"\n\nYou are trying to update schema-level information with {mapping}.\n"
                    f"However, no schema-level target could be determined.\n"
                    f"Please specify a name where the information will be stored: "
                    f".update('name', **{mapping})"
                )
        else:
            msg = (
                f"\n\nYou are trying to update information with {mapping}.\n"
                f"Please specify a name where the information will be stored: "
                f".update('name', **{mapping})"
            )
        super().__init__(msg)


def callable_to_column_mapping(columns: Iterable[str],
                               transform: Callable[[str], str]) -> dict[str, str]:
    """Apply a renaming function across a set of keys."""
    return {c: transform(c) for c in columns}


def _pre_process_info(info: InfoTypes | None) -> list[Info]:
    if info is None:
        return []
    if isinstance(info, Info):
        return [copy.deepcopy(info)]
    if isinstance(info, Iterable):
        items = list(info)
        if not all(isinstance(i, Info) for i in items):
            raise TypeError("'info' must contain only Info instances")
        return [copy.deepcopy(i) for i in items]
    raise TypeError("'info' must be either Info or Iterable[Info]")
