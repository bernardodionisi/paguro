from __future__ import annotations

import polars as pl

from typing import Any, TypedDict, TypeVar, Union

from collections.abc import Callable, Iterable, Mapping, Sequence

from paguro.ashi.repr.string.styled.styled_str import StStr

T = TypeVar("T")
Selector = Union[int, str, tuple[str, ...]]


class StyleDict(TypedDict, total=False):
    color: str | None
    background: str | None
    bold: bool
    dim: bool
    italic: bool
    underline: bool
    blink: bool
    inverted: bool
    hidden: bool
    strikethrough: bool


class AffixConfig(TypedDict, total=False):
    prefix: str
    suffix: str
    start_level: int  # default 0 (apply from root unless specified)
    apply_to_deeper_levels: bool  # default False


class StyleMaps(TypedDict, total=False):
    key: dict[Selector, StyleDict]
    value: dict[Selector, StyleDict]
    default_key: StyleDict
    default_value: StyleDict


class AffixMaps(TypedDict, total=False):
    key: dict[Selector, AffixConfig]


# Accept "partial" user inputs (any Mapping is fine)
StyleMapsLike = Mapping[str, Any]
AffixMapsLike = Mapping[str, Any]

# -----------------------
# Defaults & Normalizers
# -----------------------

DEFAULT_STYLES: StyleMaps = {
    "key": {},
    "value": {},
    "default_key": {},
    "default_value": {},
}

DEFAULT_AFFIXES: AffixMaps = {
    "key": {},
}


def _normalize_style_maps(
        styles: StyleMaps | StyleMapsLike | None,
) -> StyleMaps:
    """
    Accepts partial styles and returns a full StyleMaps dict with safe defaults.
    Supports legacy 'default' entries INSIDE the 'key'/'value' maps.
    """
    base: StyleMaps = {
        "key": {},
        "value": {},
        "default_key": {},
        "default_value": {},
    }
    if not styles:
        return base

    default_key: StyleDict = dict(  # type: ignore
        styles.get("default_key", {})
    )
    default_value: StyleDict = dict(  # type: ignore
        styles.get("default_value", {})
    )

    # Copy to avoid mutating caller input
    out: StyleMaps = {
        "key": dict(styles.get("key", {})),
        "value": dict(styles.get("value", {})),
        "default_key": default_key,
        "default_value": default_value,
        # type: ignore[arg-type]
    }
    return out


def _normalize_affix_maps(
        affixes: AffixMaps | AffixMapsLike | None,
) -> AffixMaps:
    """
    Accepts partial affixes and returns a full AffixMaps dict with safe defaults.
    Also supports legacy 'default' selector inside the map (handled by resolver).
    """
    if not affixes:
        return {"key": {}}
    return {"key": dict(affixes.get("key", {}))}  # type: ignore[arg-type]


# -----------------------
# Utilities
# -----------------------


def _as_path_tuple(*parts: Iterable[Any]) -> tuple[str, ...]:
    """
    Normalize mixed path items to a tuple[str, ...] suitable for path-based selectors.
    """
    out: list[str] = []
    for p in parts:
        if isinstance(p, tuple):
            out.extend(str(x) for x in p)
        else:
            out.append(str(p))
    return tuple(out)


def _pick_path_first(
        *,
        mapping: dict[Selector, Any],
        key: Any,
        depth: int,
        path: tuple[str, ...],
        path_based: bool,
) -> Any | None:
    """
    Generic resolver: prefer longest path match; else key; else depth.
    """
    if path_based:
        # Include empty () so callers can set a root-wide rule if desired.
        for i in range(len(path), -1, -1):
            sel = path[:i]
            if sel in mapping:
                return mapping[sel]
    if key in mapping:
        return mapping[key]
    if depth in mapping:
        return mapping[depth]
    return None


def _affix_applies(cfg: AffixConfig, depth: int) -> bool:
    start = cfg.get("start_level", 0)
    deeper = cfg.get("apply_to_deeper_levels", False)
    return depth >= start and (deeper or depth == start)


# -----------------------
# Resolvers (with legacy 'default' support)
# -----------------------


def resolve_style(
        *,
        styles_map: dict[Selector, StyleDict],
        key: Any,
        depth: int,
        path: tuple[str, ...],
        path_based: bool,
        default_style: StyleDict,
) -> StyleDict:
    """
    Resolve style by longest path -> key -> depth -> 'default' in-map -> top-level default.
    Merge result over top-level default.
    """
    match = _pick_path_first(
        mapping=styles_map, key=key, depth=depth, path=path, path_based=path_based
    )
    if match is None:
        # Legacy support: allow 'default' inside the map
        match = styles_map.get("default")

    if not match:
        return default_style

    if not default_style:
        return match
    merged = dict(default_style)
    merged.update(match)

    # return merged
    return StyleDict(**merged)  # type: ignore[typeddict-item]


def resolve_affix(
        *,
        affix_map: dict[Selector, AffixConfig],
        key: Any,
        depth: int,
        path: tuple[str, ...],
        path_based: bool,
) -> AffixConfig:
    """
    Resolve affix by longest path -> key -> depth -> 'default' in-map, then gate with depth rules.
    """
    match = _pick_path_first(mapping=affix_map, key=key, depth=depth, path=path,
                             path_based=path_based)
    if match is None:
        match = affix_map.get("default")

    if match and _affix_applies(match, depth):
        return match
    return {}


# -----------------------
# Render helpers
# -----------------------


def apply_affix_to_key(key: Any, affix_cfg: AffixConfig) -> str:
    if not affix_cfg:
        return str(key)
    return (
        f"{affix_cfg.get('prefix', '')}{key}{affix_cfg.get('suffix', '')}"
    )


def apply_style(item: Any, style: StyleDict) -> StStr:
    return StStr(item).set_style(**style)


def format_value(
        value: T, custom_formatters: dict[type, Callable[[Any], str]]
) -> T | str:
    for data_type, formatter in custom_formatters.items():
        if isinstance(value, data_type):
            return formatter(value)
    return value


# -----------------------
# Public APIs
# -----------------------


def style_nested_structure(
        data: Any,
        *,
        styles: StyleMaps | StyleMapsLike | None = None,
        affixes: AffixMaps | AffixMapsLike | None = None,
        custom_formatters: dict[type, Callable[[Any], str]] | None = None,
        max_depth: int | float = float("inf"),
        style_lists: bool = False,  # placeholder for future list styling
        path_based_styling: bool = True,
) -> Any:
    """
    Full styling pass:
      - Decorates keys with affixes (prefix/suffix rules)
      - Styles keys and values independently using (partial) style maps
      - Polars: LazyFrame -> string; DataFrame passthrough unless custom formatter is provided
    """
    styles_norm = _normalize_style_maps(styles)
    affixes_norm = _normalize_affix_maps(affixes)
    custom_formatters = custom_formatters or {}

    def rec(
            item: Any,
            depth: int,
            parent_key: Any | None,
            path: tuple[str, ...],
    ) -> Any:
        if depth > max_depth:
            return item

        # Dict-like
        if isinstance(item, Mapping):
            out: dict[Any, Any] = {}
            for k, v in item.items():
                k_path = _as_path_tuple(*path, k)

                # Key affixes
                k_affix = resolve_affix(
                    affix_map=affixes_norm["key"],
                    key=k,
                    depth=depth,
                    path=k_path,
                    path_based=path_based_styling,
                )
                rendered_key = apply_affix_to_key(k, k_affix)

                # Key style
                k_style = resolve_style(
                    styles_map=styles_norm["key"],
                    key=k,
                    depth=depth,
                    path=k_path,
                    path_based=path_based_styling,
                    default_style=styles_norm["default_key"],
                )
                styled_key = apply_style(rendered_key, k_style)

                out[styled_key] = rec(v, depth + 1, k, k_path)
            return out

        # Optional: list support (preserve structure)
        if isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray, StStr)
        ):
            return [
                rec(
                    v, depth + 1, parent_key, _as_path_tuple(*path, str(i))
                )
                for i, v in enumerate(item)
            ]

        # Leaves
        if isinstance(item, pl.LazyFrame):
            item = item.__repr__()

        formatted = format_value(item, custom_formatters)

        if isinstance(item, pl.DataFrame) and (
                pl.DataFrame not in custom_formatters
        ):
            return item

        v_style = resolve_style(
            styles_map=styles_norm["value"],
            key=parent_key,
            depth=depth,
            path=path,
            path_based=path_based_styling,
            default_style=styles_norm["default_value"],
        )
        return apply_style(formatted, v_style)

    return rec(data, depth=0, parent_key=None, path=())


def affix_nested_structure(
        data: Any,
        *,
        affixes: AffixMaps | AffixMapsLike | None = None,
        max_depth: int | float = float("inf"),
        path_based_styling: bool = True,
) -> Any:
    """
    Structural pass only:
      - Decorates keys with affixes (prefix/suffix rules)
      - No visual styling
      - Polars: LazyFrame -> string; DataFrame passthrough
    """
    affixes_norm = _normalize_affix_maps(affixes)

    def rec(item: Any, depth: int, path: tuple[str, ...]) -> Any:
        if depth > max_depth:
            return item

        if isinstance(item, Mapping):
            out: dict[Any, Any] = {}
            for k, v in item.items():
                k_path = _as_path_tuple(*path, k)
                k_affix = resolve_affix(
                    affix_map=affixes_norm["key"],
                    key=k,
                    depth=depth,
                    path=k_path,
                    path_based=path_based_styling,
                )
                rendered_key = apply_affix_to_key(k, k_affix)
                out[rendered_key] = rec(v, depth + 1, k_path)
            return out

        if isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray, StStr)
        ):
            return [
                rec(v, depth + 1, _as_path_tuple(*path, str(i)))
                for i, v in enumerate(item)
            ]

        if isinstance(item, pl.LazyFrame):
            return item.__repr__()
        return item

    return rec(data, depth=0, path=())


def render_nested_structure(
        data: Any,
        *,
        supports_styling: bool,
        styles: StyleMaps | StyleMapsLike | None = None,
        affixes: AffixMaps | AffixMapsLike | None = None,
        custom_formatters: dict[type, Callable[[Any], str]] | None = None,
        max_depth: int | float = float("inf"),
        path_based_styling: bool = True,
) -> Any:
    """
    Dispatcher:
      - supports_styling=True  -> full styling pass
      - supports_styling=False -> affix-only pass
    """
    if supports_styling and styles:
        return style_nested_structure(
            data,
            styles=styles,
            affixes=affixes,
            custom_formatters=custom_formatters,
            max_depth=max_depth,
            path_based_styling=path_based_styling,
        )
    else:
        return affix_nested_structure(
            data,
            affixes=affixes,
            max_depth=max_depth,
            path_based_styling=path_based_styling,
        )
