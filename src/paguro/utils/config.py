# This module has been adapted from:
#     https://github.com/pola-rs/polars/blob/main/py-polars/polars/config.py
#
# py-polars/polars/config.py is distributed with the following license
#
# '''
# Copyright (c) 2025 Ritchie Vink
# Some portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# '''

from __future__ import annotations

import contextlib
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from polars._utils.various import normalize_filepath

from paguro.typing import ValidationMode
from paguro.utils.dependencies import json

if TYPE_CHECKING:
    import sys
    from types import TracebackType

    if sys.version_info >= (3, 11):
        from typing import Self, Unpack
    else:
        from typing_extensions import Self, Unpack

__all__ = ["Config"]

# note: register all Config-specific environment variable names here; need to constrain
# which 'POLARS_' environment variables are recognized, as there are other lower-level
# and/or unstable settings that should not be saved or reset with the Config vars.
_PAGURO_CFG_ENV_VARS = {
    "ASHI_WIDTH_CHARS",
    "ASHI_STYLED",
    "PAGURO_DATASET_ASHI_REPR",
    # validation
    "PAGURO_AUTO_VALIDATION_MODE",
    "PAGURO_ENABLE_VALIDATION_DECORATORPAGURO_INVERTED_VALIDATION_FILTER",
}

StyleLiterals = Literal["force", "force:truecolor", "force:256", "force:16"]


class ConfigParameters(TypedDict, total=False):
    """Parameters supported by the polars Config."""

    # display
    width_chars: int | None
    styled: bool | StyleLiterals | None
    dataset_ashi_repr: bool | None

    # validation
    auto_validation_mode: bool | ValidationMode | None
    auto_validation_limit: int | None

    _enable_validation_decorator: bool | None
    _inverted_validation_filter: bool | None

    # display
    set_width_chars: int | None
    set_styled: bool | None
    set_dataset_ashi_repr: bool | None

    # validation
    set_auto_validation_mode: bool | ValidationMode | None
    set_auto_validation_limit: int | None

    _set_enable_validation_decorator: bool | None
    _set_inverted_validation_filter: bool | None


class Config(contextlib.ContextDecorator):
    """
    Configuration.

    Notes
    -----
    Can also be used as a context manager OR a function decorator in order to
    temporarily scope the lifetime of specific options.

    Constructors
    ------------

    Display
    -------

    Representation
    --------------

    Validation
    ----------

    Settings
    --------

    """

    _context_options: ConfigParameters | None = None
    _original_state: str = ""

    def __init__(
            self,
            *,
            restore_defaults: bool = False,
            apply_on_context_enter: bool = False,
            **options: Unpack[ConfigParameters],
    ) -> None:
        """
        Initialise a Config object instance for context manager usage.

        Any `options` kwargs should correspond to the available named "set_*"
        methods, but are allowed to omit the "set_" prefix for brevity.

        Parameters
        ----------
        restore_defaults
            set all options to their default values (this is applied before
            setting any other options).
        apply_on_context_enter
            defer applying the options until a context is entered. This allows you
            to create multiple `Config` instances with different options, and then
            reuse them independently as context managers or function decorators
            with specific bundles of parameters.
        **options
            keyword args that will set the option; equivalent to calling the
            named "set_<option>" method with the given value.
        """
        # save original state _before_ any changes are made
        self._original_state = self.save()
        if restore_defaults:
            self.restore_defaults()

        if apply_on_context_enter:
            # defer setting options; apply only on entering a new context
            self._context_options = options
        else:
            # apply the given options immediately
            self._set_config_params(**options)
            self._context_options = None

    def __enter__(self) -> Self:
        """Support setting Config options that are reset on scope exit."""
        self._original_state = self._original_state or self.save()
        if self._context_options:
            self._set_config_params(**self._context_options)
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
    ) -> None:
        """Reset any Config options that were set within the scope."""
        self.restore_defaults().load(self._original_state)
        self._original_state = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return False
        return (self._original_state == other._original_state) and (
                self._context_options == other._context_options
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def _set_config_params(
            self, **options: Unpack[ConfigParameters]
    ) -> None:
        for opt, value in options.items():
            if not hasattr(self, opt) and not opt.startswith("set_"):
                opt = f"set_{opt}"
            if not hasattr(self, opt):
                msg = f"`Config` has no option {opt!r}"
                raise AttributeError(msg)
            getattr(self, opt)(value)

    @classmethod
    def load(cls, cfg: str) -> Config:
        """
        Load (and set) previously saved Config options from a JSON string.

        Parameters
        ----------
        cfg : str
            JSON string produced by `Config.save()`.

        See Also
        --------
        load_from_file : Load (and set) Config options from a JSON file.
        save : Save the current set of Config options as a JSON string or file.

        Group
        -----
            Settings
        """
        try:
            options = json.loads(cfg)
        except json.JSONDecodeError as err:
            msg = "invalid Config string (did you mean to use `load_from_file`?)"
            raise ValueError(msg) from err

        cfg_load = Config()
        opts = options.get("environment", {})
        for key, opt in opts.items():
            if opt is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = opt

        for cfg_methodname, value in options.get("direct", {}).items():
            if hasattr(cfg_load, cfg_methodname):
                getattr(cfg_load, cfg_methodname)(value)
        return cfg_load

    @classmethod
    def load_from_file(cls, file: Path | str) -> Config:
        """
        Load (and set) previously saved Config options from file.

        Parameters
        ----------
        file : Path | str
            File path to a JSON string produced by `Config.save()`.

        See Also
        --------
        load : Load (and set) Config options from a JSON string.
        save : Save the current set of Config options as a JSON string or file.

        Group
        -----
            Settings
        """
        try:
            options = Path(normalize_filepath(file)).read_text()
        except OSError as err:
            msg = (
                f"invalid Config file (did you mean to use `load`?)\n{err}"
            )
            raise ValueError(msg) from err

        return cls.load(options)

    @classmethod
    def restore_defaults(cls) -> type[Config]:
        """
        Reset all polars Config settings to their default state.

        Notes
        -----
        This method operates by removing all Config options from the environment,
        and then setting any local (non-env) options back to their default value.

        Group
        -----
            Settings
        """
        # unset all Config environment variables
        for var in _PAGURO_CFG_ENV_VARS:
            os.environ.pop(var, None)

        return cls

    @classmethod
    def save(cls, if_set: bool = False) -> str:
        """
        Save the current set of Config options as a JSON string.

        Parameters
        ----------
        if_set
            By default this will save the state of all configuration options; set
            to `False` to save only those that have been set to a non-default value.

        See Also
        --------
        load : Load (and set) Config options from a JSON string.
        load_from_file : Load (and set) Config options from a JSON file.
        save_to_file : Save the current set of Config options as a JSON file.

        Examples
        --------
        >>> json_state = pg.Config.save()

        Group
        -----
            Settings

        Returns
        -------
        str
            JSON string containing current Config options.
        """
        environment_vars = {
            key: os.environ.get(key)
            for key in sorted(_PAGURO_CFG_ENV_VARS)
            if not if_set or (os.environ.get(key) is not None)
        }

        options = json.dumps(
            {"environment": environment_vars},
            separators=(",", ":"),
        )
        return options

    @classmethod
    def save_to_file(cls, file: Path | str) -> None:
        """
        Save the current set of Config options as a JSON file.

        Parameters
        ----------
        file
            Optional path to a file into which the JSON string will be written.
            Leave as `None` to return the JSON string directly.

        See Also
        --------
        load : Load (and set) Config options from a JSON string.
        load_from_file : Load (and set) Config options from a JSON file.
        save : Save the current set of Config options as a JSON string.

        Group
        -----
            Settings
        """
        file = Path(normalize_filepath(file)).resolve()
        file.write_text(cls.save())

    @classmethod
    def state(cls, if_set: bool = False) -> dict[str, str | None]:
        """
        Show the current state of all Config variables in the environment as a dict.

        Parameters
        ----------
        if_set
            By default this will show the state of all `Config` environment variables.
            change this to `True` to restrict the returned dictionary to include only
            those that have been set to a specific value.

        Group
        -----
            Settings
        """
        config_state = {
            var: os.environ.get(var)
            for var in sorted(_PAGURO_CFG_ENV_VARS)
            if not if_set or (os.environ.get(var) is not None)
        }

        return config_state

    # ------------------------------------------------------------------

    @classmethod
    def set_width_chars(cls, width_chars: int | None) -> type[Config]:
        """
        Width.

        Group
        -----
            Representation
        """
        if isinstance(width_chars, int):
            if width_chars < 40:
                msg = "width_chars must be >= 40"
                raise ValueError(msg)

        if width_chars is None:
            os.environ.pop("ASHI_WIDTH_CHARS", None)
        else:
            os.environ["ASHI_WIDTH_CHARS"] = str(width_chars)
        return cls

    @classmethod
    def set_styled(
            cls,
            styled: (
                    bool
                    | StyleLiterals
                    | None
            )
    ) -> type[Config]:
        """
        Styled.

        Group
        -----
            Representation
        """
        if styled is None:
            os.environ.pop("ASHI_STYLED", None)
        else:
            if isinstance(styled, bool):
                styled = can_style(styled=styled, warn=True)
                os.environ["ASHI_STYLED"] = str(int(styled))
            else:
                if styled == "force":
                    styled = "force:truecolor"
                os.environ["ASHI_STYLED"] = str(styled)

        return cls

    # ------------------------------------------------------------------

    @classmethod
    def set_dataset_ashi_repr(
            cls,

            active: bool | None,
    ) -> type[Config]:
        """
        Ashi representation of the dataset.

        Group
        -----
            Display
        """
        if active is None:
            os.environ.pop("PAGURO_DATASET_ASHI_REPR", None)
        else:
            os.environ["PAGURO_DATASET_ASHI_REPR"] = str(int(active))
        return cls

    # ------------------------------------------------------------------

    @classmethod
    def set_auto_validation_mode(
            cls, mode: bool | ValidationMode | None,
    ) -> type[Config]:
        """
        Validation mode.

        Group
        -----
            Validation
        """
        if isinstance(mode, bool):
            mode = "all" if mode else None

        if mode is None:
            os.environ.pop("PAGURO_AUTO_VALIDATION_MODE", None)
        else:
            _modes = (
                "schema",
                "data",
                "none",
                "all"
            )
            if mode not in _modes:
                raise ValueError(f"Invalid validation mode: choose from {_modes}")
            os.environ["PAGURO_AUTO_VALIDATION_MODE"] = mode
        return cls

    @classmethod
    def set_auto_validation_limit(
            cls,
            limit: int | None,
    ) -> type[Config]:
        """
        Limit the number of rows when collecting validation errors in auto mode.

        Group
        -----
            Validation
        """

        if limit is None:
            os.environ.pop("PAGURO_AUTO_VALIDATION_LIMIT", None)
        else:
            limit = int(limit)
            if limit <= 0:
                raise ValueError(
                    f"Invalid limit: {limit}, limit must be > 0",
                )
            os.environ["PAGURO_AUTO_VALIDATION_LIMIT"] = str(limit)
        return cls

    @classmethod
    def _set_enable_validation_decorator(
            cls, active: bool | None
    ) -> type[Config]:
        """
        Enable validation decorator.

        Group
        -----
            Validation
        """
        if active is None:
            os.environ.pop("PAGURO_ENABLE_VALIDATION_DECORATOR", None)
        else:
            os.environ["PAGURO_ENABLE_VALIDATION_DECORATOR"] = str(
                int(active)
            )
        return cls

    @classmethod
    def _set_inverted_validation_filter(
            cls, inverted: bool | None
    ) -> type[Config]:
        """
        Inverted Validation Filter.

        Group
        -----
            Validation
        """
        if inverted is None:
            os.environ.pop("PAGURO_INVERTED_VALIDATION_FILTER", None)
        else:
            os.environ["PAGURO_INVERTED_VALIDATION_FILTER"] = str(
                int(inverted)
            )
        return cls


def can_style(styled: bool, warn: bool = True) -> bool:
    from paguro.ashi.terminal import terminal_detector

    if styled:
        td = terminal_detector()
        if not (td.is_unix and td.supports_styling):
            reason = ""
            if not td.is_unix:
                reason += "- You are not on a Unix system\n"
            if not td.supports_styling:
                reason += (
                    "- Your terminal/console does not support styling\n"
                )

            if warn:
                warnings.warn(
                    f"Unfortunately styling is not supported because:\n{reason}",
                    stacklevel=2,
                )
            styled = False
    return styled


# def should_style() -> bool:
#     _cs: bool = can_style(styled=True, warn=False)
#     return bool(int(os.environ.get("ASHI_STYLED", _cs)))

def should_style() -> bool:
    ashi_styled = os.environ.get("ASHI_STYLED")
    if ashi_styled is None:
        return can_style(styled=True, warn=False)
    if ashi_styled == "1" or ashi_styled.startswith("force"):
        return True
    return False


def _forced_color_system() -> Literal["truecolor", "256", "16", "unknown"]:
    ashi_styled: str | None = os.environ.get("ASHI_STYLED")
    if ashi_styled is None:
        return "unknown"

    if "truecolor" in ashi_styled:
        return "truecolor"
    elif "256" in ashi_styled:
        return "256"
    elif "16" in ashi_styled:
        return "16"
    else:
        return "unknown"
