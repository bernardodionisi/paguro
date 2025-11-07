# This module has been adapted from:
#     https://github.com/pola-rs/polars/blob/main/py-polars/polars/meta/versions.py
#
# py-polars/polars/meta/versions.py is distributed with the following license
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

import sys

from paguro._version import __version__



def show_versions() -> None:
    """Print out version of `paguro` and dependencies to stdout."""
    # note: we import 'platform' here as a micro-optimisation for initial import
    import platform

    deps = _get_dependency_info()
    core_properties = ("paguro", "Platform", "Python")
    keylen = max(len(x) for x in [*core_properties, *deps.keys()]) + 1

    print("\n-------- Version info ---------")
    print(f"{'paguro:':{keylen}s} {__version__}")

    print("\n---- Required dependencies ----")
    print(f"{'polars:':{keylen}s} {_get_dependency_version('polars')}")

    print("\n---- Optional dependencies ----")
    for name, v in deps.items():
        print(f"{name:{keylen}s} {v}")

    print("\n-------------------------------")

    print(f"{'Platform:':{keylen}s} {platform.platform()}")
    print(f"{'Python:':{keylen}s} {sys.version}\n")


def _get_dependency_info() -> dict[str, str]:
    # from pyproject.toml [all]
    opt_deps = ["pyarrow", "numpy", "rich"]
    return {f"{name}:": _get_dependency_version(name) for name in opt_deps}


def _get_dependency_version(dep_name: str) -> str:
    # note: we import 'importlib' here as a significiant optimisation for initial import
    import importlib
    import importlib.metadata

    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    else:
        module_version = importlib.metadata.version(
            dep_name
        )  # pragma: no cover

    return module_version


# ---------------------------- docs ------------------------------------


def show_versions_docs() -> None:
    """Print out version of `paguro` and Docs dependencies to stdout."""
    # note: we import 'platform' here as a micro-optimisation for initial import

    deps = _get_dependency_info_docs()
    core_properties = ("paguro",)
    keylen = max(len(x) for x in [*core_properties, *deps.keys()]) + 1

    print("-------- Version info ---------")
    print(f"{'paguro:':{keylen}s} {__version__}")

    print("\n---- Documentation dependencies ----")
    for name, v in deps.items():
        print(f"{name:{keylen}s} {v}")


def _get_dependency_info_docs() -> dict[str, str]:
    # from pyproject.toml [dpcs]
    docs_deps = [
        "sphinx",
        "sphinx_immaterial",
        "sphinx_design",
        "sphinx_last_updated_by_git",
        "ipykernel",
    ]
    return {
        f"{name}:": _get_dependency_version(name) for name in docs_deps
    }
