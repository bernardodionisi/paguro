from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def set_doc_string(
        doc: str | None = None,
        *,
        parameters: str | None = None,
        additional_parameters: str | None = None,
):
    def deco(func: Callable[P, R]) -> Callable[P, R]:

        if doc is None:
            this_doc = func.__doc__
        else:
            this_doc = doc

        if parameters is not None:
            func.__doc__ = (
                this_doc
                .replace("{{ Parameters }}", parameters)
            )
        if additional_parameters is not None:
            func.__doc__ = (
                this_doc
                .replace(
                    "{{ AdditionalParameters }}",
                    additional_parameters
                )
            )
        return func

    return deco


# :ref:`Examples <tutorials-parameters-validate - mode>`

VALID_COLUMNS_SHARED_PARAMETERS = """
        required
            Whether there must be a column with the specified name in the target data.
        allow_nulls
            Whether the column is allowed to contain null values.
        unique
            Whether the values in the column are unique: the number of unique values 
            in the column equal the number of rows in the data.
        
            Typically a useful check to have if the column uniquely identifies 
            the rows of the data. 
        constraints
            Constraints on the values of the column
             
            Constraints can be passed in 2 alternative ways:
                - as name-Polars Expressions
                    Where the key is the custom name for the constraint (your choice) and
                    the value the polars.all` expression.
                    Within the validator `all` refers to the specified column.
                    
                    .. code-block:: python
                    
                        pg.vcol("a", a_is_ge=pl.all().ge(1))
        
                - as method-argument
                    key: the name of the polars expression method
                    value: the argument to pass in the polars expression
                        
                    .. code-block:: python
        
                        pg.vcol("a", ge=1)
"""

# ----------------------------------------------------------------------

VALIDATE_PARAMS = """
        Parameters
        ----------
        data
        mode
            - `"schema"` : validate the schema
            - `"data"` : validate the data content
            - `"all"`  : validate both schema and data content
            - `"none"`
        keep_columns
            The columns to keep along the validated column in the validation errors.
            It is useful to specify columns to inspect the rows of the target data tha
            do not satisfy the various constraints.
        collect
        on_success
        on_failure
        cast
"""
