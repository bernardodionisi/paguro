from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import polars as pl

from paguro.ashi import Box
from paguro.ashi.repr.string.box.templates import COLLECTION_BOXES
from paguro.dataset.utils.utils import _unnest
from paguro.eda.skim.utils.base import _base_config
from paguro.eda.skim.utils.config import unpack_single_config_and_get_exprs
from paguro.eda.skim.utils.hist import add_hist_to_data

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import ColumnNameOrSelector

    from paguro.collection.collection import Collection
    from paguro.dataset.dataset import Dataset
    from paguro.dataset.lazydataset import LazyDataset


def skim(
    data: pl.DataFrame | pl.LazyFrame | Dataset | LazyDataset,
    config: list[tuple] | None = None,
    *,
    by: str | list[str] | None = None,
    hist: bool = False,
    unnest_structs: bool | ColumnNameOrSelector = False,
) -> Collection:
    """
    Generate a rich, structured and readable summary of a dataset.

    `skim` is a flexible alternative to `DataFrame.describe()` that
    produces a (configurable) summary of a dataset,
    allowing grouped statistics across user-defined column sets,
    optionally broken down by row-level groups.

    The summary output adapts to the type and structure of each column, and can be
    customized to group columns arbitrarily—not just by data type—as long as the
    specified statistics make sense for the selected columns.

    Parameters
    ----------
    data
        The input data to summarize.

    config
        A list of group definitions that control how columns are selected and which
        summary statistics are computed for each group. Each entry is a 3-tuple:

            (group_title, column_selector, summary_stats)

        - `group_label` (str): A label shown in the output summary.
        - `column_selector` (str, list[str], or ColumnSelector):
            Specifies which columns belong to this group. Can be a column name,
            list of names, or a Polars-style selector (e.g., `cs.numeric()`,
            `cs.starts_with("meta_")`, etc.).
        - `summary_stats` (list): A list of summaries to compute. Each item can be:
            - A string: the name of a standard Polars expression method (e.g.,`"mean"`).
            - A Polars expression: applied directly to the selected columns.
            - A `(label, expr)` tuple: a custom label and expression pair.

        Grouping by data type is a common pattern, but not required.
        Columns can be grouped in any way, provided the corresponding
        statistics are valid for those columns.

        If `config` is `None`, a default configuration is used,
        which includes predefined groupings for common types such as
        numeric, boolean, string, categorical, date, datetime, and duration.

    by
        One or more column names to group by. If specified, statistics are computed
        separately for each group.

    hist
        Whether to include inline histograms for numeric columns. Needs numpy.

    unnest_structs
        If `True`, struct columns are unnested before summary computation.
        Can also be a column name or selector to unnest specific struct fields.
    """
    name: str | None = None
    if not isinstance(data, (pl.LazyFrame, pl.DataFrame)):
        name = data._name
        data = data._data

    name = _name_with_counts(name=name, data=data)

    if unnest_structs:
        data = _unnest(
            data=data,
            columns=None
            if isinstance(unnest_structs, bool)
            else unnest_structs,
            separator="|",
        )

    if isinstance(by, str):
        by = [by]

    out: dict = _skim(data=data, config=config, by=by, hist=hist)

    from paguro.collection.collection import Collection

    collection = Collection(data=out, name=name)

    try:
        skim_box: Callable[..., Box] | None = COLLECTION_BOXES.get("skim")
        if skim_box is not None:
            collection._box = skim_box(display_rounding=1, by=by)
        else:
            collection._box = Box()
    except Exception as e:
        warnings.warn(
            f"Unable to set box for skim: {e}",
            stacklevel=2,
        )
    # def _prettify_quantile(string: str) -> str:
    #     return string.split(".")[1].replace(")", "") + "%"

    return collection


def _name_with_counts(
    name: str | None, data: pl.LazyFrame | pl.DataFrame
) -> str:
    row_count, column_count = _get_row_and_column_counts(data)
    counts = f"rows: {row_count}, columns: {column_count}"

    if name is None:
        name = counts
    else:
        name += f"\n{counts}"
    return name


def _get_row_and_column_counts(
    data: pl.LazyFrame | pl.DataFrame,
) -> tuple[int, int]:
    if isinstance(data, pl.DataFrame):
        row_count, column_count = data.shape
    else:
        row_count, *_ = data.select(pl.len()).collect().row(0)
        column_count = len(data.collect_schema())

    return row_count, column_count


def _skim(
    data: pl.LazyFrame | pl.DataFrame,
    config: list[tuple] | None,
    *,
    by: list[str] | None,
    hist: bool,
) -> dict[str, pl.DataFrame]:
    if config is None:
        config = _base_config()

    data = data.lazy()

    schema: dict = {k: v for k, v in data.collect_schema().items()}

    titles, frames = [], []
    for i in config:
        title, columns, out_names, exprs = (
            unpack_single_config_and_get_exprs(
                data=data, single_config=i, by=by
            )
        )

        if by is not None:
            temp = data.select(*by, *columns)
        else:
            temp = data.select(columns)

        stats_data: pl.DataFrame | None = _get_stats_data(
            data=temp, names=out_names, exprs=exprs, by=by
        )

        if stats_data is None:
            continue

        stats_data = add_other_info(
            data_selected=temp,
            stats_data=stats_data,
            schema=schema,
            by=by,
            # selector=columns,
            hist=hist,
        )

        frames.append(stats_data)
        titles.append(title)

    return dict(zip(titles, frames, strict=False))


def add_other_info(
    *,
    data_selected: pl.LazyFrame | pl.DataFrame,
    stats_data: pl.DataFrame,
    schema: pl.Schema | dict,
    by: list | None,
    # selector: None | cs.Selector,
    hist: bool | int,
) -> pl.DataFrame:
    stats_data = stats_data.with_columns(
        pl.col("column")
        .replace_strict(
            {k: str(v) for k, v in schema.items()},
            # return_dtype=pl.DataType
        )
        .alias("data_type")
    )

    stats_data = add_hist_to_data(
        data=data_selected.collect()
        if isinstance(data_selected, pl.LazyFrame)
        else data_selected,
        stats_data=stats_data,
        by=by,
        hist=hist,
        # selector=selector,
    )
    return stats_data


# ----------------------------------------------------------------------


def _get_stats_data(
    data: pl.LazyFrame,
    exprs: list[pl.Expr],
    names: dict[str, list[str]],
    by: list[str] | None,
) -> pl.DataFrame | None:
    if by is None:
        _stats_data = data.select(exprs).collect()
    else:
        _stats_data = data.group_by(by).agg(exprs).collect()

    return _reshape_collected_stats_data(
        data=_stats_data,
        names_dict=names,
        by=by,
    )


def _reshape_collected_stats_data(
    data: pl.DataFrame,
    names_dict: dict,
    by: str | list[str] | None,
) -> pl.DataFrame | None:
    # for each column, create a separate DataFrame (of descriptive stats)
    out = []
    for k, v in names_dict.items():
        temp = _reshape_single_collected_stats_data(
            data=data, column_names=v, original_column_name=k, by=by
        )

        out.append(temp)

    if not out:
        return None

    data = pl.concat(out, how="vertical_relaxed")
    return data


def _reshape_single_collected_stats_data(
    data: pl.DataFrame,
    column_names: list[str],  # of the group data
    original_column_name: str,
    by: str | list[str] | None,
) -> pl.DataFrame:
    if by is None:
        return _reshape_single_collected_stats_data_no_groups(
            data=data,
            column_names=column_names,
            original_column_name=original_column_name,
        )

    out = []
    for _, d in data.group_by(by):
        temp = _reshape_single_collected_stats_data_no_groups(
            data=d,
            column_names=column_names,
            original_column_name=original_column_name,
        )

        temp = pl.concat([d.select(by), temp], how="horizontal")

        out.append(temp)

    return pl.concat(out, how="vertical_relaxed")


def _reshape_single_collected_stats_data_no_groups(
    data: pl.DataFrame,
    column_names: list[str],  # of the group data
    original_column_name: str,
) -> pl.DataFrame:
    data = (
        data.select(column_names)
        .rename(
            dict(
                zip(
                    column_names,
                    [i.split("::")[0] for i in column_names],
                    strict=False,
                )
            )
        )  # rename the metrics columns
        .insert_column(
            0,
            pl.Series([original_column_name] * data.shape[0]).alias(
                "column"
            ),
        )
    )
    return data
