from __future__ import annotations

import polars as pl
import polars.selectors as cs

from paguro.eda.skim.utils.pl_upcast_join import join_frames_with_upcast
from paguro.utils.dependencies import _NUMPY_AVAILABLE
from paguro.utils.dependencies import numpy as np


def add_hist_to_data(
    data: pl.DataFrame,
    *,
    stats_data: pl.DataFrame,
    by: list[str] | None,
    hist: bool | int,
    # selector: cs.Selector,
) -> pl.DataFrame:
    if not _NUMPY_AVAILABLE:
        msg = "Please install numpy to use 'hist'"
        raise ImportError(msg)

    symbols = " ▁▂▃▄▅▆▇█"

    cols_for_hist = cs.numeric()
    if by is not None:
        cols_for_hist -= cs.matches("|".join(by))

    if len(data.select(cols_for_hist).columns) == 0:
        return stats_data

    if not isinstance(hist, bool) or hist:
        data = (
            data
            # .select(selector)
            .with_columns(cs.decimal().cast(pl.Float64))
        )

        n_bins = hist if not isinstance(hist, bool) else None

        hist_frame: pl.DataFrame = get_unicode_hist(
            data=data, by=by, n_bins=n_bins, symbols=symbols
        )

        on = ["column"]
        if by is not None:
            if isinstance(by, str):
                on.append(by)
            else:
                on.extend(by)

        stats_data = join_frames_with_upcast(
            [stats_data, hist_frame], on=on, how="left"
        )

    return stats_data


def get_unicode_hist(
    data: pl.DataFrame,
    n_bins: int | None,
    by: str | list[str] | None,
    symbols: str,
) -> pl.DataFrame:
    if by is None:
        out = _get_unicode_hist_dict(
            data=data, n_bins=n_bins, symbols=symbols
        )
    else:
        out = _get_unicode_hist_by(
            data=data, by=by, n_bins=n_bins, symbols=symbols
        )

    return pl.DataFrame(out)


def _get_unicode_hist_dict(
    data: pl.DataFrame, n_bins: int | None, symbols: str
) -> list[dict]:
    out = []

    for i in data.columns:
        out.append(
            {
                "column": i,
                "hist": _plot_unicode_distribution_np(
                    arr=data[i], bins=n_bins, symbols=symbols
                ),
            }
        )

    return out


def _get_unicode_hist_by(
    data: pl.DataFrame,
    n_bins: int | None,
    by: str | list[str],
    symbols: str,
) -> list[dict]:
    if isinstance(by, str):
        by = [by]

    out = []

    # first lets find the number of bins in th
    # all column (to make comparable across groups)
    n_bins_all = {}
    for i in data.columns:
        if i in by:
            continue
        _, edges = _get_counts_edges_np(
            arr=data[i], n_bins=n_bins, symbols=symbols
        )
        n_bins_all[i] = edges

    for group_name, group_data in data.group_by(by):
        for col, n_bins in n_bins_all.items():
            temp = {
                "column": col,
                "hist": _plot_unicode_distribution_np(
                    arr=group_data[col], bins=n_bins, symbols=symbols
                ),
            }

            temp.update(
                dict(zip(by, group_name, strict=False))
            )  # TODO: fix since by could contain "column"
            out.append(temp)

    return out


def _get_counts_edges_np(arr, n_bins: int | None, symbols: str) -> tuple:
    if n_bins is None:
        n_bins = _get_n_bins_np(arr=arr, n_bins=n_bins, symbols=symbols)
    return np.histogram(arr, bins=n_bins)


def _get_n_bins_np(
    arr, n_bins: int | None, symbols: str
) -> int:  # returns hist when return_hist=True
    if isinstance(arr, pl.Series):
        arr = arr.drop_nulls().to_numpy()

    if n_bins is None:
        # calculate the number of bins using the Freedman-Diaconis rule
        iqr = np.subtract(*np.percentile(arr, [75, 25]))
        if iqr == 0:
            n_bins = 1
        else:
            bin_width = 2 * (iqr / (len(arr) ** (1 / 3)))
            max_bins = len(symbols)
            n_bins = max(
                2, min(max_bins, int((arr.max() - arr.min()) / bin_width))
            )
    else:
        if n_bins > len(symbols):
            msg = f"max n_bins should be {len(symbols)}"
            raise ValueError(msg)

    return n_bins


def _get_hist_np(arr, bins, symbols: str) -> str:
    try:
        import numpy as np
    except ImportError:
        msg = "Please install numpy to use 'hist'"
        raise ImportError(msg)

    # compute the histogram using the calculated number of bins
    counts, _ = np.histogram(arr, bins=bins)

    # map histogram bin counts to symbols
    max_count = max(counts) if counts.any() else 1
    normalized_counts = (counts / max_count * (len(symbols) - 1)).astype(
        int
    )
    distribution_plot = (
        "│" + "".join(symbols[count] for count in normalized_counts) + "│"
    )

    return distribution_plot


def _plot_unicode_distribution_np(arr, bins, symbols: str):
    """
    Generates a text-based distribution plot for a Polars Series.

    The idea for this feature has been sourced from a package called 'skimpy'
    """
    # TODO: polars only implementation

    if isinstance(arr, pl.Series):
        arr = arr.drop_nulls().to_numpy()

    if bins is None:
        bins = _get_n_bins_np(arr=arr, n_bins=bins, symbols=symbols)

    return _get_hist_np(arr=arr, bins=bins, symbols=symbols)
