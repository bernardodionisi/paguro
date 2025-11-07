from __future__ import annotations

import pathlib
import os

import paguro as pg
import polars as pl

from paguro.shared.various import _write_data_repr_to_svg

pg.Config.set_styled(False)
pg.Config.set_width_chars(60)


def _save_validation_error_html(
        file_name: str,
        *,
        errors_only: bool,
        validator,
        **kwargs,
) -> None:
    import pathlib
    import os

    _build_dir = pathlib.Path(
        os.environ.get("SPHINX_BUILD_DIR", "docs/build")
    )

    examples_path = _build_dir / "_images" / "examples" / "validation-errors"
    examples_path.mkdir(parents=True, exist_ok=True)

    file_path = examples_path / file_name

    with pg.Config(styled="force", width_chars=60):
        try:
            validator.validate(**kwargs)
        except (pg.exceptions.ValidationError,
                pg.exceptions.RelationValidationError) as e:
            e.write_html(
                path=file_path,
                errors_only=errors_only,
                color="#747C8C",
                background="#21232B"
            )


def _save_dataset_svg(
        data: pl.DataFrame,
        file_name: str,
        *,
        name: str | None = None,
        font_size: int = 20,
        line_height: int = 25,
) -> None:
    _build_dir = pathlib.Path(
        os.environ.get("SPHINX_BUILD_DIR", "docs/build")
    )

    examples_path = _build_dir / "_images" / "examples" / "datasets"
    examples_path.mkdir(parents=True, exist_ok=True)

    file_path = examples_path / file_name

    if name is None:
        name = pathlib.Path(file_name).stem
    with pl.Config(tbl_hide_dataframe_shape=True):
        return _write_data_repr_to_svg(
            data=data,
            title=name,
            path=file_path,
            font_size=font_size,
            line_height=line_height,
        )


# --------------------------- data -------------------------------------


# IMPORTANT: if you modify customers also modify customers in docs/pages/_tutorials !!!

customers = pl.DataFrame(
    {
        "id": ["C001", "C002", "C003", "C004"],
        "name": ["Alice Wong", "Bob Smith", "Carol Jones", None],
        "email": ["alice@company.com", None, "caroljones", "david@company.com"],
        "age": [29, 34, 41, -5],
    }
)

# IMPORTANT: if you modify orders also modify orders in docs/pages/_tutorials !!!

orders = (
    pl.DataFrame(
        {
            "id": [1001, 1002, 1003, 1004],
            "customer_id": ["C001", "C002", "C003", "C005"],
            "order_date": ["2024-03-10", "2025-01-01", "2025-03-15", None],
            "delivery_date": ["2024-03-14", "2024-09-01", "2025-03-18", "2025-03-20"],
            "total_amount": [None, 180, -50, 120],
        }
    ).with_columns(
        pl.col("order_date", "delivery_date").cast(pl.Date),
    )
)

# IMPORTANT: if you modify orders_simple also modify orders_simple in docs/pages/_tutorials !!!

orders_simple = pl.DataFrame(
    {
        "id": [1001, 1002, 1003, 1004],
        "total_amount": [None, 180, -50, 120],
    }
)

# IMPORTANT: if you modify customers_nested also modify customers_nested in docs/pages/_tutorials !!!

customers_nested = pl.DataFrame([
    {
        "id": "C001",
        "contact": {"name": "Alice Wong", "email": "alice@company.com"},
        "meta": {"age": 29, "country": "US"},
    },
    {
        "id": "C002",
        "contact": {"name": "Bob Smith", "email": None},
        "meta": {"age": 34, "country": "Canada"},
    },
    {
        "id": "C003",
        "contact": {"name": "Carol Jones", "email": "caroljones"},
        "meta": {"age": 41, "country": "US"},
    },
    {
        "id": "C004",
        "contact": {"name": None, "email": "david@company.com"},
        "meta": {"age": -5, "country": "England"},
    },
])

# IMPORTANT: if you modify orders_nested also modify orders_nested in docs/pages/_tutorials !!!

orders_nested = (
    orders
    .with_columns(
        customer_id=pl.DataFrame(customers_nested).to_struct()
    )
    .rename({"customer_id": "customer"})
)

# ----------------------------------------------------------------------

_save_dataset_svg(orders, "orders.svg")
_save_dataset_svg(orders_simple, "orders_simple.svg")
_save_dataset_svg(customers, "customers.svg")
_save_dataset_svg(customers_nested, "customers_nested.svg")
_save_dataset_svg(orders_nested, "orders_nested.svg")
