import pytest

import paguro as pg
import polars as pl

from paguro.ashi.info.info_collection import InfoCollection
from paguro.validation.shared.preprocessing.duplicates import DuplicateNameError


def test_get_predicates():
    v = pg.Validation(
        pg.vcol("a", ge=1, c0=pl.all().ge(1)),
        pg.vframe(c0=pl.col("a") > pl.col("b"))
    )

    assert len(v.gather_predicates()) == 4


def test_duplicate_names():
    with pytest.raises(DuplicateNameError):
        pg.Validation(
            pg.vcol("a", ge=1, c0=pl.all().ge(1)),
            pg.vcol("a", ge=1, c0=pl.all().ge(1))
        )


@pytest.fixture
def data():
    data = {
        "a": [1, 2, 3],
        "b": ["a", "b", "@"],
        "c": [{"c1": 1, "c2": 2}, {"c1": 1, "c2": 2}, {"c1": 1, "c2": 2}],
    }
    return data


@pytest.fixture
def validation() -> pg.Validation:
    validation = pg.Validation(
        pg.vcol("a", ge=2, c0=pl.all().ge(1)),
        pg.vcol("b", str_contains="@"),
        pg.vcol.Struct(
            name="c",
            fields=[
                pg.Validation(pg.vframe(pl.col("c1") > pl.col("c2")))
            ]
        ),
        pg.vframe(
            pl.col("a") == pl.col("c").struct.field("c1"),
            c1=pl.col("a") == pl.col("c").struct.field("c1")
        )
    )
    return validation


@pytest.fixture
def validation_with_transform() -> pg.Validation:
    validation_with_transform = pg.Validation(
        pl.col("a").ge(1),
        pg.vcol("a", ge=2, c0=pl.all().ge(1)),
        pg.vcol("b", str_contains="@"),
        pg.vcol.Struct(
            name="c",
            fields=[
                pg.Validation(
                    pg.vcol("c1", le=0),
                    pg.vframe(pl.col("c1") > pl.col("c2"))
                ),
            ]
        ),
        pg.vframe(
            pl.col("a") == pl.col("c").struct.field("c1"),
            pg.vframe(
                pg.vcol("a_alias", ge=3),
                transform=pl.col("a").alias("a_alias"),
            ),
            c1=pl.col("a") == pl.col("c").struct.field("c1"),
        )
    )
    return validation_with_transform


def validation_round_trip(validation: pg.Validation) -> pg.Validation:
    return pg.Validation.deserialize(validation.serialize())


def test_name(validation):
    assert validation.with_name("name").name == "name"
    assert validation.name is None
    validation.name = "name"
    assert validation.name == 'name'


def test_info(validation):
    assert isinstance(validation.with_info(a='a').info, InfoCollection)


# def test_validation_fingerprint(validation):
#     assert (
#             validation._fingerprint(include_info=True) ==
#             ''
#     )


def test_validation_serialization(validation):
    assert (
            validation._fingerprint(include_info=True) ==
            validation_round_trip(validation)._fingerprint(include_info=True)
    )


def test_validation_with_transform_serialization(validation_with_transform):
    assert (
            validation_with_transform._fingerprint(
                include_info=True) ==
            validation_round_trip(validation_with_transform)._fingerprint(
                include_info=True)
    )


def test_validation(validation, data):
    with pytest.raises(pg.exceptions.ValidationError):
        validation.validate(data)
