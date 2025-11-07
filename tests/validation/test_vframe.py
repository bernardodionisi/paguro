from __future__ import annotations

import pytest

import paguro as pg


def test_vframe_construction():
    vf = pg.vframe()
    assert vf._name is None

    vf = pg.vframe(name="name of the validation step")
    assert vf._name == "name of the validation step"

    # defaults

    assert vf._unique is None

def test_unique_failing():
    with pytest.raises(pg.exceptions.ValidationError):
        (
            pg.vframe(unique="a")
            .validate(
                data={
                    "a": [1, 2, 2]
                }
            )
        )

