import paguro as pg


def test_init_collection():
    cl = pg.Collection({"d0": {"a": [1, 2, 3]}, "d1": {"b": [4, 5, 6], }})
    assert isinstance(cl, pg.Collection)
