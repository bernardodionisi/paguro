import paguro as pg


def test_dataset_validation_roundtrip():
    ds = (
        pg.Dataset({"a": [1, 2, 3]})
        .with_validation(pg.vcol("a", ge=1))
    )

    new_ds = pg.Dataset._from_paguro_metadata_dict(
        frame=ds._data,
        paguro_metadata=ds._get_serialized_paguro_metadata_dict(use_pyarrow_format=False)
    )

    assert (
            ds.validation._fingerprint(include_info=False) ==
            new_ds.validation._fingerprint(include_info=False)
    )


def test_lazydataset_validation_roundtrip():
    lds = (
        pg.LazyDataset({"a": [1, 2, 3]})
        .with_validation(pg.vcol("a", ge=1))
    )

    new_ds = pg.LazyDataset._from_paguro_metadata_dict(
        frame=lds._data,
        paguro_metadata=lds._get_serialized_paguro_metadata_dict(use_pyarrow_format=False)
    )

    assert (
            lds.validation._fingerprint(include_info=False) ==
            new_ds.validation._fingerprint(include_info=False)
    )


def test_dataset_info_roundtrip():
    ds = (
        pg.Dataset({"a": [1, 2, 3]})
        .with_info("dataset", description="dataset description")
        .with_info("columns", a="columna description")

    )

    paguro_metadata = ds._get_serialized_paguro_metadata_dict(use_pyarrow_format=False)

    new_ds = pg.Dataset._from_paguro_metadata_dict(
        frame=ds._data,
        paguro_metadata=paguro_metadata
    )
    assert new_ds._info == ds._info


def test_lazydataset_info_roundtrip():
    lds = (
        pg.LazyDataset({"a": [1, 2, 3]})
        .with_info("dataset", description="dataset description")
        .with_info("columns", a="columna description")

    )

    paguro_metadata = lds._get_serialized_paguro_metadata_dict(use_pyarrow_format=False)

    new_ds = pg.LazyDataset._from_paguro_metadata_dict(
        frame=lds._data,
        paguro_metadata=paguro_metadata
    )
    assert new_ds._info == lds._info
