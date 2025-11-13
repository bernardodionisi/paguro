# - VFrameModel

"""
Model-based validation using :obj:`VFrameModel <paguro.models.vfm.VFrameModel>`
"""

# = Defining a model

"""

"""

# .. ipython:: python

import paguro as pg
from paguro.models import vfm

# END

# * Collecting the model from the data

# .. ipython:: python

import polars as pl

customers = pl.DataFrame(
    {
        "id": ["C001", "C002", "C003", "C004"],
        "name": ["Alice Wong", "Bob Smith", "Carol Jones", None],
        "email": ["alice@company.com", None, "caroljones", "david@company.com"],
        "age": [29, 34, 41, -5],
    }
)

print(customers)

# END

# .. ipython:: python


print(
    vfm.collect_model_blueprint(
        customers,
        root_class_name="Customers"
    )
)


# END


# .. ipython:: python

class Customers(vfm.VFrameModel):
    id = pg.vcol(allow_nulls=False)
    name = pg.vcol(allow_nulls=False)
    email = pg.vcol(allow_nulls=False)
    age = pg.vcol(allow_nulls=False)


# END


# .. ipython:: python

try:
    customers = pg.Dataset(customers).with_model(Customers)
except pg.exceptions.ValidationError as e:
    print(e)

# END
