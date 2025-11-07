from __future__ import annotations

from paguro.validation.valid_column import VCol
from paguro.validation.valid_column.valid_column import ValidColumn

from paguro.validation.valid_frame import VFrame
from paguro.validation.valid_frame.valid_frame import ValidFrame

from paguro.validation.valid_relations import VRelations
from paguro.validation.valid_relations.valid_relations import ValidRelations

# :class:`VCol <paguro.validation.valid_column.VCol>`.
vcol: VCol = VCol()
"""
Constructor for a :class:`ValidColumn <paguro.validation.ValidColumn>` validator.


You can construct a :class:`ValidColumn <paguro.validation.ValidColumn>` validator using `vcol` in 3 different ways:

- **Call**: using `vcol` as a function `vcol()`
- **Dot Name**: chaining your column name to the `vcol` object: `vcol.your_column_name()`
- **By DataType**: specifying the *dtype* you expect your column to have: `vcol.Integer()`

.. paguro-warning:: Note

    *Call* is the most general way of using :ref:`vcol <vcol-call-block>`, 
    it allows you to create any :class:`ValidColumn <paguro.validation.ValidColumn>` 
    that you can create using :ref:`dot name <vcol-dotname-block>` 
    and  :ref:`by data type <vcol-bydatatype-block>`
    
    The only difference is that when using *call* `vcol()`, 
    you are free to specify (or not specify!) the parameters: `name` and `dtype`.
    Using *Dot Name* the column `name` will be set to the attribute that you pass.
    *By DataType* the `dtype` will be set to the one specified by the method, 
    (i.e. :ref:`vcol.Int64() <vcol-int64-block>` will set the `dtype=polars.Int64`, 
    :ref:`vcol.Integer() <vcol-integer-block>` will allow the dtype to be any 
    of the polars integer dtypes).
    
    If you want to specify validation constraints for the fields in a struct column 
    you must use :ref:`vcol.Struct() <vcol-struct-block>`.

Let's see in more details the three different ways and what parameters you can pass 
to specify the constraints for your column.


Call
----

.. _vcol-call-block:

.. paguro-call:: 
    :collapsible: open
        
    .. _vcol-call-block:
    
    .. automethod:: paguro.validation.valid_column.VCol.__call__
       :noindex:

Dot Name
--------

.. _vcol-dotname-block:

.. paguro-getattr:: .*
    :collapsible:
    
    `pg.vcol.your_column_name()`
    
By DataType
-----------

.. _vcol-bydatatype-block:

.. paguro-nested:: Nested
    :collapsible: open

    .. _vcol-struct-block:

    .. paguro-struct:: Struct
        :collapsible:
        
        .. automethod:: paguro.validation.valid_column.VCol.Struct
           :noindex:
    
    .. paguro-array:: Array
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Array
           :noindex:
    
    .. paguro-list:: List
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.List
           :noindex:

.. paguro-temporal:: Temporal
    :collapsible: open

    .. paguro-date:: Date
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Date
           :noindex:
    
    .. paguro-datetime:: DateTime
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.DateTime
           :noindex:
    
    .. paguro-duration:: Duration
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Duration
           :noindex:
    
    .. paguro-time:: Time
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Time
           :noindex:
       
.. paguro-stringouter:: String
    :collapsible: open

    .. paguro-string:: String
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.String
           :noindex:

    .. paguro-enum:: Enum
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Enum
           :noindex:

    .. paguro-categorical:: Categorical
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Categorical
           :noindex:

.. paguro-numeric:: Numeric
    :collapsible: open
        
    .. paguro-integer:: Integer
        :collapsible:

        .. _vcol-integer-block:

        .. automethod:: paguro.validation.valid_column.VCol.Integer
           :noindex:

    .. paguro-integer:: Int8
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.Int8
           :noindex:

    .. paguro-integer:: Int16
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.Int16
           :noindex:

    .. paguro-integer:: Int32
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.Int32
           :noindex:

    .. paguro-integer:: Int64
        :collapsible:
        
        .. _vcol-int64-block:

        .. automethod:: paguro.validation.valid_column.VCol.Int64
           :noindex:

    .. paguro-integer:: Int128
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.Int128
           :noindex:
    
        
    .. paguro-uinteger:: UInteger
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.UInteger
           :noindex:

    .. paguro-uinteger:: UInt8
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.UInt8
           :noindex:

    .. paguro-uinteger:: UInt16
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.UInt16
           :noindex:

    .. paguro-uinteger:: UInt32
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.UInt32
           :noindex:

    .. paguro-uinteger:: UInt64
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.UInt64
           :noindex:

    .. paguro-uinteger:: UInt128
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.UInt128
           :noindex:
    
    .. paguro-float:: Float
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.Float
           :noindex:

    .. paguro-float:: Float32
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.Float32
           :noindex:

    .. paguro-float:: Float64
        :collapsible:
            
        .. automethod:: paguro.validation.valid_column.VCol.Float64
           :noindex:
               
    .. paguro-decimal:: Decimal
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Decimal
           :noindex:

.. paguro-other:: Other
    :collapsible: open

    .. paguro-binary:: Binary
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Binary
           :noindex:

    .. paguro-boolean:: Boolean
        :collapsible:
    
        .. automethod:: paguro.validation.valid_column.VCol.Boolean
           :noindex:
"""

vframe: VFrame = VFrame()
"""
Constructor for a :class:`ValidFrame <paguro.validation.ValidFrame>` validator.

Call
----

.. paguro-call:: 
    :collapsible: open
    
    .. automethod:: paguro.validation.valid_frame.VFrame.__call__
       :noindex:

Dot Name
--------

.. paguro-getattr:: .*
    :collapsible:
    
"""

vrelations: VRelations = VRelations()
"""
Constructor for a :class:`ValidRelations <paguro.validation.ValidRelations>` validator.

Call
----

.. paguro-call:: 
    :collapsible: open
    
    .. automethod:: paguro.validation.valid_relations.VRelations.__call__
       :noindex:

"""

__all__ = [

    "vcol",
    "vframe",
    "vrelations",

    "ValidColumn",
    "ValidFrame",
    "ValidRelations",
]
