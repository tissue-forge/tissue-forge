Reactions and Species
----------------------

.. currentmodule:: tissue_forge.state

.. autoclass:: Species

    .. automethod:: __str__

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__

    .. autoproperty:: id

    .. autoproperty:: name

    .. autoproperty:: species_type

    .. autoproperty:: compartment

    .. autoproperty:: initial_amount

    .. autoproperty:: initial_concentration

    .. autoproperty:: substance_units

    .. autoproperty:: spatial_size_units

    .. autoproperty:: units

    .. autoproperty:: has_only_substance_units

    .. autoproperty:: boundary_condition

    .. autoproperty:: charge

    .. autoproperty:: constant

    .. autoproperty:: conversion_factor


.. autoclass:: SpeciesValue

    .. autoproperty:: boundary_condition

    .. autoproperty:: initial_amount

    .. autoproperty:: initial_concentration

    .. autoproperty:: constant

    .. autoproperty:: value

    .. automethod:: secrete


.. autoclass:: SpeciesList

    .. automethod:: __str__

    .. automethod:: __len__

    .. automethod:: __getattr__

    .. automethod:: __setattr__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: item

    .. automethod:: insert

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__


.. autoclass:: StateVector

    .. automethod:: __str__

    .. automethod:: __len__

    .. automethod:: __getattr__

    .. automethod:: __setattr__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__

    .. autoattribute:: species
