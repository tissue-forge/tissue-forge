Forces
-------

.. currentmodule:: tissue_forge.tissue_forge

.. autoclass:: Force

    .. automethod:: bind_species

    .. automethod:: berendsen_tstat

    .. automethod:: random

    .. automethod:: friction

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__

.. autoclass:: _CustomForce
    :show-inheritance:

    .. automethod:: fromForce

.. autoclass:: CustomForce
    :show-inheritance:

    .. autoproperty:: value

    .. autoproperty:: period

    .. automethod:: fromForce

.. autoclass:: ForceSum
    :show-inheritance:

    .. autoattribute:: f1

    .. autoattribute:: f2

    .. automethod:: fromForce

.. autoclass:: Berendsen
    :show-inheritance:

    .. autoattribute:: itau

    .. automethod:: fromForce

.. autoclass:: Gaussian
    :show-inheritance:

    .. autoattribute:: std

    .. autoattribute:: mean

    .. autoattribute:: durration_steps

    .. automethod:: fromForce

.. autoclass:: Friction
    :show-inheritance:

    .. autoattribute:: coef

    .. automethod:: fromForce
