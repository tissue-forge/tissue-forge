Potentials
-----------

.. currentmodule:: tissue_forge.tissue_forge

.. autoclass:: _Potential

    .. autoproperty:: min

    .. autoproperty:: max

    .. autoproperty:: cutoff

    .. autoproperty:: domain

    .. autoproperty:: intervals

    .. autoproperty:: bound

    .. autoproperty:: r0

    .. autoproperty:: shifted

    .. autoproperty:: periodic

    .. autoproperty:: r_square

    .. method:: __call__

        Alias of :meth:`_call`

    .. automethod:: _call

    .. automethod:: plot

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__

    .. automethod:: lennard_jones_12_6

    .. automethod:: lennard_jones_12_6_coulomb

    .. automethod:: ewald

    .. automethod:: coulomb

    .. automethod:: coulombR

    .. automethod:: harmonic

    .. automethod:: linear

    .. automethod:: harmonic_angle

    .. automethod:: harmonic_dihedral

    .. automethod:: cosine_dihedral

    .. automethod:: well

    .. automethod:: glj

    .. automethod:: morse

    .. automethod:: overlapping_sphere

    .. automethod:: power

    .. automethod:: dpd

.. autoclass:: Potential
    :show-inheritance:

    .. autoattribute:: custom

.. autoclass:: DPDPotential
    :show-inheritance:

    .. autoattribute:: alpha

    .. autoattribute:: gamma

    .. autoattribute:: sigma

    .. automethod:: fromPot

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__
