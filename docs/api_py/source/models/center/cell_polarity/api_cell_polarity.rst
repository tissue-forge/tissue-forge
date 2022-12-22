.. _api_cell_polarity:

Cell Polarity
--------------

.. module:: tissue_forge.models.center

This is the API Reference page for the module: :mod:`cell_polarity`.
For details on the mathematics and modeling concepts, see the
:ref:`Cell Polarity Module Documentation <docs_main:cell_polarity>`.

.. moduleauthor:: T.J. Sego <tjsego@iu.edu>


.. module:: tissue_forge.models.center.cell_polarity


.. autofunction:: getVectorAB

.. autofunction:: getVectorPCP

.. autofunction:: setVectorAB

.. autofunction:: setVectorPCP

.. autofunction:: registerParticle

.. autofunction:: unregister

.. autofunction:: registerType

.. autofunction:: getInitMode

.. autofunction:: setInitMode

.. autofunction:: getInitPolarAB

.. autofunction:: setInitPolarAB

.. autofunction:: getInitPolarPCP

.. autofunction:: setInitPolarPCP

.. autofunction:: createPersistentForce

.. autofunction:: setDrawVectors

.. autofunction:: setArrowColors

.. autofunction:: setArrowScale

.. autofunction:: setArrowLength

.. autofunction:: load

.. autofunction:: createContactPotential

.. autoclass:: ContactPotential(tissue_forge.Potential)

    .. autoattribute:: couplingFlat

    .. autoattribute:: couplingOrtho

    .. autoattribute:: couplingLateral

    .. autoattribute:: distanceCoeff

    .. autoattribute:: cType

    .. autoattribute:: mag

    .. autoattribute:: rate

    .. autoattribute:: bendingCoeff

.. autoclass:: PersistentForce(tissue_forge.Force)

    .. autoattribute:: sensAB

    .. autoattribute:: sensPCP
