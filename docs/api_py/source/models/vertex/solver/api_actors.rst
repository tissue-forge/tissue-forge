.. _api_vertex_solver_actors:

Actors
^^^^^^^

.. currentmodule:: tissue_forge.models.vertex.solver


.. autoclass:: Adhesion

    .. autoproperty:: lam

    .. automethod:: energy

    .. automethod:: force


.. autoclass:: BodyForce

    .. autoproperty:: comps

    .. automethod:: energy

    .. automethod:: force


.. autoclass:: EdgeTension

    .. autoproperty:: lam

    .. autoproperty:: order

    .. automethod:: energy

    .. automethod:: force


.. autoclass:: NormalStress

    .. autoproperty:: mag

    .. automethod:: energy

    .. automethod:: force


.. autoclass:: PerimeterConstraint

    .. autoproperty:: lam

    .. autoproperty:: constr

    .. automethod:: energy

    .. automethod:: force


.. autoclass:: SurfaceAreaConstraint

    .. autoproperty:: lam

    .. autoproperty:: constr

    .. automethod:: energy

    .. automethod:: force


.. autoclass:: SurfaceTraction

    .. autoproperty:: comps

    .. automethod:: energy

    .. automethod:: force


.. autoclass:: VolumeConstraint

    .. autoproperty:: lam

    .. autoproperty:: constr

    .. automethod:: energy

    .. automethod:: force
