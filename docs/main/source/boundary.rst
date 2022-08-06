.. _boundary:

Boundary Conditions
--------------------

.. py:currentmodule:: tissue_forge

Tissue Forge supports a number of boundary conditions on bases as detailed as individual
boundaries and particle types, to as generic as default conditions on all boundaries.
In Python, different boundary conditions can be specified via the argument ``bc`` to the
top-level :func:`init` method, where heterogeneous conditions are specified by
passing a dictionary, and homogeneous conditions are specified by passing a constant.
In C++, different boundary conditions can be specified by creating and configuring a
:class:`BoundaryConditionsArgsContainer` instance (defined in *tfBoundaryConditions.h*),
and then setting it on the member :attr:`universeConfig` of a ``Simulator::Config``
instance using the method :meth:`setBoundaryConditions` before
:ref:`initializing Tissue Forge <running_a_sim>` with ``init``.

In general, each boundary can be referred to with the names ``"left"`` and ``"right"``
for the lower and upper boundaries along the first spatial dimension,
``"bottom"`` and ``"top"`` for the lower and upper boundaries along the
second dimension, and ``"back"`` and ``"front"`` for the lower and upper boundaries
along the third dimension. Both boundaries along the first spatial dimension
can be reffered to with the name ``"x"``, along the second dimension as ``"y"``, and
along the third dimension as ``"z"``. Each type of boundary condition also has a
designated name, which can be referred to using a string, as well as a constant.

Periodic
^^^^^^^^^

The *periodic boundary condition* effectively simulates an infinite domain where any
agents that leave one side automatically appears at the opposite boundary. Also, agents
near a boundary can interact with the agents near the opposite boundary, (*e.g.*
a repulsive interaction can occur between agents near the left and right boundaries).
Periodic boundary conditions also determine how chemical
:ref:`fluxes operate <flux-label>`.
The periodic boundary condition can be employed using the name ``"periodic"`` and
constant ``BOUNDARY_PERIODIC``. ::

    import tissue_forge as tf
    tf.init(bc={'x':'periodic', 'z' : tf.BOUNDARY_PERIODIC})

Free-slip and No-slip
^^^^^^^^^^^^^^^^^^^^^^

*Free-slip* and *no-slip boundary conditions* reflect particles that impact a boundary
back into the simulation domain. Free-slip boundaries are essentially equivalent to a
boundary moving at the same tangential velocity *with* the simulation
objects, and can be thought of as each impacting agent colliding with an equivlent
ghost agent with the same tangent velocity at the boundary. No-slip boundaries are
equivalent to a stationary wall, in that impacting particles bounce straight back,
inverting their velocity.
Free-slip and no-slip boundary conditions can be employed using the names
``"freeslip"`` and ``"noslip"`` and constants ``BOUNDARY_FREESLIP`` and
``BOUNDARY_NO_SLIP``, respectively. ::

    tf.init(bc={'front':'freeslip', 'back' : tf.BOUNDARY_FREESLIP,
                'left': 'noslip', 'right': tf.BOUNDARY_NO_SLIP})


Velocity
^^^^^^^^^

A *velocity boundary condition* models a simulation domain with a moving boundary.
For example, the no-slip boundary condition is a particularization of the velocity boundary
conditon to zero velocity.
The velocity boundary condition can be employed with the name ``"velocity"``. ::

    tf.init(bc={'top': {'velocity': [-1, 0, 0]})
  

Potential
^^^^^^^^^^

Tissue Forge supports implementing a *potential boundary condition* as an interaction
between a boundary and a particle type according to a :ref:`potential <potentials>`.
When a boundary condition is designated as a potential,
a potential can later be :ref:`bound <binding_boundaries_and_types>`
to the boundary and types of particles.
The potential boundary condition can be employed with the name ``"potential"``
and constant ``BOUNDARY_POTENTIAL``. ::

    tf.init(bc={'top': 'potential', 'bottom': tf.BOUNDARY_POTENTIAL})
    # Bind potentials later for the top and bottom boundaries!

Reset
^^^^^^

A *reset boundary condition* is an additional condition that can be added to periodic
boundary conditions for modeling convection through a domain. When a particle crosses
a periodic boundary that has a reset condition, all :ref:`species <species-label>`
attached to the particle are reset to their initial concentration. Applications that
employ the reset boundary condition to model convection generate simulations that have
no total change in bulk material, but can have significant total chemical flux into and
out of the domain, depending on the sources, sinks and reactions in the simulation domain.
The reset boundary condition can be employed with the name ``"reset"``. ::

    # Initialize a domain like a section of a tunnel, with flow along the x-direction
    tf.init(dim=[10, 5, 5],
            bc={'x': ('periodic', 'reset'), 'y': 'no_slip', 'z': 'no_slip'})
