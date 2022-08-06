.. _making_things_move:

.. py:currentmodule:: tissue_forge

Making Things Move
------------------

Tissue Forge updates the velocity and position of an object by calculating the
net force acting on it. Working with forces in Tissue Forge is very flexible,
whether using one of the variety of built-in :ref:`potentials <potentials>`
and :ref:`forces <forces>` provided by Tissue Forge or designing and
implementing a custom force.

Conservative forces are usually a kind of :py:attr:`Potential` object, where the
force is described in terms of its potential energy function. Long-range,
fluid, and most bonded interactions are examples of forces based on
conservative potential energy functions. All potential-based forces contribute
to the total potential energy of a system in a simulation.

How forces affect the trajectory of a particle occurs according to the dynamics
of the type of the particle. Tissue Forge supports simulating Newtonian and
Langevin (overdamped) mechanics on the basis of individual particles, which is
described using the particle type attribute ``dynamics`` and defaults to
Newtonian. In general, integrating the universe in time consists of updating
the position :math:`\mathbf{r}_i` of each :math:`i\mathrm{th}` particle according
to its mass :math:`m_i` and the total force exerted on it :math:`\mathbf{f}_i`.

For Newtonian mechanics, particle acceleration is proportional to total force,

.. math::

    f_i = m_i \frac{d^2 \mathbf{r}_i} {dt^2}

For overdamped mechanics, particle velocity is proportional to total force,

.. math::

    f_i = m_i \frac{d \mathbf{r}_i} {dt}

In both cases, Tissue Forge uses the current :attr:`mass <ParticleHandle.mass>`
of each particle.
