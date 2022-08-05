.. _universe:

.. py:currentmodule:: tissue_forge

The Tissue Forge Universe
--------------------------

The universe is everything according to a model as instructed to Tissue Forge.
It is the spacetime of a simulation and contains every model object therein,
where objects interact, processes are performed, and events occur.
The universe can be thought of as the repository for all representations of
physical objects in a simulation.

Very little exists in the universe unless it is asserted to Tissue Forge.
The universe does not exist without space and a description of its
boundaries, and a time is always assigned to the current configuration
of the universe (even if it is empty). Otherwise, the universe according
to Tissue Forge is a void unless instructed otherwise.

The universe is automatically created when :ref:`initializing a simulation <running_a_sim>`,
and exactly one universe always exists. When objects are created and
assigned a position (*e.g.*, when creating
:ref:`particles <creating_particles_and_types>`),
they are placed at the prescribed position in the universe according to a
Cartesian :ref:`coordinate system<coordinate_systems>`.
Whatever objects may be in the universe, their dynamics will only occur
if the universe is :ref:`integrated in time <running_a_sim>`.
What occurs when an object attempts to cross a boundary of the universe
is governed by prescribed :ref:`boundary conditions <boundary>`. The universe
has no intrinsic units except those implied by the objects and interactions
it contains.

For convenience, the universe can always be accessed as a top-level variable
:py:attr:`Universe` in Python.
The universe contains a number of useful methods that
provide information about both the universe and everything in it in its
current configuration. For example, the current time and time step can be
accessed with the universe properties :attr:`time <Universe.time>` and
:attr:`dt <Universe.dt>`, respectively (static methods :meth:`Universe::getTime`
and :meth:`Universe::getDt`, respectively, in C++). Likewise, an inventory of
all particles and bonds can be :ref:`accessed <accessing>` with the universe methods
:meth:`particles <Universe.particles>` and :meth:`bonds <Universe.bonds>`.
