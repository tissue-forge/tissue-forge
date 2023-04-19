.. _numerical_details:

.. py:currentmodule:: tissue_forge

Numerical Details
==================

Tissue Forge implements a number of methods, conventions and strategies to provide a
broad range of modeling and simulation features and necessary computational performance
to support both interactive and high-cost simulation execution.
This section describes the numerical details of Tissue Forge as relevant to building and
executing models and simulations.

.. _cutoff_distance:

Cutoff Distance
^^^^^^^^^^^^^^^^

Tissue Forge imposes a maximum distance within which two particles can interact, with the
exception of :ref:`bonded interactions<bonded_interactions>`, which are always imposed.
If a :ref:`potential<potentials>` or :ref:`flux<flux>` describes an interaction between the
:ref:`type(s)<particle_types>` of two particles, the interaction can only occur if the two
particles are separated by a distance less than the global cutoff distance.
The global cutoff distance can be set when :ref:`initializing Tissue Forge<running_a_sim>`,
and afterwards can be accessed but not changed (*e.g.*, :py:attr:`Universe` property
:attr:`cutoff <Universe.cutoff>` in Python).

- In Python, the cutoff distance can be set with the :func:`init` keyword argument ``cutoff``.
- In C++, the cutoff distance can be set with the ``Universe::Config`` member ``cutoff``.
- In C, the cutoff distance can be set with the ``tfUniverseConfigHandle`` method ``tfUniverseConfig_setCutoff``.

.. _space_discretization:

Space Discretization
^^^^^^^^^^^^^^^^^^^^^

Much of Tissue Forge's computational performance derives from task-based parallelism.
The simulation domain of every Tissue Forge simulation is discretized into a grid of connected subspaces,
called *cells*, and interactions are calculated between particles within the same cell, and
between particles in connected cells (*i.e.*, that share at least one point).
However, interactions are not considered between particles in disconnected cells.
As such, imposing a :ref:`cutoff distance<cutoff_distance>` greater than the smallest length
of a cell can lead to unexpected behavior in that Tissue Forge will not implement some defined interactions
(*e.g.*, interactions between particles within the cutoff distance, but in disconnected cells).
Tissue Forge provides the ability to control the discretization of the simulation domain under the
requirement that the discretization must contain at least three cells along each direction, and
also with the recommendation that cells should be as cube-like as possible (cells with faces that
are significantly different in size can introduce loss of interactions, due to reasons associated
with improved computational performance).

- In Python, the discretization can be set with the :func:`init` keyword argument ``cells``.
- In C++, the discretization can be set with the ``Universe::Config`` member ``spaceGridSize``.
- In C, the discretization can be set with the ``tfUniverseConfigHandle`` method ``tfUniverseConfig_setCells``.

.. _large_particles:

Large Particles
^^^^^^^^^^^^^^^^

A *large particle* is a particle with a radius greater than the :ref:`cutoff distance<cutoff_distance>`.
Tissue Forge provides limited support for large particles, and their simulation imposes significantly more
computational cost than an ordinary particle. Large particles cannot interact with each other via
:ref:`potentials<potentials>`, and :ref:`flux<flux>` transport involving large particles does not occur.
Furthermore, a particle that is determined to be a large particle during its creation is assumed to be
a large particle for its lifetime, and likewise particles that are not large particles during their creation
are assumed to remain as not large particles.
In general, model specifications that prescribe large particles can be adjusted using a number of
mechanisms provided by Tissue Forge, including changing the size of the simulation domain, overall scale
of particle sizes and cutoff distance.

.. _flux_steps

Flux Sub-stepping
^^^^^^^^^^^^^^^^^^

Excessively fast inter-particle :ref:`flux<flux>` transport can cause numerical instabilities. 
Tissue Forge provides optional sub-stepping of inter-particle transport to support 
fast, stable transport over a simulation step. 
Flux sub-stepping breaks up the computations of inter-particle transport into equal, 
smaller periods of simulation time over a simulation step. 
For example, a simulation step period of 0.01 and ten flux sub-steps performs 
ten flux sub-steps with period 0.001 for every simulation step. 
Increase in computational cost is proportional to the number of flux sub-steps. 

- In Python, the number of flux steps per simulation step can be set with the :func:`init` keyword argument ``flux_steps``.
- In C++, the number of flux steps per simulation step can be set with the ``Universe::Config`` member ``nr_fluxsteps``.
- In C, the number of flux steps per simulation step can be set with the ``tfUniverseConfigHandle`` method ``tfUniverseConfig_setNumFluxSteps``.
