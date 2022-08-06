.. _binding:

.. py:currentmodule:: tissue_forge

Binding
-------

Binding objects and processes together is one of the key ways to create a
Tissue Forge simulation. Binding connects a process (*e.g.*, a
:ref:`potential <potentials>`, :ref:`force <forces>`) with one
or more objects that the process acts on.
Binding in Tissue Forge is done with static methods in the module
:py:mod:`bind` (``bind`` namespace in C++), and methods that implement
binding of processes to objects, in general, only return a code
indicating success or failure of the binding procedure.

Binding Interactions Between Particles by Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An interaction between particles by pairs of types can be implemented
using the :py:mod:`bind` method :py:meth:`types <bind.types>`. ::

    import tissue_forge as tf
    ...
    # Bind an interaction between particles of type
    #   "A" and "B" according to the potential "pot"
    tf.bind.types(pot, A, B)

Binding Interactions Between Particles by Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An interaction between two particles can be implemented
using the :py:mod:`bind` method :py:meth:`particles <bind.particles>`,
which creates a :ref:`bond <bonded_interactions>`. ::

    # Bind an interaction between particles "p0" and "p1"
    #   according to the potential "pot_bond"
    tf.bind.particles(pot_bond, p0, p1)

.. _binding_with_clusters:

Binding Interactions within and Between Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Binding an interaction between particles by pairs of types
can be particularized to only occuring between particles of
the same :ref:`cluster <clusters-label>`. The :py:mod:`bind` method
:py:meth:`types <bind.types>` provides a fourth, optional argument
``bound`` that, when set to ``True``, only binds an interaction between
particles of a pair of types that are in the same cluster. ::

    # Bind an interaction between particle types "A" and "B" in the
    #   same cluster according to potential "potb"
    tf.bind.types(potb, A, B, bound=True)

.. _binding_boundaries_and_types:

Binding Interactions with Boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tissue Forge supports enforcing :ref:`boundary conditions <boundary>` on
particles as an interaction between a particle and a boundary according
to a potential. Binding an interaction between a particle by type and a
boundary can be implemented using the :py:mod:`bind` method
:meth:`boundary_condition <bind.boundaryCondition>`. ::

    tf.init(bc={'top': 'potential'})
    ...
    # Bind an interaction between the top boundary and particle type
    #   "A" according to potential "pot"
    tf.bind.boundary_condition(pot, tf.Universe.boundary_conditions.top, A)

Binding Forces to Particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Binding a :ref:`force <forces>` to a particle type can be implemented
using the :py:mod:`bind` method :meth:`force <bind.force>`. ::

    # Bind force "f" to act on particles of type "C"
    tf.bind.force(f, C)

Binding Species to Forces
^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`Species <species-label>` can be bound to forces such that the magnitude
of the force, when applied to a particle, is multiplied by the concentration
of the species attached to a particle. In the simplest cast, a species can be
bound to a force when binding the force to a particle type. ::

    # Bind force "f1" to act on particles ot type "D", and bind species "S1" to "f1"
    tf.bind.force(f1, D, 'S1')

Binding of species to forces can occur at a second, finer level of granularity,
specifically related to force arithmetic. Since Tissue Forge supports combining forces
using addition operations (see :ref:`Creating Forces <creating_forces-label>`), it
is possible to bind different species to two forces, and then apply them both to all
particles of a particle type, ::

    # Bind species "S2" to force "f2" and species "S3" to force "f3"
    f2.bind_species('S2')
    f3.bind_species('S3')
    # Apply both forces to particle type "D"
    f23 = f2 + f3
    tf.bind.force(f23, D)

Now suppose that the combined forces ``f2`` and ``f3`` are to be applied to another
particle type, but also that a species should be bound to the result of their addition.
Tissue Forge uses the actual objects created during instantiation when they are bound to
other objects, which means that subsequent binding operations can have upstream effects
on previous binding operations. In the case of binding a species to the combined forces
``f2`` and ``f3``, binding the species to the previously bound ``f23`` would also affect
its application to all particles of type ``D``. ::

    # This affects the previous binding of "f23" to "D"
    tf.bind.force(f23, E, 'S4')

Instead, a new object must be created by addition if it is to be bound to a
particle type and exclusively bound to by a species. ::

    # Bind a species 'S4' to the sum of f2 and f3 and apply it to a particle type "E"
    # without affecting previously binding them to "D"
    f23_bound = f2 + f3
    tf.bind.force(f23_bound, E, 'S4')

This approach, when executed correctly, provides the ability to construct arbitrarily
complex hierarchies of species-regulated forces on particles according to local
chemical conditions.
