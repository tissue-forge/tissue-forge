.. _become:

.. py:currentmodule:: tissue_forge

Changing Type
=============

One of the hallmarks of biological cells, and biological objects in general is
that they change *phenotype* over time. Tissue Forge implements the concept of a
particle phenotype analogously to a class in a programming language (*i.e.*, the
particle type): for a particular phenotype, we have an instance of an object, and the
phenotype of the object defines the set of interactions and processes in which
that object participates. Furthermore, Tissue Forge supports the process of a
change in particle type, using the particle method
:py:meth:`become <ParticleHandle.become>`.

:py:meth:`become <ParticleHandle.become>` changes the *type* of a particle, and
generally neglects the :ref:`state <flux>` of the particle on the condition that
the state defined in the respective particle type definitions of a particle of one
type becoming another type are compatible. If the state of the two particle types
are incompatible, then the state variables of the initial particle type that
are incompatible with the state of the new particle type are destroyed, and
the state variables of the new particle type that are incompatible with the
initial particle type are created and initialized according to their default
value. For example, ::

    import tissue_forge as tf
    class AType(tf.ParticleTypeSpec):
        species = ['S1', 'S2']

    class BType(tf.ParticleTypeSpec):
        species = ['S2', 'S3']

    A = AType.get(); B = BType.get()
    a = A();
    a.become(B)

Here the types ``A`` and ``B`` only share species ``S2``, but species
``S1`` only exists in ``A`` and ``S3`` only exists in ``B``. In such cases,
the common state variable ``S2`` is unchanged during the change from ``A`` to ``B``,
while the change destroys the species ``S1`` (because it does not exist in ``B``),
and creates a new ``S3`` and initialize it to its default value (if specified in
an initial assignment).
