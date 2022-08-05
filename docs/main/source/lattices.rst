.. _lattices:

.. py:currentmodule:: tissue_forge

Lattices
---------

Tissue Forge provides built-in methods to quickly assemble lattices
of :ref:`particles <creating_particles_and_types>` in Python. Along
with a number of convenience methods to generate standard lattices
with particular :ref:`particle types <creating_particles_and_types>`
and :ref:`bonded interactions <bonded_interactions>`, Tissue Forge
also provides basic infrastructure to create custom unit cells
and construct lattices with them. All built-in methods are provided
in the :doc:`lattice <docs_api_py:api_lattice>` module.

In general, constructing a lattice of
:ref:`particles <creating_particles_and_types>` consists of two steps:

#. Constructing a unit cell
#. Creating the lattice

Built-in Unit Cells
^^^^^^^^^^^^^^^^^^^^

Creating a lattice with a built-in unit cell is as simple as calling
the corresponding :doc:`lattice <docs_api_py:api_lattice>` function
while providing necessary and optional specification for the cell.
For example, the method :func:`lattice.sq` constructs a two-dimensional
square unit cell. ::

    import tissue_forge as tf

    class LatticeType(tf.ParticleTypeSpec):
        """A particle type for lattices"""
        radius = 0.1

    lattice_type = LatticeType.get()  # Get the particle type
    # Construct a square unit cell with
    # - a lattice constant of 3 times the particle type radius
    # - all particles of type "LatticeType"
    unit_cell_sq = tf.lattice.sq(3*lattice_type.radius, lattice_type)

All methods that construct built-in unit cells support optional features to
add to the corresponding unit cell, like adding bonded interactions
to pairs of particles. Specifying bonded interactions among particles
in a lattice requires providing at least a callable that takes two
:py:class:`particles <ParticleHandle>` as arguments and returns a newly created
:py:class:`bond <BondHandle>`. ::

    # Create a potential for bonded interactions in a lattice
    pot = tf.Potential.power(r0=2.0 * lattice_type.radius, alpha=2)
    # Construct a body center cubic with bonded interactions
    uc_bcc = tf.lattice.bcc(3*lattice_type.radius,
                            lattice_type,
                            lambda i, j: tf.Bond.create(pot, i, j))

Each method provides even more granularity, like specifying particular
groups of bonded interactions to create or not create. ::

    # Construct a face centered cubic with only bonds between corner and center particles
    uc_fcc = tf.lattice.fcc(3*lattice_type.radius,
                            lattice_type,
                            lambda i, j: tf.Bond.create(pot, i, j),
                            (False, True))

Currently, Tissue Forge provides built-in methods to construct the following
unit cells,

* Square (2D): :func:`lattice.sq`
* Hexagonal (2D): :func:`lattice.hex2d`
* Simple cubic (3D): :func:`lattice.sc`
* Body centered cubic (3D): :func:`lattice.bcc`
* Face centered cubic (3D): :func:`lattice.fcc`
* Hexagonal close pack (3D): :func:`lattice.hcp`

For details on all built-in unit cells, see the
:doc:`Tissue Forge Python API Reference <docs_api_py:api_lattices>`.

Custom Unit Cells
^^^^^^^^^^^^^^^^^^

Custom unit cells can be designed for creating arbitrarily complex lattices
of Tissue Forge objects. Custom unit cells are created by constructing a
:py:class:`unitcell <lattice.unitcell>` instance. A
:py:class:`unitcell <lattice.unitcell>` includes a prescription of a
box that defines its spatial extent, as well as details about the
:ref:`particles <creating_particles_and_types>` that constitute it.

The box of a :py:class:`unitcell <lattice.unitcell>` is defined by
three vectors that define a right-handed coordinate system, each of which
defines the extent of the box along its particular direction such that
patterning the :py:class:`unitcell <lattice.unitcell>` along a particular
direction places a :py:class:`unitcell <lattice.unitcell>` instances at
intervals according to the vector. For example, a
:py:class:`unitcell <lattice.unitcell>` that generates a lattice with
spatial intervals of ``1``, ``2``, and ``3`` along the ``x``-, ``y``-
and ``z``-directions, respectively, has vectors ``[1, 0, 0]``,
``[0, 2, 0]`` and ``[0, 0, 3]``. The :py:class:`unitcell <lattice.unitcell>`
definition supports both two- and three-dimensional unit cells, though
even two-dimensional :py:class:`unitcell <lattice.unitcell>` instances require
three vectors for their box definition, each with three coordinates. Rather,
:py:class:`unitcell <lattice.unitcell>` instances are declared two- or
three-dimensional using the integer argument ``dimensions``.

The :py:class:`particles <ParticleHandle>` of a :py:class:`unitcell <lattice.unitcell>`
are defined by declaring the number of :py:class:`particles <ParticleHandle>` and the
position and :py:class:`type <ParticleType>` of each :py:class:`particles <ParticleHandle>`.
The position of each particle is defined with respect to the origin of the
coordinate system of the :py:class:`unitcell <lattice.unitcell>`.
For example, to create a :py:class:`unitcell <lattice.unitcell>` for a
two-dimensional square lattice with unit length of ``1``, ::

    # Construct a 2D square unit cell
    uc_sq_custom = tf.lattice.unitcell(N=1,                   # One particle
                                       a1=[1, 0, 0],          # Length 1 along x
                                       a2=[0, 1, 0],          # Length 1 along y
                                       a3=[0, 0, 1],          # Length 1 along z
                                       dimensions=2,          # 2D
                                       types=[lattice_type],  # Of type "lattice_type"
                                       position=[[0, 0, 0]])  # One particle at the origin

The :py:class:`unitcell <lattice.unitcell>` also supports embedding information
about :ref:`bonded interactions <bonded_interactions>` between
:py:class:`particles <ParticleHandle>` of each :py:class:`unitcell <lattice.unitcell>`
when used to create a lattice. Bonded interactions can be attached to a
:py:class:`unitcell <lattice.unitcell>` definition by specifying a tuple,
each element of which contains three pieces of information in a
:py:class:`BondRule <lattice.BondRule>`,

#. a callable that takes two :py:class:`particles <ParticleHandle>` as arguments
   and returns a newly created :py:class:`bond <BondHandle>`
#. a tuple of two integers identify the index of each
   :py:class:`particles <ParticleHandle>` of the bond, according to the ordering of
   arguments passed to the :py:class:`unitcell <lattice.unitcell>` constructor,
   where at least the first integer refers to a :py:class:`particle <ParticleHandle>`
   in the current :py:class:`unitcell <lattice.unitcell>`.
#. a lattice offset vector referring to the displacement from the current
   :py:class:`unitcell <lattice.unitcell>` to the :py:class:`unitcell <lattice.unitcell>`
   to which the second :py:class:`particle <ParticleHandle>` of the bond belongs,
   where an offset vector of ``[0, 0, 0]`` refers to the current
   :py:class:`unitcell <lattice.unitcell>`.

For example, to create a two-dimensional square lattice with unit length of ``1`` and
bonds between all particles in a lattice, ::

    # Create a callable for constructing uniform bonded interactions in a lattice
    bond_callable = lambda i, j: tf.Bond.create(pot, i, j)
    # Construct a 2D square unit cell with bonded interactions
    uc_sq_bonded = tf.lattice.unitcell(
        N=1,                   # One particle
        a1=[1, 0, 0],          # Length 1 along x
        a2=[0, 1, 0],          # Length 1 along y
        a3=[0, 0, 1],          # Length 1 along z
        dimensions=2,          # 2D
        types=[lattice_type],  # Of type "lattice_type"
        position=[[0, 0, 0]],  # One particle at the origin
        bonds=[                # Declare bonded interactions...
            tf.lattice.BondRule(bond_callable, (0, 0), (1, 0, 0)),  # ... +x cell
            tf.lattice.BondRule(bond_callable, (0, 0), (0, 1, 0)),  # ... +y cell
            tf.lattice.BondRule(bond_callable, (0, 0), (0, 0, 1))]  # ... +z cell
    )

Creating a Lattice
^^^^^^^^^^^^^^^^^^^

Creating a lattice with an available :py:class:`unitcell <lattice.unitcell>` is as simple
as calling :py:func:`lattice.create_lattice` while providing details about the patterning
of the lattice, and also optionally about where to place the lattice. In the simplest case,
passing a :py:class:`unitcell <lattice.unitcell>` and an integer ``n`` creates a lattice
consisting of ``n`` instances of the :py:class:`unitcell <lattice.unitcell>` in each
direction (in the ``xy`` plane for two-dimensional unit cells) centered at the center of
the :ref:`universe <universe>`. ::

    # Create a body center cubic lattice at the origin with 10 unit cells per direction
    tf.lattice.create_lattice(uc_bcc, 10)

Instead of passing an integer for the number of unit cells, the number of unit cells
can be passed per unit cell direction. ::

    # Create a square lattice with 10 cells along x, and 20 cells along y
    tf.lattice.create_lattice(unit_cell_sq, [10, 20])

Passing a position as a third optional argument instructs Tissue Forge about where to
begin constructing the lattice. ::

    # Create a face center cubic lattice beginning at (1, 2, 3) with 15 unit cells per direction
    tf.lattice.create_lattice(uc_fcc, 15, [1, 2, 3])
