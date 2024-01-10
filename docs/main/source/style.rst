.. _style:

.. py:currentmodule:: tissue_forge

Style
------

All renderable objects in Tissue Forge have a ``style`` attribute, which can refer
to a :py:class:`rendering.Style` object (``Style`` in the ``rendering`` namespace in C++).
A :py:class:`rendering.Style` object behaves like a container for a variety of style
descriptors. Each instance of an object with a ``style`` automatically inherits the style of
its type, which can then be individually manipulated. The ``style`` attribute
currently supports setting the color (:meth:`setColor <rendering.Style.setColor>`) and
visibility (:meth:`setVisible <rendering.Style.setVisible>`) of its parent object.

.. _styling_particles:

Styling Particles
^^^^^^^^^^^^^^^^^^

Each :class:`ParticleType` instance is initialized with a :py:class:`rendering.Style`
instance that is stored in its :attr:`style <ParticleType.style>` attribute. By default,
all :class:`particles <Particle>` are rendered according to the style of their type,
which can be changed at any time during simulation. However, each :class:`Particle`
instance also has a style that, when set, provides instructions on how to uniquely
render the :class:`Particle` instance. The style of a :class:`Particle` can be set
using the :attr:`style <ParticleHandle.style>` attribute of its handle. ::

    import tissue_forge as tf

    class MyParticleType(tf.ParticleTypeSpec):
        pass

    my_particle_type = MyParticleType.get()
    my_particle = my_particle_type()
    my_particle.style = tf.rendering.Style()
    my_particle.style.visible = True

Styling Particle Types in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ParticleType` has a special procedure for specifying the style of
a type as a class definition in Python. The :attr:`style <ParticleType.style>`
attribute of a :class:`ParticleType` subclass can be defined in Python as a
dictionary with key-value pairs for particle type class definitions. The color
of a type can be specified with the key ``"color"`` and value of the name of a
color as a string. The visibility of a type can be specified with key
``"visible"`` and value of a Boolean. ::

    class MyBlueParticleType(tf.ParticleTypeSpec):
        style = {'color': 'CornflowerBlue', 'visible': False}

Styling Bonds
^^^^^^^^^^^^^^

:class:`Bonds <BondHandle>`, :class:`angles <AngleHandle>`, and :class:`dihedrals <DihedralHandle>`
have a :py:class:`rendering.Style` instance.
All instances of each class share the same :py:class:`rendering.Style` instance by default.
To customize the appearance of an individual bond, angle, or dihedral,
create and set a new style instance using the :ref:`same procedure as for particles <styling_particles>`.::

    # Create a bond between particles "part_a" and "part_b"
    bh: tf.BondHandle = tf.Bond.create(potential, part_a, part_b)
    # Make the bond invisible
    bond_style = tf.rendering.Style()
    bond_style.visible = False
    bh.style = bond_style
