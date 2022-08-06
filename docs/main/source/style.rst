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

Styling Particle Types in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ParticleType` has a special procedure for specifying the style of
a type as a class definition in Python. The :attr:`style <ParticleType.style>`
attribute of a :class:`ParticleType` subclass can be defined in Python as a
dictionary with key-value pairs for particle type class definitions. The color
of a type can be specified with the key ``"color"`` and value of the name of a
color as a string. The visibility of a type can be specified with key
``"visible"`` and value of a Boolean. ::

    import tissue_forge as tf

    class MyParticleType(tf.ParticleTypeSpec):
        style = {'color': 'CornflowerBlue', 'visible': False}

    my_particle_type = MyParticleType.get()
    my_particle = my_particle_type()
    my_particle.style.visible = True
