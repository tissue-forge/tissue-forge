.. _events:

.. py:currentmodule:: tissue_forge

Events
-------

An *event* is a set of procedures that occurs when a condition is satisfied (triggered).
Tissue Forge provides a robust event system with an ever-increasing library of
built-in events, as well as support for defining fully customized events.
In Tissue Forge, the procedures that correspond to an event are specified in a
user-specified, custom function. Each type of built-in event corresponds to a
particular condition by which Tissue Forge will evaluate the custom function,
as well as to a particular set of simulation information that Tissue Forge will
provide to the custom function.

The custom function that performs the set of procedures of an event is called
the *invoke method*. Aside from the condition that corresponds to a particular built-in
event, the condition of each event can be further customized by also specifying a
*predicate method*, which is another custom function that, when evaluated, tells
Tissue Forge whether the event is triggered. Both invoke and predicate methods take as
argument an instance of a specialized class of a base class :py:class:`event.Event`
(``Event`` from the ``event`` namespace in C++). In C++, pointers to invoke and
predicate methods can be created using the template ``EventMethodT``, where the template
parameter is the class of the corresponding event. Invoke methods return
``1`` if an error occurred during evaluation, and otherwise ``0``. Predicate methods
return ``1`` if the event should occur, ``0`` if the event should not occur, and a
negative value if an error occurred.

Working with Events
^^^^^^^^^^^^^^^^^^^^

In the most basic (and also most robust) case, Tissue Forge provides a basic
:py:class:`event.Event` class that, when created, is evaluated at every simulation step.
A :py:class:`event.Event` instance has no built-in predicate method, and its invoke method is
evaluated at every simulation step unless a custom predicate method is provided.
As such, the :py:class:`event.Event` class is the standard class for implementing custom
events for a particular model and simulation. An :py:class:`event.Event` instance
can be created with the top-level method :func:`event.on_event`
(``event::onEvent`` in C++). ::

    import tissue_forge as tf
    ...
    # Invoke method: destroy the first listed particle in the universe
    def destroy_first_invoke(event):
        particle = tf.Universe.particles[0]
        particle.destroy()
        return 0

    tf.event.on_event(invoke_method=destroy_first_invoke)

In this example, a particle is destroyed at every simulation step, which could be
problematic in the case where no particles exist in the universe. Assigning a predicate
method to the event could solve such a problem that describes appropriate conditions
for the event to occur, ::

    # Predicate method: first particle is destroyed if there are more than ten particles
    def destroy_first_predicate(event):
        predicate = len(tf.Universe.particles) > 10
        return int(predicate)

    tf.event.on_event(invoke_method=destroy_first_invoke,
                      predicate_method=destroy_first_predicate)

Events that are called repeatedly can also be designated for removal from the event
system using the :py:class:`event.Event` method :meth:`remove <event.EventBase.remove>` in
an invoke method. ::

    class SplittingType(tf.ParticleTypeSpec):
        pass

    splitting_type = SplittingType.get()

    # Split once each step until 100 particles are created
    def split_to_onehundred(event):
        particles = splitting_type.items()
        num_particles = len(particles)
        if num_particles == 0:
            splitting_type()
        elif num_particles >= 100:
            event.remove()
        else:
            particles[0].split()
        return 0

    tf.event.on_event(invoke_method=split_to_onehundred)

.. note::

    In Python, unhandled exceptions that occur in invoke and predicate methods only stop
    execution of the method, and might not produce any notification of an error. Basic
    exception handling is strongly encouraged to detect and report when errors occur in
    custom functions.

Timed Events
^^^^^^^^^^^^^

The built-in event :py:attr:`event.TimeEvent` (``event::TimeEvent`` in C++)
repeatedly occurs with a prescribed period. By default, the period of evaluation is
approximately implemented as the first simulation time at which at an amount
of time at least as great as the period has elapsed since the last evaluation
of the event. :py:attr:`event.TimeEvent` instances can be created with the top-level
method :func:`event.on_time` (``event::onTimeEvent`` in C++). ::

    def split_regular(event):
        splitting_type()
        return 0

    tf.event.on_time(invoke_method=split_regular, period=10.0)

The period of evaluation can also be implemented stochastically using the
optional keyword argument ``distribution``, which names a built-in distribution
by which Tissue Forge will generate the next time of evaluation from the event
period. Currently, Tissue Forge supports the Poisson distribution, which has
the name `"exponential"`. ::

    def split_random(event):
        splitting_type()
        return 0

    tf.event.on_time(invoke_method=split_random, period=10.0, distribution="exponential")

:py:class:`event.TimeEvent` instances can also be generated for only a particular period
in simulation. The optional keyword argument ``start_time`` (default 0.0)
defines the first time in simulation when the event can occur, and the optional
keyword argument ``end_time`` (default forever) defines the last time in
simulation when the event can occur. ::

    def destroy_for_a_while(event):
        particles = splitting_type.items()
        if len(particles) > 0:
            particles[0].destroy()
        return 0

    tf.event.on_time(invoke_method=destroy_for_a_while, period=10.0,
                     start_time=20.0, end_time=30.0)

Events with Particles
^^^^^^^^^^^^^^^^^^^^^^

Tissue Forge provides built-in events that operate on individual particles on
the basis of particle type. In addition to working with a custom invoke
method and optional predicate method, particle events select a particle
from a prescribed particle type. These event instances have the attributes
:attr:`targetType` and :attr:`targetParticle` that are set to the particle
type and particle that correspond to an event.

The :py:class:`event.ParticleEvent` (``event::ParticleEvent``) is a particle event
that functions much the same as :py:class:`event.Event`. A :py:class:`event.ParticleEvent`
instance has an invoke method and optional predicate method, and is
evaluated at every simulation step. However, a :py:class:`event.ParticleEvent`
instance also has an associated particle type and, on evaluation, an
associated particle. :py:class:`event.ParticleEvent` instances can be created with
the top-level method :func:`on_particle` (``event::onParticleEvent`` in C++). ::

    def split_selected(event):
        selected_particle = event.targetParticle
        selected_particle.split()
        return 0

    tf.event.on_particle(splitting_type, invoke_method=split_selected)

By default, a particle is randomly selected during the evaluation of a
particle event according to a uniform distribution. The largest particle
(*i.e.*, the cluster with the most constituent particles) can also be selected
using the optional keyword argument ``selector`` and passing ``"largest"``. ::

    def invoke_destroy_largest(event):
        event.targetParticle.destroy()
        return 0

    tf.event.on_particle(splitting_type, invoke_method=invoke_destroy_largest,
                         selector="largest")

The particle event :py:class:`event.ParticleTimeEvent` (:class:`event::ParticleTimeEvent` in C++)
functions is a combination of :py:class:`event.TimeEvent` and
:py:class:`event.ParticleEvent`, and can be created with the top-level method
:func:`event.on_particletime` (``event::onParticleTimeEvent`` in C++) with all
of the combined corresponding arguments. ::

    def split_selected_later(event):
        event.targetParticle.split()
        return 0

    tf.event.on_particletime(splitting_type, period=10.0,
                             invoke_method=split_selected_later, start_time=20.0)

.. _events_input_driven:

Input-Driven Events
^^^^^^^^^^^^^^^^^^^^

Tissue Forge provides an event :py:class:`event.KeyEvent` (``event::KeyEvent`` in C++) that
occurs each time a key on the keyboard is pressed. :py:class:`event.KeyEvent` instances
do not support a custom predicate method. The name of the key that triggered
the event is available as the :py:class:`event.KeyEvent` string attribute
:attr:`key_name <event.KeyEvent.key_name>`, as are key modifier flags
(``Alt``: :attr:`key_alt <event.KeyEvent.key_alt>`,
``Ctrl``: :attr:`key_ctrl <event.KeyEvent.key_ctrl>`,
``Shift``: :attr:`key_shift <event.KeyEvent.key_shift>`).
One :py:class:`event.KeyEvent` instance in Python can be
created with the top-level method :func:`event.on_keypress`. In C++, an arbitrary number of
invoke methods can be assigned as keyboard callbacks using the static method
``event::KeyEvent::addHandler``. ::

    # key "d" destroys a particle; key "c" creates a particle
    def do_key_actions(event):
        if event.key_name == "d":
            particles = splitting_type.items()
            if len(particles) > 0:
                particles[0].destroy()
        elif event.key_name == "c":
            splitting_type()
        return 0

    tf.on_keypress(do_key_actions)
