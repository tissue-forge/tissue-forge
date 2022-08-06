Events
-------

.. currentmodule:: tissue_forge.event

.. autoclass:: EventBase

    .. autoattribute:: last_fired

    .. autoattribute:: times_fired

    .. automethod:: remove

.. autoclass:: _Event
    :show-inheritance:

.. autoclass:: Event
    :show-inheritance:

.. autofunction:: on_event


.. autoclass:: _TimeEvent

.. autoclass:: TimeEvent
    :show-inheritance:

    .. autoattribute:: next_time

    .. autoattribute:: period

    .. autoattribute:: start_time

    .. autoattribute:: end_time

.. autofunction:: on_time


.. autoclass:: _ParticleEvent
    :show-inheritance:

    .. autoattribute:: targetType

    .. autoattribute:: targetParticle

.. autoclass:: ParticleEvent
    :show-inheritance:

.. autofunction:: on_particle


.. autoclass:: _ParticleTimeEvent
    :show-inheritance:

    .. autoattribute:: targetType

    .. autoattribute:: targetParticle

    .. autoattribute:: next_time

    .. autoattribute:: period

    .. autoattribute:: start_time

    .. autoattribute:: end_time

.. autoclass:: ParticleTimeEvent
    :show-inheritance:

.. autofunction:: on_particletime


.. autoclass:: KeyEvent

    .. autoproperty:: key_name

    .. autoproperty:: key_alt

    .. autoproperty:: key_ctrl

    .. autoproperty:: key_shift

.. autofunction:: on_keypress
