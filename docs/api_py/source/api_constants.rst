Constants
----------

.. currentmodule:: tissue_forge

.. _geometry_constants_label:

Geometry Constants
^^^^^^^^^^^^^^^^^^^

.. autoclass:: PointsType
    :members:

Boundary Condition Constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: BOUNDARY_NONE
    :module: tissue_forge

    no boundary conditions

.. attribute:: PERIODIC_X
    :module: tissue_forge

    periodic in the x (first) direction

.. attribute:: PERIODIC_Y
    :module: tissue_forge

    periodic in the y (second) direction

.. attribute:: PERIODIC_Z
    :module: tissue_forge

    periodic in the z (third) direction

.. attribute:: PERIODIC_FULL
    :module: tissue_forge

    periodic in the all directions

.. attribute:: FREESLIP_X
    :module: tissue_forge

    free slip in the x (first) direction

.. attribute:: FREESLIP_Y
    :module: tissue_forge

    free slip in the y (second) direction

.. attribute:: FREESLIP_Z
    :module: tissue_forge

    free slip in the z (third) direction

.. attribute:: FREESLIP_FULL
    :module: tissue_forge

    free slip in the all directions

Integrator Constants
^^^^^^^^^^^^^^^^^^^^^

.. attribute:: FORWARD_EULER
    :module: tissue_forge

    Integrator constant: Forward Euler.

    Recommended, most tested, standard single-step.

.. attribute:: RUNGE_KUTTA_4
    :module: tissue_forge

    Integrator constant: Runge-Kutta.

    Experimental Runge-Kutta-4.

Particle Dynamics Constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: Newtonian
    :module: tissue_forge

    Newtonian dynamics.

.. attribute:: Overdamped
    :module: tissue_forge

    Overdamped dynamics.

Potential Constants
^^^^^^^^^^^^^^^^^^^^

.. attribute:: potential
    :module: tissue_forge.Potential.Kind

    Potential kind

.. attribute:: dpd
    :module: tissue_forge.Potential.Kind

    Dissipative particle dynamics kind
