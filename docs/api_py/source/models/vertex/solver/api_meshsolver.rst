.. _api_vertex_solver_meshsolver:

Solver
^^^^^^^

.. currentmodule:: tissue_forge.models.vertex.solver


.. autoclass:: MeshSolverTimers

    .. automethod:: reset

    .. automethod:: str


.. autoclass:: MeshSolver

    .. autoproperty:: timers

    .. automethod:: init

    .. automethod:: get

    .. automethod:: compact

    .. automethod:: engine_lock

    .. automethod:: engine_unlock

    .. automethod:: is_dirty

    .. automethod:: set_dirty

    .. automethod:: get_mesh

    .. automethod:: register_type

    .. automethod:: find_surface_from_name

    .. automethod:: find_body_from_name

    .. automethod:: get_body_type

    .. automethod:: get_surface_type

    .. automethod:: num_body_types

    .. automethod:: num_surface_types

    .. automethod:: num_vertices

    .. automethod:: num_surfaces

    .. automethod:: num_bodies

    .. automethod:: size_vertices

    .. automethod:: size_surfaces

    .. automethod:: size_bodies

    .. automethod:: position_changed

    .. automethod:: update

    .. automethod:: get_log

    .. automethod:: log
