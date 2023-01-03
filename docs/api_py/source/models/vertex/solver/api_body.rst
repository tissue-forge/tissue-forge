.. _api_vertex_solver_body:

Body
^^^^^

.. currentmodule:: tissue_forge.models.vertex.solver


.. autoclass:: Body

    .. autoproperty:: id

    .. autoproperty:: actors

    .. autoproperty:: surfaces

    .. autoproperty:: vertices

    .. autoproperty:: neighbor_bodies

    .. autoproperty:: density

    .. autoproperty:: centroid

    .. autoproperty:: velocity

    .. autoproperty:: area

    .. autoproperty:: volume

    .. autoproperty:: mass

    .. autoproperty:: body_forces

    .. autoproperty:: surface_area_constraints

    .. autoproperty:: volume_constraints

    .. automethod:: create

    .. automethod:: definedBy

    .. automethod:: destroy

    .. automethod:: validate

    .. automethod:: position_changed

    .. automethod:: toString

    .. automethod:: update_internals

    .. automethod:: add

    .. automethod:: remove

    .. automethod:: replace

    .. automethod:: type

    .. automethod:: become

    .. automethod:: find_vertex

    .. automethod:: find_surface

    .. automethod:: neighbor_surfaces

    .. automethod:: get_vertex_area

    .. automethod:: get_vertex_volume

    .. automethod:: get_vertex_mass

    .. automethod:: find_interface

    .. automethod:: contact_area

    .. automethod:: is_outside

    .. automethod:: split

    .. automethod:: destroy_c

    .. automethod:: __str__

    .. automethod:: __lt__

    .. automethod:: __gt__

    .. automethod:: __le__

    .. automethod:: __ge__

    .. automethod:: __eq__

    .. automethod:: __ne__


.. autoclass:: BodyHandle

    .. autoproperty:: id

    .. autoproperty:: body

    .. autoproperty:: surfaces

    .. autoproperty:: vertices

    .. autoproperty:: neighbor_bodies

    .. autoproperty:: density

    .. autoproperty:: centroid

    .. autoproperty:: velocity

    .. autoproperty:: area

    .. autoproperty:: volume

    .. autoproperty:: mass

    .. autoproperty:: body_forces

    .. autoproperty:: surface_area_constraints

    .. autoproperty:: volume_constraints

    .. automethod:: definedBy

    .. automethod:: destroy

    .. automethod:: validate

    .. automethod:: position_changed

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: add

    .. automethod:: remove

    .. automethod:: replace

    .. automethod:: type

    .. automethod:: become

    .. automethod:: find_vertex

    .. automethod:: find_surface

    .. automethod:: neighbor_surfaces

    .. automethod:: get_vertex_area

    .. automethod:: get_vertex_volume

    .. automethod:: get_vertex_mass

    .. automethod:: find_interface

    .. automethod:: contact_area

    .. automethod:: is_outside

    .. automethod:: split

    .. automethod:: __str__

    .. automethod:: __lt__

    .. automethod:: __gt__

    .. automethod:: __le__

    .. automethod:: __ge__

    .. automethod:: __eq__

    .. automethod:: __ne__


.. autoclass:: BodyTypeSpec

    .. autoattribute:: density

    .. autoattribute:: body_force_comps

    .. autoattribute:: surface_area_lam

    .. autoattribute:: surface_area_val

    .. autoattribute:: volume_lam

    .. autoattribute:: volume_val

    .. autoattribute:: adhesion

    .. automethod:: get

    .. automethod:: body_force

    .. automethod:: surface_area_constaint

    .. automethod:: volume_constraint

    .. automethod:: bind_adhesion


.. autoclass:: BodyType(MeshObjType)

    .. autoproperty:: name

    .. autoproperty:: registered

    .. autoproperty:: density

    .. autoproperty:: instances

    .. autoproperty:: instance_ids

    .. autoproperty:: num_instances

    .. autoproperty:: body_forces

    .. autoproperty:: surface_area_constraints

    .. autoproperty:: volume_constraints

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: find_from_name

    .. automethod:: register_type

    .. automethod:: get

    .. automethod:: add

    .. automethod:: remove

    .. automethod:: __call__

    .. automethod:: extend

    .. automethod:: extrude

    .. automethod:: __str__

    .. automethod:: __lt__

    .. automethod:: __gt__

    .. automethod:: __le__

    .. automethod:: __ge__

    .. automethod:: __eq__

    .. automethod:: __ne__

    .. automethod:: __len__

    .. automethod:: __getitem__

    .. automethod:: __contains__
