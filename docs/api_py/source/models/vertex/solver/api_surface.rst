.. _api_vertex_solver_surface:

Surface
^^^^^^^^

.. currentmodule:: tissue_forge.models.vertex.solver


.. autoclass:: Surface

    .. autoproperty:: id

    .. autoproperty:: actors

    .. autoproperty:: style

    .. autoproperty:: bodies

    .. autoproperty:: vertices

    .. autoproperty:: neighbor_surfaces

    .. autoproperty:: density

    .. autoproperty:: normal

    .. autoproperty:: unnormalized_normal

    .. autoproperty:: centroid

    .. autoproperty:: velocity

    .. autoproperty:: area

    .. autoproperty:: perimeter

    .. autoproperty:: normal_stresses

    .. autoproperty:: perimeter_constraints

    .. autoproperty:: surface_area_constraints

    .. autoproperty:: surface_tractions

    .. autoproperty:: edge_tensions

    .. autoproperty:: adhesions

    .. autoproperty:: convex_polygon_constraints

    .. autoproperty:: flat_surface_constraints

    .. automethod:: create

    .. automethod:: defines

    .. automethod:: definedBy

    .. automethod:: destroy

    .. automethod:: destroy_c

    .. automethod:: validate

    .. automethod:: position_changed

    .. automethod:: toString

    .. automethod:: add

    .. automethod:: remove

    .. automethod:: insert

    .. automethod:: remove

    .. automethod:: replace

    .. automethod:: refresh_bodies

    .. automethod:: type

    .. automethod:: become

    .. automethod:: find_vertex

    .. automethod:: find_body

    .. automethod:: neighbor_vertices

    .. automethod:: connected_surfaces

    .. automethod:: connecting_vertices

    .. automethod:: contiguous_vertex_labels

    .. automethod:: num_shared_contiguous_vertex_sets

    .. automethod:: shared_contiguous_vertices

    .. automethod:: volume_sense

    .. automethod:: get_volume_contr

    .. automethod:: get_outward_normal

    .. automethod:: get_vertex_area

    .. automethod:: get_vertex_mass

    .. automethod:: triangle_normal

    .. automethod:: normal_distance

    .. automethod:: is_outside

    .. automethod:: contains

    .. automethod:: sew

    .. automethod:: merge

    .. automethod:: extend

    .. automethod:: extrude

    .. automethod:: split

    .. automethod:: __str__

    .. automethod:: __lt__

    .. automethod:: __gt__

    .. automethod:: __le__

    .. automethod:: __ge__

    .. automethod:: __eq__

    .. automethod:: __ne__


.. autoclass:: SurfaceHandle

    .. autoproperty:: id

    .. autoproperty:: surface

    .. autoproperty:: bodies

    .. autoproperty:: vertices

    .. autoproperty:: neighbor_surfaces

    .. autoproperty:: density

    .. autoproperty:: normal

    .. autoproperty:: unnormalized_normal

    .. autoproperty:: centroid

    .. autoproperty:: velocity

    .. autoproperty:: area

    .. autoproperty:: perimeter

    .. autoproperty:: style

    .. autoproperty:: normal_stresses

    .. autoproperty:: perimeter_constraints

    .. autoproperty:: surface_area_constraints

    .. autoproperty:: surface_tractions

    .. autoproperty:: edge_tensions

    .. autoproperty:: adhesions

    .. autoproperty:: convex_polygon_constraints

    .. autoproperty:: flat_surface_constraints

    .. automethod:: defines

    .. automethod:: definedBy

    .. automethod:: destroy

    .. automethod:: validate

    .. automethod:: position_changed

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: add

    .. automethod:: remove

    .. automethod:: insert

    .. automethod:: replace

    .. automethod:: refresh_bodies

    .. automethod:: type

    .. automethod:: become

    .. automethod:: find_vertex

    .. automethod:: find_body

    .. automethod:: neighbor_vertices

    .. automethod:: connected_surfaces

    .. automethod:: connecting_vertices

    .. automethod:: contiguous_vertex_labels

    .. automethod:: num_shared_contiguous_vertex_sets

    .. automethod:: shared_contiguous_vertices

    .. automethod:: volume_sense

    .. automethod:: get_volume_contr

    .. automethod:: get_outward_normal

    .. automethod:: get_vertex_area

    .. automethod:: get_vertex_mass

    .. automethod:: triangle_normal

    .. automethod:: normal_distance

    .. automethod:: is_outside

    .. automethod:: contains

    .. automethod:: merge

    .. automethod:: extrude

    .. automethod:: split

    .. automethod:: __str__

    .. automethod:: __lt__

    .. automethod:: __gt__

    .. automethod:: __le__

    .. automethod:: __ge__

    .. automethod:: __eq__

    .. automethod:: __ne__


.. autoclass:: SurfaceTypeSpec

    .. autoattribute:: density

    .. autoattribute:: edge_tension_lam

    .. autoattribute:: edge_tension_order

    .. autoattribute:: normal_stress_mag

    .. autoattribute:: perimeter_lam

    .. autoattribute:: perimeter_val

    .. autoattribute:: surface_area_lam

    .. autoattribute:: surface_area_val

    .. autoattribute:: surface_traction_comps

    .. autoattribute:: adhesion

    .. automethod:: get

    .. automethod:: edge_tension

    .. automethod:: normal_stress

    .. automethod:: perimeter_constaint

    .. automethod:: surface_area_constaint

    .. automethod:: surface_traction

    .. automethod:: bind_adhesion


.. autoclass:: SurfaceType(MeshObjType)

    .. autoproperty:: name

    .. autoproperty:: registered

    .. autoproperty:: style

    .. autoproperty:: density

    .. autoproperty:: normal_stresses

    .. autoproperty:: surface_area_constraints

    .. autoproperty:: surface_tractions

    .. autoproperty:: edge_tensions

    .. autoproperty:: adhesions

    .. autoproperty:: convex_polygon_constraints

    .. autoproperty:: flat_surface_constraints

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: find_from_name

    .. automethod:: register_type

    .. automethod:: get

    .. automethod:: add

    .. automethod:: remove

    .. automethod:: n_polygon

    .. automethod:: replace

    .. automethod:: __call__

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
