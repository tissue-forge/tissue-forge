/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

%{

#include <models/vertex/solver/tfSurface.h>
%}

vertex_solver_MeshObj_prep_py(TissueForge::models::vertex::Surface)

/////////////
// Surface //
/////////////

%rename(refresh_bodies) TissueForge::models::vertex::Surface::refreshBodies;
%rename(find_vertex) TissueForge::models::vertex::Surface::findVertex;
%rename(find_body) TissueForge::models::vertex::Surface::findBody;
%rename(neighbor_vertices) TissueForge::models::vertex::Surface::neighborVertices;
%rename(connected_surfaces) TissueForge::models::vertex::Surface::connectedSurfaces;
%rename(connecting_vertices) TissueForge::models::vertex::Surface::connectingVertices;
%rename(contiguous_vertex_labels) TissueForge::models::vertex::Surface::contiguousVertexLabels;
%rename(num_shared_contiguous_vertex_sets) TissueForge::models::vertex::Surface::numSharedContiguousVertexSets;
%rename(shared_contiguous_vertices) TissueForge::models::vertex::Surface::sharedContiguousVertices;
%rename(volume_sense) TissueForge::models::vertex::Surface::volumeSense;
%rename(get_volume_contr) TissueForge::models::vertex::Surface::getVolumeContr;
%rename(get_outward_normal) TissueForge::models::vertex::Surface::getOutwardNormal;
%rename(get_vertex_area) TissueForge::models::vertex::Surface::getVertexArea;
%rename(get_vertex_mass) TissueForge::models::vertex::Surface::getVertexMass;
%rename(triangle_normal) TissueForge::models::vertex::Surface::triangleNormal;
%rename(normal_distance) TissueForge::models::vertex::Surface::normalDistance;
%rename(is_outside) TissueForge::models::vertex::Surface::isOutside;
%rename(position_changed) TissueForge::models::vertex::Surface::positionChanged;

%rename(_neighborSurfaces) TissueForge::models::vertex::Surface::neighborSurfaces;
%rename(_destroy_o) TissueForge::models::vertex::Surface::destroy(Surface*);
%rename(_destroy_h) TissueForge::models::vertex::Surface::destroy(SurfaceHandle&);
%rename(_sew_o1) TissueForge::models::vertex::Surface::sew(Surface*, Surface*, const FloatP_t&);
%rename(_sew_o2) TissueForge::models::vertex::Surface::sew(std::vector<Surface*>, const FloatP_t&);
%rename(_sew_h1) TissueForge::models::vertex::Surface::sew(const SurfaceHandle&, const SurfaceHandle&, const FloatP_t&);
%rename(_sew_h2) TissueForge::models::vertex::Surface::sew(std::vector<SurfaceHandle>, const FloatP_t&);

///////////////////
// SurfaceHandle //
///////////////////

%rename(_surface) TissueForge::models::vertex::SurfaceHandle::surface;
%rename(refresh_bodies) TissueForge::models::vertex::SurfaceHandle::refreshBodies;
%rename(find_vertex) TissueForge::models::vertex::SurfaceHandle::findVertex;
%rename(find_body) TissueForge::models::vertex::SurfaceHandle::findBody;
%rename(neighbor_vertices) TissueForge::models::vertex::SurfaceHandle::neighborVertices;
%rename(connected_surfaces) TissueForge::models::vertex::SurfaceHandle::connectedSurfaces;
%rename(connecting_vertices) TissueForge::models::vertex::SurfaceHandle::connectingVertices;
%rename(contiguous_vertex_labels) TissueForge::models::vertex::SurfaceHandle::contiguousVertexLabels;
%rename(num_shared_contiguous_vertex_sets) TissueForge::models::vertex::SurfaceHandle::numSharedContiguousVertexSets;
%rename(shared_contiguous_vertices) TissueForge::models::vertex::SurfaceHandle::sharedContiguousVertices;
%rename(volume_sense) TissueForge::models::vertex::SurfaceHandle::volumeSense;
%rename(get_volume_contr) TissueForge::models::vertex::SurfaceHandle::getVolumeContr;
%rename(get_outward_normal) TissueForge::models::vertex::SurfaceHandle::getOutwardNormal;
%rename(get_vertex_area) TissueForge::models::vertex::SurfaceHandle::getVertexArea;
%rename(get_vertex_mass) TissueForge::models::vertex::SurfaceHandle::getVertexMass;
%rename(triangle_normal) TissueForge::models::vertex::SurfaceHandle::triangleNormal;
%rename(normal_distance) TissueForge::models::vertex::SurfaceHandle::normalDistance;
%rename(is_outside) TissueForge::models::vertex::SurfaceHandle::isOutside;
%rename(position_changed) TissueForge::models::vertex::SurfaceHandle::positionChanged;

%rename(_neighborSurfaces) TissueForge::models::vertex::SurfaceHandle::neighborSurfaces;

/////////////////
// SurfaceType //
/////////////////

%rename(_isRegistered) TissueForge::models::vertex::SurfaceType::isRegistered;
%rename(_getInstances) TissueForge::models::vertex::SurfaceType::getInstances;
%rename(_getInstanceIds) TissueForge::models::vertex::SurfaceType::getInstanceIds;
%rename(_getNumInstances) TissueForge::models::vertex::SurfaceType::getNumInstances;
%rename(_operator_vertices) TissueForge::models::vertex::SurfaceType::operator() (const std::vector<VertexHandle>&);
%rename(_operator_positions) TissueForge::models::vertex::SurfaceType::operator() (const std::vector<FVector3>&);
%rename(_operator_iofacedata) TissueForge::models::vertex::SurfaceType::operator() (TissueForge::io::ThreeDFFaceData*);
%rename(_n_polygon) TissueForge::models::vertex::SurfaceType::nPolygon;
%rename(_replace) TissueForge::models::vertex::SurfaceType::replace;

%rename(find_from_name) TissueForge::models::vertex::SurfaceType::findFromName;
%rename(register_type) TissueForge::models::vertex::SurfaceType::registerType;

%rename(_vertex_solver_Surface) TissueForge::models::vertex::Surface;
%rename(_vertex_solver_SurfaceHandle) TissueForge::models::vertex::SurfaceHandle;
%rename(_vertex_solver_SurfaceType) TissueForge::models::vertex::SurfaceType;

%include <models/vertex/solver/tfSurface.h>

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Surface)
vertex_solver_MeshObjHandle_extend_py(TissueForge::models::vertex::SurfaceHandle)
vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::SurfaceType)

%extend TissueForge::models::vertex::Surface {
    %pythoncode %{
        @property
        def bodies(self):
            """bodies defined by the surface"""
            return self.getBodies()

        @property
        def vertices(self):
            """vertices that define the surface"""
            return self.getVertices()

        @property
        def neighbor_surfaces(self):
            """surfaces that share at least one vertex"""
            return self._neighborSurfaces()

        @property
        def density(self):
            """density of the surface; only used in 2D simulation"""
            return self.getDensity()

        @density.setter
        def density(self, _density):
            self.setDensity(_density)

        @property
        def mass(self):
            """mass of the surface; only used in 2D simulation"""
            return self.getMass()

        @property
        def normal(self):
            """normal of the surface"""
            return self.getNormal()

        @property
        def unnormalized_normal(self):
            """unnormalized normal of the surface"""
            return self.getUnnormalizedNormal()

        @property
        def centroid(self):
            """centroid of the surface"""
            return self.getCentroid()

        @property
        def velocity(self):
            """velocity of the surface"""
            return self.getVelocity()

        @property
        def area(self):
            """area of the surface"""
            return self.getArea()

        @property
        def perimeter(self):
            """perimeter of the surface"""
            return self.getPerimeter()

        @property
        def normal_stresses(self):
            """normal stresses bound to the surface"""
            return _vertex_solver_MeshObjActor_getNormalStress(self)

        @property
        def surface_area_constraints(self):
            """surface area constraints bound to the surface"""
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def surface_tractions(self):
            """surface tractions bound to the surface"""
            return _vertex_solver_MeshObjActor_getSurfaceTraction(self)

        @property
        def edge_tensions(self):
            """edge tensions bound to the surface"""
            return _vertex_solver_MeshObjActor_getEdgeTension(self)

        @property
        def adhesions(self):
            """adhesions bound to the surface"""
            return _vertex_solver_MeshObjActor_getAdhesion(self)

        @property
        def convex_polygon_constraints(self):
            """convex polygon constraints bound to the surface"""
            return _vertex_solver_MeshObjActor_getConvexPolygonConstraint(self)

        @property
        def flat_surface_constraints(self):
            """flat surface constraints bound to the surface"""
            return _vertex_solver_MeshObjActor_getFlatSurfaceConstraint(self)

        @classmethod
        def destroy_c(cls, s):
            """
            Destroy a surface. 
            
            Any resulting vertices without a surface are also destroyed. 

            :param s: surface to destroy or its handle
            """

            if isinstance(s, _vertex_solver_Surface):
                return cls._destroy_o(s)
            elif isinstance(s, _vertex_solver_SurfaceHandle):
                return cls._destroy_h(s)
            else:
                raise TypeError

        @classmethod
        def sew(cls, s1=None, s2=None, surfs=None, dist_cf: float = None):
            """
            Sew either two surfaces or a set of surfaces. 
        
            All vertices are merged that are a distance apart less than a distance criterion. 
            
            The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 

            :param s1: the surface or its handle
            :param s2: another surface or its handle
            :param surfs: a set of surfaces or their handles
            :param dist_cf: distance criterion coefficient
            """
            func = None
            args = []

            if (s1 is not None and s2 is None) or (s1 is None and s2 is not None) or (s1 is None and surfs is None):
                raise ValueError
            elif s1 is not None and type(s1) != type(s2):
                raise TypeError

            if s1 is not None:
                args.extend([s1, s2])

                if isinstance(s1, _vertex_solver_Surface):
                    func = cls._sew_o1
                if isinstance(s1, _vertex_solver_SurfaceHandle):
                    func = cls._sew_h1
                else:
                    raise TypeError
            else:
                args.append(surfs)
                if isinstance(surfs[0], _vertex_solver_Surface):
                    func = cls._sew_o2
                elif isinstance(surfs[0], _vertex_solver_SurfaceHandle):
                    func = cls._sew_h2
                else:
                    raise TypeError

            if dist_cf is not None:
                args.append(dist_cf)

            return func(*args)
    %}
}

%extend TissueForge::models::vertex::SurfaceHandle {
    %pythoncode %{
        @property
        def surface(self):
            """underlying :class:`Surface` instance, if any"""
            return self._surface()

        @property
        def bodies(self):
            """bodies defined by the surface"""
            return self.getBodies()

        @property
        def vertices(self):
            """vertices that define the surface"""
            return self.getVertices()

        @property
        def neighbor_surfaces(self):
            """surfaces that share at least one vertex"""
            return self._neighborSurfaces()

        @property
        def density(self):
            """density of the surface; only used in 2D simulation"""
            return self.getDensity()

        @density.setter
        def density(self, _density):
            self.setDensity(_density)

        @property
        def mass(self):
            """mass of the surface; only used in 2D simulation"""
            return self.getMass()

        @property
        def normal(self):
            """normal of the surface"""
            return self.getNormal()

        @property
        def unnormalized_normal(self):
            """unnormalized normal of the surface"""
            return self.getUnnormalizedNormal()

        @property
        def centroid(self):
            """centroid of the surface"""
            return self.getCentroid()

        @property
        def velocity(self):
            """velocity of the surface"""
            return self.getVelocity()

        @property
        def area(self):
            """area of the surface"""
            return self.getArea()

        @property
        def perimeter(self):
            """perimeter of the surface"""
            return self.getPerimeter()

        @property
        def style(self):
            """style of the surface, if any"""
            return self.getStyle()

        @style.setter
        def style(self, _s):
            self.setStyle(_s)

        @property
        def normal_stresses(self):
            """normal stresses bound to the surface"""
            o = self.surface
            return _vertex_solver_MeshObjActor_getNormalStress(o) if o is not None else None

        @property
        def surface_area_constraints(self):
            """surface area constraints bound to the surface"""
            o = self.surface
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(o) if o is not None else None

        @property
        def perimeter_constraints(self):
            """perimeter constraints bound to the surface"""
            o = self.surface
            return _vertex_solver_MeshObjActor_getPerimeterConstraint(o) if o is not None else None

        @property
        def surface_tractions(self):
            """surface tractions bound to the surface"""
            o = self.surface
            return _vertex_solver_MeshObjActor_getSurfaceTraction(o) if o is not None else None

        @property
        def edge_tensions(self):
            """edge tensions bound to the surface"""
            o = self.surface
            return _vertex_solver_MeshObjActor_getEdgeTension(o) if o is not None else None

        @property
        def adhesions(self):
            """adhesions bound to the surface"""
            o = self.surface
            return _vertex_solver_MeshObjActor_getAdhesion(o) if o is not None else None

        @property
        def convex_polygon_constraints(self):
            """convex polygon constraints bound to the surface"""
            o = self.surface
            return _vertex_solver_MeshObjActor_getConvexPolygonConstraint(o) if o is not None else None

        @property
        def flat_surface_constraints(self):
            """flat surface constraints bound to the surface"""
            o = self.surface
            return _vertex_solver_MeshObjActor_getFlatSurfaceConstraint(o) if o is not None else None
    %}
}

%extend TissueForge::models::vertex::SurfaceType {
    %pythoncode %{
        @property
        def normal_stresses(self):
            """normal stresses bound to the type"""
            return _vertex_solver_MeshObjActor_getNormalStress(self)

        @property
        def surface_area_constraints(self):
            """surface area constraints bound to the type"""
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def perimeter_constraints(self):
            """perimeter constraints bound to the type"""
            return _vertex_solver_MeshObjActor_getPerimeterConstraint(self)

        @property
        def surface_tractions(self):
            """surface tractions bound to the type"""
            return _vertex_solver_MeshObjActor_getSurfaceTraction(self)

        @property
        def edge_tensions(self):
            """edge tensions bound to the type"""
            return _vertex_solver_MeshObjActor_getEdgeTension(self)

        @property
        def adhesions(self):
            """adhesions bound to the type"""
            return _vertex_solver_MeshObjActor_getAdhesion(self)

        @property
        def convex_polygon_constraints(self):
            """convex polygon constraints bound to the type"""
            return _vertex_solver_MeshObjActor_getConvexPolygonConstraint(self)

        @property
        def flat_surface_constraints(self):
            """flat surface constraints bound to the type"""
            return _vertex_solver_MeshObjActor_getFlatSurfaceConstraint(self)

        def __call__(self, vertices=None, positions=None, face_data=None):
            """
            Construct a surface of this type

            :param vertices: a list of vertices
            :param positions: a list of positions
            :param face_data: 3DF face data
            """
            result = None

            if vertices is not None:
                result = self._operator_vertices(vertices)
            elif positions is not None:
                for i, p in enumerate(positions):
                    if not isinstance(p, FVector3):
                        positions[i] = FVector3(*p)
                result = self._operator_positions(positions)
            elif face_data is not None:
                result = self._operator_iofacedata(face_data)

            return result

        def n_polygon(self, n: int, center, radius: float, ax1, ax2):
            """
            Construct a N-sided polygonal surface of this type

            :param n: number of vertices
            :param center: center of polygon
            :param radius: radius of circle for placing vertices
            :param ax1: first axis defining the plane of the polygon
            :param ax2: second axis defining the plane of the polygon
            """
            if not isinstance(center, FVector3):
                center = FVector3(*center)
            if not isinstance(ax1, FVector3):
                ax1 = FVector3(*ax1)
            if not isinstance(ax2, FVector3):
                ax2 = FVector3(*ax2)
            result = self._n_polygon(n, center, radius, ax1, ax2)
            if result is not None:
                result.thisown = 0
            return result

        def replace(self, to_replace, len_cfs):
            """
            Replace a vertex with a surface. Vertices are created for the surface along every destroyed edge.

            :param to_replace: vertex to replace
            :param len_cfs: coefficient in (0, 1) along each connected edge defining how far to construct a new vertex
            """
            result = self._replace(to_replace, len_cfs)
            if result is not None:
                result.thisown = 0
            return result
    %}
}
