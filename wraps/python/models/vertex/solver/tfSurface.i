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

%template(vectorMeshSurface) std::vector<TissueForge::models::vertex::Surface*>;
%template(vectorvectorMeshSurface) std::vector<std::vector<TissueForge::models::vertex::Surface*> >;

%rename(_extend) TissueForge::models::vertex::Surface::extend(const unsigned int&, const FVector3&);
%rename(_extrude) TissueForge::models::vertex::Surface::extrude(const unsigned int&, const FloatP_t&);
%rename(_split_vertices) TissueForge::models::vertex::Surface::split(Vertex*, Vertex*);
%rename(_split_cpvecs) TissueForge::models::vertex::Surface::split(const FVector3&, const FVector3&);
%rename(destroy_c) TissueForge::models::vertex::Surface::destroy(Surface*);
%rename(refresh_bodies) TissueForge::models::vertex::Surface::refreshBodies;
%rename(find_vertex) TissueForge::models::vertex::Surface::findVertex;
%rename(find_body) TissueForge::models::vertex::Surface::findBody;
%rename(neighbor_vertices) TissueForge::models::vertex::Surface::neighborVertices;
%rename(neighbor_surfaces) TissueForge::models::vertex::Surface::neighborSurfaces;
%rename(connected_surfaces) TissueForge::models::vertex::Surface::connectedSurfaces;
%rename(contiguous_edge_labels) TissueForge::models::vertex::Surface::contiguousEdgeLabels;
%rename(num_shared_contiguous_edges) TissueForge::models::vertex::Surface::numSharedContiguousEdges;
%rename(volume_sense) TissueForge::models::vertex::Surface::volumeSense;
%rename(get_volume_contr) TissueForge::models::vertex::Surface::getVolumeContr;
%rename(get_outward_normal) TissueForge::models::vertex::Surface::getOutwardNormal;
%rename(get_vertex_area) TissueForge::models::vertex::Surface::getVertexArea;
%rename(triangle_normal) TissueForge::models::vertex::Surface::triangleNormal;
%rename(normal_distance) TissueForge::models::vertex::Surface::normalDistance;
%rename(is_outside) TissueForge::models::vertex::Surface::isOutside;
%rename(position_changed) TissueForge::models::vertex::Surface::positionChanged;

%rename(_isRegistered) TissueForge::models::vertex::SurfaceType::isRegistered;
%rename(_getInstances) TissueForge::models::vertex::SurfaceType::getInstances;
%rename(_getInstanceIds) TissueForge::models::vertex::SurfaceType::getInstanceIds;
%rename(_getNumInstances) TissueForge::models::vertex::SurfaceType::getNumInstances;
%rename(_operator_vertices) TissueForge::models::vertex::SurfaceType::operator() (std::vector<Vertex*>);
%rename(_operator_positions) TissueForge::models::vertex::SurfaceType::operator() (const std::vector<FVector3>&);
%rename(_operator_iofacedata) TissueForge::models::vertex::SurfaceType::operator() (io::ThreeDFFaceData*);
%rename(_n_polygon) TissueForge::models::vertex::SurfaceType::nPolygon;
%rename(_replace) TissueForge::models::vertex::SurfaceType::replace;

%rename(find_from_name) TissueForge::models::vertex::SurfaceType::findFromName;
%rename(register_type) TissueForge::models::vertex::SurfaceType::registerType;

%rename(_vertex_solver_Surface) TissueForge::models::vertex::Surface;
%rename(_vertex_solver_SurfaceType) TissueForge::models::vertex::SurfaceType;

%include <models/vertex/solver/tfSurface.h>

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Surface)
vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::SurfaceType)

%extend TissueForge::models::vertex::Surface {
    %pythoncode %{
        @property
        def structures(self):
            return self.getStructures()

        @property
        def bodies(self):
            return self.getBodies()

        @property
        def vertices(self):
            return self.getVertices()

        @property
        def neighbor_surfaces(self):
            return self.neighborSurfaces()

        @property
        def normal(self):
            return self.getNormal()

        @property
        def centroid(self):
            return self.getCentroid()

        @property
        def velocity(self):
            return self.getVelocity()

        @property
        def area(self):
            return self.getArea()

        @property
        def normal_stresses(self):
            return _vertex_solver_MeshObjActor_getNormalStress(self)

        @property
        def surface_area_constraints(self):
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def surface_tractions(self):
            return _vertex_solver_MeshObjActor_getSurfaceTraction(self)

        @property
        def edge_tensions(self):
            return _vertex_solver_MeshObjActor_getEdgeTension(self)

        @property
        def adhesions(self):
            return _vertex_solver_MeshObjActor_getAdhesion(self)

        def extend(self, vert_idx_start: int, position):
            """
            Create a surface from two vertices and a position

            :param vert_idx_start: index of first vertex; the next vertex is the second vertex
            :param position: position of third vertex
            """
            if not isinstance(position, FVector3):
                position = FVector3(*position)
            result = self._extend(vert_idx_start, position)
            if result is not None:
                result.thisown = 0
            return result

        def extrude(self, norm_len: float):
            """
            Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface

            :param norm_len: length along the normal to extrude
            """
            result = self._extrude(norm_len)
            if result is not None:
                result.thisown = 0
            return result

        def split(self, v1=None, v2=None, cp_pos=None, cp_norm=None):
            """
            Split into two surfaces using either two vertices of the surface or a cut plane. 

            Either both vertices or the cut plane position and normal must be specified.

            :param v1: first vertex of the split
            :param v2: second vertex of the split
            :param cp_pos: point on the cut plane
            :param cp_norm: normal of the cut plane
            """
            result = None

            if v1 is not None and v2 is not None:
                result = self._split_vertices(v1, v2)
            elif cp_pos is not None and cp_norm is not None:
                if not isinstance(cp_pos, FVector3):
                    cp_pos = FVector3(*cp_pos)
                if not isinstance(cp_norm, FVector3):
                    cp_norm = FVector3(*cp_norm)
                result = self._split_cpvecs(cp_pos, cp_norm)

            if result is not None:
                result.thisown = 0
            return result
    %}
}

%extend TissueForge::models::vertex::SurfaceType {
    %pythoncode %{
        @property
        def normal_stresses(self):
            return _vertex_solver_MeshObjActor_getNormalStress(self)

        @property
        def surface_area_constraints(self):
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def surface_tractions(self):
            return _vertex_solver_MeshObjActor_getSurfaceTraction(self)

        @property
        def edge_tensions(self):
            return _vertex_solver_MeshObjActor_getEdgeTension(self)

        @property
        def adhesions(self):
            return _vertex_solver_MeshObjActor_getAdhesion(self)

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

            if result is not None:
                result.thisown = 0
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
