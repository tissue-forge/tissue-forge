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
%rename(find_from_name) TissueForge::models::vertex::SurfaceType::findFromName;
%rename(register_type) TissueForge::models::vertex::SurfaceType::registerType;
%rename(n_polygon) TissueForge::models::vertex::SurfaceType::nPolygon;

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
    %}
}
