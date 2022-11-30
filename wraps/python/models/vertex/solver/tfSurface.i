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
