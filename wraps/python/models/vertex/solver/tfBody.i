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

#include <models/vertex/solver/tfBody.h>
%}

%template(vectorMeshBody) std::vector<TissueForge::models::vertex::Body*>;
%template(vectorvectorvectorMeshBody) std::vector<std::vector<std::vector<TissueForge::models::vertex::Body*> > >;

%rename(_vertex_solver_Body) TissueForge::models::vertex::Body;
%rename(_vertex_solver_BodyType) TissueForge::models::vertex::BodyType;

%include <models/vertex/solver/tfBody.h>

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Body)
vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::BodyType)

%extend TissueForge::models::vertex::Body {
    %pythoncode %{
        @property
        def structures(self):
            return self.getStructures()

        @property
        def surfaces(self):
            return self.getSurfaces()

        @property
        def vertices(self):
            return self.getVertices()

        @property
        def neighbor_bodies(self):
            return self.neighborBodies()

        @property
        def density(self):
            return self.getDensity()

        @density.setter
        def density(self, _density):
            self.setDensity(_density)

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
        def volume(self):
            return self.getVolume()

        @property
        def mass(self):
            return self.getMass()

        @property
        def body_forces(self):
            return _vertex_solver_MeshObjActor_getBodyForce(self)

        @property
        def surface_area_constraints(self):
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def volume_constraints(self):
            return _vertex_solver_MeshObjActor_getVolumeConstraint(self)
    %}
}

%extend TissueForge::models::vertex::BodyType {
    %pythoncode %{
        @property
        def body_forces(self):
            return _vertex_solver_MeshObjActor_getBodyForce(self)

        @property
        def surface_area_constraints(self):
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def volume_constraints(self):
            return _vertex_solver_MeshObjActor_getVolumeConstraint(self)
    %}
}
