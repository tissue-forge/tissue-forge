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

#include <models/vertex/solver/tfVertex.h>
%}

%template(vectorMeshVertex) std::vector<TissueForge::models::vertex::Vertex*>;

%rename(_vertex_solver_Vertex) TissueForge::models::vertex::Vertex;
%rename(_vertex_solver__MeshParticleType_get) TissueForge::models::vertex::MeshParticleType_get;

%include <models/vertex/solver/tfVertex.h>

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Vertex)

%extend TissueForge::models::vertex::Vertex {
    %pythoncode %{
        @property
        def structures(self):
            return self.getStructures()

        @property
        def bodies(self):
            return self.getBodies()

        @property
        def surfaces(self):
            return self.getSurfaces()

        @property
        def neighbor_vertices(self):
            return self.neighborVertices()

        @property
        def volume(self) -> float:
            return self.getVolume()
        
        @property
        def mass(self) -> float:
            return self.getMass()

        @property
        def position(self):
            return self.getPosition()

        @position.setter
        def position(self, _position):
            self.setPosition(_position)
    %}
}
