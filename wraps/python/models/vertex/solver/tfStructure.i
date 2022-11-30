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

#include <models/vertex/solver/tfStructure.h>
%}

%template(vectorMeshStructure) std::vector<TissueForge::models::vertex::Structure*>;

%rename(_vertex_solver_Structure) TissueForge::models::vertex::Structure;
%rename(_vertex_solver_StructureType) TissueForge::models::vertex::StructureType;

%include <models/vertex/solver/tfStructure.h>

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Structure)
vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::StructureType)

%extend TissueForge::models::vertex::Structure {
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
        def vertices(self):
            return self.getVertices()
    %}
}
