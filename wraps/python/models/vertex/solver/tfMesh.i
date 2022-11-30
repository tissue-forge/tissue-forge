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

#include <models/vertex/solver/tfMesh.h>
%}

%template(vectorMesh) std::vector<TissueForge::models::vertex::Mesh*>;

%rename(_vertex_solver_Mesh) TissueForge::models::vertex::Mesh;

%include <models/vertex/solver/tfMesh.h>

%extend TissueForge::models::vertex::Mesh {
    %pythoncode %{
        @property
        def quality(self):
            return self.getQuality()

        @quality.setter
        def quality(self, _quality):
            self.setQuality(_quality)
    %}
}
