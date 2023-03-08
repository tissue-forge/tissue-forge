/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#include "io/tfThreeDFMeshData.h"

%}


%include "io/tfThreeDFMeshData.h"

%extend TissueForge::io::ThreeDFMeshData{
    %pythoncode %{
        def is_in(self, *args, **kwargs) -> bool:
            """Alias of :meth:_in"""
            
            return self._in(*args, **kwargs)
        
        @property
        def vertices(self) -> vectorThreeDFVertexData_p:
            return self.getVertices()

        @property
        def edges(self) -> vectorThreeDFEdgeData_p:
            return self.getEdges()

        @property
        def faces(self) -> vectorThreeDFFaceData_p:
            return self.getFaces()

        @property
        def num_vertices(self) -> int:
            return self.getNumVertices()

        @property
        def num_edges(self) -> int:
            return self.getNumEdges()

        @property
        def num_faces(self) -> int:
            return self.getNumFaces()

        @property
        def centroid(self) -> FVector3:
            return self.getCentroid()
    %}
}
