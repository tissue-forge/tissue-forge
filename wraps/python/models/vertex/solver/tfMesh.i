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

%rename(ensure_available_vertices) TissueForge::models::vertex::Mesh::ensureAvailableVertices;
%rename(ensure_available_surfaces) TissueForge::models::vertex::Mesh::ensureAvailableSurfaces;
%rename(ensure_available_bodies) TissueForge::models::vertex::Mesh::ensureAvailableBodies;
%rename(find_vertex) TissueForge::models::vertex::Mesh::findVertex;
%rename(get_vertex_by_pid) TissueForge::models::vertex::Mesh::getVertexByPID;
%rename(get_vertex) TissueForge::models::vertex::Mesh::getVertex;
%rename(get_surface) TissueForge::models::vertex::Mesh::getSurface;
%rename(get_body) TissueForge::models::vertex::Mesh::getBody;
%rename(make_dirty) TissueForge::models::vertex::Mesh::makeDirty;

%rename(_vertex_solver_Mesh) TissueForge::models::vertex::Mesh;

%include <models/vertex/solver/tfMesh.h>

%extend TissueForge::models::vertex::Mesh {
    %pythoncode %{
        @property
        def has_quality(self) -> bool:
            """Test whether the mesh has automatic quality maintenance"""
            return self.hasQuality()

        @property
        def quality_working(self) -> bool:
            """Test whether mesh quality is currently being improved"""
            return self.qualityWorking()

        @property
        def quality(self):
            """Quality maintenance"""
            return self.getQuality()

        @quality.setter
        def quality(self, _quality):
            self.setQuality(_quality)

        @property
        def num_vertices(self) -> int:
            """Number of vertices"""
            return self.numVertices()

        @property
        def num_surfaces(self) -> int:
            """Number of surfaces"""
            return self.numSurfaces()

        @property
        def num_bodies(self) -> int:
            """Number of bodies"""
            return self.numBodies()

        @property
        def size_vertices(self) -> int:
            """Size of the list of vertices"""
            return self.sizeVertices()

        @property
        def size_surfaces(self) -> int:
            """Size of the list of surfaces"""
            return self.sizeSurfaces()

        @property
        def size_bodies(self) -> int:
            """Size of the list of bodies"""
            return self.sizeBodies()

        @property
        def is_3d(self) -> bool:
            """Test whether the mesh is 3D. A 3D mesh has at least one body"""
            return self.is3D()

        def __str__(self) -> str:
            return self.str()
    %}
}
