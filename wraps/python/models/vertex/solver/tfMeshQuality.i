/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego and Tien Comlekoglu
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

#include <models/vertex/solver/tfMeshQuality.h>
%}

%ignore TissueForge::models::vertex::MeshQualityOperation;
%ignore TissueForge::models::vertex::CustomQualityOperation;

%rename(do_quality) TissueForge::models::vertex::MeshQuality::doQuality;
%rename(exclude_vertex) TissueForge::models::vertex::MeshQuality::excludeVertex;
%rename(exclude_surface) TissueForge::models::vertex::MeshQuality::excludeSurface;
%rename(exclude_body) TissueForge::models::vertex::MeshQuality::excludeBody;
%rename(include_vertex) TissueForge::models::vertex::MeshQuality::includeVertex;
%rename(include_surface) TissueForge::models::vertex::MeshQuality::includeSurface;
%rename(include_body) TissueForge::models::vertex::MeshQuality::includeBody;

%rename(_vertex_solver_Quality) TissueForge::models::vertex::MeshQuality;

%include <models/vertex/solver/tfMeshQuality.h>

%extend TissueForge::models::vertex::MeshQuality {
    %pythoncode %{
        @property
        def vertex_merge_distance(self) -> float:
            """Distance below which two vertices are scheduled for merging"""
            return self.getVertexMergeDistance()

        @vertex_merge_distance.setter
        def vertex_merge_distance(self, _val: float):
            self.setVertexMergeDistance(_val)

        @property
        def surface_demote_area(self) -> float:
            """Area below which a surface is scheduled to become a vertex"""
            return self.getSurfaceDemoteArea()

        @surface_demote_area.setter
        def surface_demote_area(self, _val: float):
            self.setSurfaceDemoteArea(_val)

        @property
        def body_demote_volume(self) -> float:
            """Volume below which a body is scheduled to become a vertex"""
            return self.getBodyDemoteVolume()

        @body_demote_volume.setter
        def body_demote_volume(self, _val: float):
            self.setBodyDemoteVolume(_val)

        @property
        def edge_split_distance(self) -> float:
            """Distance at which two vertices are seperated when a vertex is split"""
            return self.getEdgeSplitDist()

        @edge_split_distance.setter
        def edge_split_distance(self, _val: float):
            self.setEdgeSplitDist(_val)

        @property
        def collision_2d(self) -> bool:
            """Whether 2D collisions are implemented"""
            return self.getCollision2D()

        @collision_2d.setter
        def collision_2d(self, _collision_2d: bool):
            self.setCollision2D(_collision_2d)

        def __str__(self) -> str:
            return self.str()
    %}
}
