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

%ignore TissueForge::models::vertex::Vertex::replace;
%ignore TissueForge::models::vertex::Vertex::insert;

%rename(find_surface) TissueForge::models::vertex::Vertex::findSurface;
%rename(find_body) TissueForge::models::vertex::Vertex::findBody;
%rename(shared_surfaces) TissueForge::models::vertex::Vertex::sharedSurfaces;
%rename(position_changed) TissueForge::models::vertex::Vertex::positionChanged;
%rename(update_properties) TissueForge::models::vertex::Vertex::updateProperties;
%rename(transfer_bonds_to) TissueForge::models::vertex::Vertex::transferBondsTo;

%rename(_getPartId) TissueForge::models::vertex::Vertex::getPartId;
%rename(_neighborVertices) TissueForge::models::vertex::Vertex::neighborVertices;
%rename(_replace_surface) TissueForge::models::vertex::Vertex::replace(const FVector3&, Surface*);
%rename(_replace_body) TissueForge::models::vertex::Vertex::replace(const FVector3&, Body*);
%rename(_insert_vertices) TissueForge::models::vertex::Vertex::insert(const FVector3&, Vertex*, Vertex*);
%rename(_insert_neighbors) TissueForge::models::vertex::Vertex::insert(const FVector3&, Vertex*, std::vector<Vertex*>);
%rename(_split) TissueForge::models::vertex::Vertex::split(const FVector3&);

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
        def pid(self) -> int:
            return self._getPartId()

        @property
        def neighbor_vertices(self):
            return self._neighborVertices()

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

        @staticmethod
        def replace(position, target):
            """Create a vertex at a position and replace a target surface or body with it"""
            if not isinstance(position, FVector3):
                position = FVector3(*position)

            result = None
            if isinstance(target, _vertex_solver_Surface):
                result = _vertex_solver_Vertex._replace_surface(position, target)
            elif isinstance(target, _vertex_solver_Body):
                result = _vertex_solver_Vertex._replace_body(position, target)

            if result is None:
                raise RuntimeError

            result.thisown = 0
            return result

        @staticmethod
        def insert(position, v, v2=None, nbs=None):
            """Create a vertex at a position and insert it between a vertex and either another vertex or a list of neighboring vertices"""
            if not isinstance(position, FVector3):
                position = FVector3(*position)

            result = None
            if v2 is not None:
                result = _vertex_solver_Vertex._insert_vertices(position, v, v2)
            elif nbs is not None:
                result = _vertex_solver_Vertex._insert_neighbors(position, v, nbs)

            if result is None:
                raise RuntimeError

            result.thisown = 0
            return result

        def split(self, separation):
            """
            Split a vertex into an edge
            
            The vertex must define at least one surface.
            
            New topology is governed by a cut plane at the midpoint of, and orthogonal to, the new edge. 
            Each first-order neighbor vertex is connected to the vertex of the new edge on the same side of 
            the cut plane. 
            """
            if not isinstance(separation, FVector3):
                separation = FVector3(*separation)
            result = self._split(separation)
            if result is not None:
                result.thisown = 0
            return result
    %}
}
