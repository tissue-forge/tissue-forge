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

vertex_solver_MeshObj_prep_py(TissueForge::models::vertex::Vertex)

////////////
// Vertex //
////////////

%ignore TissueForge::models::vertex::Vertex::getCachedParticleMass;

%rename(set_position) TissueForge::models::vertex::Vertex::setPosition(const FVector3&, const bool&);
%rename(find_surface) TissueForge::models::vertex::Vertex::findSurface;
%rename(find_body) TissueForge::models::vertex::Vertex::findBody;
%rename(update_connected_vertices) TissueForge::models::vertex::Vertex::updateConnectedVertices;
%rename(shared_surfaces) TissueForge::models::vertex::Vertex::sharedSurfaces;
%rename(position_changed) TissueForge::models::vertex::Vertex::positionChanged;
%rename(update_properties) TissueForge::models::vertex::Vertex::updateProperties;
%rename(transfer_bonds_to) TissueForge::models::vertex::Vertex::transferBondsTo;

%rename(_insert_o1) TissueForge::models::vertex::Vertex::insert(const FVector3&, Vertex*, Vertex*);
%rename(_insert_o2) TissueForge::models::vertex::Vertex::insert(const FVector3&, Vertex*, std::vector<Vertex*>);
%rename(_insert_h1) TissueForge::models::vertex::Vertex::insert(const FVector3&, const VertexHandle&, const VertexHandle&);
%rename(_insert_h2) TissueForge::models::vertex::Vertex::insert(const FVector3&, const VertexHandle&, const std::vector<VertexHandle>&);
%rename(_replace_o1) TissueForge::models::vertex::Vertex::replace(const FVector3&, Surface*);
%rename(_replace_o2) TissueForge::models::vertex::Vertex::replace(const FVector3&, Body*);
%rename(_replace_h1) TissueForge::models::vertex::Vertex::replace(const FVector3&, SurfaceHandle&);
%rename(_replace_h2) TissueForge::models::vertex::Vertex::replace(const FVector3&, BodyHandle&);
%rename(_getPartId) TissueForge::models::vertex::Vertex::getPartId;
%rename(_connectedVertices) TissueForge::models::vertex::Vertex::connectedVertices;

//////////////////
// VertexHandle //
//////////////////

%ignore TissueForge::models::vertex::VertexHandle::replace;
%ignore TissueForge::models::vertex::VertexHandle::insert;

%rename(set_position) TissueForge::models::vertex::VertexHandle::setPosition(const FVector3&, const bool&);
%rename(find_surface) TissueForge::models::vertex::VertexHandle::findSurface;
%rename(find_body) TissueForge::models::vertex::VertexHandle::findBody;
%rename(update_connected_vertices) TissueForge::models::vertex::VertexHandle::updateConnectedVertices;
%rename(shared_surfaces) TissueForge::models::vertex::VertexHandle::sharedSurfaces;
%rename(position_changed) TissueForge::models::vertex::VertexHandle::positionChanged;
%rename(update_properties) TissueForge::models::vertex::VertexHandle::updateProperties;
%rename(transfer_bonds_to) TissueForge::models::vertex::VertexHandle::transferBondsTo;

%rename(_vertex) TissueForge::models::vertex::VertexHandle::vertex;
%rename(_getPartId) TissueForge::models::vertex::VertexHandle::getPartId;
%rename(_connectedVertices) TissueForge::models::vertex::VertexHandle::connectedVertices;
%rename(_replace_body) TissueForge::models::vertex::VertexHandle::replace(const FVector3&, const BodyHandle&);


%rename(_vertex_solver_Vertex) TissueForge::models::vertex::Vertex;
%rename(_vertex_solver_VertexHandle) TissueForge::models::vertex::VertexHandle;
%rename(_vertex_solver__MeshParticleType_get) TissueForge::models::vertex::MeshParticleType_get;

%include <models/vertex/solver/tfVertex.h>

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Vertex)
vertex_solver_MeshObjHandle_extend_py(TissueForge::models::vertex::VertexHandle)

%extend TissueForge::models::vertex::Vertex {
    %pythoncode %{
        @property
        def bodies(self):
            """bodies defined by the vertex"""
            return self.getBodies()

        @property
        def surfaces(self):
            """surfaces defined by the vertex"""
            return self.getSurfaces()

        @property
        def pid(self) -> int:
            """id of underlying particle"""
            return self._getPartId()

        @property
        def connected_vertices(self):
            """connected vertices"""
            return self._connectedVertices()

        @property
        def area(self) -> float:
            """area of the vertex"""
            return self.getArea()

        @property
        def volume(self) -> float:
            """volume of the vertex"""
            return self.getVolume()
        
        @property
        def mass(self) -> float:
            """mass of the vertex"""
            return self.getMass()

        @property
        def position(self):
            """position of the vertex"""
            return self.getPosition()

        @position.setter
        def position(self, _position):
            self.set_position(_position)

        @property
        def velocity(self):
            """velocity of the vertex"""
            return self.getVelocity()

        @classmethod
        def insert_c(cls, pos, v1, v2=None, verts=None):
            """
            Create a vertex and inserts it between a vertices and either another vertex or a set of vertices.

            :param pos: position of new vertex
            :param v1: first vertex or its handle
            :param v2: second vertex of its handle
            :param verts: set of vertices
            :return: handle to newly created vertex, if any
            """
            pos = FVector3(*pos) if not isinstance(pos, FVector3) else pos
            if isinstance(v1, _vertex_solver_Vertex):
                if v2 is not None:
                    result = cls._insert_o1(pos, v1, v2)
                elif verts is not None:
                    result = cls._insert_o2(pos, v1, verts)
                else:
                    raise ValueError
            elif isinstance(v1, _vertex_solver_VertexHandle):
                if v2 is not None:
                    result = cls._insert_h1(pos, v1, v2)
                elif verts is not None:
                    result = cls._insert_h2(pos, v1, verts)
                else:
                    raise ValueError
            else:
                raise TypeError

            return result

        @classmethod
        def replace_c(cls, pos, surface=None, body=None):
            """
            Create a vertex and replace either a surface or body with it

            :param pos: position of new vertex
            :param surface: surface to replace or its handle
            :param body: body to replace or its handle
            :return: handle to newly created vertex, if any
            """
            pos = FVector3(*pos) if not isinstance(pos, FVector3) else pos
            if surface is not None:
                if isinstance(surface, _vertex_solver_Surface):
                    result = cls._replace_o1(pos, surface)
                elif isinstance(surface, _vertex_solver_SurfaceHandle):
                    result = cls._replace_h1(pos, surface)
                else:
                    raise TypeError
            elif body is not None:
                if isinstance(body, _vertex_solver_Body):
                    result = cls._replace_o2(pos, body)
                elif isinstance(body, _vertex_solver_BodyHandle):
                    result = cls._replace_h2(pos, body)
                else:
                    raise TypeError
            else:
                raise ValueError

            return result
    %}
}

%extend TissueForge::models::vertex::VertexHandle {
    %pythoncode %{
        @property
        def vertex(self):
            """underlying :class:`Vertex` instance, if any"""
            return self._vertex()

        @property
        def bodies(self):
            """bodies defined by the vertex"""
            return self.getBodies()

        @property
        def surfaces(self):
            """surfaces defined by the vertex"""
            return self.getSurfaces()

        @property
        def pid(self) -> int:
            """id of underlying particle"""
            return self._getPartId()

        @property
        def connected_vertices(self):
            """connected vertices"""
            return self._connectedVertices()

        @property
        def area(self) -> float:
            """area of the vertex"""
            return self.getArea()

        @property
        def volume(self) -> float:
            """volume of the vertex"""
            return self.getVolume()
        
        @property
        def mass(self) -> float:
            """mass of the vertex"""
            return self.getMass()

        @property
        def position(self):
            """position of the vertex"""
            return self.getPosition()

        @position.setter
        def position(self, _position):
            self.set_position(_position)

        @property
        def velocity(self):
            """velocity of the vertex"""
            return self.getVelocity()
    %}
}
