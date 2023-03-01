/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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

vertex_solver_MeshObj_prep_py(TissueForge::models::vertex::Body)

//////////
// Body //
//////////

%rename(_connectedBodies) TissueForge::models::vertex::Body::connectedBodies;
%rename(_adjacentBodies) TissueForge::models::vertex::Body::adjacentBodies;
%rename(update_internals) TissueForge::models::vertex::Body::updateInternals;
%rename(position_changed) TissueForge::models::vertex::Body::positionChanged;
%rename(find_vertex) TissueForge::models::vertex::Body::findVertex;
%rename(find_surface) TissueForge::models::vertex::Body::findSurface;
%rename(neighbor_surfaces) TissueForge::models::vertex::Body::neighborSurfaces;
%rename(get_vertex_area) TissueForge::models::vertex::Body::getVertexArea;
%rename(get_vertex_volume) TissueForge::models::vertex::Body::getVertexVolume;
%rename(get_vertex_mass) TissueForge::models::vertex::Body::getVertexMass;
%rename(find_interface) TissueForge::models::vertex::Body::findInterface;
%rename(contact_area) TissueForge::models::vertex::Body::contactArea;
%rename(shared_vertices) TissueForge::models::vertex::Body::sharedVertices;
%rename(is_outside) TissueForge::models::vertex::Body::isOutside;

%rename(_destroy_o) TissueForge::models::vertex::Body::destroy(Body*);
%rename(_destroy_h) TissueForge::models::vertex::Body::destroy(BodyHandle&);

////////////////
// BodyHandle //
////////////////

%rename(_body) TissueForge::models::vertex::BodyHandle::body;
%rename(_connectedBodies) TissueForge::models::vertex::BodyHandle::connectedBodies;
%rename(_adjacentBodies) TissueForge::models::vertex::BodyHandle::adjacentBodies;
%rename(update_internals) TissueForge::models::vertex::BodyHandle::updateInternals;
%rename(position_changed) TissueForge::models::vertex::BodyHandle::positionChanged;
%rename(find_vertex) TissueForge::models::vertex::BodyHandle::findVertex;
%rename(find_surface) TissueForge::models::vertex::BodyHandle::findSurface;
%rename(neighbor_surfaces) TissueForge::models::vertex::BodyHandle::neighborSurfaces;
%rename(get_vertex_area) TissueForge::models::vertex::BodyHandle::getVertexArea;
%rename(get_vertex_volume) TissueForge::models::vertex::BodyHandle::getVertexVolume;
%rename(get_vertex_mass) TissueForge::models::vertex::BodyHandle::getVertexMass;
%rename(find_interface) TissueForge::models::vertex::BodyHandle::findInterface;
%rename(contact_area) TissueForge::models::vertex::BodyHandle::contactArea;
%rename(shared_vertices) TissueForge::models::vertex::BodyHandle::sharedVertices;
%rename(is_outside) TissueForge::models::vertex::BodyHandle::isOutside;

//////////////
// BodyType //
//////////////

%rename(_isRegistered) TissueForge::models::vertex::BodyType::isRegistered;
%rename(_getInstances) TissueForge::models::vertex::BodyType::getInstances;
%rename(_getInstanceIds) TissueForge::models::vertex::BodyType::getInstanceIds;
%rename(_getNumInstances) TissueForge::models::vertex::BodyType::getNumInstances;
%rename(find_from_name) TissueForge::models::vertex::BodyType::findFromName;
%rename(register_type) TissueForge::models::vertex::BodyType::registerType;

%rename(_vertex_solver_Body) TissueForge::models::vertex::Body;
%rename(_vertex_solver_BodyHandle) TissueForge::models::vertex::BodyHandle;
%rename(_vertex_solver_BodyType) TissueForge::models::vertex::BodyType;

%include <models/vertex/solver/tfBody.h>

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Body)
vertex_solver_MeshObjHandle_extend_py(TissueForge::models::vertex::BodyHandle)
vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::BodyType)

%extend TissueForge::models::vertex::Body {
    %pythoncode %{
        @property
        def surfaces(self):
            """Surfaces that define the body"""
            return self.getSurfaces()

        @property
        def vertices(self):
            """Vertices that define the body"""
            return self.getVertices()

        @property
        def connected_bodies(self):
            """bodies in contact"""
            return self._connectedBodies()

        @property
        def adjacent_bodies(self):
            """adjacent bodies"""
            return self._adjacentBodies()

        @property
        def density(self):
            """density of the body"""
            return self.getDensity()

        @density.setter
        def density(self, _density):
            self.setDensity(_density)

        @property
        def centroid(self):
            """centroid of the body"""
            return self.getCentroid()

        @property
        def velocity(self):
            """velocity of the body"""
            return self.getVelocity()

        @property
        def area(self):
            """area of the body"""
            return self.getArea()

        @property
        def volume(self):
            """volume of the body"""
            return self.getVolume()

        @property
        def mass(self):
            """mass of the body"""
            return self.getMass()

        @property
        def body_forces(self):
            """body forces bound to the body"""
            return _vertex_solver_MeshObjActor_getBodyForce(self)

        @property
        def surface_area_constraints(self):
            """surface area constraints bound to the body"""
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def volume_constraints(self):
            """volume constraints bound to the body"""
            return _vertex_solver_MeshObjActor_getVolumeConstraint(self)

        @classmethod
        def destroy_c(cls, b):
            """
            Destroy a body. 
            
            Any resulting surfaces without a body are also destroyed. 

            :param b: body to destroy or its handle
            """
            if isinstance(b, _vertex_solver_Body):
                return cls._destroy_o(b)
            elif isinstance(b, _vertex_solver_BodyHandle):
                return cls._destroy_h(b)
            else:
                raise TypeError
    %}
}

%extend TissueForge::models::vertex::BodyHandle {
    %pythoncode %{
        @property
        def body(self):
            """underlying :class:`Body` instance, if any"""
            return self._body()

        @property
        def surfaces(self):
            """surfaces that define the body"""
            return self.getSurfaces()

        @property
        def vertices(self):
            """vertices that define the body"""
            return self.getVertices()

        @property
        def connected_bodies(self):
            """bodies in contact"""
            return self._connectedBodies()

        @property
        def adjacent_bodies(self):
            """adjacent bodies"""
            return self._adjacentBodies()

        @property
        def density(self):
            """density of the body"""
            return self.getDensity()

        @density.setter
        def density(self, _density):
            self.setDensity(_density)

        @property
        def centroid(self):
            """centroid of the body"""
            return self.getCentroid()

        @property
        def velocity(self):
            """velocity of the body"""
            return self.getVelocity()

        @property
        def area(self):
            """area of the body"""
            return self.getArea()

        @property
        def volume(self):
            """volume of the body"""
            return self.getVolume()

        @property
        def mass(self):
            """mass of the body"""
            return self.getMass()

        @property
        def body_forces(self):
            """body forces bound to the body"""
            o = self.body
            return _vertex_solver_MeshObjActor_getBodyForce(o) if o is not None else None

        @property
        def surface_area_constraints(self):
            """surface area constraints bound to the body"""
            o = self.body
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(o) if o is not None else None

        @property
        def volume_constraints(self):
            """volume constraints bound to the body"""
            o = self.body
            return _vertex_solver_MeshObjActor_getVolumeConstraint(o) if o is not None else None
    %}
}

%extend TissueForge::models::vertex::BodyType {
    %pythoncode %{
        @property
        def body_forces(self):
            """body forces bound to the type"""
            return _vertex_solver_MeshObjActor_getBodyForce(self)

        @property
        def surface_area_constraints(self):
            """surface area constraints bound to the type"""
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def volume_constraints(self):
            """volume constraints bound to the type"""
            return _vertex_solver_MeshObjActor_getVolumeConstraint(self)
    %}
}
