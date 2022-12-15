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

%rename(_neighborBodies) TissueForge::models::vertex::Body::neighborBodies;
%rename(_split) TissueForge::models::vertex::Body::split;
%rename(destroy_c) TissueForge::models::vertex::Body::destroy(Body*);
%rename(position_changed) TissueForge::models::vertex::Body::positionChanged;
%rename(find_vertex) TissueForge::models::vertex::Body::findVertex;
%rename(find_surface) TissueForge::models::vertex::Body::findSurface;
%rename(neighbor_surfaces) TissueForge::models::vertex::Body::neighborSurfaces;
%rename(get_vertex_area) TissueForge::models::vertex::Body::getVertexArea;
%rename(get_vertex_volume) TissueForge::models::vertex::Body::getVertexVolume;
%rename(get_vertex_mass) TissueForge::models::vertex::Body::getVertexMass;
%rename(find_interface) TissueForge::models::vertex::Body::findInterface;
%rename(contact_area) TissueForge::models::vertex::Body::contactArea;
%rename(is_outside) TissueForge::models::vertex::Body::isOutside;

%rename(_isRegistered) TissueForge::models::vertex::BodyType::isRegistered;
%rename(_getInstances) TissueForge::models::vertex::BodyType::getInstances;
%rename(_getInstanceIds) TissueForge::models::vertex::BodyType::getInstanceIds;
%rename(_getNumInstances) TissueForge::models::vertex::BodyType::getNumInstances;
%rename(find_from_name) TissueForge::models::vertex::BodyType::findFromName;
%rename(register_type) TissueForge::models::vertex::BodyType::registerType;

%rename(_vertex_solver_Body) TissueForge::models::vertex::Body;
%rename(_vertex_solver_BodyType) TissueForge::models::vertex::BodyType;

%include <models/vertex/solver/tfBody.h>

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Body)
vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::BodyType)

%extend TissueForge::models::vertex::Body {
    %pythoncode %{
        @property
        def surfaces(self):
            return self.getSurfaces()

        @property
        def vertices(self):
            return self.getVertices()

        @property
        def neighbor_bodies(self):
            return self._neighborBodies()

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

        def split(self, cp_pos, cp_norm, stype=None):
            """
            Split into two bodies. The split is defined by a cut plane

            :param cp_pos: position on the cut plane
            :param cp_norm: cut plane normal
            :param stype: type of newly created surface. taken from connected surfaces if not specified
            """
            if not isinstance(cp_pos, FVector3):
                cp_pos = FVector3(*cp_pos)
            if not isinstance(cp_norm, FVector3):
                cp_norm = FVector3(*cp_norm)

            result = self._split(cp_pos, cp_norm, stype)
            if result is not None:
                result.thisown = 0
            return result
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
