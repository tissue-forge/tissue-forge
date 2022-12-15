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

%module vertex_solver

%{

#include <models/vertex/solver/tfMeshObj.h>
#include <models/vertex/solver/tf_mesh_metrics.h>
%}

// Helper functions to access object members

%inline %{

static int _vertex_solver_MeshObj_getObjId(const TissueForge::models::vertex::MeshObj *self) {
    return self->objId;
}

static bool _vertex_solver_MeshObj_in(const TissueForge::models::vertex::MeshObj *self, const TissueForge::models::vertex::MeshObj *obj) {
    return self->in(obj);
}

static bool _vertex_solver_MeshObj_has(const TissueForge::models::vertex::MeshObj *self, const TissueForge::models::vertex::MeshObj *obj) {
    return self->has(obj);
}

static int _vertex_solver_MeshObjType_getId(const TissueForge::models::vertex::MeshObjType *self) {
    return self->id;
}

%}


// todo: correct so that this isn't necessary
%ignore TissueForge::models::vertex::MeshObjActor::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::MeshObjActor::force(const MeshObj *, const MeshObj *, FloatP_t *);

%rename(_vertex_solver_edgeStrain) TissueForge::models::vertex::edgeStrain;
%rename(_vertex_solver_vertexStrain) TissueForge::models::vertex::vertexStrain;


%import <models/vertex/solver/tfMeshObj.h>

%define vertex_solver_MeshObj_extend_py(name) 

%extend name {
    %pythoncode %{
        def __lt__(self, rhs) -> bool:
            return self.id < rhs.id

        def __gt__(self, rhs) -> bool:
            return rhs < self

        def __le__(self, rhs) -> bool:
            return not (self > rhs)

        def __ge__(self, rhs) -> bool:
            return not (self < rhs)

        def __eq__(self, rhs) -> bool:
            return self.id == rhs.id

        def __ne__(self, rhs) -> bool:
            return not (self == rhs)

        def __str__(self) -> str:
            return self.str()

        @property
        def id(self) -> int:
            return _vertex_solver_MeshObj_getObjId(self)

        def is_in(self, _obj) -> bool:
            return _vertex_solver_MeshObj_in(self, _obj)

        def has(self, _obj) -> bool:
            return _vertex_solver_MeshObj_has(self, _obj)
    %}
}

%enddef

%define vertex_solver_MeshObjType_extend_py(name) 

%extend name {
    %pythoncode %{
        @property
        def registered(self) -> bool:
            """Tests whether this type is registered"""
            return self._isRegistered()

        @property
        def instances(self):
            """List of instances that belong to this type"""
            return self._getInstances()

        @property
        def instance_ids(self):
            """List of instance ids that belong to this type"""
            return self._getInstanceIds()

        @property
        def num_instances(self) -> int:
            """Number of instances that belong to this type"""
            return self._getNumInstances()

        def __len__(self) -> int:
            return self.num_instances

        def __getitem__(self, index: int):
            return self.instances[index]

        def __contains__(self, item):
            return item in self.instances

        def __lt__(self, rhs) -> bool:
            return self.id < rhs.id

        def __gt__(self, rhs) -> bool:
            return rhs < self

        def __le__(self, rhs) -> bool:
            return not (self > rhs)

        def __ge__(self, rhs) -> bool:
            return not (self < rhs)

        def __eq__(self, rhs) -> bool:
            return self.id == rhs.id

        def __ne__(self, rhs) -> bool:
            return not (self == rhs)

        def __str__(self) -> str:
            return self.str()

        @property
        def id(self) -> int:
            return _vertex_solver_MeshObjType_getId(self)
    %}
}

%enddef

%include "tfMeshLogger.i"
%include "tfVertex.i"
%include "tfSurface.i"
%include "tfBody.i"
%include "tfMeshQuality.i"
%include "tfMesh.i"
%include "tfMeshSolver.i"
%include "tf_mesh_bind.i"
%include "tf_mesh_create.i"

%include <models/vertex/solver/tf_mesh_metrics.h>

%include "actors/tf_actors.i"
