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

// Mitigate clashes with core
%ignore TissueForge::models::vertex::operator<;
%ignore TissueForge::models::vertex::operator>;
%ignore TissueForge::models::vertex::operator<=;
%ignore TissueForge::models::vertex::operator>=;
%ignore TissueForge::models::vertex::operator==;
%ignore TissueForge::models::vertex::operator!=;

// Helper functions to access object members

%inline %{

static int _vertex_solver_MeshObjType_getId(const TissueForge::models::vertex::MeshObjType *self) {
    return self->id;
}

%}


%rename(_vertex_solver_edgeStrain) TissueForge::models::vertex::edgeStrain;
%rename(_vertex_solver_vertexStrain) TissueForge::models::vertex::vertexStrain;
%rename(_vertex_solver_MeshObjType) TissueForge::models::vertex::MeshObjType;
%rename(_vertex_solver_MeshObjActor) TissueForge::models::vertex::MeshObjActor;

%include <models/vertex/solver/tfMeshObj.h>

%define vertex_solver_MeshObj_prep_py(name) 

%rename(_objectId) name::objectId;

%enddef

%define vertex_solver_MeshObj_comparisons_py(name) 

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
    %}
}

%enddef

%define vertex_solver_MeshObj_extend_py(name) 

vertex_solver_MeshObj_comparisons_py(name);

%extend name {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        @property
        def id(self) -> int:
            return self._objectId()
    %}
}

%enddef

%define vertex_solver_MeshObjHandle_extend_py(name) 

vertex_solver_MeshObj_comparisons_py(name);

%extend name {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()
    %}
}

%enddef

%define vertex_solver_MeshObjType_extend_py(name) 

vertex_solver_MeshObj_comparisons_py(name)

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

        def __str__(self) -> str:
            return self.str()

        @property
        def id(self) -> int:
            return _vertex_solver_MeshObjType_getId(self)
    %}
}

%enddef

// Specify basic templates now for consistency throughout

%{

#include <models/vertex/solver/tfBody.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>
%}

%template(vectorMeshVertex) std::vector<TissueForge::models::vertex::Vertex*>;
%template(vectorMeshVertexHandle) std::vector<TissueForge::models::vertex::VertexHandle>;

%template(vectorMeshSurface) std::vector<TissueForge::models::vertex::Surface*>;
%template(vectorvectorMeshSurface) std::vector<std::vector<TissueForge::models::vertex::Surface*> >;
%template(vectorMeshSurfaceHandle) std::vector<TissueForge::models::vertex::SurfaceHandle>;
%template(vectorvectorMeshSurfaceHandle) std::vector<std::vector<TissueForge::models::vertex::SurfaceHandle> >;

%template(vectorMeshBody) std::vector<TissueForge::models::vertex::Body*>;
%template(vectorvectorvectorMeshBody) std::vector<std::vector<std::vector<TissueForge::models::vertex::Body*> > >;
%template(vectorMeshBodyHandle) std::vector<TissueForge::models::vertex::BodyHandle>;
%template(vectorvectorvectorMeshBodyHandle) std::vector<std::vector<std::vector<TissueForge::models::vertex::BodyHandle> > >;

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
