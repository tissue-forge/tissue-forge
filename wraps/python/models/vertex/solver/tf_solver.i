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
#include <models/vertex/solver/tfBody.h>
#include <models/vertex/solver/tfMesh.h>
#include <models/vertex/solver/tfMeshLogger.h>
#include <models/vertex/solver/tfMeshQuality.h>
#include <models/vertex/solver/tfMeshSolver.h>
#include <models/vertex/solver/tfStructure.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tf_mesh_metrics.h>
#include <models/vertex/solver/tf_mesh_bind.h>
#include <models/vertex/solver/tf_mesh_create.h>
#include <models/vertex/solver/actors/tfBodyForce.h>
#include <models/vertex/solver/actors/tfNormalStress.h>
#include <models/vertex/solver/actors/tfSurfaceAreaConstraint.h>
#include <models/vertex/solver/actors/tfSurfaceTraction.h>
#include <models/vertex/solver/actors/tfVolumeConstraint.h>
#include <models/vertex/solver/actors/tfEdgeTension.h>
#include <models/vertex/solver/actors/tfAdhesion.h>
%}

// Helper functions to access object members

%inline %{

static int _vertex_solver_MeshObj_getObjId(const TissueForge::models::vertex::MeshObj *self) {
    return self->objId;
}

static TissueForge::models::vertex::Mesh *_vertex_solver_MeshObj_getMesh(const TissueForge::models::vertex::MeshObj *self) {
    return self->mesh;
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

%template(vectorMeshVertex) std::vector<TissueForge::models::vertex::Vertex*>;
%template(vectorMeshSurface) std::vector<TissueForge::models::vertex::Surface*>;
%template(vectorMeshBody) std::vector<TissueForge::models::vertex::Body*>;
%template(vectorMeshStructure) std::vector<TissueForge::models::vertex::Structure*>;
%template(vectorMesh) std::vector<TissueForge::models::vertex::Mesh*>;

%template(vectorvectorMeshSurface) std::vector<std::vector<TissueForge::models::vertex::Surface*> >;
%template(vectorvectorvectorMeshBody) std::vector<std::vector<std::vector<TissueForge::models::vertex::Body*> > >;


%ignore TissueForge::models::vertex::MeshQualityOperation;
%ignore TissueForge::models::vertex::CustomQualityOperation;

// todo: correct so that this block isn't necessary
%ignore TissueForge::models::vertex::MeshObjActor::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::MeshObjActor::force(const MeshObj *, const MeshObj *, FloatP_t *);
%ignore TissueForge::models::vertex::VolumeConstraint::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::VolumeConstraint::force(const MeshObj *, const MeshObj *, FloatP_t *);
%ignore TissueForge::models::vertex::SurfaceAreaConstraint::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::SurfaceAreaConstraint::force(const MeshObj *, const MeshObj *, FloatP_t *);
%ignore TissueForge::models::vertex::BodyForce::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::BodyForce::force(const MeshObj *, const MeshObj *, FloatP_t *);
%ignore TissueForge::models::vertex::SurfaceTraction::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::SurfaceTraction::force(const MeshObj *, const MeshObj *, FloatP_t *);
%ignore TissueForge::models::vertex::NormalStress::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::NormalStress::force(const MeshObj *, const MeshObj *, FloatP_t *);
%ignore TissueForge::models::vertex::EdgeTension::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::EdgeTension::force(const MeshObj *, const MeshObj *, FloatP_t *);
%ignore TissueForge::models::vertex::Adhesion::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::Adhesion::force(const MeshObj *, const MeshObj *, FloatP_t *);


%rename(_vertex_solver_Body) TissueForge::models::vertex::Body;
%rename(_vertex_solver_BodyType) TissueForge::models::vertex::BodyType;
%rename(_vertex_solver_Mesh) TissueForge::models::vertex::Mesh;
%rename(_vertex_solver_MeshSolver) TissueForge::models::vertex::MeshSolver;
%rename(_vertex_solver_Structure) TissueForge::models::vertex::Structure;
%rename(_vertex_solver_StructureType) TissueForge::models::vertex::StructureType;
%rename(_vertex_solver_Surface) TissueForge::models::vertex::Surface;
%rename(_vertex_solver_SurfaceType) TissueForge::models::vertex::SurfaceType;
%rename(_vertex_solver_Vertex) TissueForge::models::vertex::Vertex;
%rename(_vertex_solver_Logger) TissueForge::models::vertex::MeshLogger;
%rename(_vertex_solver_Quality) TissueForge::models::vertex::MeshQuality;
%rename(_vertex_solver_BodyForce) TissueForge::models::vertex::BodyForce;
%rename(_vertex_solver_NormalStress) TissueForge::models::vertex::NormalStress;
%rename(_vertex_solver_SurfaceAreaConstraint) TissueForge::models::vertex::SurfaceAreaConstraint;
%rename(_vertex_solver_SurfaceTraction) TissueForge::models::vertex::SurfaceTraction;
%rename(_vertex_solver_VolumeConstraint) TissueForge::models::vertex::VolumeConstraint;
%rename(_vertex_solver_EdgeTension) TissueForge::models::vertex::EdgeTension;
%rename(_vertex_solver_Adhesion) TissueForge::models::vertex::Adhesion;
%rename(_vertex_solver_edgeStrain) TissueForge::models::vertex::edgeStrain;
%rename(_vertex_solver_vertexStrain) TissueForge::models::vertex::vertexStrain;

%rename(_vertex_solver__MeshParticleType_get) TissueForge::models::vertex::MeshParticleType_get;

%rename(_vertex_solver__createQuadMesh_expMesh) TissueForge::models::vertex::createQuadMesh(Mesh*, SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const char*ax_1, const char*ax_2);
%rename(_vertex_solver__createQuadMesh_defMesh) TissueForge::models::vertex::createQuadMesh(SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const char*ax_1, const char*ax_2);
%rename(_vertex_solver__createPLPDMesh_expMesh) TissueForge::models::vertex::createPLPDMesh(Mesh*, BodyType*, SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const FloatP_t&, const char*, const char*);
%rename(_vertex_solver__createPLPDMesh_defMesh) TissueForge::models::vertex::createPLPDMesh(BodyType*, SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const FloatP_t&, const char*, const char*);
%rename(_vertex_solver__createHex2DMesh_expMesh) TissueForge::models::vertex::createHex2DMesh(Mesh*, SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const FloatP_t&, const char*, const char*);
%rename(_vertex_solver__createHex2DMesh_defMesh) TissueForge::models::vertex::createHex2DMesh(SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const FloatP_t&, const char*, const char*);
%rename(_vertex_solver__createHex3DMesh_expMesh) TissueForge::models::vertex::createHex3DMesh(Mesh*, BodyType*, SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const char*, const char*);
%rename(_vertex_solver__createHex3DMesh_defMesh) TissueForge::models::vertex::createHex3DMesh(BodyType*, SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const char*, const char*);


%rename(_vertex_solver_bind_structure_type) TissueForge::models::vertex::bind::structure(MeshObjActor*, StructureType*);
%rename(_vertex_solver_bind_structure_inst) TissueForge::models::vertex::bind::structure(MeshObjActor*, Structure*);
%rename(_vertex_solver_bind_body_type) TissueForge::models::vertex::bind::body(MeshObjActor*, BodyType*);
%rename(_vertex_solver_bind_body_inst) TissueForge::models::vertex::bind::body(MeshObjActor*, Body*);
%rename(_vertex_solver_bind_surface_type) TissueForge::models::vertex::bind::surface(MeshObjActor*, SurfaceType*);
%rename(_vertex_solver_bind_surface_inst) TissueForge::models::vertex::bind::surface(MeshObjActor*, Surface*);
%rename(_vertex_solver_bind_types) TissueForge::models::vertex::bind::types(MeshObjTypePairActor*, MeshObjType*, MeshObjType*);

%import <models/vertex/solver/tfMeshObj.h>

%include <models/vertex/solver/tfSurface.h>
%include <models/vertex/solver/tfVertex.h>
%include <models/vertex/solver/tfBody.h>
%include <models/vertex/solver/tfStructure.h>
%include <models/vertex/solver/tfMesh.h>
%include <models/vertex/solver/tfMeshLogger.h>
%include <models/vertex/solver/tfMeshQuality.h>
%include <models/vertex/solver/tfMeshSolver.h>
%include <models/vertex/solver/tf_mesh_bind.h>
%include <models/vertex/solver/tf_mesh_create.h>
%include <models/vertex/solver/tf_mesh_metrics.h>
%include <models/vertex/solver/actors/tfBodyForce.h>
%include <models/vertex/solver/actors/tfNormalStress.h>
%include <models/vertex/solver/actors/tfSurfaceAreaConstraint.h>
%include <models/vertex/solver/actors/tfSurfaceTraction.h>
%include <models/vertex/solver/actors/tfVolumeConstraint.h>
%include <models/vertex/solver/actors/tfEdgeTension.h>
%include <models/vertex/solver/actors/tfAdhesion.h>

%define vertex_solver_MeshObjActor_prep(baseName) 

%inline %{

static std::vector<TissueForge::models::vertex:: ## baseName ## *> _vertex_solver_MeshObjActor_get ## baseName(TissueForge::models::vertex::MeshObj *obj) {
    return TissueForge::models::vertex::MeshObjActor::get<TissueForge::models::vertex:: ## baseName ##>(obj);
}

static std::vector<TissueForge::models::vertex:: ## baseName ## *> _vertex_solver_MeshObjActor_get ## baseName(TissueForge::models::vertex::MeshObjType *objType) {
    return TissueForge::models::vertex::MeshObjActor::get<TissueForge::models::vertex:: ## baseName ##>(objType);
}

%}

%template(vectorMesh ## baseName) std::vector<TissueForge::models::vertex:: ## baseName ## *>;

%enddef

vertex_solver_MeshObjActor_prep(BodyForce)
vertex_solver_MeshObjActor_prep(NormalStress)
vertex_solver_MeshObjActor_prep(SurfaceAreaConstraint)
vertex_solver_MeshObjActor_prep(SurfaceTraction)
vertex_solver_MeshObjActor_prep(VolumeConstraint)
vertex_solver_MeshObjActor_prep(EdgeTension)
vertex_solver_MeshObjActor_prep(Adhesion)

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

        @property
        def mesh(self):
            return _vertex_solver_MeshObj_getMesh(self)

        def is_in(self, _obj) -> bool:
            return _vertex_solver_MeshObj_in(self, _obj)

        def has(self, _obj) -> bool:
            return _vertex_solver_MeshObj_has(self, _obj)
    %}
}

%enddef

vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Vertex)
vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Surface)
vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Body)
vertex_solver_MeshObj_extend_py(TissueForge::models::vertex::Structure)

%define vertex_solver_MeshObjType_extend_py(name) 

%extend name {
    %pythoncode %{
        def __len__(self) -> int:
            return self.getNumInstances()

        def __getitem__(self, index: int):
            return self.getInstances()[index]

        def __contains__(self, item):
            return item in self.getInstances()

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

vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::SurfaceType)
vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::BodyType)
vertex_solver_MeshObjType_extend_py(TissueForge::models::vertex::StructureType)

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
        def neighbor_vertices(self):
            return self.neighborVertices()

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
    %}
}

%extend TissueForge::models::vertex::Surface {
    %pythoncode %{
        @property
        def structures(self):
            return self.getStructures()

        @property
        def bodies(self):
            return self.getBodies()

        @property
        def vertices(self):
            return self.getVertices()

        @property
        def neighbor_surfaces(self):
            return self.neighborSurfaces()

        @property
        def normal(self):
            return self.getNormal()

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
        def normal_stresses(self):
            return _vertex_solver_MeshObjActor_getNormalStress(self)

        @property
        def surface_area_constraints(self):
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def surface_tractions(self):
            return _vertex_solver_MeshObjActor_getSurfaceTraction(self)

        @property
        def edge_tensions(self):
            return _vertex_solver_MeshObjActor_getEdgeTension(self)

        @property
        def adhesions(self):
            return _vertex_solver_MeshObjActor_getAdhesion(self)
    %}
}

%extend TissueForge::models::vertex::Body {
    %pythoncode %{
        @property
        def structures(self):
            return self.getStructures()

        @property
        def surfaces(self):
            return self.getSurfaces()

        @property
        def vertices(self):
            return self.getVertices()

        @property
        def neighbor_bodies(self):
            return self.neighborBodies()

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
    %}
}

%extend TissueForge::models::vertex::Structure {
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
        def vertices(self):
            return self.getVertices()
    %}
}

%extend TissueForge::models::vertex::SurfaceType {
    %pythoncode %{
        @property
        def normal_stresses(self):
            return _vertex_solver_MeshObjActor_getNormalStress(self)

        @property
        def surface_area_constraints(self):
            return _vertex_solver_MeshObjActor_getSurfaceAreaConstraint(self)

        @property
        def surface_tractions(self):
            return _vertex_solver_MeshObjActor_getSurfaceTraction(self)

        @property
        def edge_tensions(self):
            return _vertex_solver_MeshObjActor_getEdgeTension(self)

        @property
        def adhesions(self):
            return _vertex_solver_MeshObjActor_getAdhesion(self)
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
