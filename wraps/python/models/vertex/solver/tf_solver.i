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
#include <models/vertex/solver/actors/tfBodyForce.h>
#include <models/vertex/solver/actors/tfNormalStress.h>
#include <models/vertex/solver/actors/tfSurfaceAreaConstraint.h>
#include <models/vertex/solver/actors/tfSurfaceTraction.h>
#include <models/vertex/solver/actors/tfVolumeConstraint.h>
#include <models/vertex/solver/actors/tfEdgeTension.h>
%}


%template(vectorMeshVertex) std::vector<TissueForge::models::vertex::Vertex*>;
%template(vectorMeshSurface) std::vector<TissueForge::models::vertex::Surface*>;
%template(vectorMeshBody) std::vector<TissueForge::models::vertex::Body*>;
%template(vectorMeshStructure) std::vector<TissueForge::models::vertex::Structure*>;
%template(vectorMesh) std::vector<TissueForge::models::vertex::Mesh*>;

%ignore TissueForge::models::vertex::MeshQualityOperation;
%ignore TissueForge::models::vertex::CustomQualityOperation;

// todo: correct so that this block isn't necessary
%ignore TissueForge::models::vertex::MeshObjActor::energy(MeshObj *, MeshObj *, float &);
%ignore TissueForge::models::vertex::MeshObjActor::force(MeshObj *, MeshObj *, float *);
%ignore TissueForge::models::vertex::VolumeConstraint::energy(MeshObj *, MeshObj *, float &);
%ignore TissueForge::models::vertex::VolumeConstraint::force(MeshObj *, MeshObj *, float *);
%ignore TissueForge::models::vertex::SurfaceAreaConstraint::energy(MeshObj *, MeshObj *, float &);
%ignore TissueForge::models::vertex::SurfaceAreaConstraint::force(MeshObj *, MeshObj *, float *);
%ignore TissueForge::models::vertex::BodyForce::energy(MeshObj *, MeshObj *, float &);
%ignore TissueForge::models::vertex::BodyForce::force(MeshObj *, MeshObj *, float *);
%ignore TissueForge::models::vertex::SurfaceTraction::energy(MeshObj *, MeshObj *, float &);
%ignore TissueForge::models::vertex::SurfaceTraction::force(MeshObj *, MeshObj *, float *);
%ignore TissueForge::models::vertex::NormalStress::energy(MeshObj *, MeshObj *, float &);
%ignore TissueForge::models::vertex::NormalStress::force(MeshObj *, MeshObj *, float *);
%ignore TissueForge::models::vertex::EdgeTension::energy(MeshObj *, MeshObj *, float &);
%ignore TissueForge::models::vertex::EdgeTension::force(MeshObj *, MeshObj *, float *);


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
%rename(_vertex_solver_edgeStrain) TissueForge::models::vertex::edgeStrain;
%rename(_vertex_solver_vertexStrain) TissueForge::models::vertex::vertexStrain;

%rename(_vertex_solver__MeshParticleType_get) TissueForge::models::vertex::MeshParticleType_get;


%rename(_vertex_solver_bind_structure_type) TissueForge::models::vertex::bind::structure(StructureType*, MeshObjActor*);
%rename(_vertex_solver_bind_structure_inst) TissueForge::models::vertex::bind::structure(Structure*, MeshObjActor*);
%rename(_vertex_solver_bind_body_type) TissueForge::models::vertex::bind::body(BodyType*, MeshObjActor*);
%rename(_vertex_solver_bind_body_inst) TissueForge::models::vertex::bind::body(Body*, MeshObjActor*);
%rename(_vertex_solver_bind_surface_type) TissueForge::models::vertex::bind::surface(SurfaceType*, MeshObjActor*);
%rename(_vertex_solver_bind_surface_inst) TissueForge::models::vertex::bind::surface(Surface*, MeshObjActor*);
%rename(_vertex_solver_bind_types) TissueForge::models::vertex::bind::types(MeshObjType*, MeshObjType*, MeshObjTypePairActor*);

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
%include <models/vertex/solver/tf_mesh_metrics.h>
%include <models/vertex/solver/actors/tfBodyForce.h>
%include <models/vertex/solver/actors/tfNormalStress.h>
%include <models/vertex/solver/actors/tfSurfaceAreaConstraint.h>
%include <models/vertex/solver/actors/tfSurfaceTraction.h>
%include <models/vertex/solver/actors/tfVolumeConstraint.h>
%include <models/vertex/solver/actors/tfEdgeTension.h>
