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

#include <models/vertex/solver/tf_mesh_bind.h>
%}

%rename(_vertex_solver_bind_structure_type) TissueForge::models::vertex::bind::structure(MeshObjActor*, StructureType*);
%rename(_vertex_solver_bind_structure_inst) TissueForge::models::vertex::bind::structure(MeshObjActor*, Structure*);
%rename(_vertex_solver_bind_body_type) TissueForge::models::vertex::bind::body(MeshObjActor*, BodyType*);
%rename(_vertex_solver_bind_body_inst) TissueForge::models::vertex::bind::body(MeshObjActor*, Body*);
%rename(_vertex_solver_bind_surface_type) TissueForge::models::vertex::bind::surface(MeshObjActor*, SurfaceType*);
%rename(_vertex_solver_bind_surface_inst) TissueForge::models::vertex::bind::surface(MeshObjActor*, Surface*);
%rename(_vertex_solver_bind_types) TissueForge::models::vertex::bind::types(MeshObjTypePairActor*, MeshObjType*, MeshObjType*);

%include <models/vertex/solver/tf_mesh_bind.h>
