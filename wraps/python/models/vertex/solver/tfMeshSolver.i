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

#include <models/vertex/solver/tfMeshSolver.h>
%}

%rename(engine_lock) TissueForge::models::vertex::MeshSolver::engineLock;
%rename(engine_unlock) TissueForge::models::vertex::MeshSolver::engineUnlock;
%rename(is_dirty) TissueForge::models::vertex::MeshSolver::isDirty;
%rename(set_dirty) TissueForge::models::vertex::MeshSolver::setDirty;
%rename(new_mesh) TissueForge::models::vertex::MeshSolver::newMesh;
%rename(load_mesh) TissueForge::models::vertex::MeshSolver::loadMesh;
%rename(unload_mesh) TissueForge::models::vertex::MeshSolver::unloadMesh;
%rename(num_meshes) TissueForge::models::vertex::MeshSolver::numMeshes;
%rename(get_mesh) TissueForge::models::vertex::MeshSolver::getMesh;
%rename(register_type) TissueForge::models::vertex::MeshSolver::registerType;
%rename(find_surface_from_name) TissueForge::models::vertex::MeshSolver::findSurfaceFromName;
%rename(find_body_from_name) TissueForge::models::vertex::MeshSolver::findBodyFromName;
%rename(find_structure_from_name) TissueForge::models::vertex::MeshSolver::findStructureFromName;
%rename(get_structure_type) TissueForge::models::vertex::MeshSolver::getStructureType;
%rename(get_body_type) TissueForge::models::vertex::MeshSolver::getBodyType;
%rename(get_surface_type) TissueForge::models::vertex::MeshSolver::getSurfaceType;
%rename(num_structure_types) TissueForge::models::vertex::MeshSolver::numStructureTypes;
%rename(num_body_types) TissueForge::models::vertex::MeshSolver::numBodyTypes;
%rename(num_surface_types) TissueForge::models::vertex::MeshSolver::numSurfaceTypes;
%rename(num_vertices) TissueForge::models::vertex::MeshSolver::numVertices;
%rename(num_surfaces) TissueForge::models::vertex::MeshSolver::numSurfaces;
%rename(num_bodies) TissueForge::models::vertex::MeshSolver::numBodies;
%rename(num_structures) TissueForge::models::vertex::MeshSolver::numStructures;
%rename(size_vertices) TissueForge::models::vertex::MeshSolver::sizeVertices;
%rename(size_surfaces) TissueForge::models::vertex::MeshSolver::sizeSurfaces;
%rename(size_bodies) TissueForge::models::vertex::MeshSolver::sizeBodies;
%rename(size_structures) TissueForge::models::vertex::MeshSolver::sizeStructures;
%rename(get_log) TissueForge::models::vertex::MeshSolver::getLog;

%rename(_vertex_solver_MeshSolver) TissueForge::models::vertex::MeshSolver;

%include <models/vertex/solver/tfMeshSolver.h>
