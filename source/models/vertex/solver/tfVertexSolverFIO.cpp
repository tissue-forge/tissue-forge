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

#include "tfVertexSolverFIO.h"

#include "tfMeshSolver.h"
#include "tf_mesh_io.h"

#include <tfError.h>


using namespace TissueForge;
namespace VMod = TissueForge::models::vertex;


std::string VMod::io::VertexSolverFIOModule::moduleName() { return "VertexSolver"; }

#define TF_MESH_IOTOEASY(fe, key, member) \
    fe = new TissueForge::io::IOElement(); \
    if(TissueForge::io::toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define TF_MESH_IOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || TissueForge::io::fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

HRESULT VMod::io::VertexSolverFIOModule::toFile(const TissueForge::io::MetaData &metaData, TissueForge::io::IOElement *fileElement) {

    VMod::MeshSolver *solver = VMod::MeshSolver::get();

    if(!solver) 
        return tf_error(E_FAIL, "toFile requested without a solver");

    TissueForge::io::IOElement *fe;

    // Store types

    std::vector<VMod::BodyType> bodyTypes;
    std::vector<int> bodyTypeIds;
    for(size_t i = 0; i < solver->numBodyTypes(); i++) {
        auto o = solver->getBodyType(i);
        bodyTypes.push_back(*o);
        bodyTypeIds.push_back(o->id);
    }
    TF_MESH_IOTOEASY(fe, "bodyTypes", bodyTypes);
    TF_MESH_IOTOEASY(fe, "bodyTypeIds", bodyTypeIds);

    std::vector<VMod::SurfaceType> surfaceTypes;
    std::vector<int> surfaceTypeIds;
    for(size_t i = 0; i < solver->numSurfaceTypes(); i++) {
        auto o = solver->getSurfaceType(i);
        surfaceTypes.push_back(*o);
        surfaceTypeIds.push_back(o->id);
    }
    TF_MESH_IOTOEASY(fe, "surfaceTypes", surfaceTypes);
    TF_MESH_IOTOEASY(fe, "surfaceTypeIds", surfaceTypeIds);
    TF_MESH_IOTOEASY(fe, "mesh", solver->mesh);

    return S_OK;
}

HRESULT VMod::io::VertexSolverFIOModule::fromFile(const TissueForge::io::MetaData &metaData, const TissueForge::io::IOElement &fileElement) {

    VMod::MeshSolver *solver = VMod::MeshSolver::get();
    if(!solver && (VMod::MeshSolver::init() != S_OK || (solver = VMod::MeshSolver::get()) == NULL)) 
        return tf_error(E_FAIL, "Could not initialize solver");

    if(clearImport() != S_OK) 
        return tf_error(E_FAIL, "Failed clearing previous import");
    VMod::io::VertexSolverFIOModule::importSummary = new VMod::io::VertexSolverFIOImportSummary();

    TissueForge::io::IOChildMap::const_iterator feItr;

    // Load and register types and store their id maps

    std::vector<VMod::BodyType*> bodyTypes;
    std::vector<int> bodyTypeIds;
    TF_MESH_IOFROMEASY(feItr, fileElement.children, metaData, "bodyTypes", &bodyTypes);
    TF_MESH_IOFROMEASY(feItr, fileElement.children, metaData, "bodyTypeIds", &bodyTypeIds);
    for(size_t i = 0; i < bodyTypes.size(); i++) {
        auto t = bodyTypes[i];
        solver->registerType(t);
        VMod::io::VertexSolverFIOModule::importSummary->bodyTypeIdMap.insert({bodyTypeIds[i], t->id});
    }

    std::vector<VMod::SurfaceType*> surfaceTypes;
    std::vector<int> surfaceTypeIds;
    TF_MESH_IOFROMEASY(feItr, fileElement.children, metaData, "surfaceTypes", &surfaceTypes);
    TF_MESH_IOFROMEASY(feItr, fileElement.children, metaData, "surfaceTypeIds", &surfaceTypeIds);
    for(size_t i = 0; i < surfaceTypes.size(); i++) {
        auto t = surfaceTypes[i];
        solver->registerType(t);
        VMod::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.insert({surfaceTypeIds[i], t->id});
    }

    // Load meshes

    TF_MESH_IOFROMEASY(feItr, fileElement.children, metaData, "mesh", solver->mesh);

    return S_OK;
}

bool VMod::io::VertexSolverFIOModule::hasImport() { return VMod::io::VertexSolverFIOModule::importSummary != NULL; }

HRESULT VMod::io::VertexSolverFIOModule::clearImport() {
    if(hasImport()) {
        delete VMod::io::VertexSolverFIOModule::importSummary;
        VMod::io::VertexSolverFIOModule::importSummary = NULL;
    }
    return S_OK;
}
