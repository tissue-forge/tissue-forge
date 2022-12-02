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

#include "tfStructure.h"

#include "tfBody.h"
#include "tfMeshSolver.h"
#include "tf_mesh_io.h"
#include "tfVertexSolverFIO.h"

#include <tfError.h>
#include <tfLogger.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>


using namespace TissueForge::models::vertex;


std::vector<MeshObj*> Structure::parents() const {
    std::vector<MeshObj*> result(structures_parent.size() + bodies.size(), 0);
    for(unsigned int i = 0; i < structures_parent.size(); i++) 
        result[i] = static_cast<MeshObj*>(structures_parent[i]);
    for(unsigned int i = 0; i < bodies.size(); i++) 
        result[structures_parent.size() + i] = static_cast<MeshObj*>(bodies[i]);
    return result;
}

HRESULT Structure::addChild(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::STRUCTURE)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Structure *s = (Structure*)obj;
    if(std::find(structures_child.begin(), structures_child.end(), s) != structures_child.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    structures_child.push_back(s);
    return S_OK;
}

HRESULT Structure::addParent(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::STRUCTURE) && !TissueForge::models::vertex::check(obj, MeshObj::Type::BODY)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    if(obj->objType() == MeshObj::Type::BODY) {
        Body *b = (Body*)obj;
        if(std::find(bodies.begin(), bodies.end(), b) != bodies.end()) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }
        bodies.push_back(b);
    } 
    else {
        Structure *s = (Structure*)obj;
        if(std::find(structures_parent.begin(), structures_parent.end(), s) != structures_parent.end()) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }
        structures_parent.push_back(s);
    }
    
    return S_OK;
}

HRESULT Structure::removeChild(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::STRUCTURE)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Structure *s = (Structure*)obj;
    auto itr = std::find(structures_child.begin(), structures_child.end(), s);
    if(itr == structures_child.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    structures_child.erase(itr);
    return S_OK;
}

HRESULT Structure::removeParent(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::STRUCTURE) && !TissueForge::models::vertex::check(obj, MeshObj::Type::BODY)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    if(obj->objType() == MeshObj::Type::BODY) {
        Body *b = (Body*)obj;
        auto itr = std::find(bodies.begin(), bodies.end(), b);
        if(itr != bodies.end()) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }
        bodies.erase(itr);
    } 
    else {
        Structure *s = (Structure*)obj;
        auto itr = std::find(structures_parent.begin(), structures_parent.end(), s);
        if(itr != structures_parent.end()) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }
        structures_parent.erase(itr);
    }
    
    return S_OK;
}

std::string Structure::str() const {
    std::stringstream ss;

    ss << "Structure(";
    if(this->objId >= 0) {
        ss << "id=" << this->objId << ", typeId=" << this->typeId;
    }
    ss << ")";

    return ss.str();
}

HRESULT Structure::destroy() {
    if(this->mesh && this->mesh->remove(this) != S_OK) 
        return E_FAIL;
    return S_OK;
}

StructureType *Structure::type() const {
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getStructureType(typeId);
}

HRESULT Structure::become(StructureType *stype) {
    this->typeId = stype->id;
    return S_OK;
}

std::vector<Body*> Structure::getBodies() const {
    std::unordered_set<Body*> result(bodies.begin(), bodies.end());
    for(auto &sp : structures_parent) 
        for(auto &b : sp->getBodies()) 
            result.insert(b);
    return std::vector<Body*>(result.begin(), result.end());
}

std::vector<Surface*> Structure::getSurfaces() const {
    std::unordered_set<Surface*> result;
    for(auto &b : bodies) 
        for(auto &s : b->getSurfaces()) 
            result.insert(s);
    for(auto &sp : structures_parent) 
        for(auto &s : sp->getSurfaces()) 
            result.insert(s);
    return std::vector<Surface*>(result.begin(), result.end());
}

std::vector<Vertex*> Structure::getVertices() const {
    std::unordered_set<Vertex*> result;
    for(auto &b : bodies) 
        for(auto &v : b->getVertices()) 
            result.insert(v);
    for(auto &sp : structures_parent) 
        for(auto &v : sp->getVertices()) 
            result.insert(v);
    return std::vector<Vertex*>(result.begin(), result.end());
}

StructureType::StructureType(const bool &noReg) : 
    MeshObjType()
{
    name = "Structure";

    if(!noReg) 
        this->registerType();
}

std::string StructureType::str() const {
    std::stringstream ss;

    ss << "StructureType(id=" << this->id << ", name=" << this->name << ")";

    return ss.str();
}

StructureType *StructureType::findFromName(const std::string &_name) {
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->findStructureFromName(_name);
}

HRESULT StructureType::registerType() {
    if(isRegistered()) return S_OK;

    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return E_FAIL;

    HRESULT result = solver->registerType(this);
    if(result == S_OK) 
        on_register();

    return result;
}

bool StructureType::isRegistered() {
    return get();
}

StructureType *StructureType::get() {
    return findFromName(name);
}

std::vector<Structure*> StructureType::getInstances() {
    std::vector<Structure*> result;

    MeshSolver *solver = MeshSolver::get();
    if(solver) { 
        result.reserve(solver->numStructures());
        for(auto &m : solver->meshes) {
            for(size_t i = 0; i < m->sizeStructures(); i++) {
                Structure *s = m->getStructure(i);
                if(s) 
                    result.push_back(s);
            }
        }
    }

    return result;
}

std::vector<int> StructureType::getInstanceIds() {
    auto instances = getInstances();
    std::vector<int> result;
    result.reserve(instances.size());
    for(auto &b : instances) 
        if(b) 
            result.push_back(b->objId);
    return result;
}

unsigned int StructureType::getNumInstances() {
    return getInstances().size();
}

namespace TissueForge::io {


    #define TF_MESH_STRUCTUREIOTOEASY(fe, key, member) \
        fe = new IOElement(); \
        if(toFile(member, metaData, fe) != S_OK)  \
            return E_FAIL; \
        fe->parent = fileElement; \
        fileElement->children[key] = fe;

    #define TF_MESH_STRUCTUREIOFROMEASY(feItr, children, metaData, key, member_p) \
        feItr = children.find(key); \
        if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
            return E_FAIL;

    template <>
    HRESULT toFile(const TissueForge::models::vertex::Structure &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_MESH_STRUCTUREIOTOEASY(fe, "objId", dataElement.objId);
        TF_MESH_STRUCTUREIOTOEASY(fe, "meshId", dataElement.mesh == NULL ? -1 : dataElement.mesh->getId());
        TF_MESH_STRUCTUREIOTOEASY(fe, "typeId", dataElement.typeId);

        if(dataElement.actors.size() > 0) {
            std::vector<TissueForge::models::vertex::MeshObjActor*> actors;
            for(auto &a : dataElement.actors) 
                if(a) 
                    actors.push_back(a);
            TF_MESH_STRUCTUREIOTOEASY(fe, "actors", actors);
        }

        std::vector<int> structures_parent, bodies;
        for(auto &p : dataElement.parents()) {
            if(p->objType() == TissueForge::models::vertex::MeshObj::BODY) 
                bodies.push_back(p->objId);
            else 
                structures_parent.push_back(p->objId);
        }
        TF_MESH_STRUCTUREIOTOEASY(fe, "structures_parent", structures_parent);
        TF_MESH_STRUCTUREIOTOEASY(fe, "bodies", bodies);

        std::vector<int> structures_child_ids;
        for(auto &c : dataElement.children()) 
            structures_child_ids.push_back(c->objId);
        TF_MESH_STRUCTUREIOTOEASY(fe, "structures_child", structures_child_ids);

        fileElement->type = "Structure";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Structure **dataElement) {
        
        if(!FIO::hasImport()) 
            return tf_error(E_FAIL, "No import data available");
        else if(!TissueForge::models::vertex::io::VertexSolverFIOModule::hasImport()) 
            return tf_error(E_FAIL, "No vertex import data available");

        TissueForge::models::vertex::MeshSolver *solver = TissueForge::models::vertex::MeshSolver::get();
        if(!solver) 
            return tf_error(E_FAIL, "No vertex solver available");

        IOChildMap::const_iterator feItr;

        TissueForge::models::vertex::Mesh *mesh = NULL;
        int meshIdOld;
        unsigned int meshId;
        TF_MESH_STRUCTUREIOFROMEASY(feItr, fileElement.children, metaData, "meshId", &meshIdOld);
        if(meshIdOld >= 0) {
            auto meshId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->meshIdMap.find(meshIdOld);
            if(meshId_itr != TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->meshIdMap.end() && meshId_itr->second < solver->numMeshes()) {
                meshId = meshId_itr->second;
                mesh = solver->getMesh(meshId);
            }
        }
        if(!mesh) {
            return tf_error(E_FAIL, "Could not identify mesh");
        }

        int typeIdOld;
        TF_MESH_STRUCTUREIOFROMEASY(feItr, fileElement.children, metaData, "typeId", &typeIdOld);
        auto typeId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->structureTypeIdMap.find(typeIdOld);
        if(typeId_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->structureTypeIdMap.end()) {
            return tf_error(E_FAIL, "Could not identify type");
        }

        *dataElement = new TissueForge::models::vertex::Structure();
        (*dataElement)->typeId = typeId_itr->second;

        if(mesh->add(*dataElement) != S_OK) 
            return tf_error(E_FAIL, "Failed to add to mesh");

        int objIdOld;
        TF_MESH_STRUCTUREIOFROMEASY(feItr, fileElement.children, metaData, "objId", &objIdOld);
        TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->structureIdMap[meshId].insert({objIdOld, (*dataElement)->objId});

        if(fileElement.children.find("actors") != fileElement.children.end()) {
            TF_MESH_STRUCTUREIOFROMEASY(feItr, fileElement.children, metaData, "actors", &(*dataElement)->actors);
        }

        return S_OK;
    }

    template <>
    HRESULT toFile(const TissueForge::models::vertex::StructureType &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_MESH_STRUCTUREIOTOEASY(fe, "id", dataElement.id);

        if(dataElement.actors.size() > 0) {
            std::vector<TissueForge::models::vertex::MeshObjActor*> actors;
            for(auto &a : dataElement.actors) 
                if(a) 
                    actors.push_back(a);
            TF_MESH_STRUCTUREIOTOEASY(fe, "actors", actors);
        }

        TF_MESH_STRUCTUREIOTOEASY(fe, "name", dataElement.name);

        fileElement->type = "StructureType";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::StructureType **dataElement) {
        
        IOChildMap::const_iterator feItr;

        *dataElement = new TissueForge::models::vertex::StructureType();

        TF_MESH_STRUCTUREIOFROMEASY(feItr, fileElement.children, metaData, "name", &(*dataElement)->name);
        if(fileElement.children.find("actors") != fileElement.children.end()) {
            TF_MESH_STRUCTUREIOFROMEASY(feItr, fileElement.children, metaData, "actors", &(*dataElement)->actors);
        }

        return S_OK;
    }
}

std::string TissueForge::models::vertex::Structure::toString() {
    return TissueForge::io::toString(*this);
}

std::string TissueForge::models::vertex::StructureType::toString() {
    return TissueForge::io::toString(*this);
}

StructureType *TissueForge::models::vertex::StructureType::fromString(const std::string &str) {
    return TissueForge::io::fromString<TissueForge::models::vertex::StructureType*>(str);
}
