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

#include <tfLogger.h>


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
