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
#include <tf_util.h>


using namespace TissueForge::models::vertex;


std::vector<MeshObj*> Structure::parents() {
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

StructureType *Structure::type() {
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getStructureType(typeId);
}

std::vector<Body*> Structure::getBodies() {
    std::vector<Body*> result(bodies.begin(), bodies.end());
    for(auto &sp : structures_parent) 
        for(auto &b : sp->getBodies()) 
            result.push_back(b);
    return util::unique(result);
}

std::vector<Surface*> Structure::getSurfaces() {
    std::vector<Surface*> result;
    for(auto &b : bodies) 
        for(auto &s : b->getSurfaces()) 
            result.push_back(s);
    for(auto &sp : structures_parent) 
        for(auto &s : sp->getSurfaces()) 
            result.push_back(s);
    return util::unique(result);
}

std::vector<Vertex*> Structure::getVertices() {
    std::vector<Vertex*> result;
    for(auto &b : bodies) 
        for(auto &v : b->getVertices()) 
            result.push_back(v);
    for(auto &sp : structures_parent) 
        for(auto &v : sp->getVertices()) 
            result.push_back(v);
    return util::unique(result);
}
