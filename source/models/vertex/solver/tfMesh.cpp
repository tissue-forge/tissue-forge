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

#include "tfMesh.h"

#include "tfMeshSolver.h"

#include <tfEngine.h>
#include <tfLogger.h>

#include <algorithm>
#include <map>
#include <Magnum/Math/Intersection.h>


#define TF_MESH_GETPART(idx, inv) idx >= inv.size() ? NULL : inv[idx]


using namespace TissueForge;
using namespace TissueForge::models::vertex;


HRESULT Mesh_checkUnstoredObj(MeshObj *obj) {
    if(!obj || obj->objId >= 0 || obj->mesh) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }
    
    return S_OK;
}


HRESULT Mesh_checkStoredObj(MeshObj *obj, Mesh *mesh) {
    if(!obj || obj->objId < 0 || obj->mesh == NULL || obj->mesh != mesh) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}


template <typename T> 
HRESULT Mesh_checkObjStorage(MeshObj *obj, std::vector<T*> inv) {
    if(!Mesh_checkStoredObj(obj) || obj->objId >= inv.size()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}


template <typename T> 
int Mesh_invIdAndAlloc(std::vector<T*> &inv, std::set<unsigned int> &availIds, T *obj=NULL) {
    if(!availIds.empty()) {
        std::set<unsigned int>::iterator itr = availIds.begin();
        unsigned int objId = *itr;
        if(obj) {
            inv[objId] = obj;
            obj->objId = *itr;
        }
        availIds.erase(itr);
        return objId;
    }
    
    int res = inv.size();
    int new_size = inv.size() + TFMESHINV_INCR;
    inv.reserve(new_size);
    for(unsigned int i = res; i < new_size; i++) {
        inv.push_back(NULL);
        if(i != res) 
            availIds.insert(i);
    }
    if(obj) {
        inv[res] = obj;
        obj->objId = res;
    }
    return res;
}


#define TF_MESH_OBJINVCHECK(obj, inv) \
    { \
        if(obj->objId >= inv.size()) { \
            TF_Log(LOG_ERROR) << "Object with id " << obj->objId << " exceeds inventory (" << inv.size() << ")"; \
            return E_FAIL; \
        } \
    }


template <typename T> 
HRESULT Mesh_addObj(Mesh *mesh, std::vector<T*> &inv, std::set<unsigned int> &availIds, T *obj, MeshSolver *_solver=NULL) {
    if(Mesh_checkUnstoredObj(obj) != S_OK || !obj->validate()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    obj->objId = Mesh_invIdAndAlloc(inv, availIds, obj);
    obj->mesh = mesh;
    if(_solver) {
        std::vector<int> objIds{obj->objId};
        std::vector<MeshObj::Type> objTypes{obj->objType()};
        for(auto &p : obj->parents()) {
            objIds.push_back(p->objId);
            objTypes.push_back(p->objType());
        }
        _solver->log(mesh, MeshLogEventType::Create, objIds, objTypes);
    }
    return S_OK;
}


//////////
// Mesh //
//////////


Mesh::Mesh() : 
    _quality{new MeshQuality(this)}
{}

Mesh::~Mesh() { 
    if(_quality) { 
        delete _quality;
        _quality = 0;
    }
}

HRESULT Mesh::setQuality(MeshQuality *quality) {
    if(_quality) delete _quality;

    _quality = quality;
    return S_OK;
}

HRESULT Mesh::add(Vertex *obj) { 
    isDirty = true;
    if(_solver) 
        _solver->setDirty(true);

    return Mesh_addObj(this, this->vertices, this->vertexIdsAvail, obj, _solver);
}

HRESULT Mesh::add(Surface *obj){ 
    isDirty = true;

    for(auto &v : obj->vertices) 
        if(v->objId < 0 && add(v) != S_OK) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }

    if(Mesh_addObj(this, this->surfaces, this->surfaceIdsAvail, obj, _solver) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT Mesh::add(Body *obj){ 
    isDirty = true;

    for(auto &s : obj->surfaces) 
        if(s->objId < 0 && add((Surface*)s) != S_OK) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }

    if(Mesh_addObj(this, this->bodies, this->bodyIdsAvail, obj, _solver) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT Mesh::add(Structure *obj){ 
    isDirty = true;

    for(auto &p : obj->parents()) {
        if(p->objId >= 0) 
            continue;

        if(TissueForge::models::vertex::check(p, MeshObj::Type::STRUCTURE)) {
            if(add((Structure*)p) != S_OK) {
                TF_Log(LOG_ERROR);
                return E_FAIL;
            }
        } 
        else if(TissueForge::models::vertex::check(p, MeshObj::Type::BODY)) {
            if(add((Body*)p) != S_OK) {
                TF_Log(LOG_ERROR);
                return E_FAIL;
            }
        }
        else {
            TF_Log(LOG_ERROR) << "Could not determine type of parent";
            return E_FAIL;
        }
    }

    if(Mesh_addObj(this, this->structures, this->structureIdsAvail, obj, _solver) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}


HRESULT Mesh::removeObj(MeshObj *obj) { 
    isDirty = true;

    if(Mesh_checkStoredObj(obj, this) != S_OK) {
        TF_Log(LOG_ERROR) << "Invalid mesh object passed for remove.";
        return E_FAIL;
    } 

    if(TissueForge::models::vertex::check(obj, MeshObj::Type::VERTEX)) {
        TF_MESH_OBJINVCHECK(obj, this->vertices);
        this->vertices[obj->objId] = NULL;
        this->vertexIdsAvail.insert(obj->objId);
    } 
    else if(TissueForge::models::vertex::check(obj, MeshObj::Type::SURFACE)) {
        TF_MESH_OBJINVCHECK(obj, this->surfaces);
        this->surfaces[obj->objId] = NULL;
        this->surfaceIdsAvail.insert(obj->objId);
    } 
    else if(TissueForge::models::vertex::check(obj, MeshObj::Type::BODY)) {
        TF_MESH_OBJINVCHECK(obj, this->bodies);
        this->bodies[obj->objId] = NULL;
        this->bodyIdsAvail.insert(obj->objId);
    } 
    else if(TissueForge::models::vertex::check(obj, MeshObj::Type::STRUCTURE)) {
        TF_MESH_OBJINVCHECK(obj, this->structures);
        this->structures[obj->objId] = NULL;
        this->structureIdsAvail.insert(obj->objId);
    } 
    else {
        TF_Log(LOG_ERROR) << "Mesh object type could not be determined.";
        return E_FAIL;
    }

    if(_solver)
        _solver->log(this, MeshLogEventType::Destroy, {obj->objId}, {obj->objType()});
    obj->objId = -1;
    obj->mesh = NULL;

    for(auto &p : std::vector<MeshObj*>(obj->parents())) 
        p->removeChild(obj);
    for(auto &c : obj->children()) 
        if(removeObj(c) != S_OK) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }

    return S_OK;
}

Vertex *Mesh::findVertex(const FVector3 &pos, const FloatP_t &tol) const {

    for(auto &v : vertices)
        if(v->particle()->relativePosition(pos).length() <= tol) 
            return v;

    return NULL;
}

Vertex *Mesh::getVertex(const unsigned int &idx) const {
    return TF_MESH_GETPART(idx, vertices);
}

Surface *Mesh::getSurface(const unsigned int &idx) const {
    return TF_MESH_GETPART(idx, surfaces);
}

Body *Mesh::getBody(const unsigned int &idx) const {
    return TF_MESH_GETPART(idx, bodies);
}

Structure *Mesh::getStructure(const unsigned int &idx) const {
    return TF_MESH_GETPART(idx, structures);
}

template <typename T> 
bool Mesh_validateInv(const std::vector<T*> &inv) {
    for(auto &o : inv) 
        if(!o->validate()) 
            return false;
    return true;
}

bool Mesh::validate() {
    if(!Mesh_validateInv(this->vertices)) 
        return false;
    else if(!Mesh_validateInv(this->surfaces)) 
        return false;
    else if(!Mesh_validateInv(this->bodies)) 
        return false;
    else if(!Mesh_validateInv(this->structures)) 
        return false;
    return true;
}

HRESULT Mesh::makeDirty() {
    isDirty = true;
    if(_solver) 
        if(_solver->setDirty(true) != S_OK) 
            return E_FAIL;
    return S_OK;
}

bool Mesh::connected(const Vertex *v1, const Vertex *v2) const {
    for(auto &s1 : v1->surfaces) 
        for(auto vitr = s1->vertices.begin() + 1; vitr != s1->vertices.end(); vitr++) 
            if((*vitr == v1 && *(vitr - 1) == v2) || (*vitr == v2 && *(vitr - 1) == v1)) 
                return true;

    return false;
}

bool Mesh::connected(const Surface *s1, const Surface *s2) const {
    for(auto &v : s1->parents()) 
        if(v->in(s2)) 
            return true;
    return false;
}

bool Mesh::connected(const Body *b1, const Body *b2) const {
    for(auto &s : b1->parents()) 
        if(s->in(b2)) 
            return true;
    return false;
}

// Mesh editing

HRESULT Mesh::remove(Vertex *v) {
    return removeObj(v);
}

HRESULT Mesh::remove(Surface *s) {
    return removeObj(s);
}

HRESULT Mesh::remove(Body *b) {
    return removeObj(b);
}

HRESULT Mesh::remove(Structure *s) {
    return removeObj(s);
}
