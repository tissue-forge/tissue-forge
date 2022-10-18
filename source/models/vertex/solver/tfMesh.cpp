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

Vertex *Mesh::findVertex(const FVector3 &pos, const FloatP_t &tol) {

    for(auto &v : vertices)
        if(v->particle()->relativePosition(pos).length() <= tol) 
            return v;

    return NULL;
}

Vertex *Mesh::getVertex(const unsigned int &idx) {
    return TF_MESH_GETPART(idx, vertices);
}

Surface *Mesh::getSurface(const unsigned int &idx) {
    return TF_MESH_GETPART(idx, surfaces);
}

Body *Mesh::getBody(const unsigned int &idx) {
    return TF_MESH_GETPART(idx, bodies);
}

Structure *Mesh::getStructure(const unsigned int &idx) {
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

bool Mesh::connected(Vertex *v1, Vertex *v2) {
    for(auto &s1 : v1->surfaces) 
        for(auto vitr = s1->vertices.begin() + 1; vitr != s1->vertices.end(); vitr++) 
            if((*vitr == v1 && *(vitr - 1) == v2) || (*vitr == v2 && *(vitr - 1) == v1)) 
                return true;

    return false;
}

bool Mesh::connected(Surface *s1, Surface *s2) {
    for(auto &v : s1->parents()) 
        if(v->in(s2)) 
            return true;
    return false;
}

bool Mesh::connected(Body *b1, Body *b2) {
    for(auto &s : b1->parents()) 
        if(s->in(b2)) 
            return true;
    return false;
}

HRESULT Mesh_surfaceOutwardNormal(Surface *s, Body *b1, Body *b2, FVector3 &onorm) {
    if(b1 && b2) { 
        TF_Log(LOG_ERROR) << "Surface is twice-connected";
        return NULL;
    } 
    else if(b1) {
        onorm = s->getNormal();
    } 
    else if(b2) {
        onorm = -s->getNormal();
    } 
    else {
        onorm = s->getNormal();
    }
    return S_OK;
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

HRESULT Mesh::insert(Vertex *toInsert, Vertex *v1, Vertex *v2) {
    Vertex *v;
    std::vector<Vertex*>::iterator vitr;

    // Find the common surface(s)
    int nidx;
    Vertex *vn;
    for(auto &s1 : v1->surfaces) {
        nidx = 0;
        for(vitr = s1->vertices.begin(); vitr != s1->vertices.end(); vitr++) {
            nidx++;
            if(nidx >= s1->vertices.size()) 
                nidx -= s1->vertices.size();
            vn = s1->vertices[nidx];

            if((*vitr == v1 && vn == v2) || (*vitr == v2 && vn == v1)) {
                s1->vertices.insert(vitr + 1 == s1->vertices.end() ? s1->vertices.begin() : vitr + 1, toInsert);
                toInsert->add(s1);
                break;
            }
        }
    }

    if(toInsert->objId < 0 && add(toInsert) != S_OK) 
        return E_FAIL;

    if(_solver) { 
        if(!qualityWorking() && _solver->positionChanged() != S_OK)
            return E_FAIL;

        _solver->log(this, MeshLogEventType::Create, {v1->objId, v2->objId}, {v1->objType(), v2->objType()}, "insert");
    }

    return S_OK;
}

HRESULT Mesh::insert(Vertex *toInsert, Vertex *vf, std::vector<Vertex*> nbs) {
    for(auto &v : nbs) 
        if(insert(toInsert, vf, v) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT Mesh_SurfaceDisconnectReplace(
    Vertex *toInsert, 
    Surface *toReplace, 
    Surface *targetSurf, 
    std::vector<Vertex*> &targetSurf_vertices, 
    std::set<Vertex*> &totalToRemove) 
{
    std::vector<unsigned int> edgeLabels = targetSurf->contiguousEdgeLabels(toReplace);
    std::vector<Vertex*> toRemove;
    for(unsigned int i = 0; i < edgeLabels.size(); i++) {
        unsigned int lab = edgeLabels[i];
        if(lab > 0) {
            if(lab > 1) {
                TF_Log(LOG_ERROR) << "Replacement cannot occur over non-contiguous contacts";
                return E_FAIL;
            }
            toRemove.push_back(targetSurf_vertices[i]);
        }
    }
    
    if(toRemove.empty()) 
        return S_OK;
    
    targetSurf->insert(toInsert, toRemove[0]);
    toInsert->add(targetSurf);
    for(auto &v : toRemove) {
        targetSurf->remove(v);
        v->remove(targetSurf);
        totalToRemove.insert(v);
    }
    return S_OK;
}

HRESULT Mesh::replace(Vertex *toInsert, Surface *toReplace) {
    // For every surface connected to the replaced surface
    //      Gather every vertex connected to the replaced surface
    //      Replace all vertices with the inserted vertex
    // Remove the replaced surface from the mesh
    // Add the inserted vertex to the mesh

    // Prevent nonsensical resultant bodies
    if(toReplace->b1 && toReplace->b1->surfaces.size() < 5) { 
        TF_Log(LOG_DEBUG) << "Insufficient surfaces (" << toReplace->b1->surfaces.size() << ") in first body (" << toReplace->b1->objId << ") for replace";
        return E_FAIL;
    }
    else if(toReplace->b2 && toReplace->b2->surfaces.size() < 5) {
        TF_Log(LOG_DEBUG) << "Insufficient surfaces (" << toReplace->b2->surfaces.size() << ") in first body (" << toReplace->b2->objId << ") for replace";
        return E_FAIL;
    }

    // Gather every contacting surface
    std::vector<Surface*> connectedSurfaces = toReplace->connectedSurfaces();

    // Disconnect every vertex connected to the replaced surface
    std::set<Vertex*> totalToRemove;
    for(auto &s : connectedSurfaces) 
        if(Mesh_SurfaceDisconnectReplace(toInsert, toReplace, s, s->vertices, totalToRemove) != S_OK) 
            return E_FAIL;

    // Add the inserted vertex
    if(add(toInsert) != S_OK) 
        return E_FAIL;

    if(_solver) 
        _solver->log(this, MeshLogEventType::Create, {toInsert->objId, toReplace->objId}, {toInsert->objType(), toReplace->objType()}, "replace");

    // Remove the replaced surface and its vertices
    while(!toReplace->vertices.empty()) {
        Vertex *v = toReplace->vertices.front();
        v->remove(toReplace);
        toReplace->remove(v);
        totalToRemove.insert(v);
    }
    if(toReplace->b1) { 
        Body *b1 = toReplace->b1;
        b1->remove(toReplace);
        toReplace->remove(b1);
        b1->positionChanged();
    }
    if(toReplace->b2) { 
        Body *b2 = toReplace->b2;
        b2->remove(toReplace);
        toReplace->remove(b2);
        b2->positionChanged();
    }
    if(toReplace->destroy() != S_OK) 
        return E_FAIL;
    for(auto &v : totalToRemove) 
        if(v->destroy() != S_OK) 
            return E_FAIL;

    if(_solver) 
        if(!qualityWorking() && _solver->positionChanged() != S_OK)
            return E_FAIL;

    return S_OK;
}

HRESULT Mesh::replace(Vertex *toInsert, Body *toReplace) {
    // Detach surfaces and bodies
    std::set<Vertex*> totalToRemove;
    std::vector<Surface*> b_surfaces(toReplace->surfaces);
    for(auto &s : b_surfaces) { 
        for(auto &ns : s->neighborSurfaces()) { 
            if(ns->in(toReplace)) 
                continue;
            if(Mesh_SurfaceDisconnectReplace(toInsert, s, ns, ns->vertices, totalToRemove) != S_OK) 
                return E_FAIL;
        }

        if(s->b1 && s->b1 != toReplace) {
            if(s->b1->surfaces.size() < 5) {
                TF_Log(LOG_DEBUG) << "Insufficient surfaces (" << s->b1->surfaces.size() << ") in first body (" << s->b1->objId << ") for replace";
                return E_FAIL;
            }
            s->b1->remove(s);
            s->remove(s->b1);
        }
        if(s->b2 && s->b2 != toReplace) {
            if(s->b2->surfaces.size() < 5) {
                TF_Log(LOG_DEBUG) << "Insufficient surfaces (" << s->b2->surfaces.size() << ") in first body (" << s->b2->objId << ") for replace";
                return E_FAIL;
            }
            s->b2->remove(s);
            s->remove(s->b2);
        }
    }

    // Add the vertex
    if(add(toInsert) != S_OK) 
        return E_FAIL;

    if(_solver) 
        _solver->log(this, MeshLogEventType::Create, {toInsert->objId, toReplace->objId}, {toInsert->objType(), toReplace->objType()}, "replace");

    while(!toReplace->surfaces.empty()) {
        Surface *s = toReplace->surfaces.front();
        while(!s->vertices.empty()) {
            Vertex *v = s->vertices.front();
            s->remove(v);
            v->remove(s);
            totalToRemove.insert(v);
        }
        toReplace->remove(s);
        s->remove(toReplace);
        s->destroy();
    }
    if(toReplace->destroy() != S_OK) 
        return E_FAIL;
    for(auto &v : totalToRemove) 
        if(v->destroy() != S_OK) 
            return E_FAIL;

    if(_solver) 
        if(!qualityWorking() && _solver->positionChanged() != S_OK)
            return E_FAIL;

    return S_OK;
}

Surface *Mesh::replace(SurfaceType *toInsert, Vertex *toReplace, std::vector<FloatP_t> lenCfs) {
    std::vector<Vertex*> neighbors = toReplace->neighborVertices();
    if(lenCfs.size() != neighbors.size()) {
        TF_Log(LOG_ERROR) << "Length coefficients are inconsistent with connectivity";
        return NULL;
    } 

    for(auto &cf : lenCfs) 
        if(cf <= 0.f || cf >= 1.f) {
            TF_Log(LOG_ERROR) << "Length coefficients must be in (0, 1)";
            return NULL;
        }

    // Insert new vertices
    FVector3 pos0 = toReplace->getPosition();
    std::vector<Vertex*> insertedVertices;
    for(unsigned int i = 0; i < neighbors.size(); i++) {
        FloatP_t cf = lenCfs[i];
        if(cf <= 0.f || cf >= 1.f) {
            TF_Log(LOG_ERROR) << "Length coefficients must be in (0, 1)";
            return NULL;
        }

        Vertex *v = neighbors[i];
        FVector3 pos1 = v->getPosition();
        FVector3 pos = pos0 + (pos1 - pos0) * cf;
        MeshParticleType *ptype = MeshParticleType_get();
        ParticleHandle *ph = (*ptype)(&pos);
        Vertex *vInserted = new Vertex(ph->id);
        if(insert(vInserted, toReplace, v) != S_OK) 
            return NULL;
        insertedVertices.push_back(vInserted);
    }

    // Disconnect replaced vertex from all surfaces
    std::vector<Surface*> toReplaceSurfaces(toReplace->surfaces.begin(), toReplace->surfaces.end());
    for(auto &s : toReplaceSurfaces) {
        s->remove(toReplace);
        toReplace->remove(s);
    }

    // Create new surface; its constructor should handle internal connections
    Surface *inserted = (*toInsert)(insertedVertices);

    // Remove replaced vertex from the mesh and add inserted surface to the mesh
    add(inserted);
    if(_solver) 
        _solver->log(this, MeshLogEventType::Create, {inserted->objId, toReplace->objId}, {inserted->objType(), toReplace->objType()}, "replace"); 
    toReplace->destroy();

    if(_solver) 
        if(!qualityWorking()) 
            _solver->positionChanged();

    return inserted;
}

HRESULT Mesh::merge(Vertex *toKeep, Vertex *toRemove, const FloatP_t &lenCf) {

    // In common surfaces, just remove; in different surfaces, replace
    std::vector<Surface*> common_s, different_s;
    common_s.reserve(toRemove->surfaces.size());
    different_s.reserve(toRemove->surfaces.size());
    for(auto &s : toRemove->surfaces) {
        if(!toKeep->in(s)) 
            different_s.push_back(s);
        else 
            common_s.push_back(s);
    }
    for(auto &s : common_s) {
        s->remove(toRemove);
        toRemove->remove(s);
    }
    for(auto &s : different_s) {
        toRemove->remove(s);
        toKeep->add(s);
        s->replace(toKeep, toRemove);
    }
    
    // Set new position
    const FVector3 posToKeep = toKeep->getPosition();
    const FVector3 newPos = posToKeep + (toRemove->getPosition() - posToKeep) * lenCf;
    if(toKeep->setPosition(newPos) != S_OK) 
        return E_FAIL;

    if(_solver) 
        _solver->log(this, MeshLogEventType::Create, {toKeep->objId, toRemove->objId}, {toKeep->objType(), toRemove->objType()}, "merge");
    
    if(toRemove->transferBondsTo(toKeep) != S_OK || toRemove->destroy() != S_OK) 
        return E_FAIL;

    if(_solver) 
        if(!qualityWorking() && _solver->positionChanged() != S_OK)
            return E_FAIL;

    return S_OK;
}

HRESULT Mesh::merge(Surface *toKeep, Surface *toRemove, const std::vector<FloatP_t> &lenCfs) {
    if(toKeep->vertices.size() != toRemove->vertices.size()) {
        TF_Log(LOG_ERROR) << "Surfaces must have the same number of vertices to merge";
        return E_FAIL;
    }

    // Find vertices that are not shared
    std::vector<Vertex*> toKeepExcl;
    for(auto &v : toKeep->vertices) 
        if(!v->in(toRemove)) 
            toKeepExcl.push_back(v);

    // Ensure sufficient length cofficients
    std::vector<FloatP_t> _lenCfs = lenCfs;
    if(_lenCfs.size() < toKeepExcl.size()) {
        TF_Log(LOG_DEBUG) << "Insufficient provided length coefficients. Assuming 0.5";
        for(unsigned int i = _lenCfs.size(); i < toKeepExcl.size(); i++) 
            _lenCfs.push_back(0.5);
    }

    // Match vertex order of removed surface to kept surface by nearest distance
    std::vector<Vertex*> toRemoveOrdered;
    for(auto &kv : toKeepExcl) {
        Vertex *mv = NULL;
        FVector3 kp = kv->getPosition();
        FloatP_t bestDist = 0.f;
        for(auto &rv : toRemove->vertices) {
            FloatP_t dist = (rv->getPosition() - kp).length();
            if((!mv || dist < bestDist) && std::find(toRemoveOrdered.begin(), toRemoveOrdered.end(), rv) == toRemoveOrdered.end()) {
                bestDist = dist;
                mv = rv;
            }
        }
        if(!mv) {
            TF_Log(LOG_ERROR) << "Could not match surface vertices";
            return E_FAIL;
        }
        toRemoveOrdered.push_back(mv);
    }

    // Replace vertices in neighboring surfaces
    for(unsigned int i = 0; i < toKeepExcl.size(); i++) {
        Vertex *rv = toRemoveOrdered[i];
        Vertex *kv = toKeepExcl[i];
        std::vector<Surface*> rvSurfaces = rv->surfaces;
        for(auto &s : rvSurfaces) 
            if(s != toRemove) {
                if(!rv->in(s)) {
                    TF_Log(LOG_ERROR) << "Something went wrong during surface merge";
                    return E_FAIL;
                }
                s->replace(kv, rv);
                kv->add(s);
            }
    }

    // Replace surface in child bodies
    for(auto &b : toRemove->getBodies()) {
        if(!toKeep->in(b)) {
            b->add(toKeep);
            toKeep->add(b);
        }
        b->remove(toRemove);
        toRemove->remove(b);
    }

    // Detach removed vertices
    for(auto &v : toRemoveOrdered) {
        v->surfaces.clear();
        toRemove->remove(v);
    }

    // Move kept vertices by length coefficients
    for(unsigned int i = 0; i < toKeepExcl.size(); i++) {
        Vertex *v = toKeepExcl[i];
        FVector3 posToKeep = v->getPosition();
        FVector3 newPos = posToKeep + (toRemoveOrdered[i]->getPosition() - posToKeep) * _lenCfs[i];
        if(v->setPosition(newPos) != S_OK) 
            return E_FAIL;
    }

    if(_solver) 
        _solver->log(this, MeshLogEventType::Create, {toKeep->objId, toRemove->objId}, {toKeep->objType(), toRemove->objType()}, "merge");
    
    // Remove surface and vertices that are not shared
    if(toRemove->destroy() != S_OK) 
        return E_FAIL;
    for(auto &v : toRemoveOrdered) 
        if(v->destroy() != S_OK) 
            return E_FAIL;

    if(_solver) 
        if(!qualityWorking() && _solver->positionChanged() != S_OK)
            return E_FAIL;

    return S_OK;
}

Surface *Mesh::extend(Surface *base, const unsigned int &vertIdxStart, const FVector3 &pos) {
    // Validate indices
    if(vertIdxStart >= base->vertices.size()) {
        TF_Log(LOG_ERROR) << "Invalid vertex indices (" << vertIdxStart << ", " << base->vertices.size() << ")";
        return NULL;
    }

    // Get base vertices
    Vertex *v0 = base->vertices[vertIdxStart];
    Vertex *v1 = base->vertices[vertIdxStart == base->vertices.size() - 1 ? 0 : vertIdxStart + 1];

    // Construct new vertex at specified position
    SurfaceType *stype = base->type();
    MeshParticleType *ptype = MeshParticleType_get();
    FVector3 _pos = pos;
    ParticleHandle *ph = (*ptype)(&_pos);
    Vertex *vert = new Vertex(ph->id);

    // Construct new surface, add new parts and return
    Surface *s = (*stype)({v0, v1, vert});
    add(s);

    if(_solver) {
        if(!qualityWorking()) 
            _solver->positionChanged();

        _solver->log(this, MeshLogEventType::Create, {base->objId, s->objId}, {base->objType(), s->objType()}, "extend");
    }

    return s;
}

Surface *Mesh::extrude(Surface *base, const unsigned int &vertIdxStart, const FloatP_t &normLen) {
    // Validate indices
    if(vertIdxStart >= base->vertices.size()) {
        TF_Log(LOG_ERROR) << "Invalid vertex indices (" << vertIdxStart << ", " << base->vertices.size() << ")";
        return NULL;
    }

    // Get base vertices
    Vertex *v0 = base->vertices[vertIdxStart];
    Vertex *v1 = base->vertices[vertIdxStart == base->vertices.size() - 1 ? 0 : vertIdxStart + 1];

    // Construct new vertices
    FVector3 disp = base->normal * normLen;
    FVector3 pos2 = v0->getPosition() + disp;
    FVector3 pos3 = v1->getPosition() + disp;
    MeshParticleType *ptype = MeshParticleType_get();
    ParticleHandle *p2 = (*ptype)(&pos2);
    ParticleHandle *p3 = (*ptype)(&pos3);
    Vertex *v2 = new Vertex(p2->id);
    Vertex *v3 = new Vertex(p3->id);

    // Construct new surface, add new parts and return
    SurfaceType *stype = base->type();
    Surface *s = (*stype)({v0, v1, v2, v3});
    add(s);

    if(_solver) {
        if(!qualityWorking()) 
            _solver->positionChanged();

        _solver->log(this, MeshLogEventType::Create, {base->objId, s->objId}, {base->objType(), s->objType()}, "extrude");
    }

    return s;
}

Body *Mesh::extend(Surface *base, BodyType *btype, const FVector3 &pos) {
    // For every pair of vertices, construct a surface with a new vertex at the given position
    Vertex *vNew = new Vertex(pos);
    SurfaceType *stype = base->type();
    std::vector<Surface*> surfaces(1, base);
    for(unsigned int i = 0; i < base->vertices.size(); i++) {
        // Get base vertices
        Vertex *v0 = base->vertices[i];
        Vertex *v1 = base->vertices[i == base->vertices.size() - 1 ? 0 : i + 1];

        Surface *s = (*stype)({v0, v1, vNew});
        if(!s) 
            return NULL;
        surfaces.push_back(s);
    }

    // Construct a body from the surfaces
    Body *b = (*btype)(surfaces);
    if(!b) 
        return NULL;

    // Add new parts and return
    add(b);

    if(_solver) {
        if(!qualityWorking()) 
            _solver->positionChanged();

        _solver->log(this, MeshLogEventType::Create, {base->objId, b->objId}, {base->objType(), b->objType()}, "extend");
    }

    return b;
}

Body *Mesh::extrude(Surface *base, BodyType *btype, const FloatP_t &normLen) {
    unsigned int i, j;
    FVector3 normal;

    // Only permit if the surface has an available slot
    base->refreshBodies();
    if(Mesh_surfaceOutwardNormal(base, base->b1, base->b2, normal) != S_OK) 
        return NULL;

    std::vector<Vertex*> newVertices(base->vertices.size(), 0);
    SurfaceType *stype = base->type();
    MeshParticleType *ptype = MeshParticleType_get();
    FVector3 disp = normal * normLen;

    for(i = 0; i < base->vertices.size(); i++) {
        FVector3 pos = base->vertices[i]->getPosition() + disp;
        ParticleHandle *ph = (*ptype)(&pos);
        newVertices[i] = new Vertex(ph->id);
    }

    std::vector<Surface*> newSurfaces;
    for(i = 0; i < base->vertices.size(); i++) {
        j = i + 1 >= base->vertices.size() ? i + 1 - base->vertices.size() : i + 1;
        Surface *s = (*stype)({
            base->vertices[i], 
            base->vertices[j], 
            newVertices[j], 
            newVertices[i]
        });
        if(!s) 
            return NULL;
        newSurfaces.push_back(s);
    }
    newSurfaces.push_back(base);
    newSurfaces.push_back((*stype)(newVertices));

    Body *b = (*btype)(newSurfaces);
    if(!b) 
        return NULL;
    add(b);

    if(_solver) {
        if(!qualityWorking()) 
            _solver->positionChanged();

        _solver->log(this, MeshLogEventType::Create, {base->objId, b->objId}, {base->objType(), b->objType()}, "extrude");
    }

    return b;
}

HRESULT Mesh::sew(Surface *s1, Surface *s2, const FloatP_t &distCf) {
    // Verify that surfaces are in the mesh
    if(s1->mesh != this || s2->mesh != this) {
        TF_Log(LOG_ERROR) << "Surface not in this mesh";
        return E_FAIL;
    }

    if(Surface::sew(s1, s2, distCf) != S_OK) 
        return E_FAIL;

    if(_solver) 
        _solver->log(this, MeshLogEventType::Create, {s1->objId, s2->objId}, {s1->objType(), s2->objType()}, "sew");

    return S_OK;
}

HRESULT Mesh::sew(std::vector<Surface*> _surfaces, const FloatP_t &distCf) {
    for(std::vector<Surface*>::iterator itri = _surfaces.begin(); itri != _surfaces.end() - 1; itri++) 
        for(std::vector<Surface*>::iterator itrj = itri + 1; itrj != _surfaces.end(); itrj++) 
            if(*itri != *itrj && sew(*itri, *itrj, distCf) != S_OK) 
                return E_FAIL;

    return S_OK;
}

HRESULT Mesh::splitPlan(Vertex *v, const FVector3 &sep, std::vector<Vertex*> &verts_v, std::vector<Vertex*> &verts_new_v) {
    verts_v.clear();
    verts_new_v.clear();

    std::vector<Vertex*> nbs = v->neighborVertices();

    // Verify that 
    //  1. the vertex is in the mesh
    //  2. the vertex defines at least one surface
    if(v->mesh != this) {
        tf_error(E_FAIL, "Vertex not in this mesh");
        return 0;
    } 
    else if(nbs.size() == 0) {
        tf_error(E_FAIL, "Vertex must define a surface");
        return 0;
    }
    
    // Define a cut plane at the midpoint of and orthogonal to the new edge
    FVector3 v_pos0 = v->getPosition();
    FVector4 planeEq = FVector4::planeEquation(sep.normalized(), v_pos0);

    // Determine which neighbors will be connected to each vertex
    verts_new_v.reserve(nbs.size());
    verts_v.reserve(nbs.size());
    for(auto nv : nbs) {
        if(planeEq.distance(nv->getPosition()) >= 0) 
            verts_new_v.push_back(nv);
        else 
            verts_v.push_back(nv);
    }

    // Reject if either side of the plane has no vertices
    if(verts_new_v.empty() || verts_v.empty()) {
        verts_v.clear();
        verts_new_v.clear();
        TF_Log(LOG_DEBUG) << "No vertices on both sides of cut plane; ignoring";
        return S_OK;
    }

    return S_OK;

}

Vertex *Mesh::splitExecute(Vertex *v, const FVector3 &sep, const std::vector<Vertex*> &verts_v, const std::vector<Vertex*> &verts_new_v) {
    FVector3 v_pos0 = v->getPosition();
    FVector3 hsep = sep * 0.5;
    FVector3 v_pos1 = v_pos0 - hsep;
    FVector3 u_pos = v_pos0 + hsep;

    // Determine which surfaces the target vertex will no longer partially define
    // A surface remains partially defined by the target vertex if the target vertex has 
    // a neighbor on its own side of the cut plane that also partially defines the surface
    std::set<Surface*> u_surfs, vn_surfs;
    for(auto &nv : verts_v) 
        for(auto &s : nv->sharedSurfaces(v)) 
            vn_surfs.insert(s);
    for(auto &nv : verts_new_v) 
        for(auto &s : nv->sharedSurfaces(v)) 
            u_surfs.insert(s);
    std::set<Surface*> surfs_keep_v, surfs_remove_v;
    for(auto &s : u_surfs) {
        if(std::find(vn_surfs.begin(), vn_surfs.end(), s) == vn_surfs.end()) 
            surfs_remove_v.insert(s);
        else 
            surfs_keep_v.insert(s);
    }

    // Create and insert the new vertex
    Vertex *u = new Vertex(u_pos);
    v->setPosition(v_pos1);
    if(add(u) != S_OK) {
        tf_error(E_FAIL, "Could not add vertex");
        u->destroy();
        delete u;
        return 0;
    }

    //  Replace v with u where removing
    for(auto &s : surfs_remove_v) {
        v->remove(s);
        u->add(s);
        s->replace(u, v);
    }

    //  Insert u between v and neighbor where not removing
    for(auto &s : surfs_keep_v) {
        u->add(s);
        for(auto &nv : verts_new_v) {
            std::vector<Vertex*>::iterator verts_new_v_itr = std::find(s->vertices.begin(), s->vertices.end(), nv);
            if(verts_new_v_itr != s->vertices.end()) { 
                s->insert(u, v, *verts_new_v_itr);
                break;
            }
        }
    }

    if(_solver) {
        if(!qualityWorking()) 
            _solver->positionChanged();

        _solver->log(this, MeshLogEventType::Create, {v->objId, u->objId}, {v->objType(), u->objType()}, "split");
    }

    return u;
}

Vertex *Mesh::split(Vertex *v, const FVector3 &sep) {
    
    std::vector<Vertex*> verts_v, new_verts_v;
    Vertex *u = NULL;
    if(splitPlan(v, sep, verts_v, new_verts_v))
        u = splitExecute(v, sep, verts_v, new_verts_v);

    if(_solver) {
        if(!qualityWorking()) 
            _solver->positionChanged();

        _solver->log(this, MeshLogEventType::Create, {v->objId, u->objId}, {v->objType(), u->objType()}, "split");
    }

    return u;
}

Surface *Mesh::split(Surface *s, Vertex *v1, Vertex *v2) { 
    // Verify that vertices are in surface
    if(!v1->in(s) || !v2->in(s)) { 
        tf_error(E_FAIL, "Vertices are not part of the splitting surface");
        return NULL;
    }

    // Verify that vertices are not adjacent
    const std::vector<Vertex*> v1_nbs = v1->neighborVertices();
    if(std::find(v1_nbs.begin(), v1_nbs.end(), v2) != v1_nbs.end()) {
        tf_error(E_FAIL, "Vertices are adjacent");
        return NULL;
    }

    // Extract vertices for new surface
    std::vector<Vertex*> v_new_surf;
    v_new_surf.reserve(s->vertices.size());
    v_new_surf.push_back(v1);
    std::vector<Vertex*>::iterator v_itr = std::find(s->vertices.begin(), s->vertices.end(), v1);
    while(true) {
        v_itr++;
        if(v_itr == s->vertices.end()) 
            v_itr = s->vertices.begin();
        if(*v_itr == v2) 
            break;
        v_new_surf.push_back(*v_itr);
    }
    v_new_surf.push_back(v2);
    for(auto v_itr = v_new_surf.begin() + 1; v_itr != v_new_surf.end() - 1; v_itr++) {
        s->remove(*v_itr);
        (*v_itr)->remove(s);
    }

    // Build new surface
    Surface *s_new = (*s->type())(v_new_surf);
    if(!s_new) 
        return NULL;
    add(s_new);

    // Continue hierarchy
    for(auto &b : s->getBodies()) {
        s_new->add(b);
        b->add(s_new);
    }

    if(_solver) {
        if(!qualityWorking()) 
            _solver->positionChanged();

        _solver->log(
            this, MeshLogEventType::Create, 
            {s->objId, s_new->objId, v1->objId, v2->objId}, 
            {s->objType(), s_new->objType(), v1->objType(), v2->objType()}, 
            "split"
        );
    }

    return s_new;
}

/** Find a contiguous set of surface vertices partioned by a cut plane */
static std::vector<Vertex*> Mesh_SurfaceCutPlaneVertices(Surface *s, const FVector4 &planeEq) {
    // Calculate side of cut plane
    auto s_vertices = s->getVertices();
    std::vector<bool> planeEq_newSide;
    planeEq_newSide.reserve(s_vertices.size());
    size_t num_to_new = 0;
    for(auto &v : s_vertices) {
        const bool onNewSide = planeEq.distance(v->getPosition()) > 0;
        planeEq_newSide.push_back(onNewSide);
        if(onNewSide) 
            num_to_new++;
    }

    // If either the new or current surface has insufficient vertices, exit out
    if(num_to_new == 0 || num_to_new == s_vertices.size()) {
        return {};
    }

    // Determine insertion points
    Vertex *v_new_start = NULL;
    Vertex *v_new_end = NULL;
    std::vector<Vertex*> verts_new_s;
    verts_new_s.reserve(s_vertices.size());
    std::vector<bool>::iterator b_itr = planeEq_newSide.begin() + 1;
    std::vector<bool>::iterator b_itr_prev = b_itr - 1;
    std::vector<bool>::iterator b_itr_next = b_itr + 1;
    std::vector<Vertex*>::iterator v_itr = s_vertices.begin() + 1;
    while(!v_new_end) { 
        if(!v_new_start) {
            if(*b_itr && !*b_itr_prev) {
                v_new_start = *v_itr;
                verts_new_s.push_back(v_new_start);
            }
        }
        else {
            if(*b_itr && !*b_itr_next) 
                v_new_end = *v_itr;
            
            verts_new_s.push_back(*v_itr);
        }

        b_itr_prev = b_itr;
        b_itr = b_itr_next;
        b_itr_next = b_itr_next + 1 == planeEq_newSide.end() ? planeEq_newSide.begin() : b_itr_next + 1;
        v_itr = v_itr + 1 == s_vertices.end() ? s_vertices.begin() : v_itr + 1;
    }

    return verts_new_s;
}

/** Find the coordinates and adjacent vertices of the intersections of a surface and cut plane */
static HRESULT Mesh_SurfaceCutPlanePointsPairs(
    Surface *s, 
    const FVector4 &planeEq, 
    FVector3 &pos_start, 
    FVector3 &pos_end, 
    Vertex **v_new_start, 
    Vertex **v_old_start, 
    Vertex **v_new_end, 
    Vertex **v_old_end) 
{
    // Determine insertion points
    std::vector<Vertex*> verts_new_s = Mesh_SurfaceCutPlaneVertices(s, planeEq);
    if(verts_new_s.size() == 0) 
        return E_FAIL;
    *v_new_start = verts_new_s.front();
    *v_new_end = verts_new_s.back();

    // Determine coordinates of new vertices where cut plane intersects the surface
    auto s_vertices = s->getVertices();
    *v_old_start = (*v_new_start) == s_vertices.front() ? s_vertices.back()  : *(std::find(s_vertices.begin(), s_vertices.end(), *v_new_start) - 1);
    *v_old_end   = (*v_new_end) == s_vertices.back()    ? s_vertices.front() : *(std::find(s_vertices.begin(), s_vertices.end(), *v_new_end)   + 1);

    FVector3 pos_old_start = (*v_old_start)->getPosition();
    FVector3 pos_old_end = (*v_old_end)->getPosition();

    FVector3 rel_pos_start = (*v_new_start)->getPosition() - pos_old_start;
    FVector3 rel_pos_end   = (*v_new_end)->getPosition()   - pos_old_end;
    
    FloatP_t intersect_t_start = Magnum::Math::Intersection::planeLine(planeEq, pos_old_start, rel_pos_start);
    FloatP_t intersect_t_end   = Magnum::Math::Intersection::planeLine(planeEq, pos_old_end,   rel_pos_end);
    
    // If new vertices are indeterminant, exit out
    if(!(intersect_t_start > 0 && intersect_t_start < 1 && intersect_t_end > 0 && intersect_t_end < 1)) {
        std::string msg = "Indeterminant vertices " + cast<FloatP_t, std::string>(intersect_t_start) + ", " + cast<FloatP_t, std::string>(intersect_t_end);
        tf_error(E_FAIL, msg.c_str());
        return E_FAIL;
    }

    // Return coordinates
    pos_start = pos_old_start + intersect_t_start * rel_pos_start;
    pos_end = pos_old_end   + intersect_t_end   * rel_pos_end;

    return S_OK;
}

Surface *Mesh::split(Surface *s, const FVector3 &cp_pos, const FVector3 &cp_norm) {
    FVector4 planeEq = FVector4::planeEquation(cp_norm.normalized(), cp_pos);

    FVector3 pos_start, pos_end; 
    Vertex *v_new_start, *v_old_start, *v_new_end, *v_old_end;
    if(Mesh_SurfaceCutPlanePointsPairs(s, planeEq, pos_start, pos_end, &v_new_start, &v_old_start, &v_new_end, &v_old_end) != S_OK) 
        return NULL;

    // Create and insert new vertices
    Vertex *v_start = new Vertex(pos_start);
    Vertex *v_end   = new Vertex(pos_end);
    if(insert(v_start, v_old_start, v_new_start) != S_OK || insert(v_end, v_old_end, v_new_end) != S_OK) {
        v_start->destroy();
        v_end->destroy();
        delete v_start;
        delete v_end;
        return NULL;
    }

    // Create new surface
    return split(s, v_start, v_end);
}


struct Mesh_BodySplitEdge {
    Vertex *v_oldSide;  // Old side
    Vertex *v_newSide;  // New side
    FVector3 intersect_pt;
    std::vector<Surface*> surfaces;

    bool operator==(Mesh_BodySplitEdge o) const {
        return v_oldSide == o.v_oldSide && v_newSide == o.v_newSide;
    }

    static bool intersects(Vertex *v1, Vertex *v2, const FVector4 &planeEq, FVector3 &intersect_pt) {
        FVector3 pos_old = v1->getPosition();
        FVector3 rel_pos = v2->getPosition() - pos_old;
        FloatP_t intersect_t = Magnum::Math::Intersection::planeLine(planeEq, pos_old, rel_pos);
        
        // If new position is indeterminant, exit out
        if(!(intersect_t > 0 && intersect_t < 1)) 
            return false;

        // Return coordinates
        intersect_pt = pos_old + intersect_t * rel_pos;
        return true;
    }

    static std::vector<Surface*> extractSurfaces(Vertex *v1, Vertex *v2, Body *b) {
        std::vector<Surface*> surfaces;
        for(auto &s : b->getSurfaces()) 
            if(v1->in(s) && v2->in(s)) 
                surfaces.push_back(s);
        return surfaces;
    }

    static std::vector<Mesh_BodySplitEdge> construct(Body *b, const FVector4 &planeEq) {
        std::map<std::pair<int, int>, Mesh_BodySplitEdge> edgeMap;
        for(auto &s : b->getSurfaces()) {
            for(auto &v : s->getVertices()) {
                Vertex *va, *vb;
                Vertex *v_lower, *v_upper;
                std::tie(va, vb) = s->neighborVertices(v);

                std::vector<std::pair<Vertex*, Vertex*> > edge_cases;
                if(v->objId < va->objId) 
                    edge_cases.push_back({v, va});
                if(v->objId < vb->objId) 
                    edge_cases.push_back({v, vb});

                for(auto &ec : edge_cases) {
                    std::tie(v_lower, v_upper) = ec;

                    FVector3 intersect_pt;
                    if(intersects(v_lower, v_upper, planeEq, intersect_pt)) {
                        Mesh_BodySplitEdge edge;
                        if(planeEq.distance(v_lower->getPosition()) > 0) {
                            edge.v_oldSide = v_upper;
                            edge.v_newSide = v_lower;
                        } 
                        else {
                            edge.v_oldSide = v_lower;
                            edge.v_newSide = v_upper;
                        }
                        edge.intersect_pt = intersect_pt;
                        edge.surfaces = extractSurfaces(v_lower, v_upper, b);

                        if(edge.surfaces.size() != 2) {
                            tf_error(E_FAIL, "Incorrect number of extracted surfaces");
                            return {};
                        }

                        edgeMap.insert({{v_lower->objId, v_upper->objId}, edge});
                    }
                }
            }
        }

        std::vector<Mesh_BodySplitEdge> result;
        result.reserve(edgeMap.size());
        for(auto &itr : edgeMap) 
            result.push_back(itr.second);
        
        std::vector<Mesh_BodySplitEdge> result_sorted;
        result_sorted.reserve(result.size());
        result_sorted.push_back(result.back());
        result.pop_back();
        Surface *s_target = result_sorted[0].surfaces[1];
        while(!result.empty()) {
            std::vector<Mesh_BodySplitEdge>::iterator itr = result.begin();
            while(itr != result.end()) { 
                if(itr->surfaces[0] == s_target) { 
                    s_target = itr->surfaces[1];
                    result_sorted.push_back(*itr);
                    result.erase(itr);
                    break;
                } 
                else if(itr->surfaces[1] == s_target) {
                    s_target = itr->surfaces[0];
                    result_sorted.push_back(*itr);
                    result.erase(itr);
                    break;
                }
                itr++;
            }
        }
        
        return result_sorted;
    }

    typedef std::pair<Surface*, std::pair<Mesh_BodySplitEdge, Mesh_BodySplitEdge> > surfaceSplitPlanEl_t;

    static HRESULT surfaceSplitPlan(const std::vector<Mesh_BodySplitEdge> &edges, std::vector<surfaceSplitPlanEl_t> &result) {
        std::vector<Mesh_BodySplitEdge> edges_copy(edges);
        edges_copy.push_back(edges.front());
        for(size_t i = 0; i < edges.size(); i++) {
            Mesh_BodySplitEdge edge_i = edges_copy[i];
            Mesh_BodySplitEdge edge_j = edges_copy[i + 1];
            Surface *s = NULL;
            for(auto &si : edge_i.surfaces) {
                auto itr = std::find(edge_j.surfaces.begin(), edge_j.surfaces.end(), si);
                if(itr != edge_j.surfaces.end()) {
                    s = *itr;
                    break;
                }
            }
            if(!s) 
                return E_FAIL;
            
            Vertex *v_old = edge_i.v_oldSide;
            Vertex *v_new = edge_i.v_newSide;
            Vertex *v_old_na, *v_old_nb;
            std::tie(v_old_na, v_old_nb) = s->neighborVertices(v_old);
            if(v_old_na == v_new) 
                result.push_back({s, {edge_i, edge_j}});
            else 
                result.push_back({s, {edge_j, edge_i}});
        }
        return S_OK;
    }

    static HRESULT vertexConstructorPlan(const std::vector<surfaceSplitPlanEl_t> &splitPlan, std::vector<Mesh_BodySplitEdge> &vertexPlan) {
        vertexPlan.clear();
        for(auto itr = splitPlan.begin(); itr != splitPlan.end(); itr++) {
            auto edges = itr->second;
            auto edges_prev = itr == splitPlan.begin() ? splitPlan.back().second : (itr - 1)->second;
            if(edges.first == edges_prev.first || edges.first == edges_prev.second) 
                vertexPlan.push_back(edges.first);
            else 
                vertexPlan.push_back(edges.second);
        }
        return S_OK;
    }
};

Body *Mesh::split(Body *b, const FVector3 &cp_pos, const FVector3 &cp_norm, SurfaceType *stype) {
    FVector4 planeEq = FVector4::planeEquation(cp_norm.normalized(), cp_pos);

    // Determine which surfaces are moved to new body
    std::vector<Surface*> surfs_moved;
    for(auto &s : b->surfaces) {
        size_t num_newSide = 0;
        for(auto &v : s->vertices) 
            if(planeEq.distance(v->getPosition()) > 0) 
                num_newSide++;
        if(num_newSide == s->vertices.size()) 
            surfs_moved.push_back(s);
    }

    // Build edge list
    std::vector<Mesh_BodySplitEdge> splitEdges = Mesh_BodySplitEdge::construct(b, planeEq);

    // Split edges
    std::vector<Mesh_BodySplitEdge::surfaceSplitPlanEl_t> sSplitPlan;
    if(Mesh_BodySplitEdge::surfaceSplitPlan(splitEdges, sSplitPlan) != S_OK) 
        return NULL;
    std::vector<Mesh_BodySplitEdge> vertexPlan;
    if(Mesh_BodySplitEdge::vertexConstructorPlan(sSplitPlan, vertexPlan) != S_OK) 
        return NULL;
    std::vector<Vertex*> new_vertices;
    std::map<std::pair<int, int>, Vertex*> new_vertices_map;
    for(auto &edge : vertexPlan) {
        Vertex *v_new = new Vertex(edge.intersect_pt);
        insert(v_new, edge.v_oldSide, edge.v_newSide);
        new_vertices.push_back(v_new);
        new_vertices_map.insert({{edge.v_oldSide->objId, edge.v_newSide->objId}, v_new});
    }

    // Split surfaces
    std::vector<Surface*> new_surfs;
    new_surfs.reserve(sSplitPlan.size());
    for(size_t i = 0; i < sSplitPlan.size(); i++) {
        Surface *s = sSplitPlan[i].first;
        Vertex *v1 = new_vertices_map[{sSplitPlan[i].second.first.v_oldSide->objId,  sSplitPlan[i].second.first.v_newSide->objId}];
        Vertex *v2 = new_vertices_map[{sSplitPlan[i].second.second.v_oldSide->objId, sSplitPlan[i].second.second.v_newSide->objId}];
        Surface *s_new = split(s, v1, v2);
        if(!s_new) 
            return NULL;
        new_surfs.push_back(s_new);
    }

    // Construct interface surface
    if(!stype) 
        stype = new_surfs[0]->type();
    Surface *s_new = (*stype)(new_vertices);
    if(!s_new) 
        return NULL;
    add(s_new);
    b->add(s_new);
    s_new->add(b);
    s_new->positionChanged();

    // Transfer moved and new split surfaces to new body
    for(auto &s : surfs_moved) {
        s->remove(b);
        b->remove(s);
    }
    for(auto &s : new_surfs) {
        s->remove(b);
        b->remove(s);
    }
    b->positionChanged();

    // Construct new body
    std::vector<Surface*> new_body_surfs(surfs_moved);
    new_body_surfs.push_back(s_new);
    for(auto &s : new_surfs) 
        new_body_surfs.push_back(s);
    Body *b_new = (*b->type())(new_body_surfs);
    if(!b_new) 
        return NULL;

    add(b_new);

    if(_solver) {
        if(!qualityWorking()) 
            _solver->positionChanged();

        _solver->log(this, MeshLogEventType::Create, {b->objId, b_new->objId}, {b->objType(), b_new->objType()}, "split");
    }

    return b_new;
}
