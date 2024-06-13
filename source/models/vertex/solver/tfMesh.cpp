/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego and Tien Comlekoglu
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
#include "tf_mesh_io.h"
#include "tfVertexSolverFIO.h"

#include <tfError.h>
#include <tfEngine.h>
#include <tfLogger.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>
#include <tfTaskScheduler.h>

#include <algorithm>
#include <map>


#define TF_MESH_GETPART(idx, inv) idx >= inv.size() ? NULL : &inv[idx]


using namespace TissueForge;
using namespace TissueForge::models::vertex;


#define TF_INVALIDOBJRM_MSG "Invalid mesh object passed for remove."
#define TF_MESH_OBJCHECK(obj) !(obj) || (obj)->objectId() < 0
#define TF_MESH_OBJINVSIZECHECK(obj, sz) (obj)->_objId >= (sz)


template <typename T> 
HRESULT Mesh_checkStoredObj(T *obj) {
    if(TF_MESH_OBJCHECK(obj)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}


#define TF_MESH_OBJINVCHECK(obj, sz) \
    { \
        if(TF_MESH_OBJINVSIZECHECK(obj, sz)) { \
            TF_Log(LOG_ERROR) << "Object with id " << obj->_objId << " exceeds inventory (" << sz << ")"; \
            return E_FAIL; \
        } \
    }


//////////
// Mesh //
//////////


Mesh::Mesh() : 
    vertices{new std::vector<Vertex>()},
    surfaces{new std::vector<Surface>()},
    bodies{new std::vector<Body>()},
    nr_vertices{0}, 
    nr_surfaces{0}, 
    nr_bodies{0}, 
    _quality{new MeshQuality()}
{}

Mesh::~Mesh() { 
    if(vertices) 
        delete vertices;
    vertices = 0;

    if(surfaces) 
        delete surfaces;
    surfaces = 0;

    if(bodies) 
        delete bodies;
    bodies = 0;

    if(_quality) { 
        delete _quality;
        _quality = 0;
    }
}

std::string Mesh::str() const {
    std::stringstream ss;

    ss << "Mesh(";
    ss << "nr_vertices=" << this->nr_vertices << "(" << this->sizeVertices() << "), ";
    ss << "nr_surfaces=" << this->nr_surfaces << "(" << this->sizeSurfaces() << "), ";
    ss << "nr_bodies="   << this->nr_bodies   << "(" << this->sizeBodies()   << "), ";
    ss << "quality=" << (this->hasQuality() ? "yes" : "no");
    ss << ")";

    return ss.str();
}

HRESULT Mesh::setQuality(MeshQuality *quality) {
    if(_quality) delete _quality;

    _quality = quality;
    return S_OK;
}

HRESULT Mesh::incrementVertices(const size_t &numIncr) {
    int new_size = vertices->size() + numIncr;
    std::vector<Vertex> *new_vertices = new std::vector<Vertex>();
    new_vertices->reserve(new_size);
    for(unsigned int i = 0; i < vertices->size(); i++) 
        new_vertices->push_back((*vertices)[i]);
    for(unsigned int i = vertices->size(); i < new_size; i++) {
        new_vertices->push_back(Vertex());
        vertexIdsAvail.insert(i);
    }

    if(surfaces->size() > 0) {

        Surface *m_surfaces = &(*surfaces)[0];
        auto func_surfs = [&m_surfaces, &new_vertices](int sid) -> void {
            Surface &s = m_surfaces[sid];
            if(s._objId < 0) 
                return;
            std::vector<Vertex*> &s_vertices = s.vertices;
            for(unsigned int i = 0; i < s_vertices.size(); i++) {
                Vertex *v = s_vertices[i];
                s.vertices[i] = &(*new_vertices)[v->_objId];
            }
        };
        parallel_for(surfaces->size(), func_surfs);

    }

    auto &m_verticesByPID = verticesByPID;
    auto func_vertices = [&m_verticesByPID, &new_vertices](int vid) -> void {
        Vertex &v = (*new_vertices)[vid];
        if(v._objId < 0) 
            return;
        m_verticesByPID[v.pid]  = &(*new_vertices)[vid];
        std::vector<Vertex*> &v_connectedVertices = v._connectedVertices;
        for(unsigned int i = 0; i < v_connectedVertices.size(); i++) {
            Vertex *nv = v_connectedVertices[i];
            v._connectedVertices[i] = &(*new_vertices)[nv->_objId];
        }
    };
    parallel_for(vertices->size(), func_vertices);
    
    if(vertices) 
        delete vertices;
    vertices = new_vertices;
    return S_OK;
}

HRESULT Mesh::incrementSurfaces(const size_t &numIncr) {
    int new_size = surfaces->size() + numIncr;
    std::vector<Surface> *new_surfaces = new std::vector<Surface>();
    new_surfaces->reserve(new_size);
    for(unsigned int i = 0; i < surfaces->size(); i++) 
        new_surfaces->push_back((*surfaces)[i]);
    for(unsigned int i = surfaces->size(); i < new_size; i++) {
        new_surfaces->push_back(Surface());
        surfaceIdsAvail.insert(i);
    }

    if(vertices->size() > 0) {

        Vertex *m_vertices = &(*vertices)[0];
        auto func_verts = [&m_vertices, &new_surfaces](int vid) -> void {
            Vertex &v = m_vertices[vid];
            if(v._objId < 0) 
                return;
            std::vector<Surface*> &v_surfaces = v.surfaces;
            for(unsigned int i = 0; i < v_surfaces.size(); i++) {
                Surface *s = v_surfaces[i];
                v.surfaces[i] = &(*new_surfaces)[s->_objId];
            }
        };
        parallel_for(vertices->size(), func_verts);

    }

    if(bodies->size() > 0) {

        Body *m_bodies = &(*bodies)[0];
        auto func_bodies = [&m_bodies, &new_surfaces](int bid) -> void {
            Body &b = m_bodies[bid];
            if(b._objId < 0) 
                return;
            std::vector<Surface*> &b_surfaces = b.surfaces;
            for(unsigned int i = 0; i < b_surfaces.size(); i++) {
                Surface *s = b_surfaces[i];
                b.surfaces[i] = &(*new_surfaces)[s->_objId];
            }
        };
        parallel_for(bodies->size(), func_bodies);

    }
    
    if(surfaces) 
        delete surfaces;
    surfaces = new_surfaces;
    return S_OK;
}

HRESULT Mesh::incrementBodies(const size_t &numIncr) {
    int new_size = bodies->size() + numIncr;
    std::vector<Body> *new_bodies = new std::vector<Body>();
    new_bodies->reserve(new_size);
    for(unsigned int i = 0; i < bodies->size(); i++) 
        new_bodies->push_back((*bodies)[i]);
    for(unsigned int i = bodies->size(); i < new_size; i++) {
        new_bodies->push_back(Body());
        bodyIdsAvail.insert(i);
    }

    if(surfaces->size() > 0) {

        Surface *m_surfaces = &(*surfaces)[0];
        auto func_surfs = [&m_surfaces, &new_bodies](int sid) -> void {
            Surface &s = m_surfaces[sid];
            if(s._objId < 0) 
                return;
            if(s.b1) {
                s.b1 = &(*new_bodies)[s.b1->_objId];
            }
            if(s.b2) {
                s.b2 = &(*new_bodies)[s.b2->_objId];
            }
        };
        parallel_for(surfaces->size(), func_surfs);

    }
    
    if(bodies) 
        delete bodies;
    bodies = new_bodies;
    return S_OK;
}

HRESULT Mesh::allocateVertex(Vertex **obj) {
    // Check for available space; reallocate and reassign throughout if necessary
    if(vertexIdsAvail.empty() && incrementVertices() != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    std::set<unsigned int>::iterator itr = vertexIdsAvail.begin();
    int objId = (int)*itr;
    vertexIdsAvail.erase(itr);
    *obj = &(*vertices)[objId];
    (*obj)->_objId = objId;
    nr_vertices++;
    return S_OK;
}

HRESULT Mesh::allocateSurface(Surface **obj) {
    // Check for available space; reallocate and reassign throughout if necessary
    if(surfaceIdsAvail.empty() && incrementSurfaces() != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    std::set<unsigned int>::iterator itr = surfaceIdsAvail.begin();
    int objId = (int)*itr;
    surfaceIdsAvail.erase(itr);
    *obj = &(*surfaces)[objId];
    (*obj)->_objId = objId;
    nr_surfaces++;
    return S_OK;
}

HRESULT Mesh::allocateBody(Body **obj) {
    // Check for available space; reallocate and reassign throughout if necessary
    if(bodyIdsAvail.empty() && incrementBodies() != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    std::set<unsigned int>::iterator itr = bodyIdsAvail.begin();
    int objId = (int)*itr;
    bodyIdsAvail.erase(itr);
    *obj = &(*bodies)[objId];
    (*obj)->_objId = objId;
    nr_bodies++;
    return S_OK;
}

HRESULT Mesh::allocateVertices(Vertex ***objs, const size_t& numObjs) {
    // Check for available space; reallocate and reassign throughout if necessary
    const int numNeeded = (int)numObjs - (int)vertexIdsAvail.size();
    if(numNeeded > 0 && incrementVertices(numNeeded) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    for(unsigned int i = 0; i < numObjs; i++) {
        std::set<unsigned int>::iterator itr = vertexIdsAvail.begin();
        int objId = (int)*itr;
        vertexIdsAvail.erase(itr);
        auto obj = &(*objs)[i];
        *obj = &(*vertices)[objId];
        (*obj)->_objId = objId;
    }
    nr_vertices += numObjs;
    return S_OK;
}

HRESULT Mesh::allocateSurfaces(Surface ***objs, const size_t& numObjs) {
    // Check for available space; reallocate and reassign throughout if necessary
    const int numNeeded = (int)numObjs - (int)surfaceIdsAvail.size();
    if(numNeeded > 0 && incrementSurfaces(numNeeded) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    for(unsigned int i = 0; i < numObjs; i++) {
        std::set<unsigned int>::iterator itr = surfaceIdsAvail.begin();
        int objId = (int)*itr;
        surfaceIdsAvail.erase(itr);
        auto obj = &(*objs)[i];
        *obj = &(*surfaces)[objId];
        (*obj)->_objId = objId;
    }
    nr_surfaces += numObjs;
    return S_OK;
}

HRESULT Mesh::allocateBodies(Body ***objs, const size_t& numObjs) {
    // Check for available space; reallocate and reassign throughout if necessary
    const int numNeeded = (int)numObjs - (int)bodyIdsAvail.size();
    if(numNeeded > 0 && incrementBodies(numNeeded) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    for(unsigned int i = 0; i < numObjs; i++) {
        std::set<unsigned int>::iterator itr = bodyIdsAvail.begin();
        int objId = (int)*itr;
        bodyIdsAvail.erase(itr);
        auto obj = &(*objs)[i];
        *obj = &(*bodies)[objId];
        (*obj)->_objId = objId;
    }
    nr_bodies++;
    return S_OK;
}

HRESULT Mesh::ensureAvailableVertices(const size_t &numAlloc) {
    int nr_diff = vertices->size() - nr_vertices - numAlloc;
    return nr_diff >= 0 ? S_OK : incrementVertices(-nr_diff);
}

HRESULT Mesh::ensureAvailableSurfaces(const size_t &numAlloc) {
    int nr_diff = surfaces->size() - nr_surfaces - numAlloc;
    return nr_diff >= 0 ? S_OK : incrementSurfaces(-nr_diff);
}

HRESULT Mesh::ensureAvailableBodies(const size_t &numAlloc) {
    int nr_diff = bodies->size() - nr_bodies - numAlloc;
    return nr_diff >= 0 ? S_OK : incrementBodies(-nr_diff);
}

HRESULT Mesh::create(Vertex **obj, const unsigned int &pid) { 
    isDirty = true;
    if(_solver) 
        _solver->setDirty(true);

    if(allocateVertex(obj) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    verticesByPID[pid] = *obj;

    return S_OK;
}

HRESULT Mesh::create(Vertex*** objs, const std::vector<unsigned int>& pids) {
    isDirty = true;
    if(_solver) 
        _solver->setDirty(true);

    if(allocateVertices(objs, pids.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    auto pidsItr = pids.begin();
    for(unsigned int i = 0; i < pids.size(); i++, pidsItr++) 
        verticesByPID[*pidsItr] = (*objs)[i];

    return S_OK;
}

HRESULT Mesh::create(Surface **obj){ 
    isDirty = true;
    if(_solver) 
        _solver->setDirty(true);

    if(allocateSurface(obj) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT Mesh::create(Surface*** objs, const size_t& numObjs) {
    isDirty = true;
    if(_solver) 
        _solver->setDirty(true);

    if(allocateSurfaces(objs, numObjs) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT Mesh::create(Body **obj){ 
    isDirty = true;
    if(_solver) 
        _solver->setDirty(true);

    if(allocateBody(obj) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT Mesh::create(Body*** objs, const size_t& numObjs) {
    isDirty = true;
    if(_solver) 
        _solver->setDirty(true);

    if(allocateBodies(objs, numObjs) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

Mesh *Mesh::get() {
    MeshSolver *solver = MeshSolver::get();
    return solver ? solver->getMesh() : NULL;
}

Vertex *Mesh::findVertex(const FVector3 &pos, const FloatP_t &tol) {

    for(size_t i = 0; i < vertices->size(); i++) {
        Vertex &v = (*vertices)[i];
        if(v._objId >= 0 && v.particle()->relativePosition(pos).length() <= tol) 
            return &v;
    }

    return NULL;
}

Vertex *Mesh::getVertexByPID(const unsigned int &pid) const {

    auto itr = verticesByPID.find(pid);
    return itr == verticesByPID.end() ? NULL : itr->second;
}

Vertex *Mesh::getVertex(const unsigned int &idx) {
    auto o = TF_MESH_GETPART(idx, (*vertices));
    return o && o->objectId() >= 0 ? o : NULL;
}

Surface *Mesh::getSurface(const unsigned int &idx) {
    auto o = TF_MESH_GETPART(idx, (*surfaces));
    return o && o->objectId() >= 0 ? o : NULL;
}

Body *Mesh::getBody(const unsigned int &idx) {
    auto o = TF_MESH_GETPART(idx, (*bodies));
    return o && o->objectId() >= 0 ? o : NULL;
}

template <typename T> 
bool Mesh_validateInv(std::vector<T> &inv) {
    for(size_t i = 0; i < inv.size(); i++) {
        T &o = inv[i];
        if(o.objectId() < 0 && !o.validate()) 
            return false;
    }
    return true;
}

bool Mesh::validate() {
    if(!Mesh_validateInv(*this->vertices)) 
        return false;
    else if(!Mesh_validateInv(*this->surfaces)) 
        return false;
    else if(!Mesh_validateInv(*this->bodies)) 
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
    for(auto &s1 : v1->surfaces) {
        if(s1->vertices.front()->objectId() == v1->objectId() && s1->vertices.back()->objectId() == v2->objectId() || 
            s1->vertices.front()->objectId() == v2->objectId() && s1->vertices.back()->objectId() == v1->objectId()) 
            return true;
        for(auto vitr = s1->vertices.begin() + 1; vitr != s1->vertices.end(); vitr++) 
            if(((*vitr)->objectId() == v1->objectId() && (*(vitr - 1))->objectId() == v2->objectId()) || 
                ((*vitr)->objectId() == v2->objectId() && (*(vitr - 1))->objectId() == v1->objectId())) 
                return true;
    }

    return false;
}

bool Mesh::connected(const Surface *s1, const Surface *s2) const {
    for(auto &v : s1->vertices) 
        if(v->defines(s2)) 
            return true;
    return false;
}

bool Mesh::connected(const Body *b1, const Body *b2) const {
    for(auto &s : b1->surfaces) 
        if(s->defines(b2)) 
            return true;
    return false;
}

// Mesh editing

HRESULT Mesh::remove(Vertex *v) {
    isDirty = true;

    if(Mesh_checkStoredObj(v) != S_OK) {
        TF_Log(LOG_ERROR) << TF_INVALIDOBJRM_MSG;
        return E_FAIL;
    } 

    TF_MESH_OBJINVCHECK(v, vertices->size());
    auto& children = v->getSurfaces();
    this->vertexIdsAvail.insert(v->_objId);
    if(v->pid >= 0) {
        auto itr = verticesByPID.find(v->pid);
        if(itr != verticesByPID.end()) 
            verticesByPID.erase(itr);
    }
    if(_quality) 
        _quality->includeVertex(v->_objId);
    (*this->vertices)[v->_objId] = Vertex();
    nr_vertices--;

    if(_solver)
        _solver->log(MeshLogEventType::Destroy, {v->_objId}, {v->objType()});

    for(auto &c : children) 
        if(remove(c) != S_OK) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }

    return S_OK;
}

HRESULT Mesh::remove(Vertex** v, const size_t& numObjs) {
    isDirty = true;

    // check objects and get children, vertex ids and particle ids

    const size_t nObjs = vertices->size();
    std::vector<HRESULT> statusPool(ThreadPool::size(), S_OK);
    std::vector<std::unordered_set<Surface*> > childrenPool(ThreadPool::size());
    std::vector<std::unordered_set<unsigned int> > idsToMakeAvailPool(ThreadPool::size());
    std::vector<std::unordered_set<int> > pidsToErasePool(ThreadPool::size());
    auto& this_verticesByPID = this->verticesByPID;
    parallel_for(
        ThreadPool::size(), 
        [&v, &numObjs, &statusPool, &nObjs, &childrenPool, &idsToMakeAvailPool, &pidsToErasePool, &this_verticesByPID](int tid) -> void {
            HRESULT& statusThread = statusPool[tid];
            std::unordered_set<Surface*>& childrenThread = childrenPool[tid];
            std::unordered_set<unsigned int>& idsToMakeAvailThread = idsToMakeAvailPool[tid];
            std::unordered_set<int>& pidsToEraseThread = pidsToErasePool[tid];
            for(int i = tid; i < numObjs; i += ThreadPool::size()) {
                Vertex* obj = v[i];
                if(TF_MESH_OBJCHECK(obj) || TF_MESH_OBJINVSIZECHECK(obj, nObjs)) {
                    statusThread = E_FAIL;
                    return;
                }
                for(auto& c : obj->getSurfaces()) 
                    childrenThread.insert(c);
                idsToMakeAvailThread.insert(obj->_objId);
                auto itr = this_verticesByPID.find(obj->pid);
                if(itr != this_verticesByPID.end()) 
                    pidsToEraseThread.insert(obj->pid);
            }
        }
    );
    for(auto& statusThread : statusPool) 
        if(statusThread) {
            TF_Log(LOG_ERROR) << TF_INVALIDOBJRM_MSG;
            return E_FAIL;
        }

    size_t numChildren = 0;
    for(auto& childrenThread : childrenPool) 
        numChildren += childrenThread.size();
    std::unordered_set<Surface*> children;
    children.reserve(numChildren);
    for(auto& childrenThread : childrenPool) 
        for(auto& c : childrenThread) 
            children.insert(c);
    std::vector<Surface*> childrenVec(children.begin(), children.end());

    // make ids available and remove particle id mappings
    
    for(auto& idsToMakeAvailThread : idsToMakeAvailPool) 
        this->vertexIdsAvail.insert(idsToMakeAvailThread.begin(), idsToMakeAvailThread.end());
    for(auto& pidsToEraseThread : pidsToErasePool) 
        for(auto& pid : pidsToEraseThread) 
            this->verticesByPID.erase(pid);

    // notify quality

    if(_quality) 
        for(auto& idsToMakeAvailThread : idsToMakeAvailPool) 
            for(auto& objId : idsToMakeAvailThread) 
                _quality->includeVertex(objId);

    // clear objects

    auto& this_objs = *(this->vertices);
    parallel_for(numObjs, [&this_objs, &v](int i) -> void { this_objs[v[i]->_objId] = Vertex(); } );
    nr_vertices -= numObjs;

    // log

    if(_solver) 
        for(unsigned int i = 0; i < numObjs; i++) {
            auto obj = v[i];
            _solver->log(MeshLogEventType::Destroy, {obj->_objId}, {obj->objType()});
        }

    // remove children

    if(remove(childrenVec.data(), childrenVec.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT Mesh::remove(Surface *s) {
    isDirty = true;

    if(Mesh_checkStoredObj(s) != S_OK) {
        TF_Log(LOG_ERROR) << TF_INVALIDOBJRM_MSG;
        return E_FAIL;
    } 

    TF_MESH_OBJINVCHECK(s, surfaces->size());
    std::vector<Body*> children = s->getBodies();
    for(auto &v : s->getVertices()) 
        v->remove(s);
    this->surfaceIdsAvail.insert(s->_objId);
    if(_quality) 
        _quality->includeSurface(s->_objId);
    (*this->surfaces)[s->_objId] = Surface();
    nr_surfaces--;

    if(_solver)
        _solver->log(MeshLogEventType::Destroy, {s->_objId}, {s->objType()});

    for(auto &c : children) 
        if(remove(c) != S_OK) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }

    return S_OK;
}

HRESULT Mesh::remove(Surface** s, const size_t& numObjs) {
    isDirty = true;

    // check objects, get children, ids and surfaces by vertex

    const size_t nObjs = surfaces->size();
    std::vector<HRESULT> statusPool(ThreadPool::size(), S_OK);
    std::vector<std::unordered_set<Vertex*> > verticesPool(ThreadPool::size());
    std::vector<std::unordered_set<Body*> > childrenPool(ThreadPool::size());
    std::vector<std::unordered_set<unsigned int> > idsToMakeAvailPool(ThreadPool::size());
    std::vector<std::unordered_map<Vertex*, std::unordered_set<Surface*> > > surfacesByVerticesPool(ThreadPool::size());
    parallel_for(
        ThreadPool::size(), 
        [&s, &numObjs, &nObjs, &statusPool, &verticesPool, &childrenPool, &idsToMakeAvailPool, &surfacesByVerticesPool](int tid) -> void {
            HRESULT& statusThread = statusPool[tid];
            std::unordered_set<Vertex*>& verticesThread = verticesPool[tid];
            std::unordered_set<Body*>& childrenThread = childrenPool[tid];
            std::unordered_set<unsigned int>& idsToMakeAvailThread = idsToMakeAvailPool[tid];
            std::unordered_map<Vertex*, std::unordered_set<Surface*> >& surfacesByVerticesThread = surfacesByVerticesPool[tid];
            for(int i = tid; i < numObjs; i += ThreadPool::size()) {
                Surface* obj = s[i];
                if(TF_MESH_OBJCHECK(obj) || TF_MESH_OBJINVSIZECHECK(obj, nObjs)) {
                    statusThread = E_FAIL;
                    return;
                }
                for(auto& c : obj->getBodies()) 
                    childrenThread.insert(c);
                idsToMakeAvailThread.insert(obj->_objId);
                for(auto& v : obj->getVertices()) {
                    verticesThread.insert(v);
                    surfacesByVerticesThread[v].insert(obj);
                }
            }
        }
    );
    for(auto& statusThread : statusPool) 
        if(statusThread != S_OK) {
            TF_Log(LOG_ERROR) << TF_INVALIDOBJRM_MSG;
            return E_FAIL;
        }

    size_t numChildren = 0;
    for(auto& childrenThread : childrenPool) 
        numChildren += childrenThread.size();
    std::unordered_set<Body*> children;
    children.reserve(numChildren);
    for(auto& childrenThread : childrenPool) 
        children.insert(childrenThread.begin(), childrenThread.end());
    std::vector<Body*> childrenVec(children.begin(), children.end());

    size_t numVertices = 0;
    for(auto& verticesThread : verticesPool) 
        numVertices += verticesThread.size();
    std::unordered_set<Vertex*> vertices;
    vertices.reserve(numVertices);
    for(auto& verticesThread : verticesPool) 
        vertices.insert(verticesThread.begin(), verticesThread.end());
    std::vector<Vertex*> verticesVec(vertices.begin(), vertices.end());

    std::unordered_map<Vertex*, std::unordered_set<Surface*> > surfacesByVertices;
    for(auto& surfacesByVerticesThread : surfacesByVerticesPool) 
        for(auto& p : surfacesByVerticesThread) 
            surfacesByVertices[p.first].insert(p.second.begin(), p.second.end());

    // remove surfaces from vertices

    parallel_for(
        verticesVec.size(), 
        [&verticesVec, &surfacesByVertices](int i) -> void {
            Vertex* v = verticesVec[i];
            for(auto& s : surfacesByVertices[v]) 
                v->remove(s);
        }
    );

    // make ids available
    
    for(auto& idsToMakeAvailThread : idsToMakeAvailPool) 
        this->surfaceIdsAvail.insert(idsToMakeAvailThread.begin(), idsToMakeAvailThread.end());

    // notify quality

    if(_quality) 
        for(auto& idsToMakeAvailThread : idsToMakeAvailPool) 
            for(auto& objId : idsToMakeAvailThread) 
                _quality->includeSurface(objId);

    // clear objects

    auto& this_objs = *(this->surfaces);
    parallel_for(numObjs, [&this_objs, &s](int i) -> void { this_objs[s[i]->_objId] = Surface(); } );
    nr_surfaces -= numObjs;

    // log

    if(_solver) 
        for(unsigned int i = 0; i < numObjs; i++) {
            auto obj = s[i];
            _solver->log(MeshLogEventType::Destroy, {obj->_objId}, {obj->objType()});
        }

    // remove children

    if(remove(childrenVec.data(), childrenVec.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT Mesh::remove(Body *b) {
    isDirty = true;

    if(Mesh_checkStoredObj(b) != S_OK) {
        TF_Log(LOG_ERROR) << TF_INVALIDOBJRM_MSG;
        return E_FAIL;
    } 

    TF_MESH_OBJINVCHECK(b, bodies->size());
    for(auto &s : b->getSurfaces()) 
        s->remove(b);
    this->bodyIdsAvail.insert(b->_objId);
    if(_quality) 
        _quality->includeBody(b->_objId);
    (*this->bodies)[b->_objId] = Body();
    nr_bodies--;

    if(_solver)
        _solver->log(MeshLogEventType::Destroy, {b->_objId}, {b->objType()});

    return S_OK;
}

HRESULT Mesh::remove(Body** b, const size_t& numObjs) {
    isDirty = true;

    // check objects, get surfaces and bodies by surface

    const size_t nObjs = bodies->size();
    std::vector<HRESULT> statusPool(ThreadPool::size(), S_OK);
    std::vector<std::unordered_set<Surface*> > surfacesPool(ThreadPool::size());
    std::vector<std::unordered_set<unsigned int> > idsToMakeAvailPool(ThreadPool::size());
    std::vector<std::unordered_map<Surface*, std::unordered_set<Body*> > > bodiesBySurfacePool(ThreadPool::size());
    parallel_for(
        ThreadPool::size(), 
        [&b, &numObjs, &nObjs, &statusPool, &surfacesPool, &idsToMakeAvailPool, &bodiesBySurfacePool](int tid) -> void {
            HRESULT& statusThread = statusPool[tid];
            std::unordered_set<Surface*>& surfacesThread = surfacesPool[tid];
            std::unordered_set<unsigned int>& idsToMakeAvailThread = idsToMakeAvailPool[tid];
            std::unordered_map<Surface*, std::unordered_set<Body*> >& bodiesBySurfaceThread = bodiesBySurfacePool[tid];
            for(int i = tid; i < numObjs; i += ThreadPool::size()) {
                Body* obj = b[i];
                if(TF_MESH_OBJCHECK(obj) || TF_MESH_OBJINVSIZECHECK(obj, nObjs)) {
                    statusThread = E_FAIL;
                    return;
                }
                idsToMakeAvailThread.insert(obj->_objId);
                for(auto& s : obj->getSurfaces()) {
                    surfacesThread.insert(s);
                    bodiesBySurfaceThread[s].insert(obj);
                }
            }
        }
    );
    for(auto& statusThread : statusPool) 
        if(statusThread != S_OK) {
            TF_Log(LOG_ERROR) << TF_INVALIDOBJRM_MSG;
            return E_FAIL;
        }

    size_t numSurfaces = 0;
    for(auto& surfacesThread : surfacesPool) 
        numSurfaces += surfacesThread.size();
    std::unordered_set<Surface*> surfaces;
    surfaces.reserve(numSurfaces);
    for(auto& surfacesThread : surfacesPool) 
        surfaces.insert(surfacesThread.begin(), surfacesThread.end());
    std::vector<Surface*> surfacesVec(surfaces.begin(), surfaces.end());

    std::unordered_map<Surface*, std::unordered_set<Body*> > bodiesBySurface;
    for(auto& bodiesBySurfaceThread : bodiesBySurfacePool) 
        for(auto& p : bodiesBySurfaceThread) 
            bodiesBySurface[p.first].insert(p.second.begin(), p.second.end());

    // remove bodies from surfaces

    parallel_for(
        surfaces.size(), 
        [&surfacesVec, &bodiesBySurface](int i) -> void {
            Surface* s = surfacesVec[i];
            for(auto& b : bodiesBySurface[s]) 
                s->remove(b);
        }
    );

    // make ids available
    
    for(auto& idsToMakeAvailThread : idsToMakeAvailPool) 
        this->bodyIdsAvail.insert(idsToMakeAvailThread.begin(), idsToMakeAvailThread.end());

    // notify quality

    if(_quality) 
        for(auto& idsToMakeAvailThread : idsToMakeAvailPool) 
            for(auto& objId : idsToMakeAvailThread) 
                _quality->includeBody(objId);

    // clear objects

    auto& this_objs = *(this->bodies);
    parallel_for(numObjs, [&this_objs, &b](int i) -> void { this_objs[b[i]->_objId] = Body(); } );
    nr_bodies -= numObjs;

    // log

    if(_solver) 
        for(unsigned int i = 0; i < numObjs; i++) {
            auto obj = b[i];
            _solver->log(MeshLogEventType::Destroy, {obj->_objId}, {obj->objType()});
        }

    return S_OK;
}

namespace TissueForge::io {


    template <>
    HRESULT toFile(TissueForge::models::vertex::Mesh *dataElement, const MetaData &metaData, IOElement &fileElement) {

        std::vector<TissueForge::models::vertex::Vertex*> vertices;
        if(dataElement->numVertices() > 0) {
            vertices.reserve(dataElement->numVertices());
            for(size_t i = 0; i < dataElement->sizeVertices(); i++) {
                auto o = dataElement->getVertex(i);
                if(o) 
                    vertices.push_back(o);
            }
        }
        TF_IOTOEASY(fileElement, metaData, "vertices", vertices);

        std::vector<TissueForge::models::vertex::Surface*> surfaces;
        if(dataElement->numSurfaces() > 0) {
            surfaces.reserve(dataElement->numSurfaces());
            for(size_t i = 0; i < dataElement->sizeSurfaces(); i++) {
                auto o = dataElement->getSurface(i);
                if(o) 
                    surfaces.push_back(o);
            }
        }
        TF_IOTOEASY(fileElement, metaData, "surfaces", surfaces);

        std::vector<TissueForge::models::vertex::Body*> bodies;
        if(dataElement->numBodies() > 0) {
            bodies.reserve(dataElement->numBodies());
            for(size_t i = 0; i < dataElement->sizeBodies(); i++) {
                auto o = dataElement->getBody(i);
                if(o) 
                    bodies.push_back(o);
            }
        }
        TF_IOTOEASY(fileElement, metaData, "bodies", bodies);

        if(dataElement->hasQuality()) {
            TF_IOTOEASY(fileElement, metaData, "quality", dataElement->getQuality());
        }

        fileElement.get()->type = "Mesh";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Mesh *dataElement) {
        
        if(!FIO::hasImport()) 
            return tf_error(E_FAIL, "No import data available");
        else if(!TissueForge::models::vertex::io::VertexSolverFIOModule::hasImport()) 
            return tf_error(E_FAIL, "No vertex import data available");

        TissueForge::models::vertex::MeshSolver *solver = TissueForge::models::vertex::MeshSolver::get();
        if(!solver) 
            return tf_error(E_FAIL, "No vertex solver available");

        // Import objects

        std::vector<TissueForge::models::vertex::Vertex*> vertices;
        TF_IOFROMEASY(fileElement, metaData, "vertices", &vertices);
        std::vector<TissueForge::models::vertex::Surface*> surfaces;
        TF_IOFROMEASY(fileElement, metaData, "surfaces", &surfaces);
        std::vector<TissueForge::models::vertex::Body*> bodies;
        TF_IOFROMEASY(fileElement, metaData, "bodies", &bodies);

        // Get quality, if any

        IOChildMap fec = IOElement::children(fileElement);
        if(fec.find("quality") != fec.end()) {
            TissueForge::models::vertex::MeshQuality quality;
            TF_IOFROMEASY(fileElement, metaData, "quality", &quality);
            (*dataElement).setQuality(new TissueForge::models::vertex::MeshQuality(quality));
        }

        return S_OK;
    }
}

std::string TissueForge::models::vertex::Mesh::toString() {
    TissueForge::io::IOElement el = TissueForge::io::IOElement::create();
    std::string result;
    if(TissueForge::io::toFile(this, TissueForge::io::MetaData(), el) != S_OK) result = "";
    else result = TissueForge::io::toStr(el);
    return result;
}
