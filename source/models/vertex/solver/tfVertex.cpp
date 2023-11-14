/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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

#include "tfVertex.h"

#include "tfSurface.h"
#include "tfBody.h"
#include "tfMeshSolver.h"
#include "tf_mesh_io.h"
#include "tfVertexSolverFIO.h"
#include "tf_mesh_ops.h"

#include <tfError.h>
#include <tfLogger.h>
#include <tfEngine.h>
#include <tfTaskScheduler.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>

#include <unordered_set>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


#define Vertex_GETMESH(name, retval)                \
    Mesh *name = Mesh::get();                       \
    if(!name) {                                     \
        TF_Log(LOG_ERROR) << "Could not get mesh";  \
        return retval;                              \
    }

#define VertexHandle_INVALIDHANDLERR { tf_error(E_FAIL, "Invalid handle"); }

#define VertexHandle_GETOBJ(name, retval)                               \
    Vertex *name;                                                       \
    if(this->id < 0 || !(name = Mesh::get()->getVertex(this->id))) {    \
        VertexHandle_INVALIDHANDLERR; return retval; }


struct PairHash_VertexPtr_VertexPtr {
    size_t operator() (const std::pair<Vertex*, Vertex*>& p) const {
        return std::hash<Vertex*>()(p.first) ^ std::hash<Vertex*>()(p.second);
    }
};


////////////
// Vertex //
////////////


MeshParticleType *TissueForge::models::vertex::MeshParticleType_get() {
    TF_Log(LOG_TRACE);

    MeshParticleType tmp;
    ParticleType *result = ParticleType_FindFromName(tmp.name);
    if(result) 
        return (MeshParticleType*)result;
    
    TF_Log(LOG_DEBUG) << "Registering vertex particle type with name " << tmp.name;
    tmp.registerType();
    TF_Log(LOG_DEBUG) << "Particle types: " << _Engine.nr_types;
    
    result = ParticleType_FindFromName(tmp.name);
    if(!result) {
        TF_Log(LOG_ERROR);
        return NULL;
    }
    return (MeshParticleType*)result;
}

void Vertex::updateConnectedVertices() {
    std::unordered_set<Vertex*> result;
    Vertex *vp, *vn;

    for(auto &s : getSurfaces()) {
        std::tie(vp, vn) = s->neighborVertices(this);
        result.insert(vp);
        result.insert(vn);
    }
    this->_connectedVertices = std::vector<Vertex*>(result.begin(), result.end());
}

std::vector<Surface*> Vertex::sharedSurfaces(const Vertex *other) const {
    std::unordered_set<Surface*> result;
    for(auto &s : surfaces) 
        if(other->defines(s)) 
            result.insert(s);
    return std::vector<Surface*>(result.begin(), result.end());
}

FloatP_t Vertex::getArea() const {
    FloatP_t result = 0.f;
    for(auto &s : getSurfaces()) 
        result += s->getVertexArea(this);
    return result;
}

FloatP_t Vertex::getVolume() const {
    FloatP_t result = 0.f;
    for(auto &b : getBodies()) 
        result += b->getVertexVolume(this);
    return result;
}

FloatP_t Vertex::getMass() const {
    FloatP_t result = 0.f;
    if(MeshSolver::is3D()) 
        for(auto &b : getBodies()) 
            result += b->getVertexMass(this);
    else 
        for(auto &s : getSurfaces()) 
            result += s->getVertexMass(this);
    return result;
}

HRESULT Vertex::positionChanged() {
    if(this->pid >= 0) {
        Particle *p = this->particle()->part();
        _particlePosition = p->global_position();
        _particleVelocity = p->velocity;
        _particleMass = p->mass;
    } 
    else {
        _particleMass = 0.f;
        _particlePosition = FVector3(0);
        _particleVelocity = FVector3(0);
    }
    return S_OK;
}

HRESULT Vertex::updateProperties() {
    ParticleHandle *p = particle();
    const FloatP_t vMass = getMass();
    if(p && vMass > 0.f) {
        p->setMass(vMass);
    }
    return S_OK;
}

ParticleHandle *Vertex::particle() const {
    if(this->pid < 0) {
        TF_Log(LOG_DEBUG);
        return NULL;
    }

    Particle *p = Particle_FromId(this->pid);
    if(!p) {
        TF_Log(LOG_ERROR);
        return NULL;
    }

    return p->handle();
}

HRESULT Vertex::setPosition(const FVector3 &pos, const bool &updateChildren) {
    auto p = particle();
    if(!p) {
        TF_Log(LOG_ERROR) << "No assigned particle.";
        return E_FAIL;
    }
    p->setPosition(pos);
    _particlePosition = pos;

    if(updateChildren) 
        for(auto &s : surfaces) 
            s->positionChanged();

    return S_OK;
}

Vertex::Vertex() : 
    pid{-1}
{
    MESHOBJ_INITOBJ
}

Vertex::~Vertex() {
    MESHOBJ_DELOBJ
}

static Vertex *Vertex_create(const unsigned int &_pid) {
    Vertex_GETMESH(mesh, NULL);
    Vertex *result;
    if(mesh->create(&result, _pid) != S_OK) {
        TF_Log(LOG_ERROR);
        return NULL;
    }
    return result;
}

static std::vector<Vertex*> Vertex_create(const std::vector<unsigned int>& _pids) {
    Vertex_GETMESH(mesh, {});
    std::vector<Vertex*> result(_pids.size(), 0);
    Vertex** data = result.data();
    if(mesh->create(&data, _pids) != S_OK) {
        TF_Log(LOG_ERROR);
        return {};
    }
    return result;
}

static Vertex *Vertex_create(const FVector3 &position, int &pid) {
    MeshParticleType *ptype = MeshParticleType_get();
    Mesh *mesh = Mesh::get();
    if(!ptype) {
        TF_Log(LOG_ERROR) << "Could not instantiate particle type";
        return NULL;
    } 
    else if(!mesh) {
        TF_Log(LOG_ERROR) << "Could not get mesh";
        return NULL;
    }

    FVector3 _position = position;
    ParticleHandle *ph = (*ptype)(&_position);
    if(!ph) {
        TF_Log(LOG_ERROR) << "Could not add vertex";
        return NULL;
    }

    pid = ph->id;
    return Vertex_create(ph->id);
}

static std::vector<Vertex*> Vertex_create(const std::vector<FVector3>& positions, std::vector<unsigned int>& pids) {
    if(positions.empty()) 
        return {};

    MeshParticleType *ptype = MeshParticleType_get();
    Mesh *mesh = Mesh::get();
    if(!ptype) {
        TF_Log(LOG_ERROR) << "Could not instantiate particle type";
        return {};
    } 
    else if(!mesh) {
        TF_Log(LOG_ERROR) << "Could not get mesh";
        return {};
    }

    auto _positions = positions;
    auto _pids = ptype->factory(0, &_positions);
    if(_pids.empty()) {
        TF_Log(LOG_ERROR) << "Could not instantiate particles";
        return {};
    }
    pids = std::vector<unsigned int>(_pids.size());
    parallel_for(_pids.size(), [&pids, &_pids](int i) -> void { pids[i] = _pids[i]; });

    return Vertex_create(pids);
}

static Vertex *Vertex_create(TissueForge::io::ThreeDFVertexData *vdata, int &pid) {
    return Vertex_create(vdata->position, pid);
}

static std::vector<Vertex*> Vertex_create(const std::vector<TissueForge::io::ThreeDFVertexData*>& vdata, std::vector<unsigned int>& pids) {
    std::vector<FVector3> positions(vdata.size());

    parallel_for(vdata.size(), [&vdata, &positions](int i) -> void { positions[i] = vdata[i]->position; });

    return Vertex_create(positions, pids);
}

VertexHandle Vertex::create(const unsigned int &_pid) {
    Vertex *result;
    if(!(result = Vertex_create(_pid))) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    result->pid = _pid;
    if(_pid >= 0) {
        result->positionChanged();
    }
    return VertexHandle(result->_objId);
};

VertexHandle Vertex::create(const FVector3 &position) {
    Vertex *result;
    int _pid;
    if(!(result = Vertex_create(position, _pid))) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    result->pid = _pid;
    if(_pid >= 0) {
        result->positionChanged();
    }
    return VertexHandle(result->_objId);
}

VertexHandle Vertex::create(TissueForge::io::ThreeDFVertexData *vdata) {
    Vertex *result;
    int _pid;
    if(!(result = Vertex_create(vdata, _pid))) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    result->pid = _pid;
    if(_pid >= 0) {
        result->positionChanged();
    }
    return VertexHandle(result->_objId);
}

std::vector<VertexHandle> Vertex::create(const std::vector<unsigned int>& _pids) {
    std::vector<Vertex*> vertices = Vertex_create(_pids);
    if(vertices.empty() && !_pids.empty()) {
        TF_Log(LOG_ERROR);
        return {};
    }

    std::vector<VertexHandle> result(_pids.size());

    parallel_for(
        _pids.size(), 
        [&result, &_pids, &vertices](int i) -> void {
            unsigned int _pid = _pids[i];
            Vertex *v = vertices[i];
            v->pid = _pid;
            if(_pid >= 0) {
                v->positionChanged();
            }
            result[i] = VertexHandle(v->_objId);
        }
    );
    return result;
}

std::vector<VertexHandle> Vertex::create(const std::vector<FVector3>& positions) {
    if(positions.empty()) 
        return {};

    std::vector<unsigned int> pids;
    std::vector<Vertex*> vertices = Vertex_create(positions, pids);
    if(vertices.empty()) {
        TF_Log(LOG_ERROR);
        return {};
    }

    std::vector<VertexHandle> result(vertices.size());
    parallel_for(
        vertices.size(), 
        [&vertices, &pids, &result](int i) -> void {
            Vertex* v = vertices[i];
            v->pid = pids[i];
            if(v->pid >= 0) 
                v->positionChanged();
            result[i] = VertexHandle(v->_objId);
        }
    );
    return result;
}

std::vector<VertexHandle> Vertex::create(const std::vector<TissueForge::io::ThreeDFVertexData*>& vdata) {
    if(vdata.empty()) 
        return {};

    std::vector<unsigned int> pids;
    std::vector<Vertex*> vertices = Vertex_create(vdata, pids);
    if(vertices.empty()) {
        TF_Log(LOG_ERROR);
        return {};
    }

    std::vector<VertexHandle> result(vertices.size());
    parallel_for(
        vertices.size(), 
        [&vertices, &pids, &result](int i) -> void {
            Vertex* v = vertices[i];
            v->pid = pids[i];
            if(v->pid >= 0) 
                v->positionChanged();
            result[i] = VertexHandle(v->_objId);
        }
    );
    return result;
}

bool Vertex::defines(const Surface *obj) const { MESHBOJ_DEFINES_DEF(getVertices) }

bool Vertex::defines(const Body *obj) const { MESHBOJ_DEFINES_DEF(getVertices) }

bool Vertex::validate() {
    if(this->pid < 0) 
        return false;

    for(auto &s : surfaces) 
        if(!this->defines(s) || !s->definedBy(this)) 
            return false;

    return true;
};

std::string Vertex::str() const {
    std::stringstream ss;

    ss << "Vertex(";
    if(this->objectId() >= 0) 
        ss << "id=" << this->objectId();
    ss << ")";

    return ss.str();
}

#define VERTEX_RND_IDX(vec_size, idx) {     \
while(idx < 0) idx += vec_size;             \
while(idx >= vec_size) idx -= vec_size;     \
}

HRESULT Vertex::add(Surface *s) {
    if(std::find(surfaces.begin(), surfaces.end(), s) != surfaces.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    surfaces.push_back(s);
    return S_OK;
}

HRESULT Vertex::insert(Surface *s, const int &idx) { 
    int _idx = idx;
    VERTEX_RND_IDX(this->surfaces.size(), _idx);
    this->surfaces.insert(this->surfaces.begin() + _idx, s);
    return S_OK;
}

HRESULT Vertex::insert(Surface *s, Surface *before) {
    auto itr = std::find(this->surfaces.begin(), this->surfaces.end(), before);
    if(itr == this->surfaces.end()) 
        return E_FAIL;
    this->surfaces.insert(itr, s);
    return S_OK;
}

HRESULT Vertex::remove(Surface *s) {
    auto itr = std::find(this->surfaces.begin(), this->surfaces.end(), s);
    if(itr == this->surfaces.end()) 
        return E_FAIL;
    this->surfaces.erase(itr);
    return S_OK;
}

HRESULT Vertex::replace(Surface *toInsert, const int &idx) {
    int _idx = idx;
    VERTEX_RND_IDX(this->surfaces.size(), _idx);
    std::replace(this->surfaces.begin(), this->surfaces.end(), this->surfaces[idx], toInsert);
    return S_OK;
}

HRESULT Vertex::replace(Surface *toInsert, Surface *toRemove) {
    std::replace(this->surfaces.begin(), this->surfaces.end(), toRemove, toInsert);
    return this->defines(toInsert) ? S_OK : E_FAIL;
}

HRESULT Vertex::destroy() {
    TF_Log(LOG_TRACE) << this->_objId << "; " << this->pid;
    if(this->_objId < 0) 
        return S_OK;

    for(auto &s : getSurfaces()) 
        if(s->destroy() != S_OK) {
            TF_Log(LOG_DEBUG) << s->_objId;
            return E_FAIL;
        }
    ParticleHandle *ph = this->particle();
    if(Mesh::get()->remove(this) != S_OK) {
        TF_Log(LOG_DEBUG);
    }
    if(ph && ph->destroy() != S_OK) {
        TF_Log(LOG_DEBUG);
    }
    this->pid = -1;
    this->_connectedVertices.clear();
    return S_OK;
}

HRESULT Vertex::destroy(const std::vector<Vertex*>& toDestroy) {

    std::vector<std::unordered_set<int> > particleIdsToDestroyPool(ThreadPool::size());
    std::vector<std::unordered_set<Vertex*> > verticesToDestroyPool(ThreadPool::size());
    std::vector<std::unordered_set<Surface*> > surfacesToDestroyPool(ThreadPool::size());
    parallel_for(
        ThreadPool::size(), 
        [&particleIdsToDestroyPool, &verticesToDestroyPool, &surfacesToDestroyPool, &toDestroy](int tid) -> void {
            std::unordered_set<int>& particleIdsToDestroyThread = particleIdsToDestroyPool[tid];
            std::unordered_set<Vertex*>& verticesToDestroyThread = verticesToDestroyPool[tid];
            std::unordered_set<Surface*>& surfacesToDestroyThread = surfacesToDestroyPool[tid];
            for(int i = tid; i < toDestroy.size(); i += ThreadPool::size()) {
                Vertex* v = toDestroy[i];
                if(!v || v->objectId() < 0) 
                    continue;
                particleIdsToDestroyThread.insert(v->getPartId());
                verticesToDestroyThread.insert(v);
                for(auto& s : v->getSurfaces()) 
                    surfacesToDestroyThread.insert(s);
            }
        }
    );

    size_t numSurfaces = 0;
    for(auto& surfacesToDestroyThread : surfacesToDestroyPool) 
        numSurfaces += surfacesToDestroyThread.size();
    if(numSurfaces > 0) {
        std::unordered_set<Surface*> surfacesToDestroy;
        surfacesToDestroy.reserve(numSurfaces);
        for(auto& surfacesToDestroyThread : surfacesToDestroyPool) 
            surfacesToDestroy.insert(surfacesToDestroyThread.begin(), surfacesToDestroyThread.end());
        Surface::destroy(std::vector<Surface*>{surfacesToDestroy.begin(), surfacesToDestroy.end()});
    }

    size_t numVertices = 0;
    for(auto& verticesToDestroyThread : verticesToDestroyPool) 
        numVertices += verticesToDestroyThread.size();
    std::vector<Vertex*> verticesToDestroyVec;
    if(numVertices > 0) {
        std::unordered_set<Vertex*> verticesToDestroy;
        verticesToDestroy.reserve(numVertices);
        for(auto& verticesToDestroyThread : verticesToDestroyPool) 
            verticesToDestroy.insert(verticesToDestroyThread.begin(), verticesToDestroyThread.end());
        verticesToDestroyVec = std::vector<Vertex*>(verticesToDestroy.begin(), verticesToDestroy.end());
        Mesh::get()->remove(verticesToDestroyVec.data(), verticesToDestroyVec.size());

        std::unordered_set<int> particleIdsToDestroy;
        particleIdsToDestroy.reserve(numVertices);
        for(auto& particleIdsToDestroyThread : particleIdsToDestroyPool) 
            particleIdsToDestroy.insert(particleIdsToDestroyThread.begin(), particleIdsToDestroyThread.end());
        for(auto& pid : particleIdsToDestroy) 
            ParticleHandle(pid).destroy();
    }

    parallel_for(
        verticesToDestroyVec.size(), 
        [&verticesToDestroyVec](int i) -> void {
            Vertex* v = verticesToDestroyVec[i];
            v->pid = -1;
            v->_connectedVertices.clear();
        }
    );

    return S_OK;
}

std::vector<Body*> Vertex::getBodies() const {
    std::unordered_set<Body*> result;
    for(auto &s : surfaces) 
        for(auto &b : s->getBodies()) 
            result.insert(b);
    return std::vector<Body*>(result.begin(), result.end());
}

Surface *Vertex::findSurface(const FVector3 &dir) const {
    Surface *result = 0;

    FloatP_t bestCrit = 0;
    const FVector3 position = getPosition();

    for(auto &s : getSurfaces()) {
        const FVector3 rel_pt = s->getCentroid() - position;
        if(rel_pt.isZero()) 
            continue;
        const FloatP_t crit = rel_pt.dot(dir) / rel_pt.dot();
        if(!result || crit > bestCrit) { 
            result = s;
            bestCrit = crit;
        }
    }

    return result;
}

Body *Vertex::findBody(const FVector3 &dir) const {
    Body *result = 0;

    FloatP_t bestCrit = 0;
    const FVector3 position = getPosition();

    for(auto &b : getBodies()) {
        const FVector3 rel_pt = b->getCentroid() - position;
        if(rel_pt.isZero()) 
            continue;
        FloatP_t crit = rel_pt.dot(dir) / rel_pt.dot();
        if(!result || crit > bestCrit) { 
            result = b;
            bestCrit = crit;
        }
    }

    return result;
}

static HRESULT Vertex_destroyOrTransferBonds(
    Vertex* source, 
    Vertex* target, 
    std::vector<std::pair<uint32_t, int> >& bondsInfo, 
    std::vector<std::pair<uint32_t, int> >& anglesInfo, 
    std::vector<std::pair<uint32_t, int> >& dihedralsInfo
) {
    ParticleHandle *ph = source->particle();
    const int source_pid = source->getPartId();
    const int target_pid = target->getPartId();

    std::unordered_set<uint32_t> bonded_ids;
    bonded_ids.insert(target_pid);
    for(auto &bh : ph->getBonds()) {
        Bond *b = bh.get();
        int result = -1;
        if(b->i == source_pid && std::find(bonded_ids.begin(), bonded_ids.end(), b->j) == bonded_ids.end()) {
            result = 0;
            bonded_ids.insert(b->j);
        } 
        else if(b->j == source_pid && std::find(bonded_ids.begin(), bonded_ids.end(), b->i) == bonded_ids.end()) {
            result = 1;
            bonded_ids.insert(b->i);
        }
        bondsInfo.push_back({b->id, result});
    }
    
    for(auto &ah : ph->getAngles()) {
        Angle *a = ah.get();
        int result = -1;
        if(a->i == source_pid && a->j != target_pid && a->k != target_pid) 
            result = 0;
        else if(a->j == source_pid && a->i != target_pid && a->k != target_pid) 
            result = 1;
        else if(a->k == source_pid && a->i != target_pid && a->j != target_pid) 
            result = 2;
        anglesInfo.push_back({a->id, result});
    }
    
    for(auto &dh : ph->getDihedrals()) {
        Dihedral *d = dh.get();
        int result = -1;
        if(d->i == source_pid && d->j != target_pid && d->k != target_pid && d->l != target_pid) 
            result = 0;
        else if(d->j == source_pid && d->i != target_pid && d->k != target_pid && d->l != target_pid) 
            result = 1;
        else if(d->k == source_pid && d->i != target_pid && d->j != target_pid && d->l != target_pid) 
            result = 2;
        else if(d->l == source_pid && d->i != target_pid && d->j != target_pid && d->k != target_pid) 
            result = 3;
        dihedralsInfo.push_back({d->id, result});
    }

    return S_OK;
}

HRESULT Vertex::transferBondsTo(Vertex *other) {
    std::vector<std::pair<uint32_t, int> > bondsInfo;
    std::vector<std::pair<uint32_t, int> > anglesInfo;
    std::vector<std::pair<uint32_t, int> > dihedralsInfo;
    if(Vertex_destroyOrTransferBonds(this, other, bondsInfo, anglesInfo, dihedralsInfo) != S_OK) 
        return E_FAIL;
    for(auto& p : bondsInfo) {
        uint32_t bid;
        int result;
        std::tie(bid, result) = p;
        BondHandle bh(bid);
        if(result < 0) bh.destroy();
        else if(result == 0) bh.get()->i = other->pid;
        else if(result == 1) bh.get()->j = other->pid;
    }
    for(auto& p : anglesInfo) {
        uint32_t bid;
        int result;
        std::tie(bid, result) = p;
        AngleHandle bh(bid);
        if(result < 0) bh.destroy();
        else if(result == 0) bh.get()->i = other->pid;
        else if(result == 1) bh.get()->j = other->pid;
        else if(result == 2) bh.get()->k = other->pid;
    }
    for(auto& p : dihedralsInfo) {
        uint32_t bid;
        int result;
        std::tie(bid, result) = p;
        DihedralHandle bh(bid);
        if(result < 0) bh.destroy();
        else if(result == 0) bh.get()->i = other->pid;
        else if(result == 1) bh.get()->j = other->pid;
        else if(result == 2) bh.get()->k = other->pid;
        else if(result == 3) bh.get()->l = other->pid;
    }

    return S_OK;
}

HRESULT Vertex::transferBondsTo(const std::vector<std::pair<Vertex*, Vertex*> >& targets) {
    std::vector<std::unordered_map<uint32_t, std::unordered_map<int, int> > > bondsInfoPool(ThreadPool::size());
    std::vector<std::unordered_map<uint32_t, std::unordered_map<int, int> > > anglesInfoPool(ThreadPool::size());
    std::vector<std::unordered_map<uint32_t, std::unordered_map<int, int> > > dihedralsInfoPool(ThreadPool::size());

    parallel_for(
        ThreadPool::size(), 
        [&targets, &anglesInfoPool, &bondsInfoPool, &dihedralsInfoPool](int tid) -> void {
            std::unordered_map<uint32_t, std::unordered_map<int, int> >& bondsInfoThread = bondsInfoPool[tid];
            std::unordered_map<uint32_t, std::unordered_map<int, int> >& anglesInfoThread = anglesInfoPool[tid];
            std::unordered_map<uint32_t, std::unordered_map<int, int> >& dihedralsInfoThread = dihedralsInfoPool[tid];
            for(int i = tid; i < targets.size(); i += ThreadPool::size()) {
                Vertex* vs, *vt;
                std::tie(vs, vt) = targets[i];
                const int vtid = vt->getPartId();
                std::vector<std::pair<uint32_t, int> > _bondsInfo;
                std::vector<std::pair<uint32_t, int> > _anglesInfo;
                std::vector<std::pair<uint32_t, int> > _dihedralsInfo;
                Vertex_destroyOrTransferBonds(vs, vt, _bondsInfo, _anglesInfo, _dihedralsInfo);
                for(auto& p : _bondsInfo) {
                    uint32_t bid;
                    int result;
                    std::tie(bid, result) = p;
                    bondsInfoThread[bid][vtid] = result;
                }
                for(auto& p : _anglesInfo) {
                    uint32_t bid;
                    int result;
                    std::tie(bid, result) = p;
                    anglesInfoThread[bid][vtid] = result;
                }
                for(auto& p : _dihedralsInfo) {
                    uint32_t bid;
                    int result;
                    std::tie(bid, result) = p;
                    dihedralsInfoThread[bid][vtid] = result;
                }
            }
        }
    );

    std::unordered_map<uint32_t, std::unordered_map<int, int> > bondsInfo;
    std::unordered_map<uint32_t, std::unordered_map<int, int> > anglesInfo;
    std::unordered_map<uint32_t, std::unordered_map<int, int> > dihedralsInfo;

    for(auto& bondsInfoThread : bondsInfoPool) 
        for(auto& m : bondsInfoThread) {
            auto& mf = bondsInfo[m.first];
            for(auto& p : m.second) 
                mf[p.first] = p.second;
        }
    for(auto& anglesInfoThread : anglesInfoPool) 
        for(auto& m : anglesInfoThread) {
            auto& mf = anglesInfo[m.first];
            for(auto& p : m.second) 
                mf[p.first] = p.second;
        }
    for(auto& dihedralsInfoThread : dihedralsInfoPool) 
        for(auto& m : dihedralsInfoThread) {
            auto& mf = dihedralsInfo[m.first];
            for(auto& p : m.second) 
                mf[p.first] = p.second;
        }

    for(auto& m : bondsInfo) {
        const int bid = m.first;
        for(auto& p : m.second) {
            const int vtid = p.first;
            const int result = p.second;

            BondHandle bh(bid);
            if(result < 0) {
                bh.destroy();
                break;
            }
            else if(result == 0) bh.get()->i = vtid;
            else if(result == 1) bh.get()->j = vtid;
        }
    }
    for(auto& m : anglesInfo) {
        const int bid = m.first;
        for(auto& p : m.second) {
            const int vtid = p.first;
            const int result = p.second;

            AngleHandle bh(bid);
            if(result < 0) {
                bh.destroy();
                break;
            }
            else if(result == 0) bh.get()->i = vtid;
            else if(result == 1) bh.get()->j = vtid;
            else if(result == 2) bh.get()->k = vtid;
        }
    }
    for(auto& m : dihedralsInfo) {
        const int bid = m.first;
        for(auto& p : m.second) {
            const int vtid = p.first;
            const int result = p.second;

            DihedralHandle bh(bid);
            if(result < 0) {
                bh.destroy();
                break;
            }
            else if(result == 0) bh.get()->i = vtid;
            else if(result == 1) bh.get()->j = vtid;
            else if(result == 2) bh.get()->k = vtid;
            else if(result == 3) bh.get()->l = vtid;
        }
    }

    return S_OK;
}

static HRESULT Vertex_SurfaceDisconnectReplace(
    Vertex *toInsert, 
    Surface *toReplace, 
    Surface *targetSurf, 
    std::vector<Vertex*> &targetSurf_vertices, 
    std::set<Vertex*> &totalToRemove) 
{
    std::vector<unsigned int> edgeLabels = targetSurf->contiguousVertexLabels(toReplace);
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

HRESULT Vertex::replace(Surface *toReplace) {
    // For every surface connected to the replaced surface
    //      Gather every vertex connected to the replaced surface
    //      Replace all vertices with the inserted vertex
    // Remove the replaced surface from the mesh
    // Add the inserted vertex to the mesh

    // Gather first analysis step
    std::unordered_set<Surface*> removedSurfaces = removedSurfacesByS2V(toReplace);
    
    // Prevent destroyed bodies
    if(!removedChildrenByRemovedParents(removedSurfaces).empty()) {
        TF_Log(LOG_DEBUG) << "Insufficient surfaces";
        return E_FAIL;
    }

    // Gather remaining analysis
    std::unordered_set<Vertex*> removedVertices = orphanedVertices(removedSurfaces);
    for(auto &v : toReplace->vertices) 
        removedVertices.insert(v);
    std::unordered_set<Vertex*> connVerts;
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > vsmapRemoved;
    connectedToMapGiven(removedVertices, connVerts, vsmapRemoved);
    vsmapRemoved.erase(vsmapRemoved.find(toReplace));

    // Do replacement on every surface that stays
    for(auto &s : removedSurfaces) 
        vsmapRemoved.erase(vsmapRemoved.find(s));
    for(auto &mapEntry : vsmapRemoved) {
        Surface *s;
        std::unordered_set<Vertex*> vsRemoved;
        std::tie(s, vsRemoved) = mapEntry;
        if(vsRemoved.empty()) 
            continue;
        Vertex *v = *vsRemoved.begin();
        s->replace(this, v);
        add(s);
        v->remove(s);
        vsRemoved.erase(vsRemoved.begin());
        for(auto &vv : vsRemoved) {
            s->remove(vv);
            vv->remove(s);
        }
    }

    MeshSolver::log(MeshLogEventType::Create, {_objId, toReplace->_objId}, {objType(), toReplace->objType()}, "replace");

    // Destroy as instructed

    removedSurfaces.insert(toReplace);
    for(auto &s : removedSurfaces) 
        for(auto &b : s->getBodies()) {
            b->remove(s);
            s->remove(b);
        }
    for(auto &v : removedVertices) 
        while(!v->surfaces.empty()) {
            Surface *s = v->surfaces.front();
            v->remove(s);
            s->remove(v);
        }
    for(auto &s : removedSurfaces) 
        s->destroy();
    for(auto &v : removedVertices) 
        v->destroy();

    // Update connected objects
    updateConnectedVertices();
    for(auto &v : _connectedVertices) 
        v->updateConnectedVertices();
    std::unordered_set<Body*> connectedBodies;
    for(auto &mapEntry : vsmapRemoved) {
        mapEntry.first->positionChanged();
        for(auto &b : mapEntry.first->getBodies()) 
            connectedBodies.insert(b);
    }
    for(auto &b : connectedBodies) 
        b->positionChanged();
    for(auto &mapEntry : vsmapRemoved) 
        mapEntry.first->refreshBodies();

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

Vertex *Vertex::replace(const FVector3 &position, Surface *toReplace) {
    int _pid;
    Vertex *result = Vertex_create(position, _pid);
    if(!result) {
        TF_Log(LOG_ERROR) << "Could not create vertex";
        return NULL;
    }
    result->pid = _pid;
    result->positionChanged();
    if(result->replace(toReplace) != S_OK) {
        result->destroy();
        return NULL;
    }
    return result;
}

VertexHandle Vertex::replace(const FVector3 &position, SurfaceHandle &toReplace) {
    Vertex_GETMESH(mesh, VertexHandle());
    if(mesh->ensureAvailableVertices(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    Surface *_toReplace = toReplace.surface();
    if(!_toReplace) {
        VertexHandle_INVALIDHANDLERR;
        return VertexHandle();
    }
    Vertex *v = Vertex::replace(position, _toReplace);
    if(!v) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    toReplace.id = -1;
    return VertexHandle(v->objectId());
}

HRESULT Vertex::replace(Body *toReplace) {

    // Gather instructions

    std::unordered_set<Vertex*> removedVertices;
    std::unordered_set<Surface*> removedSurfaces;
    std::unordered_set<Body*> removedBodies;
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > replacementMap;
    transformedByB2V(toReplace, removedVertices, removedSurfaces, removedBodies, replacementMap);

    MeshSolver::log(MeshLogEventType::Create, {_objId, toReplace->_objId}, {objType(), toReplace->objType()}, "replace");
    
    // Replace as instructed
    for(auto &p : replacementMap) {
        Surface *s = p.first;
        std::unordered_set<Vertex*> cv = p.second;
        if(cv.empty()) 
            continue;
        Vertex *v = *cv.begin();
        s->replace(this, v);
        v->remove(s);
        add(s);
        cv.erase(cv.begin());
        for(auto &vv : cv) {
            s->remove(vv);
            vv->remove(s);
        }
    }

    // Destroy as instructed
    for(auto &b : removedBodies) {
        for(auto &s : b->getSurfaces()) 
            s->remove(b);
        b->destroy();
    }
    for(auto &s : removedSurfaces) {
        for(auto &b : s->getBodies()) {
            b->remove(s);
            s->remove(b);
        }
        for(auto &v : s->getVertices()) 
            v->remove(s);
        s->destroy();
    }
    for(auto &v : removedVertices) {
        for(auto &s : v->getSurfaces()) {
            s->remove(v);
            v->remove(s);
        }
        v->destroy();
    }

    // Update

    updateConnectedVertices();
    for(auto &v : _connectedVertices) 
        v->updateConnectedVertices();

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

Vertex *Vertex::replace(const FVector3 &position, Body *toReplace) {
    int _pid;
    Vertex *result = Vertex_create(position, _pid);
    if(!result) {
        TF_Log(LOG_ERROR) << "Could not create vertex";
        return NULL;
    }
    result->pid = _pid;
    result->positionChanged();
    if(result->replace(toReplace) != S_OK) {
        result->destroy();
        return NULL;
    }
    return result;
}

VertexHandle Vertex::replace(const FVector3 &position, BodyHandle &toReplace) {
    Vertex_GETMESH(mesh, VertexHandle());
    if(mesh->ensureAvailableVertices(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    Body *_toReplace = toReplace.body();
    if(!_toReplace) {
        VertexHandle_INVALIDHANDLERR;
        return VertexHandle();
    }
    Vertex *v = Vertex::replace(position, _toReplace);
    if(!v) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    toReplace.id = -1;
    return VertexHandle(v->objectId());
}

static HRESULT Vertex_merge_assm(
    Vertex* toKeep, 
    Vertex* toRemove, 
    std::unordered_set<Surface*>& common_s, 
    std::unordered_set<Surface*>& different_s
) {
    // In common surfaces, just remove; in different surfaces, replace
    auto& toRemove_surfaces = toRemove->getSurfaces();
    common_s.reserve(toRemove_surfaces.size());
    different_s.reserve(toRemove_surfaces.size());
    for(auto &s : toRemove_surfaces) {
        if(!toKeep->defines(s)) 
            different_s.insert(s);
        else {
            // Prevent invalid surface
            if(s->getVertices().size() < 4) 
                return E_FAIL;
            common_s.insert(s);
        }
    }

    return S_OK;
}

static HRESULT Vertex_merge(Vertex* toKeep, Vertex* toRemove, const FloatP_t& lenCf) {

    // In common surfaces, just remove; in different surfaces, replace
    std::unordered_set<Surface*> common_s, different_s;
    auto& toRemove_surfaces = toRemove->getSurfaces();
    std::vector<Vertex*> toRemoveConnectedVertices = toRemove->connectedVertices();
    if(Vertex_merge_assm(toKeep, toRemove, common_s, different_s) != S_OK) {
        TF_Log(LOG_DEBUG) << "Insufficient surface vertices. Ignoring";
        return E_FAIL;
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

    toKeep->updateConnectedVertices();
    std::unordered_set<Vertex*> affectedVertices;
    for(auto &v : toKeep->connectedVertices()) 
        affectedVertices.insert(v);
    for(auto &v : toRemoveConnectedVertices) 
        affectedVertices.insert(v);
    for(auto &v : affectedVertices) 
        v->updateConnectedVertices();
    
    // Set new position
    const FVector3 posToKeep = toKeep->getPosition();
    const FVector3 newPos = posToKeep + (toRemove->getPosition() - posToKeep) * lenCf;
    if(toKeep->setPosition(newPos) != S_OK) 
        return E_FAIL;

    MeshSolver::log(MeshLogEventType::Create, {toKeep->objectId(), toRemove->objectId()}, {toKeep->objType(), toRemove->objType()}, "merge");
    
    if(toRemove->transferBondsTo(toKeep) != S_OK || toRemove->destroy() != S_OK) 
        return E_FAIL;

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

static HRESULT Vertex_merge(const std::vector<std::pair<Vertex*, Vertex*> >& toMerge, const FloatP_t& lenCf) {

    std::vector<std::unordered_set<Vertex*> > verticesAffectedPool(ThreadPool::size());
    std::vector<std::unordered_map<Vertex*, std::unordered_set<Surface*> > > toRemoveSurfacesByVertexPool(ThreadPool::size()), toKeepSurfacesByVertexPool(ThreadPool::size());
    std::vector<std::unordered_map<Surface*, std::unordered_set<Vertex*> > > toRemoveVerticesBySurfacePool(ThreadPool::size());
    std::vector<std::unordered_map<Surface*, std::unordered_set<std::pair<Vertex*, Vertex*>, PairHash_VertexPtr_VertexPtr> > > toReplaceVerticesBySurfacePool(ThreadPool::size());

    parallel_for(
        ThreadPool::size(), 
        [&toMerge, &verticesAffectedPool, &toRemoveSurfacesByVertexPool, &toKeepSurfacesByVertexPool, &toRemoveVerticesBySurfacePool, &toReplaceVerticesBySurfacePool](int tid) -> void {
            std::unordered_set<Vertex*>& verticesAffectedThread = verticesAffectedPool[tid];
            std::unordered_map<Vertex*, std::unordered_set<Surface*> >& toRemoveSurfacesByVertexThread = toRemoveSurfacesByVertexPool[tid];
            std::unordered_map<Vertex*, std::unordered_set<Surface*> >& toKeepSurfacesByVertexThread = toKeepSurfacesByVertexPool[tid];
            std::unordered_map<Surface*, std::unordered_set<Vertex*> >& toRemoveVerticesBySurfaceThread = toRemoveVerticesBySurfacePool[tid];
            std::unordered_map<Surface*, std::unordered_set<std::pair<Vertex*, Vertex*>, PairHash_VertexPtr_VertexPtr> >& toReplaceVerticesBySurfaceThread = toReplaceVerticesBySurfacePool[tid];
            for(int i = tid; i < toMerge.size(); i += ThreadPool::size()) {
                Vertex* toKeep, *toRemove;
                std::tie(toKeep, toRemove) = toMerge[i];

                std::unordered_set<Surface*> common_s, different_s;
                if(Vertex_merge_assm(toKeep, toRemove, common_s, different_s) == S_OK) {
                    for(auto &s : common_s) {
                        toRemoveSurfacesByVertexThread[toRemove].insert(s);
                        toRemoveVerticesBySurfaceThread[s].insert(toRemove);
                    }
                    for(auto &s : different_s) {
                        toRemoveSurfacesByVertexThread[toRemove].insert(s);
                        toKeepSurfacesByVertexThread[toKeep].insert(s);
                        toReplaceVerticesBySurfaceThread[s].insert({toKeep, toRemove});
                    }
                }
                for(auto& v : toRemove->connectedVertices()) 
                    verticesAffectedThread.insert(v);
            }
        }
    );

    std::unordered_map<Vertex*, std::unordered_set<Surface*> > toRemoveSurfacesByVertex;
    for(auto& toRemoveSurfacesByVertexThread : toRemoveSurfacesByVertexPool) 
        for(auto& p : toRemoveSurfacesByVertexThread) 
            toRemoveSurfacesByVertex[p.first].insert(p.second.begin(), p.second.end());
    std::vector<Vertex*> toRemoveSurfacesByVertexKeys;
    toRemoveSurfacesByVertexKeys.reserve(toRemoveSurfacesByVertex.size());
    for(auto& p : toRemoveSurfacesByVertex) 
        toRemoveSurfacesByVertexKeys.push_back(p.first);
    parallel_for(
        toRemoveSurfacesByVertexKeys.size(), 
        [&toRemoveSurfacesByVertex, &toRemoveSurfacesByVertexKeys](int i) -> void {
            Vertex* v = toRemoveSurfacesByVertexKeys[i];
            for(auto& s : toRemoveSurfacesByVertex[v]) 
                v->remove(s);
        }
    );

    std::unordered_map<Vertex*, std::unordered_set<Surface*> > toKeepSurfacesByVertex;
    for(auto& toKeepSurfacesByVertexThread : toKeepSurfacesByVertexPool) 
        for(auto& p : toKeepSurfacesByVertexThread) 
            toKeepSurfacesByVertex[p.first].insert(p.second.begin(), p.second.end());
    std::vector<Vertex*> toKeepSurfacesByVertexKeys;
    toKeepSurfacesByVertexKeys.reserve(toKeepSurfacesByVertex.size());
    for(auto& p : toKeepSurfacesByVertex) 
        toKeepSurfacesByVertexKeys.push_back(p.first);
    parallel_for(
        toKeepSurfacesByVertexKeys.size(), 
        [&toKeepSurfacesByVertex, &toKeepSurfacesByVertexKeys](int i) -> void {
            Vertex* v = toKeepSurfacesByVertexKeys[i];
            for(auto& s : toKeepSurfacesByVertex[v]) 
                v->add(s);
        }
    );

    std::unordered_map<Surface*, std::unordered_set<Vertex*> > toRemoveVerticesBySurface;
    for(auto& toRemoveVerticesBySurfaceThread : toRemoveVerticesBySurfacePool) 
        for(auto& p : toRemoveVerticesBySurfaceThread) 
            toRemoveVerticesBySurface[p.first].insert(p.second.begin(), p.second.end());
    std::vector<Surface*> toRemoveVerticesBySurfaceKeys;
    toRemoveVerticesBySurfaceKeys.reserve(toRemoveVerticesBySurface.size());
    for(auto& p : toRemoveVerticesBySurface) 
        toRemoveVerticesBySurfaceKeys.push_back(p.first);
    parallel_for(
        toRemoveVerticesBySurfaceKeys.size(), 
        [&toRemoveVerticesBySurface, &toRemoveVerticesBySurfaceKeys](int i) -> void {
            Surface* s = toRemoveVerticesBySurfaceKeys[i];
            for(auto& v : toRemoveVerticesBySurface[s]) 
                s->remove(v);
        }
    );

    std::unordered_map<Surface*, std::unordered_set<std::pair<Vertex*, Vertex*>, PairHash_VertexPtr_VertexPtr> > toReplaceVerticesBySurface;
    for(auto& toReplaceVerticesBySurfaceThread : toReplaceVerticesBySurfacePool) 
        for(auto& p : toReplaceVerticesBySurfaceThread) 
            toReplaceVerticesBySurface[p.first].insert(p.second.begin(), p.second.end());
    std::vector<Surface*> toReplaceVerticesBySurfaceKeys;
    toReplaceVerticesBySurfaceKeys.reserve(toReplaceVerticesBySurface.size());
    for(auto& p : toReplaceVerticesBySurface) 
        toReplaceVerticesBySurfaceKeys.push_back(p.first);
    parallel_for(
        toReplaceVerticesBySurfaceKeys.size(), 
        [&toReplaceVerticesBySurface, &toReplaceVerticesBySurfaceKeys](int i) -> void {
            Surface* s = toReplaceVerticesBySurfaceKeys[i];
            for(auto& p : toReplaceVerticesBySurface[s]) 
                s->replace(p.first, p.second);
        }
    );

    parallel_for(
        ThreadPool::size(), 
        [&toMerge, &verticesAffectedPool](int tid) -> void {
            std::unordered_set<Vertex*>& verticesAffectedThread = verticesAffectedPool[tid];
            for(int i = tid; i < toMerge.size(); i += ThreadPool::size()) {
                Vertex* toKeep, *toRemove;
                std::tie(toKeep, toRemove) = toMerge[i];
                toKeep->updateConnectedVertices();
                for(auto& v : toKeep->connectedVertices()) 
                    verticesAffectedThread.insert(v);
            }
        }
    );

    size_t numVerticesAffected = 0;
    for(auto& verticesAffectedThread : verticesAffectedPool) 
        numVerticesAffected += verticesAffectedThread.size();
    std::unordered_set<Vertex*> verticesAffected;
    verticesAffected.reserve(numVerticesAffected);
    for(auto& verticesAffectedThread : verticesAffectedPool) 
        verticesAffected.insert(verticesAffectedThread.begin(), verticesAffectedThread.end());
    std::vector<Vertex*> verticesAffectedVec(verticesAffected.begin(), verticesAffected.end());

    parallel_for(verticesAffectedVec.size(), [&verticesAffectedVec](int i) -> void { verticesAffectedVec[i]->updateConnectedVertices(); });
    
    // Set new position

    std::vector<Vertex*> toRemoveVec(toMerge.size());
    parallel_for(
        toMerge.size(), 
        [&toMerge, &toRemoveVec, &lenCf](int i) -> void {
            Vertex* toKeep, *toRemove;
            std::tie(toKeep, toRemove) = toMerge[i];

            toRemoveVec[i] = toRemove;

            const FVector3 posToKeep = toKeep->getPosition();
            const FVector3 newPos = posToKeep + (toRemove->getPosition() - posToKeep) * lenCf;
            toKeep->setPosition(newPos);
        }
    );

    // log

    if(MeshLogger::getForwardLogging()) 
        for(auto& p : toMerge) {
            Vertex* toKeep, *toRemove;
            std::tie(toKeep, toRemove) = p;
            MeshSolver::log(MeshLogEventType::Create, {toKeep->objectId(), toRemove->objectId()}, {toKeep->objType(), toRemove->objType()}, "merge");
        }
    
    // transfer bonds
    if(Vertex::transferBondsTo(toMerge) != S_OK) 
        return E_FAIL;

    // destroy vertices
    if(Vertex::destroy(toRemoveVec) != S_OK) 
        return E_FAIL;

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

HRESULT Vertex::merge(Vertex *toRemove, const FloatP_t &lenCf) {

    // In common surfaces, just remove; in different surfaces, replace
    std::vector<Surface*> common_s, different_s;
    common_s.reserve(toRemove->surfaces.size());
    different_s.reserve(toRemove->surfaces.size());
    std::vector<Vertex*> toRemoveConnectedVertices = toRemove->connectedVertices();
    for(auto &s : toRemove->surfaces) {
        if(!defines(s)) 
            different_s.push_back(s);
        else {
            // Prevent invalid surface
            if(s->vertices.size() < 4) {
                TF_Log(LOG_DEBUG) << "Insufficient surface vertices. Ignoring";
                return E_FAIL;
            }
            common_s.push_back(s);
        }
    }
    for(auto &s : common_s) {
        s->remove(toRemove);
        toRemove->remove(s);
    }
    for(auto &s : different_s) {
        toRemove->remove(s);
        add(s);
        s->replace(this, toRemove);
    }

    updateConnectedVertices();
    std::unordered_set<Vertex*> affectedVertices;
    for(auto &v : _connectedVertices) 
        affectedVertices.insert(v);
    for(auto &v : toRemoveConnectedVertices) 
        affectedVertices.insert(v);
    for(auto &v : affectedVertices) 
        v->updateConnectedVertices();
    
    // Set new position
    const FVector3 posToKeep = getPosition();
    const FVector3 newPos = posToKeep + (toRemove->getPosition() - posToKeep) * lenCf;
    if(setPosition(newPos) != S_OK) 
        return E_FAIL;

    MeshSolver::log(MeshLogEventType::Create, {_objId, toRemove->_objId}, {objType(), toRemove->objType()}, "merge");
    
    if(toRemove->transferBondsTo(this) != S_OK || toRemove->destroy() != S_OK) 
        return E_FAIL;

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

HRESULT Vertex::merge(const std::vector<std::vector<Vertex*> >& toMerge, const FloatP_t& lenCf) {

    // iterate through "levels" of merged vertices and use Vertex_merge
    //  E.g., at level "2", loop through all elements of toMerge and construct a vector of pairs targeting the second merged vertex; do this until there are no more iterations
    int level = 0;
    int levelMax = 0;
    for(auto& toMerge_v : toMerge) 
        levelMax = std::max(levelMax, (int)toMerge_v.size() - 1);

    std::vector<std::vector<std::pair<Vertex*, Vertex*> > > toMergeLevelPool;
    auto func = [&toMerge, &level, &toMergeLevelPool](int tid) -> void {
        std::vector<std::pair<Vertex*, Vertex*> >& toMergeLevelThread = toMergeLevelPool[tid];
        for(int i = tid; i < toMerge.size(); i += ThreadPool::size()) {
            const std::vector<Vertex*>& toMerge_i = toMerge[i];
            if(toMerge_i.size() <= level + 1) 
                continue;
            auto itrKeep = toMerge_i.begin();
            toMergeLevelThread.emplace_back(*itrKeep, *(itrKeep + level + 1));
        }
    };

    for(; level < levelMax; level++) {
        toMergeLevelPool = std::vector<std::vector<std::pair<Vertex*, Vertex*> > >(ThreadPool::size());
        parallel_for(ThreadPool::size(), func);
        size_t numTargets = 0;
        for(auto& toMergeLevelThread : toMergeLevelPool) 
            numTargets += toMergeLevelThread.size();
        std::vector<std::pair<Vertex*, Vertex*> > toMergeLevel;
        toMergeLevel.reserve(numTargets);
        for(auto& toMergeLevelThread : toMergeLevelPool) 
            for(auto& p : toMergeLevelThread) 
                toMergeLevel.push_back(p);
        Vertex_merge(toMergeLevel, lenCf);
    }

    return S_OK;
}

HRESULT Vertex::insert(Vertex *v1, Vertex *v2) {

    std::vector<Vertex*>::iterator vitr;

    // Find the common surface(s)
    bool inserted = false;
    for(auto &s1 : v1->surfaces) {
        if(defines(s1)) 
            continue;
        for(vitr = s1->vertices.begin(); vitr != s1->vertices.end(); vitr++) {
            std::vector<Vertex*>::iterator vnitr = vitr + 1 == s1->vertices.end() ? s1->vertices.begin() : vitr + 1;
            
            if(((*vitr)->_objId == v1->_objId && (*vnitr)->_objId == v2->_objId) || ((*vitr)->_objId == v2->_objId && (*vnitr)->_objId == v1->_objId)) {
                s1->vertices.insert(vnitr, this);
                add(s1);
                inserted = true;
                break;
            }
        }
    }
    if(inserted) {
        updateConnectedVertices();
        for(auto &v : _connectedVertices) 
            v->updateConnectedVertices();
        std::unordered_set<Vertex*> affectedVertices;
        for(auto &v : v1->connectedVertices()) 
            affectedVertices.insert(v);
        for(auto &v : v2->connectedVertices()) 
            affectedVertices.insert(v);
        affectedVertices.erase(this);
        for(auto &v : affectedVertices) 
            v->updateConnectedVertices();
    }

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    MeshSolver::log(MeshLogEventType::Create, {_objId, v1->_objId, v2->_objId}, {objType(), v1->objType(), v2->objType()}, "insert");

    return S_OK;
}

Vertex *Vertex::insert(const FVector3 &position, Vertex *v1, Vertex *v2) {
    int _pid;
    Vertex *result = Vertex_create(position, _pid);
    if(!result) {
        TF_Log(LOG_ERROR) << "Could not create vertex";
        return NULL;
    }
    result->pid = _pid;
    result->positionChanged();
    if(result->insert(v1, v2) != S_OK) {
        result->destroy();
        return NULL;
    }
    return result;
}

VertexHandle Vertex::insert(const FVector3 &position, const VertexHandle &v1, const VertexHandle &v2) {
    Vertex_GETMESH(mesh, VertexHandle());
    if(mesh->ensureAvailableVertices(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    Vertex *_v1 = v1.vertex();
    Vertex *_v2 = v2.vertex();
    if(!_v1 || !_v2) {
        VertexHandle_INVALIDHANDLERR;
        return VertexHandle();
    }
    Vertex *v = Vertex::insert(position, _v1, _v2);
    if(!v) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    return VertexHandle(v->objectId());
}

HRESULT Vertex::insert(Vertex *vf, std::vector<Vertex*> nbs) {
    for(auto &v : nbs) 
        if(insert(vf, v) != S_OK) 
            return E_FAIL;
    return S_OK;
}

Vertex *Vertex::insert(const FVector3 &position, Vertex *vf, std::vector<Vertex*> nbs) {
    int _pid;
    Vertex *result = Vertex_create(position, _pid);
    if(!result) {
        TF_Log(LOG_ERROR) << "Could not create vertex";
        return NULL;
    }
    result->pid = _pid;
    result->positionChanged();
    if(result->insert(vf, nbs) != S_OK) {
        result->destroy();
        return NULL;
    }
    return result;
}

VertexHandle Vertex::insert(const FVector3 &position, const VertexHandle &vf, const std::vector<VertexHandle> &nbs) {
    Vertex_GETMESH(mesh, VertexHandle());
    if(mesh->ensureAvailableVertices(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    Vertex *_vf = vf.vertex();
    if(!_vf) {
        VertexHandle_INVALIDHANDLERR;
        return VertexHandle();
    }
    std::vector<Vertex*> _nbs;
    _nbs.reserve(nbs.size());
    for(auto &n : nbs) {
        Vertex *_n = n.vertex();
        if(!_n) {
            VertexHandle_INVALIDHANDLERR;
            return VertexHandle();
        }
        _nbs.push_back(_n);
    }
    Vertex *v = Vertex::insert(position, _vf, _nbs);
    if(!v) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    return VertexHandle(v->objectId());
}

HRESULT Vertex::splitPlan(const FVector3 &sep, std::vector<Vertex*> &verts_v, std::vector<Vertex*> &verts_new_v) {
    // Verify inputs
    if(sep.isZero()) 
        return tf_error(E_FAIL, "Zero separation");

    verts_v.clear();
    verts_new_v.clear();

    std::vector<Vertex*> nbs = connectedVertices();

    // Verify that the vertex defines at least one surface
    if(nbs.size() == 0) 
        return tf_error(E_FAIL, "Vertex must define a surface");
    
    // Define a cut plane at the midpoint of and orthogonal to the new edge
    FVector4 planeEq = FVector4::planeEquation(sep.normalized(), getPosition());

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

Vertex *Vertex::splitExecute(const FVector3 &sep, const std::vector<Vertex*> &verts_v, const std::vector<Vertex*> &verts_new_v) {
    FVector3 v_pos0 = getPosition();
    FVector3 hsep = sep * 0.5;
    FVector3 v_pos1 = v_pos0 - hsep;
    FVector3 u_pos = v_pos0 + hsep;

    // Determine which surfaces the target vertex will no longer partially define
    // A surface remains partially defined by the target vertex if the target vertex has 
    // a neighbor on its own side of the cut plane that also partially defines the surface
    std::set<Surface*> u_surfs, vn_surfs;
    for(auto &nv : verts_v) 
        for(auto &s : nv->sharedSurfaces(this)) 
            vn_surfs.insert(s);
    for(auto &nv : verts_new_v) 
        for(auto &s : nv->sharedSurfaces(this)) 
            u_surfs.insert(s);
    std::set<Surface*> surfs_keep_v, surfs_remove_v;
    for(auto &s : u_surfs) {
        if(std::find(vn_surfs.begin(), vn_surfs.end(), s) == vn_surfs.end()) 
            surfs_remove_v.insert(s);
        else 
            surfs_keep_v.insert(s);
    }

    // Create and insert the new vertex
    int _pid;
    Vertex *u = Vertex_create(u_pos, _pid);
    if(!u) {
        tf_error(E_FAIL, "Could not add vertex");
        return 0;
    }
    u->pid = _pid;
    u->positionChanged();
    setPosition(v_pos1);

    //  Replace v with u where removing
    for(auto &s : surfs_remove_v) {
        remove(s);
        u->add(s);
        s->replace(u, this);
    }

    //  Insert u between v and neighbor where not removing
    for(auto &s : surfs_keep_v) {
        u->add(s);
        for(auto &nv : verts_new_v) {
            std::vector<Vertex*>::iterator verts_new_v_itr = std::find(s->vertices.begin(), s->vertices.end(), nv);
            if(verts_new_v_itr != s->vertices.end()) { 
                s->insert(u, this, *verts_new_v_itr);
                break;
            }
        }
    }

    updateConnectedVertices();
    u->updateConnectedVertices();
    for(auto &nv : _connectedVertices) 
        nv->updateConnectedVertices();
    for(auto &nv : u->connectedVertices()) 
        nv->updateConnectedVertices();

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {_objId, u->_objId}, {objType(), u->objType()}, "split");

    return u;
}

Vertex *Vertex::split(const FVector3 &sep) {
    
    std::vector<Vertex*> verts_v, new_verts_v;
    Vertex *u = NULL;
    if(splitPlan(sep, verts_v, new_verts_v))
        u = splitExecute(sep, verts_v, new_verts_v);
    if(!u) {
        tf_error(E_FAIL, "Failed to split");
        return NULL;
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {_objId, u->_objId}, {objType(), u->objType()}, "split");

    return u;
}


//////////////////
// VertexHandle //
//////////////////


VertexHandle::VertexHandle(const int &_id) : id{_id} {}

Vertex *VertexHandle::vertex() const {
    Vertex *o = this->id >= 0 ? Mesh::get()->getVertex(this->id) : NULL;
    if(!o) {
        TF_Log(LOG_ERROR) << "Invalid handle";
    }
    return o;
}

bool VertexHandle::defines(const SurfaceHandle &s) const {
    VertexHandle_GETOBJ(o, false);
    Surface *_s = s.surface();
    if(!_s) {
        VertexHandle_INVALIDHANDLERR;
        return false;
    }
    return o->defines(_s);
}

bool VertexHandle::defines(const BodyHandle &b) const {
    VertexHandle_GETOBJ(o, false);
    Body *_b = b.body();
    if(!_b) {
        VertexHandle_INVALIDHANDLERR;
        return false;
    }
    return o->defines(_b);
}

HRESULT VertexHandle::destroy() {
    VertexHandle_GETOBJ(o, E_FAIL);
    HRESULT res = o->destroy();
    if(res == S_OK) 
        this->id = -1;
    return res;
}

bool VertexHandle::validate() {
    VertexHandle_GETOBJ(o, false);
    return o->validate();
}

HRESULT VertexHandle::positionChanged() {
    VertexHandle_GETOBJ(o, E_FAIL);
    return o->positionChanged();
}

std::string VertexHandle::str() const {
    std::stringstream ss;
    ss << "VertexHandle(";
    if(this->id >= 0) 
        ss  << "id=" << this->id;
    ss << ")";
    return ss.str();
}

std::string VertexHandle::toString() const {
    return TissueForge::io::toString(*this);
}

VertexHandle VertexHandle::fromString(const std::string &s) {
    return TissueForge::io::fromString<VertexHandle>(s);
}

HRESULT VertexHandle::add(const SurfaceHandle &s) const {
    VertexHandle_GETOBJ(o, E_FAIL);
    Surface *_s = s.surface();
    if(!_s) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->add(_s);
}

HRESULT VertexHandle::insert(const SurfaceHandle &s, const int &idx) const {
    VertexHandle_GETOBJ(o, E_FAIL);
    Surface *_s = s.surface();
    if(!_s) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->insert(_s, idx);
}

HRESULT VertexHandle::insert(const SurfaceHandle &s, const SurfaceHandle &before) const {
    VertexHandle_GETOBJ(o, E_FAIL);
    Surface *_s = s.surface();
    Surface *_before = before.surface();
    if(!_s || !_before) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->insert(_s, _before);
}

HRESULT VertexHandle::remove(const SurfaceHandle &s) const {
    VertexHandle_GETOBJ(o, E_FAIL);
    Surface *_s = s.surface();
    if(!_s) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->remove(_s);
}

HRESULT VertexHandle::replace(const SurfaceHandle &toInsert, const int &idx) const {
    VertexHandle_GETOBJ(o, E_FAIL);
    Surface *_toInsert = toInsert.surface();
    if(!_toInsert) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toInsert, idx);
}

HRESULT VertexHandle::replace(const SurfaceHandle &toInsert, const SurfaceHandle &toRemove) const {
    VertexHandle_GETOBJ(o, E_FAIL);
    Surface *_toInsert = toInsert.surface();
    Surface *_toRemove = toRemove.surface();
    if(!_toInsert || !_toRemove) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toInsert, _toRemove);
}

const int VertexHandle::getPartId() const {
    VertexHandle_GETOBJ(o, -1);
    return o->getPartId();
}

std::vector<BodyHandle> VertexHandle::getBodies() const {
    VertexHandle_GETOBJ(o, {});
    auto bodies = o->getBodies();
    std::vector<BodyHandle> result;
    result.reserve(result.size());
    for(auto &b : bodies) 
        result.emplace_back(b->objectId());
    return result;
}

std::vector<SurfaceHandle> VertexHandle::getSurfaces() const {
    VertexHandle_GETOBJ(o, {});
    auto& surfaces = o->getSurfaces();
    std::vector<SurfaceHandle> result;
    result.reserve(surfaces.size());
    for(auto &s : surfaces) 
        result.emplace_back(s->objectId());
    return result;
}

SurfaceHandle VertexHandle::findSurface(const FVector3 &dir) const {
    VertexHandle_GETOBJ(o, SurfaceHandle());
    Surface *s = o->findSurface(dir);
    int sid = s ? s->objectId() : -1;
    return SurfaceHandle(sid);
}

BodyHandle VertexHandle::findBody(const FVector3 &dir) const {
    VertexHandle_GETOBJ(o, BodyHandle());
    Body *b = o->findBody(dir);
    int bid = b ? b->objectId() : -1;
    return BodyHandle(bid);
}

void VertexHandle::updateConnectedVertices() const {
    VertexHandle_GETOBJ(o, );
    return o->updateConnectedVertices();
}

std::vector<VertexHandle> VertexHandle::connectedVertices() const {
    VertexHandle_GETOBJ(o, {});
    auto nbs = o->connectedVertices();
    std::vector<VertexHandle> result;
    result.reserve(nbs.size());
    for(auto &n : nbs) 
        result.emplace_back(n->objectId());
    return result;
}

std::vector<SurfaceHandle> VertexHandle::sharedSurfaces(const VertexHandle &other) const {
    VertexHandle_GETOBJ(o, {});
    Vertex *_other = other.vertex();
    if(!_other) {
        VertexHandle_INVALIDHANDLERR;
        return {};
    }
    auto ss = o->sharedSurfaces(_other);
    std::vector<SurfaceHandle> result;
    result.reserve(ss.size());
    for(auto &s : ss) 
        result.emplace_back(s->objectId());
    return result;
}

FloatP_t VertexHandle::getArea() const {
    VertexHandle_GETOBJ(o, 0);
    return o->getArea();
}

FloatP_t VertexHandle::getVolume() const {
    VertexHandle_GETOBJ(o, 0);
    return o->getVolume();
}

FloatP_t VertexHandle::getMass() const {
    VertexHandle_GETOBJ(o, 0);
    return o->getMass();
}

HRESULT VertexHandle::updateProperties() const {
    VertexHandle_GETOBJ(o, E_FAIL);
    return o->updateProperties();
}

ParticleHandle *VertexHandle::particle() const {
    VertexHandle_GETOBJ(o, NULL);
    return o->particle();
}

FVector3 VertexHandle::getPosition() const {
    VertexHandle_GETOBJ(o, FVector3());
    return o->getPosition();
}

HRESULT VertexHandle::setPosition(const FVector3 &pos, const bool &updateChildren) {
    VertexHandle_GETOBJ(o, E_FAIL);
    return o->setPosition(pos, updateChildren);
}

FVector3 VertexHandle::getVelocity() const {
    VertexHandle_GETOBJ(o, FVector3());
    return o->getVelocity();
}

HRESULT VertexHandle::transferBondsTo(const VertexHandle &other) const {
    VertexHandle_GETOBJ(o, E_FAIL);
    Vertex *_other = other.vertex();
    if(!_other) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->transferBondsTo(_other);
}

HRESULT VertexHandle::replace(SurfaceHandle &toReplace) const {
    VertexHandle_GETOBJ(o, E_FAIL);
    Surface *_toReplace = toReplace.surface();
    if(!_toReplace) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toReplace);
}

HRESULT VertexHandle::replace(BodyHandle &toReplace) {
    VertexHandle_GETOBJ(o, E_FAIL);
    Body *_toReplace = toReplace.body();
    if(!_toReplace) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toReplace);
}

HRESULT VertexHandle::merge(VertexHandle &toRemove, const FloatP_t &lenCf) {
    VertexHandle_GETOBJ(o, E_FAIL);
    Vertex *_toRemove = toRemove.vertex();
    if(!_toRemove) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    HRESULT res = o->merge(_toRemove);
    if(res == S_OK) {
        toRemove.id = -1;
    } 
    else {
        TF_Log(LOG_ERROR);
    }
    return res;
}

HRESULT VertexHandle::insert(const VertexHandle &v1, const VertexHandle &v2) {
    VertexHandle_GETOBJ(o, E_FAIL);
    Vertex *_v1 = v1.vertex();
    Vertex *_v2 = v2.vertex();
    if(!_v1 || !_v2) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->insert(_v1, _v2);
}

HRESULT VertexHandle::insert(const VertexHandle &vf, const std::vector<VertexHandle> &nbs) {
    VertexHandle_GETOBJ(o, E_FAIL);
    Vertex *_vf = vf.vertex();
    if(!_vf) {
        VertexHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    std::vector<Vertex*> _nbs;
    _nbs.reserve(nbs.size());
    for(auto &n : nbs) {
        Vertex *_n = n.vertex();
        if(!_n) {
            VertexHandle_INVALIDHANDLERR;
            return E_FAIL;
        }
        _nbs.push_back(_n);
    }
    return o->insert(_vf, _nbs);
}

HRESULT VertexHandle::splitPlan(const FVector3 &sep, const std::vector<VertexHandle> &verts_v, const std::vector<VertexHandle> &verts_new_v) {
    VertexHandle_GETOBJ(o, E_FAIL);
    std::vector<Vertex*> _verts_v;
    _verts_v.reserve(verts_v.size());
    for(auto &v : verts_v) {
        Vertex *_v = v.vertex();
        if(!_v) {
            VertexHandle_INVALIDHANDLERR;
            return E_FAIL;
        }
        _verts_v.push_back(_v);
    }
    std::vector<Vertex*> _verts_new_v;
    _verts_new_v.reserve(verts_new_v.size());
    for(auto &v : verts_new_v) {
        Vertex *_v = v.vertex();
        if(!_v) {
            VertexHandle_INVALIDHANDLERR;
            return E_FAIL;
        }
        _verts_new_v.push_back(_v);
    }
    return o->splitPlan(sep, _verts_v, _verts_new_v);
}

VertexHandle VertexHandle::splitExecute(const FVector3 &sep, const std::vector<VertexHandle> &verts_v, const std::vector<VertexHandle> &verts_new_v) {
    Vertex_GETMESH(mesh, VertexHandle());
    if(mesh->ensureAvailableVertices(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    VertexHandle_GETOBJ(o, E_FAIL);
    std::vector<Vertex*> _verts_v;
    _verts_v.reserve(verts_v.size());
    for(auto &v : verts_v) {
        Vertex *_v = v.vertex();
        if(!_v) {
            VertexHandle_INVALIDHANDLERR;
            return E_FAIL;
        }
        _verts_v.push_back(_v);
    }
    std::vector<Vertex*> _verts_new_v;
    _verts_new_v.reserve(verts_new_v.size());
    for(auto &v : verts_new_v) {
        Vertex *_v = v.vertex();
        if(!_v) {
            VertexHandle_INVALIDHANDLERR;
            return E_FAIL;
        }
        _verts_new_v.push_back(_v);
    }
    Vertex *v = o->splitExecute(sep, _verts_v, _verts_new_v);
    if(!v) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    return VertexHandle(v->objectId());
}

VertexHandle VertexHandle::split(const FVector3 &sep) {
    Vertex_GETMESH(mesh, VertexHandle());
    if(mesh->ensureAvailableVertices(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    VertexHandle_GETOBJ(o, VertexHandle());
    Vertex *v = o->split(sep);
    if(!v) {
        TF_Log(LOG_ERROR);
        return VertexHandle();
    }
    return VertexHandle(v->objectId());
}


////////
// io //
////////


namespace TissueForge::io {


    template <>
    HRESULT toFile(TissueForge::models::vertex::Vertex *dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "objId", dataElement->objectId());

        ParticleHandle *ph = dataElement->particle();
        if(ph == NULL) {
            TF_IOTOEASY(fileElement, metaData, "pid", -1);
        } 
        else {
            TF_IOTOEASY(fileElement, metaData, "pid", ph->getId());
        }

        auto& surfaces = dataElement->getSurfaces();
        std::vector<int> surfaceIds;
        surfaceIds.reserve(surfaces.size());
        for(auto &s : surfaces) 
            surfaceIds.push_back(s->objectId());
        TF_IOTOEASY(fileElement, metaData, "surfaces", surfaceIds);

        fileElement.get()->type = "Vertex";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Vertex **dataElement) {
        
        if(!FIO::hasImport()) 
            return tf_error(E_FAIL, "No import data available");
        else if(!TissueForge::models::vertex::io::VertexSolverFIOModule::hasImport()) 
            return tf_error(E_FAIL, "No vertex import data available");

        int pidOld;
        TF_IOFROMEASY(fileElement, metaData, "pid", &pidOld);
        auto idItr = FIO::importSummary->particleIdMap.find(pidOld);
        if(idItr == FIO::importSummary->particleIdMap.end() || idItr->second < 0) 
            return tf_error(E_FAIL, "Could not locate particle to import");

        *dataElement = TissueForge::models::vertex::Vertex::create(idItr->second).vertex();
        if(!(*dataElement)) {
            return tf_error(E_FAIL, "Failed to add vertex");
        }

        int objIdOld;
        TF_IOFROMEASY(fileElement, metaData, "objId", &objIdOld);
        TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->vertexIdMap.insert({objIdOld, (*dataElement)->objectId()});

        return S_OK;
    }

    template <>
    HRESULT toFile(const TissueForge::models::vertex::VertexHandle &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);

        fileElement.get()->type = "VertexHandle";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::VertexHandle *dataElement) {

        int id;
        TF_IOFROMEASY(fileElement, metaData, "id", &id);
        *dataElement = TissueForge::models::vertex::VertexHandle(id);

        return S_OK;
    }
}

std::string TissueForge::models::vertex::Vertex::toString() {
    TissueForge::io::IOElement el = TissueForge::io::IOElement::create();
    std::string result;
    if(TissueForge::io::toFile(this, TissueForge::io::MetaData(), el) == S_OK) 
        result = TissueForge::io::toStr(el);
    else 
        result = "";
    return result;
}
