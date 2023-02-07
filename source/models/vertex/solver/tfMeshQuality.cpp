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

#include "tfMeshQuality.h"

#include "tfMesh.h"
#include "tfMeshSolver.h"
#include "tf_mesh_metrics.h"
#include "tf_mesh_io.h"
#include "tfVertexSolverFIO.h"
#include "tf_mesh_ops.h"

#include <tfError.h>
#include <tfUniverse.h>
#include <tf_metrics.h>
#include <tfTaskScheduler.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>

#include <Magnum/Math/Math.h>

#include <atomic>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


//////////////////////////
// MeshQualityOperation //
//////////////////////////


HRESULT MeshQualityOperation_checkChain( 
    MeshQualityOperation *op, 
    std::vector<MeshQualityOperation*> &ops
) {
    for(auto &t : op->targets) {
        MeshQualityOperation *t_op = ops[t];
        if(t_op && op != t_op) 
            op->appendNext(t_op);
        
    }

    return S_OK;
}

MeshQualityOperation::MeshQualityOperation(Mesh *_mesh) : 
    flags{Flag::None}, 
    mesh{_mesh}
{}

HRESULT MeshQualityOperation::appendNext(MeshQualityOperation *_next) {
    // Prevent loops
    std::set<MeshQualityOperation*> us = this->upstreams();
    if(std::find(us.begin(), us.end(), _next) != us.end()) 
        return S_OK;

    _next->lock.lock();
    _next->prev.insert(this);
    _next->lock.unlock();

    next.insert(_next);
    
    return S_OK;
}

HRESULT MeshQualityOperation::removeNext(MeshQualityOperation *_next) {
    auto itr = std::find(next.begin(), next.end(), _next);
    if(itr == next.end()) 
        return E_FAIL;
    next.erase(itr);

    itr = std::find(_next->prev.begin(), _next->prev.end(), this);
    if(itr != _next->prev.end()) {
        _next->lock.lock();
        _next->prev.erase(itr);
        _next->lock.unlock();
    }

    return S_OK;
}

static void MeshQuality_upstreams(const MeshQualityOperation *op, std::set<MeshQualityOperation*> &ops) {
    for(auto *op_u : op->prev) {
        if(std::find(ops.begin(), ops.end(), op_u) == ops.end()) {
            ops.insert(op_u);
            MeshQuality_upstreams(op_u, ops);
        }
    }
}

std::set<MeshQualityOperation*> MeshQualityOperation::upstreams() const {
    std::set<MeshQualityOperation*> result;
    MeshQuality_upstreams(this, result);
    return result;
}

static void MeshQuality_downstreams(const MeshQualityOperation *op, std::set<MeshQualityOperation*> &ops) {
    for(auto *op_d : op->next) {
        if(std::find(ops.begin(), ops.end(), op_d) == ops.end()) {
            ops.insert(op_d);
            MeshQuality_downstreams(op_d, ops);
        }
    }
}

std::set<MeshQualityOperation*> MeshQualityOperation::downstreams() const {
    std::set<MeshQualityOperation*> result;
    MeshQuality_downstreams(this, result);
    return result;
}

template <typename T> 
std::set<MeshQualityOperation*> MeshQualityOperation_headOperations(const T &ops) {
    std::set<MeshQualityOperation*> result;
    for(auto &op : ops) 
        if(op->prev.empty()) 
            result.insert(op);
    return result;
}

std::set<MeshQualityOperation*> MeshQualityOperation::headOperations() const { 
    std::set<MeshQualityOperation*> us = upstreams();
    us.insert(const_cast<MeshQualityOperation*>(this));
    return MeshQualityOperation_headOperations(std::vector<MeshQualityOperation*>(us.begin(), us.end()));
}


////////////////
// Operations //
////////////////


/** Merges two vertices */
struct VertexMergeOperation : MeshQualityOperation {

    int vId, v2Id;
    Vertex *v, *v2;
    std::unordered_set<int> affectedChildren;

    VertexMergeOperation(Mesh *_mesh, Vertex *_source, Vertex *_target) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        vId = _source->objectId();
        v2Id = _target->objectId();
        for(auto &s : _source->sharedSurfaces(_target)) 
            targets.push_back(s->objectId());
    };

    void prep() override {
        v =  mesh->getVertex(vId);
        v2 = mesh->getVertex(v2Id);
        for(auto &t : targets) 
            for(auto &b : mesh->getSurface(t)->getBodies()) 
                affectedChildren.insert(b->objectId());
    }

    std::vector<int> implement() override { 
        MeshSolver::engineLock();

        HRESULT res = v->merge(v2);
        
        MeshSolver::engineUnlock();

        if(res == S_OK) {
            next.clear();
            return std::vector<int>(affectedChildren.begin(), affectedChildren.end());
        }
        return {};
    }
};


/** Creates and inserts a vertex between two vertices */
struct VertexCreateInsertOperation : MeshQualityOperation {

    int v1Id, v2Id;
    Vertex *v1, *v2;
    std::unordered_set<int> affectedChildren;

    VertexCreateInsertOperation(Mesh *_mesh, Vertex *_source, Vertex *_target) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        v1Id = _source->objectId();
        v2Id = _target->objectId();
        for(auto &s : _source->sharedSurfaces(_target)) 
            targets.push_back(s->objectId());
    };

    size_t numNewVertices() const override { return 1; }

    void prep() override {
        v1 = mesh->getVertex(v1Id);
        v2 = mesh->getVertex(v2Id);
        for(auto &t : targets) 
            for(auto &b : mesh->getSurface(t)->getBodies()) 
                affectedChildren.insert(b->objectId());
    }

    std::vector<int> implement() override { 

        MeshSolver::engineLock();
        
        HRESULT res = Vertex::insert((v1->getPosition() + v2->getPosition()) * 0.5, v1, v2) != NULL ? S_OK : E_FAIL;
        
        MeshSolver::engineUnlock();

        if(res == S_OK) {
            next.clear();
            return std::vector<int>(affectedChildren.begin(), affectedChildren.end());
        }
        return {};
    }
};

/** Inserts a vertex between two vertices */
struct VertexInsertOperation : MeshQualityOperation {

    int vId, vaId, vbId;
    Vertex *v, *va, *vb;
    std::unordered_set<int> affectedChildren;

    VertexInsertOperation(Mesh *_mesh, Vertex *_source, Surface *_target, Vertex *_va, Vertex *_vb) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        vId = _source->objectId();
        vaId = _va->objectId();
        vbId = _vb->objectId();

        std::unordered_set<int> target_surfs = {_target->objectId()};
        for(auto &s : _source->getSurfaces()) 
            target_surfs.insert(s->objectId());
        for(auto &s : _target->connectedSurfaces({_va, _vb})) 
            target_surfs.insert(s->objectId());
        targets = std::vector<int>(target_surfs.begin(), target_surfs.end());
    };

    void prep() override {
        v = mesh->getVertex(vId);
        va = mesh->getVertex(vaId);
        vb = mesh->getVertex(vbId);
        for(auto &t : targets) 
            for(auto &b : mesh->getSurface(t)->getBodies()) 
                affectedChildren.insert(b->objectId());
    }

    std::vector<int> implement() override {
        MeshSolver::engineLock();

        HRESULT res = v->insert(va, vb);

        MeshSolver::engineUnlock();

        if(res == S_OK) {
            next.clear();
            return std::vector<int>(affectedChildren.begin(), affectedChildren.end());
        }
        return {};
    }
};

/** Converts a body to a vertex */
struct BodyDemoteOperation : MeshQualityOperation {

    int toReplaceId;
    Body *toReplace;

    BodyDemoteOperation(Mesh *_mesh, Body *_source) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        toReplaceId = _source->objectId();
        for(auto &nb : adjacentTo(removedBodiesByB2V(_source))) 
            targets.push_back(nb->objectId());
    }

    size_t numNewVertices() const override { return 1; };

    void prep() override {
        toReplace = mesh->getBody(toReplaceId);
    }

    std::vector<int> implement() override {
        const FVector3 toReplaceCentroid = toReplace->getCentroid();

        MeshSolver::engineLock();

        HRESULT res = Vertex::replace(toReplaceCentroid, toReplace) != NULL ? S_OK : E_FAIL;

        MeshSolver::engineUnlock();

        if(res == S_OK) 
            next.clear();
        return {};
    };
};

/** Converts a surface to a vertex */
struct SurfaceDemoteOperation : MeshQualityOperation {

    int toReplaceId;
    Surface *toReplace;
    std::unordered_set<int> affectedChildren;

    SurfaceDemoteOperation(Mesh *_mesh, Surface *_source) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        toReplaceId = _source->objectId();
        std::unordered_set<Surface*> removed = removedSurfacesByS2V(_source);
        for(auto &nb : removed) 
            targets.push_back(nb->objectId());
        for(auto &nb : connectedSurfacesToS2V(removed)) 
            if(nb->objectId() != _source->objectId()) 
                targets.push_back(nb->objectId());
    }

    size_t numNewVertices() const override { return 1; };

    void prep() override {
        toReplace = mesh->getSurface(toReplaceId);
        for(auto &t : targets) 
            for(auto &b : mesh->getSurface(t)->getBodies()) 
                affectedChildren.insert(b->objectId());
    }

    std::vector<int> implement() override {
        const FVector3 toReplaceCentroid = toReplace->getCentroid();

        MeshSolver::engineLock();
        
        HRESULT res = Vertex::replace(toReplaceCentroid, toReplace) != NULL ? S_OK : E_FAIL;

        MeshSolver::engineUnlock();

        if(res == S_OK) {
            next.clear();
            return std::vector<int>(affectedChildren.begin(), affectedChildren.end());
        }
        return {};
    }
};

/** Converts a surface edge to a vertex */
struct EdgeDemoteOperation : MeshQualityOperation {

    int vId, v2Id;
    Vertex *v, *v2;
    std::unordered_set<int> affectedChildren;

    EdgeDemoteOperation(Mesh *_mesh, Vertex *_v1, Vertex *_v2) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        vId = _v1->objectId();
        v2Id = _v2->objectId();

        std::unordered_set<int> targets_set{vId, v2Id};
        for(auto &c : _v1->connectedVertices()) 
            targets_set.insert(c->objectId());
        for(auto &c : _v2->connectedVertices()) 
            targets_set.insert(c->objectId());
        targets = std::vector<int>(targets_set.begin(), targets_set.end());
    }

    void prep() override {
        v = mesh->getVertex(vId);
        v2 = mesh->getVertex(v2Id);
        for(auto &t : targets) 
            for(auto &s : mesh->getVertex(t)->getSurfaces()) 
                affectedChildren.insert(s->objectId());
    }

    std::vector<int> implement() override {
        MeshSolver::engineLock();
        HRESULT res = v->merge(v2);
        MeshSolver::engineUnlock();
        if(res == S_OK) {
            next.clear();
            return std::vector<int>(affectedChildren.begin(), affectedChildren.end());
        }
        return {};
    }
};

/** Converts a vertex to an edge */
struct EdgeInsertOperation : MeshQualityOperation {

    int v0Id, v1Id, v2Id;
    Vertex *v0, *v1, *v2;
    std::unordered_set<int> affectedChildren;

    EdgeInsertOperation(Mesh *_mesh, Vertex *_source, Vertex *_target1, Vertex *_target2) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        v0Id = _source->objectId();
        v1Id = _target1->objectId();
        v2Id = _target2->objectId();
        std::unordered_set<int> _targets;
        for(auto &s : _source->sharedSurfaces(_target1)) 
            _targets.insert(s->objectId());
        for(auto &s : _source->sharedSurfaces(_target2)) 
            _targets.insert(s->objectId());
        targets = std::vector<int>(_targets.begin(), _targets.end());
    }

    void prep() override {
        v0 = mesh->getVertex(v0Id);
        v1 = mesh->getVertex(v1Id);
        v2 = mesh->getVertex(v2Id);
        for(auto &t : targets) 
            for(auto &s : mesh->getVertex(t)->getSurfaces()) 
                affectedChildren.insert(s->objectId());
    }

    size_t numNewVertices() const override { return 2; };

    std::vector<int> implement() override {
        const FVector3 pos0 = v0->getPosition();
        const FVector3 pos1 = (pos0 + v1->getPosition()) * 0.5;
        const FVector3 pos2 = (pos0 + v2->getPosition()) * 0.5;

        MeshSolver::engineLock();

        Vertex::insert(pos1, v0, v1);
        Vertex::insert(pos2, v0, v2);
        v0->destroy();

        MeshSolver::engineUnlock();

        next.clear();
        return std::vector<int>(affectedChildren.begin(), affectedChildren.end());
    }

};


static bool MeshQuality_vertexSplitTest(
    Vertex *v, 
    Mesh *m, 
    const FloatP_t &edgeSplitDist, 
    FVector3 &sep, 
    const std::vector<Vertex*> &v_nbs, 
    std::vector<Vertex*> &vert_nbs, 
    std::vector<Vertex*> &new_vert_nbs
) {
    Particle *p = v->particle()->part();

    // Calculate current relative force
    FVector3 force_rel(0);
    for(auto &vn : v_nbs) {
        force_rel += vn->particle()->getForce();
    }
    force_rel -= p->force * v_nbs.size();
    FPTYPE mask[] = {
        (p->flags & PARTICLE_FROZEN_X) ? 0.0f : 1.0f,
        (p->flags & PARTICLE_FROZEN_Y) ? 0.0f : 1.0f,
        (p->flags & PARTICLE_FROZEN_Z) ? 0.0f : 1.0f
    };
    for(int k = 0; k < 3; k++) force_rel[k] *= mask[k];
    if(force_rel.isZero()) 
        return false;

    sep = force_rel.normalized() * edgeSplitDist;
    if(sep.isZero()) 
        return false;

    // Get split plan along direction of force
    v->splitPlan(sep, vert_nbs, new_vert_nbs);

    // Enforce that a split must create a new shared edge
    if(vert_nbs.size() < 2 || new_vert_nbs.size() < 2) 
        return false;

    // Calculate relative force on each vertex of a new edge
    FVector3 vert_force_rel, new_vert_force_rel;
    for(auto &vn : vert_nbs) 
        vert_force_rel += vn->particle()->getForce();
    for(auto &vn : new_vert_nbs) 
        new_vert_force_rel += vn->particle()->getForce();
    vert_force_rel -= p->force * 0.5 * vert_nbs.size();
    new_vert_force_rel -= p->force * 0.5 * new_vert_nbs.size();

    // Test whether the new edge would be in tension and return true if so
    return sep.dot(vert_force_rel) < 0 && sep.dot(new_vert_force_rel) > 0;
}

/** Splits a vertex into an edge */
struct VertexSplitOperation : MeshQualityOperation { 

    int vId;
    FVector3 sep;
    Vertex *v;
    std::vector<int> vert_nbsIds, new_vert_nbsIds;
    std::vector<Vertex*> vert_nbs, new_vert_nbs;
    std::unordered_set<int> affectedChildren;

    VertexSplitOperation(Mesh *_mesh, Vertex *_source, const FVector3 &_sep, std::vector<Vertex*> _vert_nbs, std::vector<Vertex*> _new_vert_nbs) : 
        MeshQualityOperation(_mesh), 
        sep{_sep}
    {
        flags = MeshQualityOperation::Flag::Active;
        vId = _source->objectId();
        for(auto &_v : _vert_nbs) {
            vert_nbsIds.push_back(_v->objectId());
            targets.push_back(_v->objectId());
        }
        for(auto &_v : _new_vert_nbs) {
            new_vert_nbsIds.push_back(_v->objectId());
            targets.push_back(_v->objectId());
        }
    }

    size_t numNewVertices() const override { return 1; };

    void prep() override {
        v = mesh->getVertex(vId);
        vert_nbs.reserve(vert_nbsIds.size());
        new_vert_nbs.reserve(new_vert_nbsIds.size());
        for(auto &vn : vert_nbsIds) 
            vert_nbs.push_back(mesh->getVertex(vn));
        for(auto &vn : new_vert_nbsIds) 
            new_vert_nbs.push_back(mesh->getVertex(vn));
        for(auto &t : targets) 
            for(auto &s : mesh->getVertex(t)->getSurfaces()) 
                affectedChildren.insert(s->objectId());
    }

    std::vector<int> implement() override { 

        // Create a candidate vertex
        MeshSolver::engineLock();
        Vertex *new_v = v->splitExecute(sep, vert_nbs, new_vert_nbs);
        MeshSolver::engineUnlock();

        // Only invalidate if a vertex was created, since some requested configurations are invalid and subsequently ignored
        if(new_v) {
            next.clear();
            return std::vector<int>(affectedChildren.begin(), affectedChildren.end());
        }
        return {};
    }
};


/////////////////
// MeshQuality //
/////////////////


HRESULT MeshQuality_constructChains(std::vector<MeshQualityOperation*> &ops) {
    auto func = [&ops](int i) -> void {
        MeshQualityOperation *op = ops[i];
        if(!op) return;
        MeshQualityOperation_checkChain(op, ops);
    };
    parallel_for(ops.size(), func);
    return S_OK;
}

static HRESULT MeshQuality_constructOperationsVertex(
    Mesh *mesh, 
    const std::vector<bool> &passMask, 
    const FloatP_t &edgeSplitDist, 
    const FloatP_t &vertexMergeDist, 
    std::vector<MeshQualityOperation*> &ops_active, 
    std::vector<MeshQualityOperation*> &op_heads
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeVertices(), 0);
    const FloatP_t vertexMergeDist2 = vertexMergeDist * vertexMergeDist;

    auto check_verts = [&mesh, &passMask, &ops, edgeSplitDist, vertexMergeDist2](int i) -> void {
        if(passMask[i]) return;

        Vertex *v = mesh->getVertex(i);
        if(!v) return;

        MeshQualityOperation *op = NULL;

        // Check for vertex split if the vertex defines four or more surfaces

        std::vector<Vertex*> v_nbs = v->connectedVertices();
        if(v_nbs.size() > 3) {
            FVector3 sep;
            std::vector<Vertex*> vert_nbs, new_vert_nbs;
            if(MeshQuality_vertexSplitTest(v, mesh, edgeSplitDist, sep, v_nbs, vert_nbs, new_vert_nbs)) {
                ops[i] = new VertexSplitOperation(mesh, v, sep, vert_nbs, new_vert_nbs);
                return;
            }

        }

        // Check vertex merge distance for neighbor vertices with a greater id

        FVector3 vpos = v->getPosition();
        for(auto &nv : v_nbs) {

            if(v->objectId() < nv->objectId()) {

                FVector3 nvpos = nv->getPosition();
                FVector3 nvrelPos = metrics::relativePosition(vpos, nvpos);
                FloatP_t nvdist2 = nvrelPos.dot();
                if(nvdist2 < vertexMergeDist2) {
                    TF_Log(LOG_TRACE) << v->objectId() << ", " << nv->objectId() << ", " << nvrelPos;
                    
                    ops[i] = new EdgeDemoteOperation(mesh, v, nv);
                    return;
                }

            }
        }

    };
    parallel_for(mesh->sizeVertices(), check_verts);

    if(MeshQuality_constructChains(ops) != S_OK) { 
        for(size_t i = 0; i < ops.size(); i++) delete ops[i];
        ops.clear();
        return E_FAIL;
    }

    ops_active.clear();
    ops_active.reserve(ops.size());
    for(auto &op : ops) 
        if(op) 
            ops_active.push_back(op);
    std::set<MeshQualityOperation*> op_heads_set = MeshQualityOperation_headOperations(ops_active);
    op_heads = std::vector<MeshQualityOperation*>(op_heads_set.begin(), op_heads_set.end());

    return S_OK;
}

static HRESULT MeshQuality_constructOperationsSurface(
    Mesh *mesh, 
    const std::vector<bool> &passMask, 
    const FloatP_t &surfaceDemoteArea, 
    const bool &collision2D, 
    std::vector<MeshQualityOperation*> &ops_active, 
    std::vector<MeshQualityOperation*> &op_heads
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeSurfaces(), 0);

    auto check_surfs = [&mesh, &passMask, &ops, surfaceDemoteArea, collision2D](int i) -> void {
        if(passMask[i]) return;

        Surface *s = mesh->getSurface(i);
        if(!s) return;

        // Check for surface demote

        FloatP_t sArea = s->getArea();

        if(sArea < surfaceDemoteArea) {
            TF_Log(LOG_TRACE) << sArea;

            ops[i] = new SurfaceDemoteOperation(mesh, s);
            return;
        }

        if(!collision2D) 
            return;

        // Check for edge penetration

        //  Determine neighborhood search distance
        FloatP_t nbsSearchDist2 = 0;
        FVector3 centroid = s->getCentroid();
        for(auto &v : s->getVertices()) {
            FloatP_t thisVertDist2 = (centroid - v->getPosition()).dot();
            if(thisVertDist2 > nbsSearchDist2) 
                nbsSearchDist2 = thisVertDist2;
        }

        //  Get neighbors
        ParticleList nbs = metrics::neighborhoodParticles(centroid, FPTYPE_SQRT(nbsSearchDist2));

        //  Test each neighbor
        for(size_t j = 0; j < nbs.nr_parts; j++) {
            ParticleHandle *nb = nbs.item(j);
            Vertex *v_nb = mesh->getVertexByPID(nb->id);
            if(!v_nb) 
                continue;

            //  No self-intersecting
            if(v_nb->defines(s)) 
                continue;
            
            Vertex *va, *vb;
            if(s->contains(nb->getPosition(), &va, &vb)) {
                ops[i] = new VertexInsertOperation(mesh, v_nb, s, va, vb);
                return;
            }
        }

    };
    parallel_for(mesh->sizeSurfaces(), check_surfs);
    
    if(MeshQuality_constructChains(ops) != S_OK) { 
        for(size_t i = 0; i < ops.size(); i++) delete ops[i];
        ops.clear();
        return E_FAIL;
    }

    ops_active.clear();
    ops_active.reserve(ops.size());
    for(auto &op : ops) 
        if(op) 
            ops_active.push_back(op);
    std::set<MeshQualityOperation*> op_heads_set = MeshQualityOperation_headOperations(ops_active);
    op_heads = std::vector<MeshQualityOperation*>(op_heads_set.begin(), op_heads_set.end());

    return S_OK;
}

static HRESULT MeshQuality_constructOperationsBody(
    Mesh *mesh, 
    const std::vector<bool> &passMask, 
    const FloatP_t &bodyDemoteVolume, 
    std::vector<MeshQualityOperation*> &ops_active, 
    std::vector<MeshQualityOperation*> &op_heads
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeBodies(), 0);

    auto check_bodys = [&mesh, &passMask, &ops, bodyDemoteVolume](int i) -> void {
        if(passMask[i]) return;

        Body *b = mesh->getBody(i);
        if(!b) return;

        FloatP_t bvol = b->getVolume();
        
        if(bvol < bodyDemoteVolume) {
            TF_Log(LOG_TRACE) << bvol;

            ops[i] = new BodyDemoteOperation(mesh, b);
            return;
        }
    };
    parallel_for(mesh->sizeBodies(), check_bodys);
    
    if(MeshQuality_constructChains(ops) != S_OK) { 
        for(size_t i = 0; i < ops.size(); i++) delete ops[i];
        ops.clear();
        return E_FAIL;
    }

    ops_active.clear();
    ops_active.reserve(ops.size());
    for(auto &op : ops) 
        if(op) 
            ops_active.push_back(op);
    std::set<MeshQualityOperation*> op_heads_set = MeshQualityOperation_headOperations(ops_active);
    op_heads = std::vector<MeshQualityOperation*>(op_heads_set.begin(), op_heads_set.end());

    return S_OK;
}

static HRESULT MeshQuality_doOperations(MeshQualityOperation *op, std::unordered_set<int> &affectedChildren) {
    if(!op->check()) { 
        for(auto &n : op->next) {
            auto itr = std::find(n->prev.begin(), n->prev.end(), op);
            if(itr != n->prev.end()) {
                n->lock.lock();
                n->prev.erase(itr);
                n->lock.unlock();
            }
        }
        return S_OK;
    }

    for(auto &i : op->implement()) 
        affectedChildren.insert(i);

    for(auto &n : op->next) {
        auto itr = std::find(n->prev.begin(), n->prev.end(), op);
        HRESULT res = S_OK;

        n->lock.lock();
        n->prev.erase(itr);
        if(n->prev.empty()) 
            res = MeshQuality_doOperations(n, affectedChildren);
        n->lock.unlock();

        if(res != S_OK) 
            return res;
    }

    return S_OK;
}

static HRESULT MeshQuality_doOperations(
    Mesh *mesh, 
    std::vector<MeshQualityOperation*> &op_active, 
    std::vector<MeshQualityOperation*> &op_heads, 
    std::vector<int> &affectedChildren) 
{
    std::atomic<size_t> atomic_numNewVertices = 0;
    std::atomic<size_t> atomic_numNewSurfaces = 0;
    std::atomic<size_t> atomic_numNewBodies = 0;
    
    auto func_count = [&op_active, &atomic_numNewVertices, &atomic_numNewSurfaces, &atomic_numNewBodies](int tid) -> void {
        size_t _numNewVertices = 0;
        size_t _numNewSurfaces = 0;
        size_t _numNewBodies = 0;

        for(int i = tid; i < op_active.size();) {
            auto op = op_active[i];
            _numNewVertices += op->numNewVertices();
            _numNewSurfaces += op->numNewSurfaces();
            _numNewBodies += op->numNewBodies();
            i += ThreadPool::size();
        }

        atomic_numNewVertices.fetch_add(_numNewVertices);
        atomic_numNewSurfaces.fetch_add(_numNewSurfaces);
        atomic_numNewBodies.fetch_add(_numNewBodies);
    };
    parallel_for(ThreadPool::size(), func_count);

    mesh->ensureAvailableVertices(atomic_numNewVertices);
    mesh->ensureAvailableSurfaces(atomic_numNewSurfaces);
    mesh->ensureAvailableBodies(atomic_numNewBodies);
    
    parallel_for(op_active.size(), [&op_active](int i) -> void { op_active[i]->prep(); });

    static std::mutex affectedChildrenLock;
    std::unordered_set<int> affectedChildrenSet;
    auto func_do = [&op_heads, &affectedChildrenSet](int tid) -> void {
        std::unordered_set<int> affectedChildren_tid;
        for(int i = tid; i < op_heads.size();) {
            MeshQuality_doOperations(op_heads[i], affectedChildren_tid);
            i += ThreadPool::size();
        }
        affectedChildrenLock.lock();
        for(auto &i : affectedChildren_tid) 
            affectedChildrenSet.insert(i);
        affectedChildrenLock.unlock();
    };
    parallel_for(ThreadPool::size(), func_do);
    affectedChildren = std::vector<int>(affectedChildrenSet.begin(), affectedChildrenSet.end());

    return S_OK;
}

static HRESULT MeshQuality_clearOperations(std::vector<MeshQualityOperation*> &ops) {
    auto func = [&ops](int i) -> void {
        delete ops[i];
        ops[i] = NULL;
    };
    parallel_for(ops.size(), func);

    return S_OK;
}

MeshQuality::MeshQuality(
    const FloatP_t &vertexMergeDistCf, 
    const FloatP_t &surfaceDemoteAreaCf, 
    const FloatP_t &bodyDemoteVolumeCf, 
    const FloatP_t &_edgeSplitDistCf
) : 
    _working{false},
    collision2D{true}
{
    FloatP_t uvolu = Universe::dim().product();
    FloatP_t uleng = std::cbrt(uvolu);
    FloatP_t uarea = uleng * uleng;

    vertexMergeDist = uleng * vertexMergeDistCf;
    surfaceDemoteArea = uarea * surfaceDemoteAreaCf;
    bodyDemoteVolume = uvolu * bodyDemoteVolumeCf;
    edgeSplitDist = _edgeSplitDistCf * vertexMergeDist;
}

std::string MeshQuality::str() const {
    std::stringstream ss;

    ss << "MeshQuality(";
    ss << "vertexMergeDist="    << this->vertexMergeDist                << ", ";
    ss << "surfaceDemoteArea="  << this->surfaceDemoteArea              << ", ";
    ss << "bodyDemoteVolume="   << this->bodyDemoteVolume               << ", ";
    ss << "edgeSplitDist="      << this->edgeSplitDist                  << ", ";
    ss << "collision2D="        << (this->collision2D ? "yes" : "no")   << ", ";
    ss << "working="            << (this->_working    ? "yes" : "no");
    ss << ")";

    return ss.str();
}

HRESULT MeshQuality::doQuality() { 

    _working = true;

    Mesh *mesh = Mesh::get();

    std::vector<MeshQualityOperation*> op_active, op_heads;
    std::vector<int> affectedChildren;
    std::vector<bool> passMask;

    // Vertex checks
    
    passMask = std::vector<bool>(mesh->sizeVertices(), false);
    for(auto &i : excludedVertices) 
        if(i < passMask.size()) 
            passMask[i] = true;
    if(MeshQuality_constructOperationsVertex(mesh, passMask, edgeSplitDist, vertexMergeDist, op_active, op_heads) != S_OK || 
        MeshQuality_doOperations(mesh, op_active, op_heads, affectedChildren) != S_OK || 
        MeshQuality_clearOperations(op_active) != S_OK) {
        _working = false;
        return E_FAIL;
    }

    // Surface checks

    passMask = std::vector<bool>(mesh->sizeSurfaces(), false);
    std::unordered_set<int> affectedBodiesImpl;
    for(auto &i : affectedChildren) {
        passMask[i] = true;
        Surface *s = mesh->getSurface(i);
        if(s) 
            for(auto &b : s->getBodies()) 
                affectedBodiesImpl.insert(b->objectId());
    }
    for(auto &i : excludedSurfaces) 
        if(i < passMask.size()) 
            passMask[i] = true;
    if(MeshQuality_constructOperationsSurface(mesh, passMask, surfaceDemoteArea, collision2D, op_active, op_heads) != S_OK || 
        MeshQuality_doOperations(mesh, op_active, op_heads, affectedChildren) != S_OK || 
        MeshQuality_clearOperations(op_active) != S_OK) {
        _working = false;
        return E_FAIL;
    }

    // Body checks

    passMask = std::vector<bool>(mesh->sizeBodies(), false);
    for(auto &i : affectedChildren) 
        passMask[i] = true;
    for(auto &i : affectedBodiesImpl) 
        passMask[i] = true;
    for(auto &i : excludedBodies) 
        if(i < passMask.size()) 
            passMask[i] = true;
    if(MeshQuality_constructOperationsBody(mesh, passMask, bodyDemoteVolume, op_active, op_heads) != S_OK || 
        MeshQuality_doOperations(mesh, op_active, op_heads, affectedChildren) != S_OK || 
        MeshQuality_clearOperations(op_active) != S_OK) {
        _working = false;
        return E_FAIL;
    }

    _working = false;
    
    return S_OK;
}

HRESULT MeshQuality::setVertexMergeDistance(const FloatP_t &_val) {
    if(_val < 0) 
        return E_FAIL;
    vertexMergeDist = _val;
    return S_OK;
}

HRESULT MeshQuality::setSurfaceDemoteArea(const FloatP_t &_val) {
    if(_val < 0) 
        return E_FAIL;
    surfaceDemoteArea = _val;
    return S_OK;
}

HRESULT MeshQuality::setBodyDemoteVolume(const FloatP_t &_val) {
    if(_val < 0) 
        return E_FAIL;
    bodyDemoteVolume = _val;
    return S_OK;
}

HRESULT MeshQuality::setEdgeSplitDist(const FloatP_t &_val) {
    if(_val <= 0) 
        return E_FAIL;
    edgeSplitDist = _val;
    return S_OK;
}

HRESULT MeshQuality::setCollision2D(const bool &_collision2D) {
    collision2D = _collision2D;
    return S_OK;
}

HRESULT MeshQuality::excludeVertex(const unsigned int &id) {
    excludedVertices.insert(id);
    return S_OK;
}

HRESULT MeshQuality::excludeSurface(const unsigned int &id) {
    excludedSurfaces.insert(id);
    return S_OK;
}

HRESULT MeshQuality::excludeBody(const unsigned int &id) {
    excludedBodies.insert(id);
    return S_OK;
}

HRESULT MeshQuality::includeVertex(const unsigned int &id) {
    excludedVertices.erase(id);
    return S_OK;
}

HRESULT MeshQuality::includeSurface(const unsigned int &id) {
    excludedSurfaces.erase(id);
    return S_OK;
}

HRESULT MeshQuality::includeBody(const unsigned int &id) {
    excludedBodies.erase(id);
    return S_OK;
}

namespace TissueForge::io {


    #define TF_MESH_MESHQUALITYIOTOEASY(fe, key, member) \
        fe = new IOElement(); \
        if(toFile(member, metaData, fe) != S_OK)  \
            return E_FAIL; \
        fe->parent = fileElement; \
        fileElement->children[key] = fe;

    #define TF_MESH_MESHQUALITYIOFROMEASY(feItr, children, metaData, key, member_p) \
        feItr = children.find(key); \
        if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
            return E_FAIL;

    template <>
    HRESULT toFile(const TissueForge::models::vertex::MeshQuality &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_MESH_MESHQUALITYIOTOEASY(fe, "vertexMergeDist", dataElement.getVertexMergeDistance());
        TF_MESH_MESHQUALITYIOTOEASY(fe, "surfaceDemoteArea", dataElement.getSurfaceDemoteArea());
        TF_MESH_MESHQUALITYIOTOEASY(fe, "bodyDemoteVolume", dataElement.getBodyDemoteVolume());
        TF_MESH_MESHQUALITYIOTOEASY(fe, "edgeSplitDist", dataElement.getEdgeSplitDist());
        TF_MESH_MESHQUALITYIOTOEASY(fe, "collision2D", dataElement.getCollision2D());
        TF_MESH_MESHQUALITYIOTOEASY(fe, "excludedVertices", dataElement.getExcludedVertices());
        TF_MESH_MESHQUALITYIOTOEASY(fe, "excludedSurfaces", dataElement.getExcludedSurfaces());
        TF_MESH_MESHQUALITYIOTOEASY(fe, "excludedBodies", dataElement.getExcludedBodies());

        fileElement->type = "MeshQuality";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::MeshQuality *dataElement) {
        
        IOChildMap::const_iterator feItr;

        FloatP_t vertexMergeDist;
        TF_MESH_MESHQUALITYIOFROMEASY(feItr, fileElement.children, metaData, "vertexMergeDist", &vertexMergeDist);
        dataElement->setVertexMergeDistance(vertexMergeDist);

        FloatP_t surfaceDemoteArea;
        TF_MESH_MESHQUALITYIOFROMEASY(feItr, fileElement.children, metaData, "surfaceDemoteArea", &surfaceDemoteArea);
        dataElement->setSurfaceDemoteArea(surfaceDemoteArea);

        FloatP_t bodyDemoteVolume;
        TF_MESH_MESHQUALITYIOFROMEASY(feItr, fileElement.children, metaData, "bodyDemoteVolume", &bodyDemoteVolume);
        dataElement->setBodyDemoteVolume(bodyDemoteVolume);

        FloatP_t edgeSplitDist;
        TF_MESH_MESHQUALITYIOFROMEASY(feItr, fileElement.children, metaData, "edgeSplitDist", &edgeSplitDist);
        dataElement->setEdgeSplitDist(edgeSplitDist);

        bool collision2D;
        TF_MESH_MESHQUALITYIOFROMEASY(feItr, fileElement.children, metaData, "collision2D", &collision2D);
        dataElement->setCollision2D(collision2D);

        if(FIO::hasImport() && TissueForge::models::vertex::io::VertexSolverFIOModule::hasImport()) {
            std::unordered_set<unsigned int> excludedVertices;
            TF_MESH_MESHQUALITYIOFROMEASY(feItr, fileElement.children, metaData, "excludedVertices", &excludedVertices);
            for(auto &oldId : excludedVertices) {
                auto id_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->vertexIdMap.find(oldId);
                if(id_itr != TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->vertexIdMap.end()) 
                    dataElement->excludeVertex(id_itr->second);
            }

            std::unordered_set<unsigned int> excludedSurfaces;
            TF_MESH_MESHQUALITYIOFROMEASY(feItr, fileElement.children, metaData, "excludedSurfaces", &excludedSurfaces);
            for(auto &oldId : excludedSurfaces) {
                auto id_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceIdMap.find(oldId);
                if(id_itr != TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceIdMap.end()) 
                    dataElement->excludeSurface(id_itr->second);
            }

            std::unordered_set<unsigned int> excludedBodies;
            TF_MESH_MESHQUALITYIOFROMEASY(feItr, fileElement.children, metaData, "excludedBodies", &excludedBodies);
            for(auto &oldId : excludedBodies) {
                auto id_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->bodyIdMap.find(oldId);
                if(id_itr != TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->bodyIdMap.end()) 
                    dataElement->excludeBody(id_itr->second);
            }
        }

        return S_OK;
    }
}

std::string TissueForge::models::vertex::MeshQuality::toString() {
    return TissueForge::io::toString(*this);
}

MeshQuality MeshQuality::fromString(const std::string &s) {
    return TissueForge::io::fromString<MeshQuality>(s);
}
