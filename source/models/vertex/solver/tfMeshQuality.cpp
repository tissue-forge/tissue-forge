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

#include <tfUniverse.h>
#include <tf_metrics.h>
#include <tfTaskScheduler.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


//////////////////////////
// MeshQualityOperation //
//////////////////////////


static HRESULT MeshQualityOperation_checkChain( 
    MeshQualityOperation *op, 
    std::vector<MeshQualityOperation*> &ops
) {
    for(auto &t : op->targets) {
        auto t_id = t->objId;
        MeshQualityOperation *t_op = ops[t->objId];
        if(t_op) 
            op->appendNext(t_op);
        
    }

    return S_OK;
}

MeshQualityOperation::MeshQualityOperation(Mesh *_mesh) : 
    flags{Flag::None}, 
    source{NULL}, 
    mesh{_mesh}
{}

HRESULT MeshQualityOperation::appendNext(MeshQualityOperation *_next) {
    // If this op has no upstreams, make sure this doesn't create a cycle
    if(prev.empty()) {
        std::set<MeshQualityOperation*> next_heads = _next->headOperations();
        if(std::find(next_heads.begin(), next_heads.end(), this) != next_heads.end()) 
            return S_OK;
    }

    _next->lock.lock();
    _next->prev.push_back(this);
    _next->lock.unlock();

    next.push_back(_next);
    
    return E_NOTIMPL;
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
        ops.insert(op_u);
        MeshQuality_upstreams(op_u, ops);
    }
}

std::set<MeshQualityOperation*> MeshQualityOperation::upstreams() const {
    std::set<MeshQualityOperation*> result;
    MeshQuality_upstreams(this, result);
    return result;
}

static void MeshQuality_downstreams(const MeshQualityOperation *op, std::set<MeshQualityOperation*> &ops) {
    for(auto *op_d : op->next) {
        ops.insert(op_d);
        MeshQuality_downstreams(op_d, ops);
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
    return MeshQualityOperation_headOperations(std::vector<MeshQualityOperation*>{const_cast<MeshQualityOperation*>(this)});
}


////////////////
// Operations //
////////////////


/** Removes an object */
template <typename T> 
struct MeshObjectRemoveOperation : MeshQualityOperation {

    MeshObjectRemoveOperation(Mesh *_mesh, T *_source) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Active;
        source = _source;
    }

    HRESULT implement() override { 
        MeshSolver::engineLock();
        
        HRESULT res = mesh->remove((T*)source); 
        
        MeshSolver::engineUnlock();

        next.clear();
        
        return res;
    }
};

/** Removes a vertex */
using VertexRemoveOperation = MeshObjectRemoveOperation<Vertex>;

/** Removes a surface */
using SurfaceRemoveOperation = MeshObjectRemoveOperation<Surface>;

/** Removes a body */
using BodyRemoveOperation = MeshObjectRemoveOperation<Body>;

/** Removes a structure */
using StructureRemoveOperation = MeshObjectRemoveOperation<Structure>;


/** Merges two vertices */
struct VertexMergeOperation : MeshQualityOperation {

    VertexMergeOperation(Mesh *_mesh, Vertex *_source, Vertex *_target) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        source = _source;
        targets = {_target};
    };

    HRESULT implement() override { 
        MeshSolver::engineLock();
        
        HRESULT res = mesh->merge((Vertex*)source, (Vertex*)targets[0]);
        
        MeshSolver::engineUnlock();

        next.clear();
        
        return res;
    }
};


/** Inserts a vertex between two vertices */
struct VertexInsertOperation : MeshQualityOperation {

    VertexInsertOperation(Mesh *_mesh, Vertex *_source, Vertex *_target) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        source = _source;
        targets = {_target};
    };

    HRESULT implement() override { 
        Vertex *v1 = (Vertex*)source;
        Vertex *v2 = (Vertex*)targets[0];
        Vertex *toInsert = new Vertex();
        toInsert->setPosition((v1->getPosition() + v2->getPosition()) * 0.5);

        MeshSolver::engineLock();
        
        HRESULT res = mesh->insert(toInsert, v1, v2);
        
        MeshSolver::engineUnlock();

        next.clear();
        
        return res;
    }
};

/** Converts a surface to a vertex */
struct SurfaceDemoteOperation : MeshQualityOperation {

    SurfaceDemoteOperation(Mesh *_mesh, Surface *_source) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        source = _source;
    }

    HRESULT implement() override {
        Surface *toReplace = (Surface*)source;
        Vertex *toInsert = new Vertex();
        toInsert->setPosition(toReplace->getCentroid());
        
        MeshSolver::engineLock();
        
        HRESULT res = mesh->replace(toInsert, toReplace);

        MeshSolver::engineUnlock();

        next.clear();
        
        return res;
    }
};

/** Converts a vertex to an edge */
struct EdgeInsertOperation : MeshQualityOperation {

    EdgeInsertOperation(Mesh *_mesh, Vertex *_source, Vertex *_target1, Vertex *_target2) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        source = _source;
        targets = {_target1, _target2};
    }

    HRESULT implement() override {
        Vertex *v0 = (Vertex*)source;
        Vertex *v1 = (Vertex*)targets[0];
        Vertex *v2 = (Vertex*)targets[1];

        MeshSolver::engineLock();

        Vertex *v01 = new Vertex();
        Vertex *v02 = new Vertex();

        FVector3 pos0 = v0->getPosition();
        v01->setPosition((pos0 + v1->getPosition()) * 0.5);
        v02->setPosition((pos0 + v2->getPosition()) * 0.5);
        mesh->insert(v01, v0, v1);
        mesh->insert(v02, v0, v2);
        mesh->remove(v0);

        MeshSolver::engineUnlock();

        next.clear();

        return S_OK;
    }

};

/** Splits a vertex into an edge */
struct VertexSplitOperation : MeshQualityOperation { 

    FVector3 sep;

    VertexSplitOperation(Mesh *_mesh, Vertex *_source, const FVector3 &_sep) : 
        MeshQualityOperation(_mesh), 
        sep{_sep}
    {
        flags = MeshQualityOperation::Flag::Active;
        source = _source;
        targets = vectorToBase(_source->neighborVertices());
    }

    HRESULT implement() override { 
        MeshSolver::engineLock();

        Vertex *new_v = mesh->split((Vertex*)source, sep);

        MeshSolver::engineUnlock();

        // Only invalidate if a vertex was created, since some requested configurations are invalid and subsequently ignored
        if(new_v) 
            next.clear();

        return S_OK;
    }
};


/////////////////
// MeshQuality //
/////////////////


static HRESULT MeshQuality_constructChains(std::vector<MeshQualityOperation*> &ops) {
    auto func = [&ops](int i) -> void {
        MeshQualityOperation *op = ops[i];
        if(!op) return;
        MeshQualityOperation_checkChain(op, ops);
    };
    parallel_for(ops.size(), func);
    return S_OK;
}

static std::vector<MeshQualityOperation*> MeshQuality_constructOperationsVertex(
    Mesh *mesh, 
    const FloatP_t &vertexMergeDist, 
    const FloatP_t &edgeSplitStrain, 
    const FloatP_t &edgeSplitDist
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeVertices(), 0);
    auto vertexMergeDist2 = vertexMergeDist * vertexMergeDist;

    auto check_verts = [&mesh, &ops, vertexMergeDist2, edgeSplitStrain, edgeSplitDist](int i) -> void {
        Vertex *v = mesh->getVertex(i);
        if(!v) return;

        MeshQualityOperation *op = NULL;

        FVector3 vpos = v->getPosition();

        // Check for vertex merge

        for(auto &nv : v->neighborVertices()) { 
            if(nv->objId < v->objId) 
                continue;

            FVector3 relPos = metrics::relativePosition(vpos, nv->getPosition());
            if(relPos.dot(relPos) < vertexMergeDist2) {
                TF_Log(LOG_TRACE) << relPos;
                
                ops[i] = new VertexMergeOperation(mesh, v, nv);
                return;
            }
        }

        // Check for vertex split if the vertex defines four or more surfaces

        if(v->getSurfaces().size() >= 4) {

            FVector3 evals;
            FMatrix3 evecs;
            std::tie(evals, evecs) = metrics::eigenVecsVals(vertexStrain(v));

            size_t ei;
            FloatP_t ev = 0.0;
            for(size_t i = 0; i < 3; i++) {
                FloatP_t evi = evals[i];
                if(evi > ev) {
                    ei = i;
                    ev = evi;
                }
            }
            if(ev > edgeSplitStrain) {
                TF_Log(LOG_TRACE) << evals;
                TF_Log(LOG_TRACE) << evecs;
                TF_Log(LOG_TRACE) << ei;

                ops[i] = new VertexSplitOperation(mesh, v, evecs[ei] * edgeSplitDist);
                return;
            }

        }

    };
    parallel_for(mesh->sizeVertices(), check_verts);

    if(MeshQuality_constructChains(ops) != S_OK) { 
        for(size_t i = 0; i < ops.size(); i++) delete ops[i];
        ops.clear();
    }

    return ops;
}

static std::vector<MeshQualityOperation*> MeshQuality_constructOperationsSurface(
    Mesh *mesh, 
    const FloatP_t &surfaceDemoteArea
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeSurfaces(), 0);

    auto check_surfs = [&mesh, &ops, surfaceDemoteArea](int i) -> void {
        Surface *s = mesh->getSurface(i);
        if(!s) return;

        // Check for surface demote

        FloatP_t sArea = s->getArea();

        if(sArea < surfaceDemoteArea) {
            TF_Log(LOG_TRACE) << sArea;

            ops[i] = new SurfaceDemoteOperation(mesh, s);
            return;
        }

    };
    parallel_for(mesh->sizeSurfaces(), check_surfs);
    
    if(MeshQuality_constructChains(ops) != S_OK) { 
        for(size_t i = 0; i < ops.size(); i++) delete ops[i];
        ops.clear();
    }
    
    return ops;
}

static std::vector<MeshQualityOperation*> MeshQuality_constructOperationsBody(
    Mesh *mesh
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeBodies(), 0);

    auto check_bodys = [&mesh, &ops](int i) -> void {
        Body *b = mesh->getBody(i);
        if(!b) return;
    };
    parallel_for(mesh->sizeBodies(), check_bodys);
    
    if(MeshQuality_constructChains(ops) != S_OK) { 
        for(size_t i = 0; i < ops.size(); i++) delete ops[i];
        ops.clear();
    }
    
    return ops;
}

static HRESULT MeshQuality_doOperations(MeshQualityOperation *op) {
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

    op->implement();

    for(auto &n : op->next) {
        auto itr = std::find(n->prev.begin(), n->prev.end(), op);
        HRESULT res = S_OK;

        n->lock.lock();
        n->prev.erase(itr);
        if(n->prev.empty()) 
            res = MeshQuality_doOperations(n);
        n->lock.unlock();

        if(res != S_OK) 
            return res;
    }

    return S_OK;
}

static HRESULT MeshQuality_doOperations(const std::vector<MeshQualityOperation*> &ops) {
    std::vector<MeshQualityOperation*> ops_active;
    ops_active.reserve(ops.size());
    for(auto &op : ops) 
        if(op) 
            ops_active.push_back(op);
    std::set<MeshQualityOperation*> op_heads = MeshQualityOperation_headOperations(ops_active);
    std::vector<MeshQualityOperation*> op_heads_v(op_heads.begin(), op_heads.end());

    auto func = [&op_heads_v](int i) -> void {
        MeshQuality_doOperations(op_heads_v[i]);
    };
    parallel_for(op_heads_v.size(), func);

    return S_OK;
}

static HRESULT MeshQuality_clearOperations(std::vector<MeshQualityOperation*> &ops) {
    auto func = [&ops](int i) -> void {
        MeshQualityOperation *op = ops[i];
        if(op) {
            delete ops[i];
            ops[i] = NULL;
        }
    };
    parallel_for(ops.size(), func);

    return S_OK;
}

MeshQuality::MeshQuality(
    Mesh *_mesh, 
    const FloatP_t &vertexMergeDistCf, 
    const FloatP_t &surfaceDemoteAreaCf, 
    const FloatP_t &_edgeSplitStrain, 
    const FloatP_t &_edgeSplitDistCf
) : 
    mesh{_mesh}, 
    edgeSplitStrain{_edgeSplitStrain}
{
    FloatP_t uvolu = Universe::dim().product();
    FloatP_t uleng = std::cbrt(uvolu);
    FloatP_t uarea = uleng * uleng;

    vertexMergeDist = uleng * vertexMergeDistCf;
    surfaceDemoteArea = uarea * surfaceDemoteAreaCf;
    edgeSplitDist = _edgeSplitDistCf * vertexMergeDist;
}

HRESULT MeshQuality::doQuality() { 

    Mesh *_mesh = mesh;

    // Vertex checks
    
    std::vector<MeshQualityOperation*> op_verts = MeshQuality_constructOperationsVertex(mesh, vertexMergeDist, edgeSplitStrain, edgeSplitDist);
    if(MeshQuality_doOperations(op_verts) != S_OK || MeshQuality_clearOperations(op_verts) != S_OK) 
        return E_FAIL;

    // Surface checks
    
    std::vector<MeshQualityOperation*> op_surfs = MeshQuality_constructOperationsSurface(mesh, surfaceDemoteArea);
    if(MeshQuality_doOperations(op_surfs) != S_OK || MeshQuality_clearOperations(op_surfs) != S_OK) 
        return E_FAIL;

    // Body checks

    std::vector<MeshQualityOperation*> op_bodys = MeshQuality_constructOperationsBody(mesh);
    if(MeshQuality_doOperations(op_bodys) != S_OK || MeshQuality_clearOperations(op_bodys) != S_OK) 
        return E_FAIL;

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

HRESULT MeshQuality::setEdgeSplitStrain(const FloatP_t &_val) {
    if(_val < 0) 
        return E_FAIL;
    edgeSplitStrain = _val;
    return S_OK;
}

HRESULT MeshQuality::setEdgeSplitDist(const FloatP_t &_val) {
    if(_val <= 1.0) 
        return E_FAIL;
    edgeSplitDist = _val;
    return S_OK;
}
