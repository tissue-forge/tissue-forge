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

#include <tfError.h>
#include <tfUniverse.h>
#include <tf_metrics.h>
#include <tfTaskScheduler.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>


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
        MeshQualityOperation *t_op = ops[t->objId];
        if(t_op && op != t_op) 
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
        Vertex *v = (Vertex*)source;

        MeshSolver::engineLock();
        
        HRESULT res = v->merge((Vertex*)targets[0]);
        
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

        MeshSolver::engineLock();
        
        HRESULT res = Vertex::insert((v1->getPosition() + v2->getPosition()) * 0.5, v1, v2) != NULL ? S_OK : E_FAIL;
        
        MeshSolver::engineUnlock();

        next.clear();
        
        return res;
    }
};

/** Converts a body to a vertex */
struct BodyDemoteOperation : MeshQualityOperation {

    BodyDemoteOperation(Mesh *_mesh, Body *_source) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        source = _source;
        targets = vectorToBase(_source->neighborBodies());
    }

    HRESULT implement() override {
        Body *toReplace = (Body*)source;

        MeshSolver::engineLock();

        HRESULT res = Vertex::replace(toReplace->getCentroid(), toReplace) != NULL ? S_OK : E_FAIL;

        MeshSolver::engineUnlock();

        if(res == S_OK) 
            next.clear();

        return res;
    };
};

/** Converts a surface to a vertex */
struct SurfaceDemoteOperation : MeshQualityOperation {

    SurfaceDemoteOperation(Mesh *_mesh, Surface *_source) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        source = _source;
    }

    HRESULT implement() override {
        Surface *toReplace = (Surface*)source;
        
        MeshSolver::engineLock();
        
        HRESULT res = Vertex::replace(toReplace->getCentroid(), toReplace) != NULL ? S_OK : E_FAIL;

        MeshSolver::engineUnlock();

        next.clear();
        
        return res;
    }
};

/** Converts a surface edge to a vertex */
struct EdgeDemoteOperation : MeshQualityOperation {

    Vertex *v2;

    EdgeDemoteOperation(Mesh *_mesh, Vertex *_v1, Vertex *_v2) : MeshQualityOperation(_mesh) {
        flags = MeshQualityOperation::Flag::Active;
        source = _v1;
        v2 = _v2;

        std::unordered_set<MeshObj*> targets_set;
        for(auto &c : _v1->children()) 
            targets_set.insert(c);
        for(auto &c : _v2->children()) 
            targets_set.insert(c);
        targets = std::vector<MeshObj*>(targets_set.begin(), targets_set.end());
    }

    HRESULT implement() override {
        Vertex *v = (Vertex*)source;

        MeshSolver::engineLock();
        HRESULT res = v->merge(v2);
        MeshSolver::engineUnlock();
        if(res == S_OK) 
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

        FVector3 pos0 = v0->getPosition();
        Vertex::insert((pos0 + v1->getPosition()) * 0.5, v0, v1);
        Vertex::insert((pos0 + v2->getPosition()) * 0.5, v0, v2);
        mesh->remove(v0);

        MeshSolver::engineUnlock();

        next.clear();

        return S_OK;
    }

};


static bool MeshQuality_vertexSplitTest(
    Vertex *v, 
    Mesh *m, 
    const FloatP_t &edgeSplitDist, 
    FVector3 &sep, 
    std::vector<Vertex*> &vert_nbs, 
    std::vector<Vertex*> &new_vert_nbs
) {
    Particle *p = v->particle()->part();

    // Calculate current relative force
    std::vector<Vertex*> v_nbs = v->neighborVertices();
    FVector3 force_rel(0);
    for(auto &vn : v_nbs) {
        force_rel += vn->particle()->getForce();
    }
    force_rel -= p->force * v_nbs.size();
    if(force_rel.isZero()) 
        return false;

    sep = force_rel.normalized() * edgeSplitDist;
    
    FPTYPE mask[] = {
        (p->flags & PARTICLE_FROZEN_X) ? 0.0f : 1.0f,
        (p->flags & PARTICLE_FROZEN_Y) ? 0.0f : 1.0f,
        (p->flags & PARTICLE_FROZEN_Z) ? 0.0f : 1.0f
    };
    for(int k = 0; k < 3; k++) sep[k] *= mask[k];

    // Get split plan along direction of force
    v->splitPlan(sep, vert_nbs, new_vert_nbs);

    // Enforce that a split must create a new surface
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

    FVector3 sep;
    std::vector<Vertex*> vert_nbs, new_vert_nbs;

    VertexSplitOperation(Mesh *_mesh, Vertex *_source, const FVector3 &_sep, std::vector<Vertex*> _vert_nbs, std::vector<Vertex*> _new_vert_nbs) : 
        MeshQualityOperation(_mesh), 
        sep{_sep}, 
        vert_nbs{_vert_nbs}, 
        new_vert_nbs{_new_vert_nbs}
    {
        flags = MeshQualityOperation::Flag::Active;
        source = _source;
        targets = vectorToBase(_vert_nbs);
        for(auto &mo : vectorToBase(_new_vert_nbs)) 
            targets.push_back(mo);
    }

    HRESULT implement() override { 

        Vertex *v = (Vertex*)source;

        // Create a candidate vertex
        MeshSolver::engineLock();
        Vertex *new_v = v->splitExecute(sep, vert_nbs, new_vert_nbs);
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
    const FloatP_t &edgeSplitDist
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeVertices(), 0);

    auto check_verts = [&mesh, &ops, edgeSplitDist](int i) -> void {
        Vertex *v = mesh->getVertex(i);
        if(!v) return;

        MeshQualityOperation *op = NULL;

        // Check for vertex split if the vertex defines four or more surfaces

        if(v->getSurfaces().size() >= 4) {
            FVector3 sep;
            std::vector<Vertex*> vert_nbs, new_vert_nbs;
            if(MeshQuality_vertexSplitTest(v, mesh, edgeSplitDist, sep, vert_nbs, new_vert_nbs)) {
                ops[i] = new VertexSplitOperation(mesh, v, sep, vert_nbs, new_vert_nbs);
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
    const FloatP_t &surfaceDemoteArea, 
    const FloatP_t &vertexMergeDist
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeSurfaces(), 0);
    const FloatP_t vertexMergeDist2 = vertexMergeDist * vertexMergeDist;

    auto check_surfs = [&mesh, &ops, surfaceDemoteArea, vertexMergeDist2](int i) -> void {
        Surface *s = mesh->getSurface(i);
        if(!s) return;

        // Check for surface demote

        FloatP_t sArea = s->getArea();

        if(sArea < surfaceDemoteArea) {
            TF_Log(LOG_TRACE) << sArea;

            ops[i] = new SurfaceDemoteOperation(mesh, s);
            return;
        }

        // Check vertex merge distance for vertices where this surface is the first child
        auto vertices = s->getVertices();
        for(auto itr = vertices.begin(); itr != vertices.end(); itr++) {
            Vertex *v = *itr;
            FVector3 vpos = v->getPosition();

            Vertex *nv = itr + 1 == vertices.end() ? vertices.front() : *(itr + 1);

            if(v->objId < nv->objId) {

                FVector3 nvpos = nv->getPosition();
                FVector3 nvrelPos = metrics::relativePosition(vpos, nvpos);
                FloatP_t nvdist2 = nvrelPos.dot();
                if(nvdist2 < vertexMergeDist2) {
                    TF_Log(LOG_TRACE) << nvrelPos;
                    
                    ops[i] = new EdgeDemoteOperation(mesh, v, nv);
                    return;
                }

            }

            Vertex *pv = itr == vertices.begin() ? vertices.back() : *(itr - 1);

            if(v->objId < pv->objId) {

                FVector3 pvpos = pv->getPosition();
                FVector3 pvrelPos = metrics::relativePosition(vpos, pvpos);
                FloatP_t pvdist2 = pvrelPos.dot();
                if(pvdist2 < vertexMergeDist2) {
                    TF_Log(LOG_TRACE) << pvrelPos;
                    
                    ops[i] = new EdgeDemoteOperation(mesh, v, pv);
                    return;
                }

            }
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
    Mesh *mesh, 
    const FloatP_t &bodyDemoteVolume
) {
    std::vector<MeshQualityOperation*> ops(mesh->sizeBodies(), 0);

    auto check_bodys = [&mesh, &ops, bodyDemoteVolume](int i) -> void {
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
    const FloatP_t &bodyDemoteVolumeCf, 
    const FloatP_t &_edgeSplitDistCf
) : 
    mesh{_mesh}, 
    _working{false}
{
    FloatP_t uvolu = Universe::dim().product();
    FloatP_t uleng = std::cbrt(uvolu);
    FloatP_t uarea = uleng * uleng;

    vertexMergeDist = uleng * vertexMergeDistCf;
    surfaceDemoteArea = uarea * surfaceDemoteAreaCf;
    bodyDemoteVolume = uvolu * bodyDemoteVolumeCf;
    edgeSplitDist = _edgeSplitDistCf * vertexMergeDist;
}

HRESULT MeshQuality::doQuality() { 

    _working = true;

    Mesh *_mesh = mesh;

    // Vertex checks
    
    std::vector<MeshQualityOperation*> op_verts = MeshQuality_constructOperationsVertex(mesh, edgeSplitDist);
    if(MeshQuality_doOperations(op_verts) != S_OK || MeshQuality_clearOperations(op_verts) != S_OK) {
        _working = false;
        return E_FAIL;
    }

    // Surface checks
    
    std::vector<MeshQualityOperation*> op_surfs = MeshQuality_constructOperationsSurface(mesh, surfaceDemoteArea, vertexMergeDist);
    if(MeshQuality_doOperations(op_surfs) != S_OK || MeshQuality_clearOperations(op_surfs) != S_OK) {
        _working = false;
        return E_FAIL;
    }

    // Body checks

    std::vector<MeshQualityOperation*> op_bodys = MeshQuality_constructOperationsBody(mesh, bodyDemoteVolume);
    if(MeshQuality_doOperations(op_bodys) != S_OK || MeshQuality_clearOperations(op_bodys) != S_OK) {
        _working = false;
        return E_FAIL;
    }

    _working = false;
    
    return S_OK;
}

const int MeshQuality::getMeshId() const {
    return mesh ? mesh->getId() : -1;
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
        TF_MESH_MESHQUALITYIOTOEASY(fe, "meshId", dataElement.getMeshId());

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

        return S_OK;
    }
}

std::string TissueForge::models::vertex::MeshQuality::toString() {
    return TissueForge::io::toString(*this);
}
