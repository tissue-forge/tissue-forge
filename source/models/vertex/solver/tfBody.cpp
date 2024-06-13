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

#include "tfBody.h"

#include "tfVertex.h"
#include "tfSurface.h"
#include "tfMeshSolver.h"
#include "tf_mesh_io.h"
#include "tfVertexSolverFIO.h"

#include <tfError.h>
#include <tfLogger.h>
#include <tfTaskScheduler.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>

#include <Magnum/Math/Math.h>
#include <Magnum/Math/Intersection.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


#define Body_GETMESH(name, retval)                  \
    Mesh *name = Mesh::get();                       \
    if(!name) {                                     \
        TF_Log(LOG_ERROR) << "Could not get mesh";  \
        return retval;                              \
    }

#define BodyHandle_INVALIDHANDLERR { tf_error(E_FAIL, "Invalid handle"); }

#define BodyHandle_GETOBJ(name, retval)                             \
    Body *name;                                                     \
    if(this->id < 0 || !(name = Mesh::get()->getBody(this->id))) {  \
        BodyHandle_INVALIDHANDLERR; return retval; }


//////////
// Body //
//////////


static HRESULT Body_loadSurfaces(Body* body, std::vector<Surface*> _surfaces) { 
    Body_GETMESH(mesh, E_FAIL);

    if(_surfaces.size() >= 4) {
        for(auto &s : _surfaces) {
            body->add(s);
            s->add(body);
        }

        return S_OK;
    } 
    else {
        TF_Log(LOG_ERROR) << "A body requires at least 4 surfaces";
        return E_FAIL;
    }
};

static HRESULT Body_loadSurfaces(Body* body, const std::vector<SurfaceHandle> &_surfaces) { 
    std::vector<Surface*> surfaces;
    surfaces.reserve(_surfaces.size());
    for(auto &_s : _surfaces) {
        Surface *s = _s.surface();
        if(!s) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }
        surfaces.push_back(s);
    }
    return Body_loadSurfaces(body, surfaces);
};


Body::Body() : 
    centroid{0.f}, 
    area{0.f}, 
    volume{0.f}, 
    density{0.f}, 
    typeId{-1},
    species{NULL}
{
    MESHOBJ_INITOBJ
}

Body::~Body() {
    MESHOBJ_DELOBJ
}

static Body *Body_create(const std::vector<SurfaceHandle> &_surfaces) {
    Body_GETMESH(mesh, NULL);
    Body *result;
    if(mesh->create(&result) != S_OK || Body_loadSurfaces(result, _surfaces) != S_OK) {
        TF_Log(LOG_ERROR);
        return NULL;
    }
    return result;
};

BodyHandle Body::create(const std::vector<SurfaceHandle> &_surfaces) {
    Body *b = Body_create(_surfaces);
    if(!b) {
        TF_Log(LOG_ERROR);
        return BodyHandle();
    }
    return BodyHandle(b->objectId());
};

BodyHandle Body::create(TissueForge::io::ThreeDFMeshData *ioMesh) {
    Body_GETMESH(mesh, BodyHandle());
    if(mesh->ensureAvailableSurfaces(ioMesh->faces.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return BodyHandle();
    }

    std::vector<SurfaceHandle> _surfaces;
    bool failed = false;
    for(auto f : ioMesh->faces) {
        SurfaceHandle s = Surface::create(f);
        if(!s) {
            TF_Log(LOG_ERROR) << "Failed to create surface";
            failed = true;
            break;
        }
        _surfaces.push_back(s);
    }

    if(failed) {
        TF_Log(LOG_ERROR);
        for(auto &s : _surfaces) {
            s.destroy();
        }
        return BodyHandle();
    }

    return create(_surfaces);
}

bool Body::definedBy(const Surface *obj) const { MESHOBJ_DEFINEDBY_DEF }

bool Body::definedBy(const Vertex *obj) const { MESHOBJ_DEFINEDBY_DEF }

std::string Body::str() const {
    std::stringstream ss;

    ss << "Body(";
    if(this->objectId() >= 0) {
        ss << "id=" << this->objectId() << ", typeId=" << this->typeId;
    }
    ss << ")";

    return ss.str();
}

void Body::updateInternals() {
    for(auto &v : getVertices()) 
        v->positionChanged();
    for(auto &s : getSurfaces()) 
        s->positionChanged();

    centroid = FVector3(0.f);
    area = 0.f;
    volume = 0.f;

    for(auto &s : surfaces) {
        centroid += s->getCentroid() * s->getArea();
        area += s->getArea();
    }
    centroid /= area;

    for(auto &s : surfaces) {
        s->refreshBodies();
        volume += s->getVolumeContr(this);
    }

}

#define BODY_RND_IDX(vec_size, idx) {       \
while(idx < 0) idx += vec_size;             \
while(idx >= vec_size) idx -= vec_size;     \
}

HRESULT Body::add(Surface *s) {
    if(std::find(surfaces.begin(), surfaces.end(), s) != surfaces.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    surfaces.push_back(s);
    return S_OK;
}

HRESULT Body::remove(Surface *s) {
    auto itr = std::find(surfaces.begin(), surfaces.end(), s);
    if(itr == surfaces.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }
    
    surfaces.erase(itr);
    return S_OK;
}

HRESULT Body::replace(Surface *toInsert, Surface *toRemove) {
    std::replace(this->surfaces.begin(), this->surfaces.end(), toRemove, toInsert);
    return toInsert->defines(this) ? S_OK : E_FAIL;
}

HRESULT Body::destroy() {
    if(this->typeId >= 0 && this->type()->remove(BodyHandle(this->_objId)) != S_OK) 
        return E_FAIL;
    if(this->_objId >= 0) 
        Mesh::get()->remove(this);
    return S_OK;
}

HRESULT Body::destroy(Body *target) {
    auto& surfaces = target->getSurfaces();
    if(target->destroy() != S_OK) 
        return E_FAIL;

    for(auto &s : surfaces) 
        if(s->getBodies().size() == 0) 
            Surface::destroy(s);

    return S_OK;
}

HRESULT Body::destroy(BodyHandle &target) {
    Body *_target = target.body();
    if(!_target) {
        BodyHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    HRESULT res = Body::destroy(_target);
    if(res == S_OK) 
        target.id = -1;
    return res;
}

HRESULT Body::destroy(const std::vector<Body*>& toRemove) {
    // Filter valid targets

    std::vector<std::unordered_set<Body*> > bodiesToRemovePool(ThreadPool::size());
    std::vector<std::unordered_set<BodyType*> > bodyTypesPool(ThreadPool::size());
    std::vector<std::unordered_map<BodyType*, std::unordered_set<Body*> > > bodiesByTypePool(ThreadPool::size());
    std::vector<std::unordered_set<Surface*> > surfacesToRemovePool(ThreadPool::size());
    parallel_for(
        ThreadPool::size(), 
        [&toRemove, &bodiesToRemovePool, &bodyTypesPool, &bodiesByTypePool, &surfacesToRemovePool](int tid) -> void {
            std::unordered_set<Body*>& bodiesToRemoveThread = bodiesToRemovePool[tid];
            std::unordered_set<BodyType*>& bodyTypesThread = bodyTypesPool[tid];
            std::unordered_map<BodyType*, std::unordered_set<Body*> >& bodiesByTypeThread = bodiesByTypePool[tid];
            std::unordered_set<Surface*>& surfacesToRemoveThread = surfacesToRemovePool[tid];
            for(int i = tid; i < toRemove.size(); i += ThreadPool::size()) {
                Body* b = toRemove[i];
                if(b->_objId >= 0) {
                    bodiesToRemoveThread.insert(b);
                    if(b->typeId >= 0) {
                        BodyType* btype = b->type();
                        bodyTypesThread.insert(btype);
                        bodiesByTypeThread[btype].insert(b);
                    }
                    for(auto& s : b->getSurfaces()) 
                        if(s->getBodies().size() == 1) 
                            surfacesToRemoveThread.insert(s);
                }
            }
        }
    );
    size_t numBodies = 0;
    for(auto& bodiesToRemoveThread : bodiesToRemovePool) 
        numBodies += bodiesToRemoveThread.size();
    std::unordered_set<Body*> bodiesToRemove;
    bodiesToRemove.reserve(numBodies);
    for(auto& bodiesToRemoveThread : bodiesToRemovePool) 
        bodiesToRemove.insert(bodiesToRemoveThread.begin(), bodiesToRemoveThread.end());
    std::vector<Body*> bodiesToRemoveVec(bodiesToRemove.begin(), bodiesToRemove.end());

    // Get affected surfaces

    std::vector<std::unordered_set<Surface*> > affectedSurfacesPool(ThreadPool::size());
    parallel_for(
        ThreadPool::size(), 
        [&bodiesToRemoveVec, &affectedSurfacesPool](int tid) -> void {
            std::unordered_set<Surface*>& affectedSurfacesThread = affectedSurfacesPool[tid];
            for(int i = tid; i < bodiesToRemoveVec.size(); i += ThreadPool::size()) 
                for(auto& s : bodiesToRemoveVec[i]->getSurfaces()) 
                    affectedSurfacesThread.insert(s);
        }
    );

    // Remove from types

    size_t numBodyTypes = 0;
    for(auto& bodyTypesThread : bodyTypesPool) 
        numBodyTypes += bodyTypesThread.size();
    std::unordered_set<BodyType*> bodyTypes;
    bodyTypes.reserve(numBodyTypes);
    for(auto& bodyTypesThread : bodyTypesPool) 
        bodyTypes.insert(bodyTypesThread.begin(), bodyTypesThread.end());
    std::unordered_map<BodyType*, size_t> numBodiesByType;
    for(auto& stype : bodyTypes) {
        size_t& nBodies = numBodiesByType[stype];
        for(auto& bodiesByTypeThread : bodiesByTypePool) 
            nBodies += bodiesByTypeThread[stype].size();
    }
    std::unordered_map<BodyType*, std::unordered_set<BodyHandle> > bodiesByType;
    for(auto& btype : bodyTypes) {
        std::unordered_set<BodyHandle>& thisBodiesByType = bodiesByType[btype];
        thisBodiesByType.reserve(numBodiesByType[btype]);
        for(auto& bodiesByTypeThread : bodiesByTypePool) 
            for(auto& s : bodiesByTypeThread[btype]) 
                thisBodiesByType.emplace(s->objectId());
        btype->remove({thisBodiesByType.begin(), thisBodiesByType.end()});
    }

    // Destroy bodies

    Mesh::get()->remove(bodiesToRemoveVec.data(), bodiesToRemoveVec.size());

    // Destroy orphaned surfaces

    size_t numSurfaces = 0;
    for(auto& surfacesToRemoveThread : surfacesToRemovePool) 
        numSurfaces += surfacesToRemoveThread.size();
    std::vector<Surface*> surfacesToRemove;
    surfacesToRemove.reserve(numSurfaces);
    for(auto& surfacesToRemoveThread : surfacesToRemovePool) 
        for(auto& s : surfacesToRemoveThread) 
            surfacesToRemove.push_back(s);
    Surface::destroy(surfacesToRemove);

    return S_OK;
}

bool Body::validate() {
    if(surfaces.size() < 3) 
        return false;

    for(auto &s : surfaces) 
        if(!s->defines(this) || !this->definedBy(s)) 
            return false;

    return true;
}

HRESULT Body::positionChanged() { 
    centroid = FVector3(0.f);
    area = 0.f;
    volume = 0.f;

    for(auto &s : surfaces) {
        centroid += s->getCentroid() * s->getArea();
        area += s->getArea();
        volume += s->getVolumeContr(this);
    }
    centroid /= area;

    return S_OK;
}

BodyType *Body::type() const {
    if(typeId < 0) 
        return NULL;
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getBodyType(typeId);
}

HRESULT Body::become(BodyType *btype) {
    if(this->typeId >= 0 && this->type()->remove(BodyHandle(this->_objId)) != S_OK) {
        return tf_error(E_FAIL, "Failed to become");
    }
    return btype->add(BodyHandle(this->_objId));
}

std::vector<Vertex*> Body::getVertices() const {
    std::unordered_set<Vertex*> result;

    for(auto &s : surfaces) 
        for(auto &v : s->vertices) 
            result.insert(v);

    return std::vector<Vertex*>(result.begin(), result.end());
}

Vertex *Body::findVertex(const FVector3 &dir) const {
    Vertex *result = 0;

    FloatP_t bestCrit = 0;

    for(auto &v : getVertices()) {
        const FVector3 rel_pt = v->getPosition() - centroid;
        if(rel_pt.isZero()) 
            continue;
        FloatP_t crit = rel_pt.dot(dir) / rel_pt.dot();
        if(!result || crit > bestCrit) { 
            result = v;
            bestCrit = crit;
        }
    }

    return result;
}

Surface *Body::findSurface(const FVector3 &dir) const {
    Surface *result = 0;

    FloatP_t bestCrit = 0;

    for(auto &s : getSurfaces()) {
        const FVector3 rel_pt = s->getCentroid() - centroid;
        if(rel_pt.isZero()) 
            continue;
        FloatP_t crit = rel_pt.dot(dir) / rel_pt.dot();
        if(!result || crit > bestCrit) { 
            result = s;
            bestCrit = crit;
        }
    }

    return result;
}

std::vector<Body*> Body::connectedBodies() const {
    std::unordered_set<Body*> result;
    for(auto &s : surfaces) 
        for(auto &b : s->getBodies()) 
            result.insert(b);
    result.erase(result.find(const_cast<Body*>(this)));
    return std::vector<Body*>(result.begin(), result.end());
}

std::vector<Body*> Body::adjacentBodies() const {
    std::unordered_set<Body*> result;
    for(auto &v : getVertices()) 
        for(auto &b : v->getBodies()) 
            result.insert(b);
    result.erase(result.find(const_cast<Body*>(this)));
    return std::vector<Body*>(result.begin(), result.end());
}

std::vector<Surface*> Body::neighborSurfaces(const Surface *s) const {
    std::unordered_set<Surface*> result;
    for(auto &so : surfaces) {
        if(so->objectId() == s->objectId()) 
            continue;
        for(auto &v : s->vertices) 
            if(v->defines(so)) { 
                result.insert(so);
                break;
            }
    }
    return std::vector<Surface*>(result.begin(), result.end());
}

FVector3 Body::getVelocity() const {
    FVector3 result;
    for(auto &v : getVertices()) 
        result += v->particle()->getVelocity() * getVertexMass(v);
    return result / getMass();
}

FloatP_t Body::getVertexArea(const Vertex *v) const {
    FloatP_t result = FPTYPE_ZERO;
    for(auto &s : surfaces) 
        result += s->getVertexArea(v);
    return result;
}

FloatP_t Body::getVertexVolume(const Vertex *v) const {
    if(area == 0.f) 
        return 0.f;
    return getVertexArea(v) / area * volume;
}

std::vector<Surface*> Body::findInterface(const Body *b) const {
    std::vector<Surface*> result;
    for(auto &s : surfaces) 
        if(s->defines(b)) 
            result.push_back(s);
    return result;
}

FloatP_t Body::contactArea(const Body *other) const {
    FloatP_t result = 0.f;
    for(auto &s : surfaces) 
        if(s->defines(other)) 
            result += s->area;
    return result;
}

std::vector<Vertex*> Body::sharedVertices(const Body *other) const {
    std::vector<Vertex*> verts = getVertices();
    std::vector<Vertex*> result;
    for(auto &v : other->getVertices()) 
        if(std::find(verts.begin(), verts.end(), v) != verts.end()) 
            result.push_back(v);
    return result;
}

bool Body::isOutside(const FVector3 &pos) const {
    // Test against outward-facing normal of nearest surface
    const FVector3 rel_pos = pos - centroid;
    return rel_pos.dot(findSurface(rel_pos)->getOutwardNormal(this)) > 0;
}

struct Body_BodySplitEdge {
    VertexHandle v_oldSide;  // Old side
    VertexHandle v_newSide;  // New side
    FVector3 intersect_pt;
    std::vector<SurfaceHandle> surfaces;

    bool operator==(Body_BodySplitEdge o) const {
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

    static std::vector<SurfaceHandle> extractSurfaces(Vertex *v1, Vertex *v2, Body *b) {
        std::vector<SurfaceHandle> surfaces;
        for(auto &s : b->getSurfaces()) 
            if(v1->defines(s) && v2->defines(s)) 
                surfaces.emplace_back(s->objectId());
        return surfaces;
    }

    static std::vector<Body_BodySplitEdge> construct(Body *b, const FVector4 &planeEq) {
        std::map<std::pair<int, int>, Body_BodySplitEdge> edgeMap;
        for(auto &s : b->getSurfaces()) {
            for(auto &_v : s->getVertices()) {
                Vertex *_va, *_vb;
                VertexHandle v_lower, v_upper;
                std::tie(_va, _vb) = s->neighborVertices(_v);
                VertexHandle v(_v->objectId());
                VertexHandle va(_va->objectId());
                VertexHandle vb(_vb->objectId());

                std::vector<std::pair<VertexHandle, VertexHandle> > edge_cases;
                if(v < va) 
                    edge_cases.push_back({v, va});
                if(v < vb) 
                    edge_cases.push_back({v, vb});

                for(auto &ec : edge_cases) {
                    std::tie(v_lower, v_upper) = ec;
                    Vertex *_v_lower = v_lower.vertex();
                    Vertex *_v_upper = v_upper.vertex();

                    FVector3 intersect_pt;
                    if(intersects(_v_lower, _v_upper, planeEq, intersect_pt)) {
                        Body_BodySplitEdge edge;
                        if(planeEq.distance(_v_lower->getPosition()) > 0) {
                            edge.v_oldSide = v_upper;
                            edge.v_newSide = v_lower;
                        } 
                        else {
                            edge.v_oldSide = v_lower;
                            edge.v_newSide = v_upper;
                        }
                        edge.intersect_pt = intersect_pt;
                        edge.surfaces = extractSurfaces(_v_lower, _v_upper, b);

                        if(edge.surfaces.size() != 2) {
                            tf_error(E_FAIL, "Incorrect number of extracted surfaces");
                            return {};
                        }

                        edgeMap.insert({{v_lower.id, v_upper.id}, edge});
                    }
                }
            }
        }

        std::vector<Body_BodySplitEdge> result;
        result.reserve(edgeMap.size());
        for(auto &itr : edgeMap) 
            result.push_back(itr.second);
        
        std::vector<Body_BodySplitEdge> result_sorted;
        result_sorted.reserve(result.size());
        result_sorted.push_back(result.back());
        result.pop_back();
        SurfaceHandle s_target = result_sorted[0].surfaces[1];
        while(!result.empty()) {
            std::vector<Body_BodySplitEdge>::iterator itr = result.begin();
            while(itr != result.end()) { 
                if(itr->surfaces[0].id == s_target.id) { 
                    s_target = itr->surfaces[1];
                    result_sorted.push_back(*itr);
                    result.erase(itr);
                    break;
                } 
                else if(itr->surfaces[1].id == s_target.id) {
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

    typedef std::pair<SurfaceHandle, std::pair<Body_BodySplitEdge, Body_BodySplitEdge> > surfaceSplitPlanEl_t;

    static HRESULT surfaceSplitPlan(const std::vector<Body_BodySplitEdge> &edges, std::vector<surfaceSplitPlanEl_t> &result) {
        std::vector<Body_BodySplitEdge> edges_copy(edges);
        edges_copy.push_back(edges.front());
        for(size_t i = 0; i < edges.size(); i++) {
            Body_BodySplitEdge edge_i = edges_copy[i];
            Body_BodySplitEdge edge_j = edges_copy[i + 1];
            SurfaceHandle s;
            for(auto &si : edge_i.surfaces) {
                auto itr = std::find(edge_j.surfaces.begin(), edge_j.surfaces.end(), si);
                if(itr != edge_j.surfaces.end()) {
                    s = *itr;
                    break;
                }
            }
            if(!s) 
                return E_FAIL;
            
            VertexHandle v_old = edge_i.v_oldSide;
            VertexHandle v_new = edge_i.v_newSide;
            VertexHandle v_old_na = std::get<0>(s.neighborVertices(v_old));
            if(v_old_na.id == v_new.id) 
                result.push_back({SurfaceHandle(s.id), {edge_i, edge_j}});
            else 
                result.push_back({SurfaceHandle(s.id), {edge_j, edge_i}});
        }
        return S_OK;
    }

    static HRESULT vertexConstructorPlan(const std::vector<surfaceSplitPlanEl_t> &splitPlan, std::vector<Body_BodySplitEdge> &vertexPlan) {
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

Body *Body::split(const FVector3 &cp_pos, const FVector3 &cp_norm, SurfaceType *stype) {
    Body_GETMESH(mesh, NULL);

    if(cp_norm.isZero()) {
        tf_error(E_FAIL, "Zero normal");
        return 0;
    }

    FVector4 planeEq = FVector4::planeEquation(cp_norm.normalized(), cp_pos);

    // Determine which surfaces are moved to new body
    std::vector<SurfaceHandle> surfs_moved;
    for(auto &s : surfaces) {
        size_t num_newSide = 0;
        for(auto &v : s->vertices) 
            if(planeEq.distance(v->getPosition()) > 0) 
                num_newSide++;
        if(num_newSide == s->vertices.size()) 
            surfs_moved.emplace_back(s->objectId());
    }

    // Build edge list
    std::vector<Body_BodySplitEdge> splitEdges = Body_BodySplitEdge::construct(this, planeEq);

    // Split edges
    std::vector<Body_BodySplitEdge::surfaceSplitPlanEl_t> sSplitPlan;
    if(Body_BodySplitEdge::surfaceSplitPlan(splitEdges, sSplitPlan) != S_OK) 
        return NULL;
    std::vector<Body_BodySplitEdge> vertexPlan;
    if(Body_BodySplitEdge::vertexConstructorPlan(sSplitPlan, vertexPlan) != S_OK) 
        return NULL;
    if(mesh->ensureAvailableVertices(vertexPlan.size()) != S_OK || mesh->ensureAvailableSurfaces(sSplitPlan.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return NULL;
    }
    std::vector<VertexHandle> new_vertices;
    std::map<std::pair<int, int>, VertexHandle> new_vertices_map;
    bool failed = false;
    for(auto &edge : vertexPlan) {
        VertexHandle v_new = Vertex::create(edge.intersect_pt);
        if(!v_new) {
            TF_Log(LOG_ERROR) << "Failed to create a vertex";
            failed = true;
            break;
        }
        new_vertices.push_back(v_new);
    }
    if(failed) {
        for(auto &v : new_vertices) {
            v.destroy();
        }
        return NULL;
    }
    for(size_t i = 0; i < vertexPlan.size(); i++) {
        auto edge = vertexPlan[i];
        auto v_new = new_vertices[i];
        v_new.insert(edge.v_oldSide, edge.v_newSide);
        new_vertices_map.insert({{edge.v_oldSide.id, edge.v_newSide.id}, v_new});
    }

    // Split surfaces
    std::vector<SurfaceHandle> new_surfs;
    new_surfs.reserve(sSplitPlan.size());
    for(size_t i = 0; i < sSplitPlan.size(); i++) {
        SurfaceHandle s = sSplitPlan[i].first;
        VertexHandle v1 = new_vertices_map[{sSplitPlan[i].second.first.v_oldSide.id,  sSplitPlan[i].second.first.v_newSide.id}];
        VertexHandle v2 = new_vertices_map[{sSplitPlan[i].second.second.v_oldSide.id, sSplitPlan[i].second.second.v_newSide.id}];
        SurfaceHandle s_new = s.split(v1, v2);
        if(!s_new) 
            return NULL;
        new_surfs.push_back(s_new);
    }

    // Construct interface surface
    if(!stype) 
        stype = new_surfs[0].type();
    SurfaceHandle s_new = (*stype)(new_vertices);
    if(!s_new) {
        return NULL;
    }
    Surface *_s_new = s_new.surface();
    add(_s_new);
    _s_new->add(this);
    _s_new->positionChanged();

    // Transfer moved and new split surfaces to new body
    for(auto &s : surfs_moved) {
        Surface *_s = s.surface();
        _s->remove(this);
        remove(_s);
    }
    for(auto &s : new_surfs) {
        Surface *_s = s.surface();
        _s->remove(this);
        remove(_s);
    }
    positionChanged();

    // Construct new body
    std::vector<SurfaceHandle> new_body_surfs(surfs_moved);
    new_body_surfs.push_back(s_new);
    for(auto &s : new_surfs) 
        new_body_surfs.push_back(s);
    BodyHandle b_new = (*type())(new_body_surfs);
    if(!b_new) {
        return NULL;
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {objectId(), b_new.id}, {objType(), b_new.objType()}, "split");

    return b_new.body();
}


////////////////
// BodyHandle //
////////////////


BodyHandle::BodyHandle(const int &_id) : id{_id} {}

Body *BodyHandle::body() const {
    Body *o = this->id >= 0 ? Mesh::get()->getBody(this->id) : NULL;
    if(!o) {
        TF_Log(LOG_ERROR) << "Invalid handle";
    }
    return o;
}

bool BodyHandle::definedBy(const VertexHandle &v) const {
    BodyHandle_GETOBJ(o, false);
    Vertex *_v = v.vertex();
    if(!_v) {
        BodyHandle_INVALIDHANDLERR;
        return false;
    }
    return o->definedBy(_v);
}

bool BodyHandle::definedBy(const SurfaceHandle &s) const {
    BodyHandle_GETOBJ(o, false);
    Surface *_s = s.surface();
    if(!_s) {
        BodyHandle_INVALIDHANDLERR;
        return false;
    }
    return o->definedBy(_s);
}

HRESULT BodyHandle::destroy() {
    BodyHandle_GETOBJ(o, E_FAIL);
    HRESULT res = o->destroy();
    if(res == S_OK) 
        this->id = -1;
    return res;
}

bool BodyHandle::validate() {
    BodyHandle_GETOBJ(o, false);
    return o->validate();
}

HRESULT BodyHandle::positionChanged() {
    BodyHandle_GETOBJ(o, E_FAIL);
    return o->positionChanged();
}

std::string BodyHandle::str() const {
    std::stringstream ss;

    ss << "BodyHandle(";
    Body *_b = this->body();
    if(_b) {
        ss << "id=" << this->id << ", typeId=" << _b->typeId;
    }
    ss << ")";

    return ss.str();
}

std::string BodyHandle::toString() {
    return TissueForge::io::toString(*this);
}

BodyHandle BodyHandle::fromString(const std::string &s) {
    return TissueForge::io::fromString<BodyHandle>(s);
}

HRESULT BodyHandle::add(const SurfaceHandle &s) {
    BodyHandle_GETOBJ(o, E_FAIL);
    Surface *_s = s.surface();
    if(!_s) {
        BodyHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->add(_s);
}

HRESULT BodyHandle::remove(const SurfaceHandle &s) {
    BodyHandle_GETOBJ(o, E_FAIL);
    Surface *_s = s.surface();
    if(!_s) {
        BodyHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->remove(_s);
}

HRESULT BodyHandle::replace(const SurfaceHandle &toInsert, const SurfaceHandle &toRemove) {
    BodyHandle_GETOBJ(o, E_FAIL);
    Surface *_toInsert = toInsert.surface();
    Surface *_toRemove = toRemove.surface();
    if(!_toInsert || !_toRemove) {
        BodyHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toInsert, _toRemove);
}

BodyType *BodyHandle::type() const {
    BodyHandle_GETOBJ(o, NULL);
    return o->type();
}

HRESULT BodyHandle::become(BodyType *btype) {
    BodyHandle_GETOBJ(o, E_FAIL);
    return o->become(btype);
}

std::vector<SurfaceHandle> BodyHandle::getSurfaces() const {
    BodyHandle_GETOBJ(o, {});
    auto& _result = o->getSurfaces();
    std::vector<SurfaceHandle> result;
    result.reserve(_result.size());
    for(auto &_s : _result) 
        result.push_back(_s ? SurfaceHandle(_s->objectId()) : SurfaceHandle());
    return result;
}

std::vector<VertexHandle> BodyHandle::getVertices() const {
    BodyHandle_GETOBJ(o, {});
    std::vector<Vertex*> _result = o->getVertices();
    std::vector<VertexHandle> result;
    result.reserve(_result.size());
    for(auto &_v : _result) 
        result.push_back(_v ? VertexHandle(_v->objectId()) : VertexHandle());
    return result;
}

VertexHandle BodyHandle::findVertex(const FVector3 &dir) const {
    BodyHandle_GETOBJ(o, VertexHandle());
    Vertex *v = o->findVertex(dir);
    return v ? VertexHandle(v->objectId()) : VertexHandle();
}

SurfaceHandle BodyHandle::findSurface(const FVector3 &dir) const {
    BodyHandle_GETOBJ(o, SurfaceHandle());
    Surface *s = o->findSurface(dir);
    return s ? SurfaceHandle(s->objectId()) : SurfaceHandle();
}

std::vector<BodyHandle> BodyHandle::connectedBodies() const {
    BodyHandle_GETOBJ(o, {});
    std::vector<Body*> _result = o->connectedBodies();
    std::vector<BodyHandle> result;
    result.reserve(_result.size());
    for(auto &_b : _result) 
        result.push_back(_b ? BodyHandle(_b->_objId) : BodyHandle());
    return result;
}

std::vector<BodyHandle> BodyHandle::adjacentBodies() const {
    BodyHandle_GETOBJ(o, {});
    std::vector<Body*> _result = o->adjacentBodies();
    std::vector<BodyHandle> result;
    result.reserve(_result.size());
    for(auto &_b : _result) 
        result.push_back(_b ? BodyHandle(_b->_objId) : BodyHandle());
    return result;
}

std::vector<SurfaceHandle> BodyHandle::neighborSurfaces(const SurfaceHandle &s) const {
    BodyHandle_GETOBJ(o, {});
    Surface *_s = s.surface();
    if(!_s) {
        BodyHandle_INVALIDHANDLERR;
        return {};
    }
    std::vector<Surface*> _result = o->neighborSurfaces(_s);
    std::vector<SurfaceHandle> result;
    for(auto &_s : _result) 
        result.push_back(_s ? SurfaceHandle(_s->objectId()) : SurfaceHandle());
    return result;
}

FloatP_t BodyHandle::getDensity() const {
    BodyHandle_GETOBJ(o, 0);
    return o->getDensity();
}

void BodyHandle::setDensity(const FloatP_t &_density) {
    BodyHandle_GETOBJ(o, );
    o->setDensity(_density);
}

FVector3 BodyHandle::getCentroid() const {
    BodyHandle_GETOBJ(o, FVector3());
    return o->getCentroid();
}

FVector3 BodyHandle::getVelocity() const {
    BodyHandle_GETOBJ(o, FVector3());
    return o->getVelocity();
}

FloatP_t BodyHandle::getArea() const {
    BodyHandle_GETOBJ(o, 0);
    return o->getArea();
}

FloatP_t BodyHandle::getVolume() const {
    BodyHandle_GETOBJ(o, 0);
    return o->getVolume();
}

FloatP_t BodyHandle::getMass() const {
    BodyHandle_GETOBJ(o, 0);
    return o->getMass();
}

FloatP_t BodyHandle::getVertexArea(const VertexHandle &v) const {
    BodyHandle_GETOBJ(o, 0);
    Vertex *_v = v.vertex();
    if(!_v) {
        BodyHandle_INVALIDHANDLERR;
        return 0;
    }
    return o->getVertexArea(_v);
}

FloatP_t BodyHandle::getVertexVolume(const VertexHandle &v) const {
    BodyHandle_GETOBJ(o, 0);
    Vertex *_v = v.vertex();
    if(!_v) {
        BodyHandle_INVALIDHANDLERR;
        return 0;
    }
    return o->getVertexVolume(_v);
}

std::vector<VertexHandle> BodyHandle::sharedVertices(const BodyHandle &other) const {
    BodyHandle_GETOBJ(o, {});
    Body *_other = other.body();
    if(!_other) {
        BodyHandle_INVALIDHANDLERR;
        return {};
    }
    std::vector<Vertex*> _result = o->sharedVertices(_other);
    std::vector<VertexHandle> result;
    for(auto &_other : _result) 
        result.push_back(_other ? VertexHandle(_other->objectId()) : VertexHandle());
    return result;
}

FloatP_t BodyHandle::getVertexMass(const VertexHandle &v) const {
    BodyHandle_GETOBJ(o, 0);
    Vertex *_v = v.vertex();
    if(!_v) {
        BodyHandle_INVALIDHANDLERR;
        return 0;
    }
    return o->getVertexMass(_v);
}

HRESULT BodyHandle::setSpecies(state::StateVector *s) const {
    BodyHandle_GETOBJ(o, E_FAIL);
    o->species = s;
    return S_OK;
}

state::StateVector *BodyHandle::getSpecies() const {
    BodyHandle_GETOBJ(o, {});
    return o->species;
}

std::vector<SurfaceHandle> BodyHandle::findInterface(const BodyHandle &b) const {
    BodyHandle_GETOBJ(o, {});
    Body *_b = b.body();
    if(!_b) {
        BodyHandle_INVALIDHANDLERR;
        return {};
    }
    std::vector<Surface*> _result = o->findInterface(_b);
    std::vector<SurfaceHandle> result;
    result.reserve(_result.size());
    for(auto &_s : _result) 
        result.push_back(_s ? SurfaceHandle(_s->objectId()) : SurfaceHandle());
    return result;
}

FloatP_t BodyHandle::contactArea(const BodyHandle &other) const {
    BodyHandle_GETOBJ(o, 0);
    Body *_other = other.body();
    if(!_other) {
        BodyHandle_INVALIDHANDLERR;
        return 0;
    }
    return o->contactArea(_other);
}

bool BodyHandle::isOutside(const FVector3 &pos) const {
    BodyHandle_GETOBJ(o, false);
    return o->isOutside(pos);
}

BodyHandle BodyHandle::split(const FVector3 &cp_pos, const FVector3 &cp_norm, SurfaceType *stype) {
    Body_GETMESH(mesh, BodyHandle());
    if(mesh->ensureAvailableBodies(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return BodyHandle();
    }
    BodyHandle_GETOBJ(o, BodyHandle());
    Body *_result = o->split(cp_pos, cp_norm, stype);
    return _result ? BodyHandle(_result->objectId()) : BodyHandle();
}


//////////////
// BodyType //
//////////////


static BodyHandle BodyType_fromSurfaces(BodyType *btype, const std::vector<SurfaceHandle> &surfaces) {
    // Verify that at least 4 surfaces are given
    if(surfaces.size() < 4) {
        TF_Log(LOG_ERROR) << "A body requires at least 4 surfaces";
        return BodyHandle();
    }
    // Verify that every parent vertex is in at least two surfaces
    // todo: current vertex condition is necessary for body construction, but is it sufficient?
    for(auto &s : surfaces) 
        for(auto &v : s.getVertices()) 
            if(v.getSurfaces().size() < 2) {
                TF_Log(LOG_ERROR) << "Detected insufficient connectivity";
                return BodyHandle();
            }

    BodyHandle b = Body::create(surfaces);
    Body *_b = b.body();
    if(!_b || btype->add(b) != S_OK) {
        TF_Log(LOG_ERROR) << "Failed to create instance";
        return BodyHandle();
    }
    _b->setDensity(btype->density);
    _b->updateInternals();
    return b;
}

BodyType::BodyType(const bool &noReg) : 
    MeshObjType()
{
    name = "Body";
    density = 0.f;

    if(!noReg) 
        this->registerType();
}

std::string BodyType::str() const {
    std::stringstream ss;

    ss << "BodyType(id=" << this->id << ", name=" << this->name << ")";

    return ss.str();
}

BodyType *BodyType::findFromName(const std::string &_name) {
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->findBodyFromName(_name);
}

HRESULT BodyType::registerType() {
    if(isRegistered()) return S_OK;

    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return E_FAIL;

    HRESULT result = solver->registerType(this);
    if(result == S_OK) 
        on_register();

    return result;
}

bool BodyType::isRegistered() {
    return get();
}

BodyType *BodyType::get() {
    return findFromName(name);
}

HRESULT BodyType::add(const BodyHandle &i) {
    if(!i) 
        return tf_error(E_FAIL, "Invalid object");
    Body *_i = i.body();
    if(!_i) 
        return tf_error(E_FAIL, "Object not registered");

    BodyType *iType = _i->type();
    if(iType) 
        iType->remove(i);

    _i->typeId = this->id;
    this->_instanceIds.push_back(i.id);
    return S_OK;
}

HRESULT BodyType::remove(const BodyHandle &i) {
    if(!i) 
        return tf_error(E_FAIL, "Invalid object");
    Body *_i = i.body();
    if(!_i) 
        return tf_error(E_FAIL, "Object not registered");

    auto itr = std::find(this->_instanceIds.begin(), this->_instanceIds.end(), i.id);
    if(itr == this->_instanceIds.end()) 
        return tf_error(E_FAIL, "Instance not of this type");

    this->_instanceIds.erase(itr);
    _i->typeId = -1;
    return S_OK;
}

HRESULT BodyType::remove(const std::vector<BodyHandle>& i) {
    std::vector<Body*> _i(i.size());
    std::vector<HRESULT> resultPool(ThreadPool::size(), S_OK);
    parallel_for(ThreadPool::size(), [&i, &_i, &resultPool](int tid) -> void {
        for(int j = tid; j < _i.size(); j += ThreadPool::size()) {
            BodyHandle ij = i[j];
            if(!ij) {
                resultPool[tid] = E_FAIL;
                return;
            }

            Body* _ij = ij.body();
            if(!_ij) {
                resultPool[tid] = E_FAIL;
                return;
            }

            _i[j] = _ij;
        }
    });
    for(auto& resultThread : resultPool) 
        if(resultThread != S_OK) {
            return tf_error(resultThread, "Failed to add object");
        }

    for(auto& s : i) {
        auto itr = std::find(this->_instanceIds.begin(), this->_instanceIds.end(), s.id);
        if(itr == this->_instanceIds.end()) 
            return tf_error(E_FAIL, "Instance not of this type");
        this->_instanceIds.erase(itr);
    }

    parallel_for(_i.size(), [&_i](int j) -> void { _i[j]->typeId = -1; });
    return S_OK;
}

std::vector<BodyHandle> BodyType::getInstances() {
    std::vector<BodyHandle> result;

    Mesh *m = Mesh::get();
    if(m) { 
        result.reserve(_instanceIds.size());
        for(size_t i = 0; i < m->sizeBodies(); i++) {
            Body *b = m->getBody(i);
            if(b && b->typeId == this->id) 
                result.emplace_back(i);
        }
    }

    return result;
}

unsigned int BodyType::getNumInstances() {
    return getInstances().size();
}

BodyHandle BodyType::operator() (const std::vector<SurfaceHandle> &surfaces) {
    return BodyType_fromSurfaces(this, surfaces);
}

BodyHandle BodyType::operator() (TissueForge::io::ThreeDFMeshData* ioMesh, SurfaceType *stype) {
    std::vector<SurfaceHandle> surfaces;
    surfaces.reserve(ioMesh->faces.size());
    for(auto &f : ioMesh->faces) 
        surfaces.push_back((*stype)(f));
    for(auto &si : surfaces) 
        for(auto &sj : surfaces) 
            if(si != sj && Surface::sew(si, sj) != S_OK) 
                return BodyHandle();
    return BodyType_fromSurfaces(this, surfaces);
}

BodyHandle BodyType::extend(const SurfaceHandle &base, const FVector3 &pos) {
    Body_GETMESH(mesh, BodyHandle());
    std::vector<VertexHandle> base_vertices = base.getVertices();
    if(mesh->ensureAvailableSurfaces(base_vertices.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return BodyHandle();
    }

    // For every pair of vertices, construct a surface with a new vertex at the given position
    VertexHandle vNew = Vertex::create(pos);
    SurfaceType *stype = base.type();
    std::vector<SurfaceHandle> surfaces(1, base);
    for(unsigned int i = 0; i < base_vertices.size(); i++) {
        SurfaceHandle s = (*stype)({
            base_vertices[i], 
            base_vertices[i == base_vertices.size() - 1 ? 0 : i + 1], 
            vNew
        });
        if(!s) 
            return BodyHandle();
        surfaces.push_back(s);
    }

    // Construct a body from the surfaces
    BodyHandle b = (*this)(surfaces);
    if(!b) {
        return BodyHandle();
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {base.id, b.id}, {base.objType(), b.objType()}, "extend");

    return b;
}

HRESULT Body_surfaceOutwardNormal(Surface *s, Body *b1, Body *b2, FVector3 &onorm) {
    if(b1 && b2) { 
        TF_Log(LOG_ERROR) << "Surface is twice-connected";
        return E_FAIL;
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

BodyHandle BodyType::extrude(const SurfaceHandle &base, const FloatP_t &normLen) {
    Body_GETMESH(mesh, BodyHandle());
    std::vector<VertexHandle> base_vertices = base.getVertices();
    if(mesh->ensureAvailableVertices(base_vertices.size()) != S_OK || 
        mesh->ensureAvailableSurfaces(base_vertices.size() + 1) != S_OK || 
        mesh->ensureAvailableBodies(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return BodyHandle();
    }

    unsigned int i, j;
    FVector3 normal;

    // Only permit if the surface has an available slot
    Surface *_base = base.surface();
    _base->refreshBodies();
    if(Body_surfaceOutwardNormal(_base, _base->b1, _base->b2, normal) != S_OK) 
        return BodyHandle();

    std::vector<VertexHandle> newVertices;
    newVertices.reserve(base_vertices.size());
    SurfaceType *stype = _base->type();
    FVector3 disp = normal * normLen;

    for(i = 0; i < base_vertices.size(); i++) 
        newVertices.push_back(Vertex::create(base_vertices[i].getPosition() + disp));

    std::vector<SurfaceHandle> newSurfaces;
    for(i = 0; i < base_vertices.size(); i++) {
        j = i + 1 >= base_vertices.size() ? i + 1 - base_vertices.size() : i + 1;
        SurfaceHandle s = (*stype)({
            base_vertices[i], 
            base_vertices[j], 
            newVertices[j], 
            newVertices[i]
        });
        if(!s) 
            return BodyHandle();
        newSurfaces.push_back(s);
    }
    newSurfaces.push_back(base);
    newSurfaces.push_back((*stype)(newVertices));

    BodyHandle b = (*this)(newSurfaces);
    if(!b) {
        return BodyHandle();
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {base.id, b.id}, {base.objType(), b.objType()}, "extrude");

    return b;
}


////////
// io //
////////


namespace TissueForge::io {


    template <>
    HRESULT toFile(TissueForge::models::vertex::Body *dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "objId", dataElement->objectId());
        TF_IOTOEASY(fileElement, metaData, "typeId", dataElement->typeId);

        if(dataElement->actors.size() > 0) {
            std::vector<TissueForge::models::vertex::MeshObjActor*> actors;
            for(auto &a : dataElement->actors) 
                if(a) 
                    actors.push_back(a);
            TF_IOTOEASY(fileElement, metaData, "actors", actors);
        }

        std::vector<int> surfaces;
        for(auto &s : dataElement->getSurfaces()) 
            surfaces.push_back(s->objectId());
        TF_IOTOEASY(fileElement, metaData, "surfaces", surfaces);

        TF_IOTOEASY(fileElement, metaData, "centroid", dataElement->getCentroid());
        TF_IOTOEASY(fileElement, metaData, "area", dataElement->getArea());
        TF_IOTOEASY(fileElement, metaData, "volume", dataElement->getVolume());
        TF_IOTOEASY(fileElement, metaData, "density", dataElement->getDensity());
        TF_IOTOEASY(fileElement, metaData, "typeId", dataElement->typeId);

        if(dataElement->species) {
            TF_IOTOEASY(fileElement, metaData, "species", *dataElement->species);
        }

        fileElement.get()->type = "Body";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Body **dataElement) {
        
        if(!FIO::hasImport()) 
            return tf_error(E_FAIL, "No import data available");
        else if(!TissueForge::models::vertex::io::VertexSolverFIOModule::hasImport()) 
            return tf_error(E_FAIL, "No vertex import data available");

        TissueForge::models::vertex::MeshSolver *solver = TissueForge::models::vertex::MeshSolver::get();
        if(!solver) 
            return tf_error(E_FAIL, "No vertex solver available");
        Body_GETMESH(mesh, E_FAIL);

        int typeIdOld;
        TF_IOFROMEASY(fileElement, metaData, "typeId", &typeIdOld);
        auto typeId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->bodyTypeIdMap.find(typeIdOld);
        if(typeId_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->bodyTypeIdMap.end()) {
            return tf_error(E_FAIL, "Could not identify type");
        }
        TissueForge::models::vertex::BodyType *detype = solver->getBodyType(typeId_itr->second);

        std::vector<TissueForge::models::vertex::SurfaceHandle> surfaces;
        std::vector<int> surfacesIds;
        for(auto &surfaceIdOld : surfacesIds) {
            auto surfaceId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceIdMap.find(surfaceIdOld);
            if(surfaceId_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceIdMap.end()) {
                return tf_error(E_FAIL, "Could not identify surface");
            }
            surfaces.emplace_back(surfaceId_itr->second);
        }

        *dataElement = (*detype)(surfaces).body();

        if(!(*dataElement) || (*dataElement)->typeId < 0) {
            if((*dataElement)) {
                (*dataElement)->destroy();
            }
            return tf_error(E_FAIL, "Failed to add body");
        }

        int objIdOld;
        TF_IOFROMEASY(fileElement, metaData, "objId", &objIdOld);
        TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->bodyIdMap.insert({objIdOld, (*dataElement)->objectId()});

        IOChildMap fec = IOElement::children(fileElement);

        if(fec.find("actors") != fec.end()) {
            TF_IOFROMEASY(fileElement, metaData, "actors", &(*dataElement)->actors);
        }

        if(fec.find("species") != fec.end()) {
            TF_IOFROMEASY(fileElement, metaData, "species", &(*dataElement)->species);
        }

        return S_OK;
    }

    template <>
    HRESULT toFile(const TissueForge::models::vertex::BodyHandle &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);

        fileElement.get()->type = "BodyHandle";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::BodyHandle *dataElement) {
        
        IOChildMap::const_iterator feItr;

        int id;
        TF_IOFROMEASY(fileElement, metaData, "id", &id);
        *dataElement = TissueForge::models::vertex::BodyHandle(id);

        return S_OK;
    }

    template <>
    HRESULT toFile(const TissueForge::models::vertex::BodyType &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);

        if(dataElement.actors.size() > 0) {
            std::vector<TissueForge::models::vertex::MeshObjActor*> actors;
            for(auto &a : dataElement.actors) 
                if(a) 
                    actors.push_back(a);
            TF_IOTOEASY(fileElement, metaData, "actors", actors);
        }

        TF_IOTOEASY(fileElement, metaData, "name", dataElement.name);
        TF_IOTOEASY(fileElement, metaData, "density", dataElement.density);

        fileElement.get()->type = "BodyType";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::BodyType **dataElement) {
        
        *dataElement = new TissueForge::models::vertex::BodyType();

        TF_IOFROMEASY(fileElement, metaData, "name", &(*dataElement)->name);
        IOChildMap fec = IOElement::children(fileElement);
        if(fec.find("actors") != fec.end()) {
            TF_IOFROMEASY(fileElement, metaData, "actors", &(*dataElement)->actors);
        }

        return S_OK;
    }
}

std::string TissueForge::models::vertex::Body::toString() {
    TissueForge::io::IOElement el = TissueForge::io::IOElement::create();
    std::string result;
    if(TissueForge::io::toFile(this, TissueForge::io::MetaData(), el) == S_OK) 
        result = TissueForge::io::toStr(el);
    else 
        result = "";
    return result;
}

std::string TissueForge::models::vertex::BodyType::toString() {
    return TissueForge::io::toString(*this);
}

BodyType *TissueForge::models::vertex::BodyType::fromString(const std::string &str) {
    return TissueForge::io::fromString<TissueForge::models::vertex::BodyType*>(str);
}
