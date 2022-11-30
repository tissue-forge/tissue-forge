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

#include "tfBody.h"

#include "tfVertex.h"
#include "tfSurface.h"
#include "tfStructure.h"
#include "tfMeshSolver.h"

#include <tfLogger.h>

#include <Magnum/Math/Math.h>
#include <Magnum/Math/Intersection.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


void Body::_updateInternal() {
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


static HRESULT Body_loadSurfaces(Body* body, std::vector<Surface*> _surfaces) { 
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


Body::Body() : 
    MeshObj(), 
    centroid{0.f}, 
    area{0.f}, 
    volume{0.f}, 
    density{0.f}, 
    species{NULL}
{}

Body::Body(std::vector<Surface*> _surfaces) : 
    Body() 
{
    if(Body_loadSurfaces(this, _surfaces) == S_OK) 
        _updateInternal();
};

Body::Body(io::ThreeDFMeshData *ioMesh) : 
    Body()
{
    std::vector<Surface*> _surfaces;
    for(auto f : ioMesh->faces) 
        _surfaces.push_back(new Surface(f));

    if(Body_loadSurfaces(this, _surfaces) == S_OK) 
        _updateInternal();
}

std::vector<MeshObj*> Body::parents() const { return TissueForge::models::vertex::vectorToBase(surfaces); }

std::vector<MeshObj*> Body::children() const { return TissueForge::models::vertex::vectorToBase(structures); }

HRESULT Body::addChild(MeshObj *obj) { 
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::STRUCTURE)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Structure *s = (Structure*)obj;
    if(std::find(structures.begin(), structures.end(), s) != structures.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    structures.push_back(s);
    return S_OK;
}

HRESULT Body::addParent(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::SURFACE)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Surface *s = (Surface*)obj;
    if(std::find(surfaces.begin(), surfaces.end(), s) != surfaces.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    surfaces.push_back(s);
    return S_OK;
}

HRESULT Body::removeChild(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::STRUCTURE)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Structure *s = (Structure*)obj;
    auto itr = std::find(structures.begin(), structures.end(), s);
    if(itr == structures.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    structures.erase(itr);
    return S_OK;
}

HRESULT Body::removeParent(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::SURFACE)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Surface *s = (Surface*)obj;
    auto itr = std::find(surfaces.begin(), surfaces.end(), s);
    if(itr == surfaces.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }
    
    surfaces.erase(itr);
    return S_OK;
}

std::string Body::str() const {
    std::stringstream ss;

    ss << "Body(";
    if(this->objId >= 0) {
        ss << "id=" << this->objId << ", typeId=" << this->typeId;
    }
    ss << ")";

    return ss.str();
}

#define BODY_RND_IDX(vec_size, idx) {       \
while(idx < 0) idx += vec_size;             \
while(idx >= vec_size) idx -= vec_size;     \
}

HRESULT Body::add(Surface *s) {
    return addParent(s);
}

HRESULT Body::remove(Surface *s) {
    return removeParent(s);
}

HRESULT Body::replace(Surface *toInsert, Surface *toRemove) {
    std::replace(this->surfaces.begin(), this->surfaces.end(), toRemove, toInsert);
    return toInsert->in(this) ? S_OK : E_FAIL;
}

HRESULT Body::add(Structure *s) {
    return addChild(s);
}

HRESULT Body::remove(Structure *s) {
    return removeChild(s);
}

HRESULT Body::replace(Structure *toInsert, Structure *toRemove) {
    std::replace(this->structures.begin(), this->structures.end(), toRemove, toInsert);
    return toInsert->in(this) ? S_OK : E_FAIL;
}

HRESULT Body::destroy() {
    if(this->mesh && this->mesh->remove(this) != S_OK) 
        return E_FAIL;
    return S_OK;
}

bool Body::validate() {
    return surfaces.size() >= 3;
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
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getBodyType(typeId);
}

HRESULT Body::become(BodyType *btype) {
    this->typeId = btype->id;
    return S_OK;
}

std::vector<Structure*> Body::getStructures() const {
    std::unordered_set<Structure*> result;
    for(auto &s : structures) {
        result.insert(s);
        for(auto &ss : s->getStructures()) 
            result.insert(ss);
    }
    return std::vector<Structure*>(result.begin(), result.end());
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

    FVector3 pta = centroid;
    FVector3 ptb = pta + dir;
    FloatP_t bestDist2 = 0;

    for(auto &v : getVertices()) {
        FVector3 pt = v->getPosition();
        FloatP_t dist2 = Magnum::Math::Distance::linePointSquared(pta, ptb, pt);
        if((!result || dist2 <= bestDist2) && dir.dot(pt - pta) >= 0.f) { 
            result = v;
            bestDist2 = dist2;
        }
    }

    return result;
}

Surface *Body::findSurface(const FVector3 &dir) const {
    Surface *result = 0;

    FVector3 pta = centroid;
    FVector3 ptb = pta + dir;
    FloatP_t bestDist2 = 0;

    for(auto &s : getSurfaces()) {
        FVector3 pt = s->getCentroid();
        FloatP_t dist2 = Magnum::Math::Distance::linePointSquared(pta, ptb, pt);
        if((!result || dist2 <= bestDist2) && dir.dot(pt - pta) >= 0.f) { 
            result = s;
            bestDist2 = dist2;
        }
    }

    return result;
}

std::vector<Body*> Body::neighborBodies() const {
    std::unordered_set<Body*> result;
    for(auto &s : surfaces) 
        for(auto &b : s->getBodies()) 
            if(b != this) 
                result.insert(b);
    return std::vector<Body*>(result.begin(), result.end());
}

std::vector<Surface*> Body::neighborSurfaces(const Surface *s) const {
    std::unordered_set<Surface*> result;
    for(auto &so : surfaces) {
        if(so == s) 
            continue;
        for(auto &v : s->vertices) 
            if(v->in(so)) { 
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
    FloatP_t result;
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
        if(s->in(b)) 
            result.push_back(s);
    return result;
}

FloatP_t Body::contactArea(const Body *other) const {
    FloatP_t result = 0.f;
    for(auto &s : surfaces) 
        if(s->in(other)) 
            result += s->area;
    return result;
}

bool Body::isOutside(const FVector3 &pos) const {
    // Test against outward-facing normal of nearest surface
    const FVector3 rel_pos = pos - centroid;
    return rel_pos.dot(findSurface(rel_pos)->getOutwardNormal(this)) > 0;
}

struct Body_BodySplitEdge {
    Vertex *v_oldSide;  // Old side
    Vertex *v_newSide;  // New side
    FVector3 intersect_pt;
    std::vector<Surface*> surfaces;

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

    static std::vector<Surface*> extractSurfaces(Vertex *v1, Vertex *v2, Body *b) {
        std::vector<Surface*> surfaces;
        for(auto &s : b->getSurfaces()) 
            if(v1->in(s) && v2->in(s)) 
                surfaces.push_back(s);
        return surfaces;
    }

    static std::vector<Body_BodySplitEdge> construct(Body *b, const FVector4 &planeEq) {
        std::map<std::pair<int, int>, Body_BodySplitEdge> edgeMap;
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
                        Body_BodySplitEdge edge;
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

        std::vector<Body_BodySplitEdge> result;
        result.reserve(edgeMap.size());
        for(auto &itr : edgeMap) 
            result.push_back(itr.second);
        
        std::vector<Body_BodySplitEdge> result_sorted;
        result_sorted.reserve(result.size());
        result_sorted.push_back(result.back());
        result.pop_back();
        Surface *s_target = result_sorted[0].surfaces[1];
        while(!result.empty()) {
            std::vector<Body_BodySplitEdge>::iterator itr = result.begin();
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

    typedef std::pair<Surface*, std::pair<Body_BodySplitEdge, Body_BodySplitEdge> > surfaceSplitPlanEl_t;

    static HRESULT surfaceSplitPlan(const std::vector<Body_BodySplitEdge> &edges, std::vector<surfaceSplitPlanEl_t> &result) {
        std::vector<Body_BodySplitEdge> edges_copy(edges);
        edges_copy.push_back(edges.front());
        for(size_t i = 0; i < edges.size(); i++) {
            Body_BodySplitEdge edge_i = edges_copy[i];
            Body_BodySplitEdge edge_j = edges_copy[i + 1];
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
    if(cp_norm.isZero()) {
        tf_error(E_FAIL, "Zero normal");
        return 0;
    }

    FVector4 planeEq = FVector4::planeEquation(cp_norm.normalized(), cp_pos);

    // Determine which surfaces are moved to new body
    std::vector<Surface*> surfs_moved;
    for(auto &s : surfaces) {
        size_t num_newSide = 0;
        for(auto &v : s->vertices) 
            if(planeEq.distance(v->getPosition()) > 0) 
                num_newSide++;
        if(num_newSide == s->vertices.size()) 
            surfs_moved.push_back(s);
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
    std::vector<Vertex*> new_vertices;
    std::map<std::pair<int, int>, Vertex*> new_vertices_map;
    for(auto &edge : vertexPlan) {
        Vertex *v_new = new Vertex(edge.intersect_pt);
        v_new->insert(edge.v_oldSide, edge.v_newSide);
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
        Surface *s_new = s->split(v1, v2);
        if(!s_new) 
            return NULL;
        new_surfs.push_back(s_new);
    }

    // Construct interface surface
    if(!stype) 
        stype = new_surfs[0]->type();
    Surface *s_new = (*stype)(new_vertices);
    if(!s_new || (mesh && mesh->add(s_new) != S_OK)) 
        return NULL;
    add(s_new);
    s_new->add(this);
    s_new->positionChanged();

    // Transfer moved and new split surfaces to new body
    for(auto &s : surfs_moved) {
        s->remove(this);
        remove(s);
    }
    for(auto &s : new_surfs) {
        s->remove(this);
        remove(s);
    }
    positionChanged();

    // Construct new body
    std::vector<Surface*> new_body_surfs(surfs_moved);
    new_body_surfs.push_back(s_new);
    for(auto &s : new_surfs) 
        new_body_surfs.push_back(s);
    Body *b_new = (*type())(new_body_surfs);
    if(!b_new) 
        return NULL;

    if(mesh && mesh->add(b_new) != S_OK) 
        return NULL;

    if(mesh) {
        MeshSolver *solver = MeshSolver::get();
        if(solver) {
            if(!mesh->qualityWorking()) 
                solver->positionChanged();

            solver->log(mesh, MeshLogEventType::Create, {objId, b_new->objId}, {objType(), b_new->objType()}, "split");
        }
    }

    return b_new;
}

static Body *BodyType_fromSurfaces(BodyType *btype, std::vector<Surface*> surfaces) {
    // Verify that at least 4 surfaces are given
    if(surfaces.size() < 4) {
        TF_Log(LOG_ERROR) << "A body requires at least 4 surfaces";
        return NULL;
    }
    // Verify that every parent vertex is in at least two surfaces
    // todo: current vertex condition is necessary for body construction, but is it sufficient?
    for(auto &s : surfaces) 
        for(auto &v : s->getVertices()) 
            if(v->children().size() < 2) {
                TF_Log(LOG_ERROR) << "Detected insufficient connectivity";
                return NULL;
            }

    Body *b = new Body(surfaces);
    b->typeId = btype->id;
    b->setDensity(btype->density);
    return b;
}

BodyType::BodyType(const bool &noReg) : 
    MeshObjType()
{
    name = "Body";

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

std::vector<Body*> BodyType::getInstances() {
    std::vector<Body*> result;

    MeshSolver *solver = MeshSolver::get();
    if(solver) { 
        result.reserve(solver->numBodies());
        for(auto &m : solver->meshes) {
            for(size_t i = 0; i < m->sizeBodies(); i++) {
                Body *b = m->getBody(i);
                if(b) 
                    result.push_back(b);
            }
        }
    }

    return result;
}

std::vector<int> BodyType::getInstanceIds() {
    auto instances = getInstances();
    std::vector<int> result;
    result.reserve(instances.size());
    for(auto &b : instances) 
        if(b) 
            result.push_back(b->objId);
    return result;
}

unsigned int BodyType::getNumInstances() {
    return getInstances().size();
}

Body *BodyType::operator() (std::vector<Surface*> surfaces) {
    return BodyType_fromSurfaces(this, surfaces);
}

Body *BodyType::operator() (io::ThreeDFMeshData* ioMesh, SurfaceType *stype) {
    std::vector<Surface*> surfaces;
    for(auto &f : ioMesh->faces) 
        surfaces.push_back((*stype)(f));
    for(auto &si : surfaces) 
        for(auto &sj : surfaces) 
            if(si != sj && Surface::sew(si, sj) != S_OK) 
                return NULL;
    return BodyType_fromSurfaces(this, surfaces);
}

Body *BodyType::extend(Surface *base, const FVector3 &pos) {
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
    Body *b = (*this)(surfaces);
    if(!b) 
        return NULL;

    // Add new parts and return
    if(base->mesh && base->mesh->add(b) != S_OK) 
        return NULL;

    MeshSolver *solver = base->mesh ? MeshSolver::get() : NULL;

    if(solver) {
        if(!base->mesh->qualityWorking()) 
            solver->positionChanged();

        solver->log(base->mesh, MeshLogEventType::Create, {base->objId, b->objId}, {base->objType(), b->objType()}, "extend");
    }

    return b;
}

HRESULT Body_surfaceOutwardNormal(Surface *s, Body *b1, Body *b2, FVector3 &onorm) {
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

Body *BodyType::extrude(Surface *base, const FloatP_t &normLen) {
    unsigned int i, j;
    FVector3 normal;

    // Only permit if the surface has an available slot
    base->refreshBodies();
    if(Body_surfaceOutwardNormal(base, base->b1, base->b2, normal) != S_OK) 
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

    Body *b = (*this)(newSurfaces);
    if(!b) 
        return NULL;
    if(base->mesh && base->mesh->add(b) != S_OK) 
        return NULL;

    MeshSolver *solver = base->mesh ? MeshSolver::get() : NULL;

    if(solver) {
        if(!base->mesh->qualityWorking()) 
            solver->positionChanged();

        solver->log(base->mesh, MeshLogEventType::Create, {base->objId, b->objId}, {base->objType(), b->objType()}, "extrude");
    }

    return b;
}
