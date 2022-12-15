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

#include "tfSurface.h"

#include "tfVertex.h"
#include "tfSurface.h"
#include "tfBody.h"
#include "tfStructure.h"
#include "tfMeshSolver.h"
#include "actors/tfConvexPolygonConstraint.h"
#include "actors/tfFlatSurfaceConstraint.h"
#include "tf_mesh_io.h"
#include "tfVertexSolverFIO.h"

#include <Magnum/Math/Math.h>
#include <Magnum/Math/Intersection.h>

#include <tfError.h>
#include <tfLogger.h>
#include <tf_metrics.h>
#include <tf_util.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>

#include <io/tfThreeDFVertexData.h>
#include <io/tfThreeDFEdgeData.h>

#include <unordered_set>


#define Surface_VERTEXINDEX(vertices, idx) idx >= vertices.size() ? idx - vertices.size() : (idx < 0 ? idx + vertices.size() : idx)


using namespace TissueForge;
using namespace TissueForge::models::vertex;


FVector3 triNorm(const FVector3 &p1, const FVector3 &p2, const FVector3 &p3) {
    return Magnum::Math::cross(p1 - p2, p3 - p2);
}

Surface::Surface() : 
    MeshObj(), 
    b1{NULL}, 
    b2{NULL}, 
    area{0.f}, 
    _volumeContr{0.f}, 
    species1{NULL}, 
    species2{NULL}, 
    style{NULL}
{}

Surface::Surface(std::vector<Vertex*> _vertices) : 
    Surface()
{
    if(_vertices.size() >= 3) {
        for(auto &v : _vertices) {
            add(v);
            v->add(this);
        }
    } 
    else {
        TF_Log(LOG_ERROR) << "Surfaces require at least 3 vertices (" << _vertices.size() << " given)";
    }
}

static HRESULT Surface_order3DFFaceVertices(TissueForge::io::ThreeDFFaceData *face, std::vector<TissueForge::io::ThreeDFVertexData*> &result) {
    auto vedges = face->getEdges();
    auto vverts = face->getVertices();
    result.clear();
    std::vector<int> edgesLeft;
    for(int i = 1; i < vedges.size(); edgesLeft.push_back(i), i++) {}
    
    TissueForge::io::ThreeDFVertexData *currentVertex;
    TissueForge::io::ThreeDFEdgeData *edge = vedges[0];
    currentVertex = edge->vb;
    result.push_back(edge->va);
    result.push_back(currentVertex);

    while(edgesLeft.size() > 1) { 
        int j;
        edge = 0;
        for(j = 0; j < edgesLeft.size(); j++) {
            edge = vedges[edgesLeft[j]];
            if(edge->va == currentVertex || edge->vb == currentVertex) 
                break;
        }
        if(!edge) {
            TF_Log(LOG_ERROR) << "Error importing face";
            return E_FAIL;
        } 
        else {
            currentVertex = edge->va == currentVertex ? edge->vb : edge->va;
            result.push_back(currentVertex);
            edgesLeft.erase(std::find(edgesLeft.begin(), edgesLeft.end(), edgesLeft[j]));
        }
    }
    return S_OK;
}

Surface::Surface(TissueForge::io::ThreeDFFaceData *face) : 
    Surface()
{
    std::vector<TissueForge::io::ThreeDFVertexData*> vverts;
    if(Surface_order3DFFaceVertices(face, vverts) == S_OK) {
        for(auto &vv : vverts) {
            Vertex *v = new Vertex(vv);
            add(v);
            v->add(this);
        }
    }
}

std::vector<MeshObj*> Surface::children() const {
    std::vector<MeshObj*> result;
    if(b1) 
        result.push_back((MeshObj*)b1);
    if(b2) 
        result.push_back((MeshObj*)b2);
    return result;
}

HRESULT Surface::addChild(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::BODY)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Body *b = (Body*)obj;
    
    if(b1) {
        if(b1 == b) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }

        if(b2) {
            if(b2 == b) {
                TF_Log(LOG_ERROR);
                return E_FAIL;
            }
        }
        else {
            b2 = b;
        }
    }
    else 
        b1 = b;

    return S_OK;
}

HRESULT Surface::addParent(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::VERTEX)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Vertex *v = (Vertex*)obj;
    if(std::find(vertices.begin(), vertices.end(), v) != vertices.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    vertices.push_back(v);
    return S_OK;
}

HRESULT Surface::removeChild(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::BODY)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Body *b = (Body*)obj;

    if(b1 == b) 
        b1 = NULL;
    else if(b2 == b) 
        b2 = NULL;
    else {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT Surface::removeParent(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::VERTEX)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Vertex *v = (Vertex*)obj;
    auto itr = std::find(vertices.begin(), vertices.end(), v);
    if(itr == vertices.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    vertices.erase(itr);
    return S_OK;
}

std::string Surface::str() const {
    std::stringstream ss;

    ss << "Surface(";
    if(this->objId >= 0) {
        ss << "id=" << this->objId << ", typeId=" << this->typeId;
    }
    ss << ")";

    return ss.str();
}

#define SURFACE_RND_IDX(vec_size, idx) {    \
while(idx < 0) idx += vec_size;             \
while(idx >= vec_size) idx -= vec_size;     \
}

HRESULT Surface::add(Vertex *v) {
    return addParent(v);
}

HRESULT Surface::insert(Vertex *v, const int &idx) {
    auto itr = std::find(this->vertices.begin(), this->vertices.end(), v);
    if(itr != this->vertices.end()) {
        TF_Log(LOG_DEBUG);
        return E_FAIL;
    }

    int _idx = idx;
    SURFACE_RND_IDX(this->vertices.size(), _idx);
    this->vertices.insert(this->vertices.begin() + _idx, v);
    return S_OK;
}

HRESULT Surface::insert(Vertex *v, Vertex *before) {
    auto itr = std::find(this->vertices.begin(), this->vertices.end(), before);
    if(itr == this->vertices.end()) {
        TF_Log(LOG_DEBUG);
        return E_FAIL;
    }

    this->vertices.insert(itr, v);
    return S_OK;
}

HRESULT Surface::remove(Vertex *v) {
    return removeParent(v);
}

HRESULT Surface::replace(Vertex *toInsert, const int &idx) {
    int _idx = idx;
    SURFACE_RND_IDX(this->vertices.size(), _idx);
    std::replace(this->vertices.begin(), this->vertices.end(), this->vertices[_idx], toInsert);
    return S_OK;
}

HRESULT Surface::replace(Vertex *toInsert, Vertex *toRemove) {
    std::replace(this->vertices.begin(), this->vertices.end(), toRemove, toInsert);
    return toInsert->in(this) ? S_OK : E_FAIL;
}

HRESULT Surface::add(Body *b) {
    return addChild(b);
}

HRESULT Surface::remove(Body *b) {
    return removeChild(b);
}

HRESULT Surface::replace(Body *toInsert, const int &idx) {
    int _idx = idx;
    SURFACE_RND_IDX(2, _idx);
    if(_idx == 0) 
        this->b1 = toInsert;
    else 
        this->b2 = toInsert;
    return S_OK;
}

HRESULT Surface::replace(Body *toInsert, Body *toRemove) {
    if(this->b1 == toRemove) {
        this->b1 = toInsert;
        return S_OK;
    } 
    else if(this->b2 == toRemove) {
        this->b2 = toInsert;
        return S_OK;
    } 
    return E_FAIL;
}

HRESULT Surface::destroy() {
    if((b1 && b1->destroy() != S_OK) || (b2 && b2->destroy() != S_OK)) 
        return E_FAIL;
    return S_OK;
}

HRESULT Surface::destroy(Surface *target) {
    auto vertices = target->getVertices();
    if(target->destroy() != S_OK) 
        return E_FAIL;

    for(auto &v : vertices) 
        if(v->getSurfaces().size() == 0) 
            v->destroy();

    delete target;
    target = NULL;
    return S_OK;
}

bool Surface::validate() {
    return vertices.size() >= 3;
}

HRESULT Surface::refreshBodies() {
    if(!b1 && !b2) 
        return S_OK;
    else if(normal.isZero()) {
        TF_Log(LOG_ERROR) << "Normal not set";
        return E_FAIL;
    } 
    else if(centroid.isZero()) {
        TF_Log(LOG_ERROR) << "Centroid not set";
        return E_FAIL;
    }

    Body *bo = NULL;
    Body *bi = NULL;

    FVector3 n;
    if(b1) {
        n = centroid - b1->getCentroid();
        if(n.dot(normal) > 0) 
            bo = b1;
        else 
            bi = b1;
    }
    if(b2) {
        n = centroid - b2->getCentroid();
        if(n.dot(normal) > 0) {
            if(bo) {
                TF_Log(LOG_ERROR) << "Two bodies registered on the same side (outside)";
                return E_FAIL;
            }
            bo = b2;
        }
        else {
            if(bi) {
                TF_Log(LOG_ERROR) << "Two bodies registered on the same side (inside)";
                return E_FAIL;
            }
            bi = b2;
        }
    }

    b1 = bo;
    b2 = bi;

    return S_OK;
}

SurfaceType *Surface::type() const {
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getSurfaceType(typeId);
}

HRESULT Surface::become(SurfaceType *stype) {
    this->typeId = stype->id;
    return S_OK;
}

HRESULT Surface::insert(Vertex *toInsert, Vertex *v1, Vertex *v2) {
    // Validate input
    if(std::find(vertices.begin(), vertices.end(), toInsert) != vertices.end()) {
        TF_Log(LOG_DEBUG);
        return E_FAIL;
    }

    // Handle wrap
    Vertex *ve = *vertices.rbegin();
    if((*vertices.begin() == v1 && *vertices.rbegin() == v2) || (*vertices.begin() == v2 && *vertices.rbegin() == v1)) {
        vertices.insert(vertices.begin(), toInsert);
        return S_OK;
    }

    for(std::vector<Vertex*>::iterator itr = vertices.begin(); itr != vertices.end(); itr++) {
        if(*itr == v1 || *itr == v2) {
            vertices.insert(itr + 1, toInsert);
            return S_OK;
        }
    }
    TF_Log(LOG_ERROR) << "Vertices not found.";
    return E_FAIL;
}

std::vector<Structure*> Surface::getStructures() const {
    std::unordered_set<Structure*> result;
    if(b1) 
        for(auto &s : b1->getStructures()) 
            result.insert(s);
    if(b2) 
        for(auto &s : b2->getStructures()) 
            result.insert(s);
    return std::vector<Structure*>(result.begin(), result.end());
}

std::vector<Body*> Surface::getBodies() const {
    std::vector<Body*> result;
    if(b1) 
        result.push_back(b1);
    if(b2) 
        result.push_back(b2);
    return result;
}

Vertex *Surface::findVertex(const FVector3 &dir) const {
    Vertex *result = 0;

    FloatP_t bestCrit = 0;

    for(auto &v : getVertices()) {
        const FVector3 rel_pt = v->getPosition() - centroid;
        if(rel_pt.isZero()) 
            continue;
        const FloatP_t crit = rel_pt.dot(dir) / rel_pt.dot();
        if(!result || crit > bestCrit) { 
            result = v;
            bestCrit = crit;
        }
    }

    return result;
}

Body *Surface::findBody(const FVector3 &dir) const {
    Body *result = 0;

    FloatP_t bestCrit = 0;

    for(auto &b : getBodies()) {
        const FVector3 rel_pt = b->getCentroid() - centroid;
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

std::tuple<Vertex*, Vertex*> Surface::neighborVertices(const Vertex *v) const {
    Vertex *vp = NULL;
    Vertex *vn = NULL; 

    auto itr = std::find(vertices.begin(), vertices.end(), v);
    if(itr != vertices.end()) {
        vp = itr + 1 == vertices.end() ? *vertices.begin()     : *(itr + 1);
        vn = itr == vertices.begin()   ? *(vertices.end() - 1) : *(itr - 1);
    }

    return std::make_tuple(vp, vn);
}

std::vector<Surface*> Surface::neighborSurfaces() const { 
    std::unordered_set<Surface*> result;
    if(b1) 
        for(auto &s : b1->neighborSurfaces(this)) 
            result.insert(s);
    if(b2) 
        for(auto &s : b2->neighborSurfaces(this)) 
            result.insert(s);
    return std::vector<Surface*>(result.begin(), result.end());
}

std::vector<Surface*> Surface::connectedSurfaces(const std::vector<Vertex*> &verts) const {
    std::unordered_set<Surface*> result;
    for(auto &v : verts) 
        for(auto &s : v->surfaces) 
            if(s != this) 
                result.insert(s);
    return std::vector<Surface*>(result.begin(), result.end());
}

std::vector<Surface*> Surface::connectedSurfaces() const {
    return connectedSurfaces(vertices);
}

std::vector<unsigned int> Surface::contiguousEdgeLabels(const Surface *other) const {
    std::vector<bool> sharedVertices(vertices.size(), false);
    for(unsigned int i = 0; i < sharedVertices.size(); i++) 
        if(vertices[i]->in(other)) 
            sharedVertices[i] = true;

    bool shared_c, shared_n;
    std::vector<unsigned int> result(vertices.size(), 0);
    unsigned int edgeLabel = 1;
    for(unsigned int i = 0; i < sharedVertices.size(); i++) {
        shared_c = sharedVertices[i];
        shared_n = sharedVertices[i + 1 == sharedVertices.size() ? 0 : i + 1];
        if(shared_c) 
            result[i] = edgeLabel;
        if(shared_c && !shared_n) 
            edgeLabel++;
    }

    if(result[0] > 0 && result[result.size() - 1] > 0) {
        unsigned int lastLabel = result[result.size() - 1];
        for(unsigned int i = 0; i < result.size(); i++) 
            if(result[i] == lastLabel) 
                result[i] = result[0];
    }

    return result;
}

unsigned int Surface::numSharedContiguousEdges(const Surface *other) const {
    unsigned int result = 0;
    for(auto &i : contiguousEdgeLabels(other)) 
        result = std::max(result, i);
    return result;
}

FloatP_t Surface::volumeSense(const Body *body) const {
    if(body == b1) 
        return 1.f;
    else if(body == b2) 
        return -1.f;
    return 0.f;
}

FVector3 Surface::getOutwardNormal(const Body *body) const {
    if(body == b1) 
        return normal;
    else if(body == b2) 
        return normal * -1;
    return FVector3(0);
}

FloatP_t Surface::getVertexArea(const Vertex *v) const {
    FloatP_t result = 0.f;
    
    for(unsigned int i = 0; i < vertices.size(); i++) {
        Vertex *vc = vertices[i];
        Vertex *vn = vertices[Surface_VERTEXINDEX(vertices, i + 1)];

        if(vc == v || vn == v) {
            FVector3 triNormal = triNorm(vc->getPosition(), centroid, vn->getPosition());
            result += triNormal.length();
        }
    }

    return result / 4.f;
}

FVector3 Surface::triangleNormal(const unsigned int &idx) const {
    return triNorm(vertices[idx]->getPosition(), 
                   centroid, 
                   vertices[Surface_VERTEXINDEX(vertices, idx + 1)]->getPosition());
}

FloatP_t Surface::normalDistance(const FVector3 &pos) const {
    return FVector4::planeEquation(normal, centroid).distance(pos);
}

bool Surface::isOutside(const FVector3 &pos) const {
    return normalDistance(pos) > 0;
}

HRESULT Surface::positionChanged() {
    normal = FVector3(0.f);
    centroid = FVector3(0.f);
    velocity = FVector3(0.f);
    area = 0.f;
    _volumeContr = 0.f;

    for(auto &v : vertices) {
        ParticleHandle *p = v->particle();
        centroid += p->getPosition();
        velocity += p->getVelocity();
    }
    centroid /= (FloatP_t)vertices.size();
    velocity /= (FloatP_t)vertices.size();

    for(unsigned int i = 0; i < vertices.size(); i++) {
        FVector3 triNormal = triangleNormal(i);

        _volumeContr += triNormal.dot(centroid);
        area += triNormal.length();
        normal += triNormal;
    }

    area /= 2.f;
    _volumeContr /= 6.f;
    if(normal.isZero()) 
        return tf_error(E_FAIL, "Zero normal");
    normal = normal.normalized();

    return S_OK;
}

HRESULT Surface::sew(Surface *s1, Surface *s2, const FloatP_t &distCf) {
    if(s1 == s2) 
        return S_OK;

    Mesh *mesh = s1->mesh;
    if(s2->mesh) {
        if(!mesh) 
            mesh = s2->mesh;
        else if(s2->mesh != mesh) 
            return tf_error(E_FAIL, "Surfaces not in the same mesh");
    }

    if(s1->positionChanged() != S_OK || s2->positionChanged() != S_OK) 
        return E_FAIL;

    // Pre-calculate vertex positions
    std::vector<FVector3> s2_positions;
    s2_positions.reserve(s2->vertices.size());
    for(auto &v : s2->vertices) 
        s2_positions.push_back(v->getPosition());

    // Find vertices to merge
    FloatP_t distCrit2 = distCf * distCf * 0.5 * (s1->area + s2->area);
    std::vector<int> indicesMatched(s1->vertices.size(), -1);
    size_t numMatched = 0;
    for(int i = 0; i < s1->vertices.size(); i++) {
        Vertex *vi = s1->vertices[i];
        if(vi->in(s2)) 
            continue;
        
        auto posi = vi->getPosition();
        FloatP_t minDist2 = distCrit2;
        int matchedIdx = -1;
        for(int j = 0; j < s2->vertices.size(); j++) { 
            Vertex *vj = s2->vertices[j];
            auto dist2 = (s2_positions[j] - posi).dot();
            if(dist2 < minDist2 && !vj->in(s1)) {
                minDist2 = dist2;
                matchedIdx = j;
            }
        }
        indicesMatched[i] = matchedIdx;
        if(matchedIdx >= 0) 
            numMatched++;
    }

    if(numMatched == 0) 
        return S_OK;

    // Merge matched vertices
    std::vector<Vertex*> toRemove;
    toRemove.reserve(s1->vertices.size());
    for(int i = 0; i < s1->vertices.size(); i++) {
        Vertex *vi = s1->vertices[i];
        int j = indicesMatched[i];
        if(j >= 0) {
            Vertex *vj = s2->vertices[j];
            for(auto &s : vj->getSurfaces()) 
                for(int k = 0; k < s->vertices.size(); k++) 
                    if(s->vertices[k] == vj) {
                        s->vertices[k] = vi;
                        if(vi->add(s) != S_OK)
                            return E_FAIL;
                        break;
                    }
            if(vj->objId >= 0)
                toRemove.push_back(vj);
        }
    }

    for(auto &vj : toRemove) {
        auto children = vj->children();
        for(auto &c : children) 
            vj->removeChild(c);
        if(vj->destroy() != S_OK) 
            return E_FAIL;
        delete vj;
    }

    MeshSolver *solver = mesh ? MeshSolver::get() : NULL;

    if(solver) 
        solver->log(mesh, MeshLogEventType::Create, {s1->objId, s2->objId}, {s1->objType(), s2->objType()}, "sew");

    return S_OK;
}

HRESULT Surface::sew(std::vector<Surface*> _surfaces, const FloatP_t &distCf) {
    for(std::vector<Surface*>::iterator itri = _surfaces.begin(); itri != _surfaces.end() - 1; itri++) 
        for(std::vector<Surface*>::iterator itrj = itri + 1; itrj != _surfaces.end(); itrj++) 
            if(*itri != *itrj && sew(*itri, *itrj, distCf) != S_OK) 
                return E_FAIL;

    return S_OK;
}

HRESULT Surface::merge(Surface *toRemove, const std::vector<FloatP_t> &lenCfs) {
    if(vertices.size() != toRemove->vertices.size()) {
        TF_Log(LOG_ERROR) << "Surfaces must have the same number of vertices to merge";
        return E_FAIL;
    }

    // Find vertices that are not shared
    std::vector<Vertex*> toKeepExcl;
    for(auto &v : vertices) 
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
        if(!in(b)) {
            b->add(this);
            add(b);
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

    MeshSolver *solver = mesh ? MeshSolver::get() : NULL;

    if(solver) 
        solver->log(mesh, MeshLogEventType::Create, {objId, toRemove->objId}, {objType(), toRemove->objType()}, "merge");
    
    // Remove surface and vertices that are not shared
    if(toRemove->destroy() != S_OK) 
        return E_FAIL;
    for(auto &v : toRemoveOrdered) 
        if(v->destroy() != S_OK) 
            return E_FAIL;

    if(solver) 
        if(!mesh->qualityWorking() && solver->positionChanged() != S_OK)
            return E_FAIL;

    return S_OK;
}

Surface *Surface::extend(const unsigned int &vertIdxStart, const FVector3 &pos) {
    // Validate indices
    if(vertIdxStart >= vertices.size()) {
        TF_Log(LOG_ERROR) << "Invalid vertex indices (" << vertIdxStart << ", " << vertices.size() << ")";
        return NULL;
    }

    // Get base vertices
    Vertex *v0 = vertices[vertIdxStart];
    Vertex *v1 = vertices[vertIdxStart == vertices.size() - 1 ? 0 : vertIdxStart + 1];

    // Construct new vertex at specified position
    MeshParticleType *ptype = MeshParticleType_get();
    FVector3 _pos = pos;
    ParticleHandle *ph = (*ptype)(&_pos);
    Vertex *vert = new Vertex(ph->id);

    // Construct new surface, add new parts and return
    Surface *s = (*type())({v0, v1, vert});
    if(mesh) 
        mesh->add(s);

    MeshSolver *solver = mesh ? MeshSolver::get() : NULL;

    if(solver) {
        if(!mesh->qualityWorking()) 
            solver->positionChanged();

        solver->log(mesh, MeshLogEventType::Create, {objId, s->objId}, {objType(), s->objType()}, "extend");
    }

    return s;
}

Surface *Surface::extrude(const unsigned int &vertIdxStart, const FloatP_t &normLen) {
    // Validate indices
    if(vertIdxStart >= vertices.size()) {
        TF_Log(LOG_ERROR) << "Invalid vertex indices (" << vertIdxStart << ", " << vertices.size() << ")";
        return NULL;
    }

    // Get base vertices
    Vertex *v0 = vertices[vertIdxStart];
    Vertex *v1 = vertices[vertIdxStart == vertices.size() - 1 ? 0 : vertIdxStart + 1];

    // Construct new vertices
    FVector3 disp = normal * normLen;
    FVector3 pos2 = v0->getPosition() + disp;
    FVector3 pos3 = v1->getPosition() + disp;
    MeshParticleType *ptype = MeshParticleType_get();
    ParticleHandle *p2 = (*ptype)(&pos2);
    ParticleHandle *p3 = (*ptype)(&pos3);
    Vertex *v2 = new Vertex(p2->id);
    Vertex *v3 = new Vertex(p3->id);

    // Construct new surface, add new parts and return
    SurfaceType *stype = type();
    Surface *s = (*stype)({v0, v1, v2, v3});
    if(mesh && mesh->add(s) != S_OK) 
        return NULL;

    MeshSolver *solver = mesh ? MeshSolver::get() : NULL;

    if(solver) {
        if(!mesh->qualityWorking()) 
            solver->positionChanged();

        solver->log(mesh, MeshLogEventType::Create, {objId, s->objId}, {objType(), s->objType()}, "extrude");
    }

    return s;
}

Surface *Surface::split(Vertex *v1, Vertex *v2) { 
    // Verify that vertices are in surface
    if(!v1->in(this) || !v2->in(this)) { 
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
    v_new_surf.reserve(vertices.size());
    v_new_surf.push_back(v1);
    std::vector<Vertex*>::iterator v_itr = std::find(vertices.begin(), vertices.end(), v1);
    while(true) {
        v_itr++;
        if(v_itr == vertices.end()) 
            v_itr = vertices.begin();
        if(*v_itr == v2) 
            break;
        v_new_surf.push_back(*v_itr);
    }
    v_new_surf.push_back(v2);
    for(auto v_itr = v_new_surf.begin() + 1; v_itr != v_new_surf.end() - 1; v_itr++) {
        remove(*v_itr);
        (*v_itr)->remove(this);
    }

    // Build new surface
    Surface *s_new = (*type())(v_new_surf);
    if(!s_new) 
        return NULL;
    if(mesh) 
        mesh->add(s_new);

    // Continue hierarchy
    for(auto &b : getBodies()) {
        s_new->add(b);
        b->add(s_new);
    }

    MeshSolver *solver = mesh ? MeshSolver::get() : NULL;

    if(solver) {
        if(!mesh->qualityWorking()) 
            solver->positionChanged();

        solver->log(
            mesh, MeshLogEventType::Create, 
            {objId, s_new->objId, v1->objId, v2->objId}, 
            {objType(), s_new->objType(), v1->objType(), v2->objType()}, 
            "split"
        );
    }

    return s_new;
}

/** Find a contiguous set of surface vertices partioned by a cut plane */
static std::vector<Vertex*> Surface_SurfaceCutPlaneVertices(Surface *s, const FVector4 &planeEq) {
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
static HRESULT Surface_SurfaceCutPlanePointsPairs(
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
    std::vector<Vertex*> verts_new_s = Surface_SurfaceCutPlaneVertices(s, planeEq);
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

Surface *Surface::split(const FVector3 &cp_pos, const FVector3 &cp_norm) {
    if(cp_norm.isZero()) {
        tf_error(E_FAIL, "Zero normal");
        return 0;
    }

    FVector4 planeEq = FVector4::planeEquation(cp_norm.normalized(), cp_pos);

    FVector3 pos_start, pos_end; 
    Vertex *v_new_start, *v_old_start, *v_new_end, *v_old_end;
    if(Surface_SurfaceCutPlanePointsPairs(this, planeEq, pos_start, pos_end, &v_new_start, &v_old_start, &v_new_end, &v_old_end) != S_OK) 
        return NULL;

    // Create and insert new vertices
    Vertex *v_start = new Vertex(pos_start);
    Vertex *v_end   = new Vertex(pos_end);
    if(v_start->insert(v_old_start, v_new_start) != S_OK || v_end->insert(v_old_end, v_new_end) != S_OK) {
        v_start->destroy();
        v_end->destroy();
        delete v_start;
        delete v_end;
        return NULL;
    }

    // Create new surface
    return split(v_start, v_end);
}

SurfaceType::SurfaceType(const FloatP_t &flatLam, const FloatP_t &convexLam, const bool &noReg) : 
    MeshObjType() 
{
    name = "Surface";

    style = NULL;
    MeshSolver *solver = MeshSolver::get();
    if(solver) {
        auto colors = color3Names();
        auto c = colors[(solver->numSurfaceTypes() - 1) % colors.size()];
        style = new rendering::Style(c);
    }

    actors.push_back(new FlatSurfaceConstraint(flatLam));
    actors.push_back(new ConvexPolygonConstraint(convexLam));

    if(!noReg) 
        this->registerType();
}

std::string SurfaceType::str() const {
    std::stringstream ss;

    ss << "SurfaceType(id=" << this->id << ", name=" << this->name << ")";

    return ss.str();
}

SurfaceType *SurfaceType::findFromName(const std::string &_name) {
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->findSurfaceFromName(_name);
}

HRESULT SurfaceType::registerType() {
    if(isRegistered()) return S_OK;

    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return E_FAIL;

    HRESULT result = solver->registerType(this);
    if(result == S_OK) 
        on_register();

    return result;
}

bool SurfaceType::isRegistered() {
    return get();
}

SurfaceType *SurfaceType::get() {
    return findFromName(name);
}

std::vector<Surface*> SurfaceType::getInstances() {
    std::vector<Surface*> result;

    MeshSolver *solver = MeshSolver::get();
    if(solver) { 
        result.reserve(solver->numSurfaces());
        for(auto &m : solver->meshes) {
            for(size_t i = 0; i < m->sizeSurfaces(); i++) {
                Surface *s = m->getSurface(i);
                if(s) 
                    result.push_back(s);
            }
        }
    }

    return result;
}

std::vector<int> SurfaceType::getInstanceIds() {
    auto instances = getInstances();
    std::vector<int> result;
    result.reserve(instances.size());
    for(auto &s : instances) 
        if(s) 
            result.push_back(s->objId);
    return result;
}

unsigned int SurfaceType::getNumInstances() {
    return getInstances().size();
}

Surface *SurfaceType::operator() (std::vector<Vertex*> _vertices) {
    Surface *s = new Surface(_vertices);
    s->typeId = this->id;
    return s;
}

Surface *SurfaceType::operator() (const std::vector<FVector3> &_positions) {
    std::vector<Vertex*> _vertices;
    for(auto &p : _positions) 
        _vertices.push_back(new Vertex(p));
    
    TF_Log(LOG_DEBUG) << "Created " << _vertices.size() << " vertices";
    
    return (*this)(_vertices);
}

Surface *SurfaceType::operator() (TissueForge::io::ThreeDFFaceData *face) {
    std::vector<TissueForge::io::ThreeDFVertexData*> vverts;
    std::vector<FVector3> _positions;
    if(Surface_order3DFFaceVertices(face, vverts) == S_OK) 
        for(auto &vv : vverts) 
            _positions.push_back(vv->position);
    
    return (*this)(_positions);
}

Surface *SurfaceType::nPolygon(const unsigned int &n, const FVector3 &center, const FloatP_t &radius, const FVector3 &ax1, const FVector3 &ax2) {
    if(ax1.isZero() || ax2.isZero()) {
        tf_error(E_FAIL, "Zero axis");
        return 0;
    }

    const FVector3 ax3 = Magnum::Math::cross(ax1, ax2);
    FMatrix4 t = FMatrix4::translation(center) * FMatrix4::from(FMatrix3(ax1.normalized(), ax2.normalized(), ax3.normalized()), FVector3(0));

    std::vector<FVector3> positions;
    positions.reserve(n);
    
    FloatP_t fact = M_PI * (FloatP_t)2 / (FloatP_t)n;
    for(size_t i = 0; i < n; i++) {
        FMatrix4 rot = FMatrix4::rotationZ(fact * (FloatP_t)i);
        FVector4 pos(radius, 0, 0, 1);
        positions.push_back((t * rot * pos).xyz());
    }

    return (*this)(positions);
}

Surface *SurfaceType::replace(Vertex *toReplace, std::vector<FloatP_t> lenCfs) {
    Mesh *_mesh = toReplace->mesh;

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
        if(vInserted->insert(toReplace, v) != S_OK) 
            return NULL;
        insertedVertices.push_back(vInserted);
    }

    // Disconnect replaced vertex from all surfaces
    std::vector<Surface*> toReplaceSurfaces = toReplace->getSurfaces();
    for(auto &s : toReplaceSurfaces) {
        s->remove(toReplace);
        toReplace->remove(s);
    }

    // Create new surface; its constructor should handle internal connections
    Surface *inserted = (*this)(insertedVertices);

    // Remove replaced vertex from the mesh and add inserted surface to the mesh
    if(_mesh && _mesh->add(inserted) != S_OK) 
        return NULL;

    MeshSolver *solver = _mesh ? MeshSolver::get() : NULL;
    if(solver) 
        solver->log(_mesh, MeshLogEventType::Create, {inserted->objId, toReplace->objId}, {inserted->objType(), toReplace->objType()}, "replace"); 
    toReplace->destroy();

    if(solver && !_mesh->qualityWorking()) 
        solver->positionChanged();

    return inserted;
}

namespace TissueForge::io {


    #define TF_MESH_SURFACEIOTOEASY(fe, key, member) \
        fe = new IOElement(); \
        if(toFile(member, metaData, fe) != S_OK)  \
            return E_FAIL; \
        fe->parent = fileElement; \
        fileElement->children[key] = fe;

    #define TF_MESH_SURFACEIOFROMEASY(feItr, children, metaData, key, member_p) \
        feItr = children.find(key); \
        if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
            return E_FAIL;

    template <>
    HRESULT toFile(const TissueForge::models::vertex::Surface &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_MESH_SURFACEIOTOEASY(fe, "objId", dataElement.objId);
        TF_MESH_SURFACEIOTOEASY(fe, "meshId", dataElement.mesh == NULL ? -1 : dataElement.mesh->getId());
        TF_MESH_SURFACEIOTOEASY(fe, "typeId", dataElement.typeId);

        if(dataElement.actors.size() > 0) {
            std::vector<TissueForge::models::vertex::MeshObjActor*> actors;
            for(auto &a : dataElement.actors) 
                if(a) 
                    actors.push_back(a);
            TF_MESH_SURFACEIOTOEASY(fe, "actors", actors);
        }

        std::vector<int> vertices;
        for(auto &p : dataElement.parents()) 
            vertices.push_back(p->objId);
        TF_MESH_SURFACEIOTOEASY(fe, "vertices", vertices);

        std::vector<int> bodies;
        for(auto &c : dataElement.children()) 
            bodies.push_back(c->objId);
        TF_MESH_SURFACEIOTOEASY(fe, "bodies", bodies);

        TF_MESH_SURFACEIOTOEASY(fe, "normal", dataElement.getNormal());
        TF_MESH_SURFACEIOTOEASY(fe, "centroid", dataElement.getCentroid());
        TF_MESH_SURFACEIOTOEASY(fe, "velocity", dataElement.getVelocity());
        TF_MESH_SURFACEIOTOEASY(fe, "area", dataElement.getArea());

        TF_MESH_SURFACEIOTOEASY(fe, "typeId", dataElement.typeId);

        if(dataElement.species1) {
            TF_MESH_SURFACEIOTOEASY(fe, "species_outward", *dataElement.species1);
        }
        if(dataElement.species2) {
            TF_MESH_SURFACEIOTOEASY(fe, "species_inward", *dataElement.species2);
        }
        if(dataElement.style) {
            TF_MESH_SURFACEIOTOEASY(fe, "style", *dataElement.style);
        }

        fileElement->type = "Surface";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Surface **dataElement) {
        
        if(!FIO::hasImport()) 
            return tf_error(E_FAIL, "No import data available");
        else if(!TissueForge::models::vertex::io::VertexSolverFIOModule::hasImport()) 
            return tf_error(E_FAIL, "No vertex import data available");

        TissueForge::models::vertex::MeshSolver *solver = TissueForge::models::vertex::MeshSolver::get();
        if(!solver) 
            return tf_error(E_FAIL, "No vertex solver available");

        IOChildMap::const_iterator feItr;

        TissueForge::models::vertex::Mesh *mesh = NULL;
        int meshIdOld;
        unsigned int meshId;
        TF_MESH_SURFACEIOFROMEASY(feItr, fileElement.children, metaData, "meshId", &meshIdOld);
        if(meshIdOld >= 0) {
            auto meshId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->meshIdMap.find(meshIdOld);
            if(meshId_itr != TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->meshIdMap.end() && meshId_itr->second < solver->numMeshes()) {
                meshId = meshId_itr->second;
                mesh = solver->getMesh(meshId);
            }
        }
        if(!mesh) {
            return tf_error(E_FAIL, "Could not identify mesh");
        }

        int typeIdOld;
        TF_MESH_SURFACEIOFROMEASY(feItr, fileElement.children, metaData, "typeId", &typeIdOld);
        auto typeId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.find(typeIdOld);
        if(typeId_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.end()) {
            return tf_error(E_FAIL, "Could not identify type");
        }
        TissueForge::models::vertex::SurfaceType *detype = solver->getSurfaceType(typeId_itr->second);

        std::vector<TissueForge::models::vertex::Vertex*> vertices;
        std::vector<int> verticesIds;
        TF_MESH_SURFACEIOFROMEASY(feItr, fileElement.children, metaData, "vertices", &verticesIds);
        for(auto &vertexIdOld : verticesIds) {
            auto vertexId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->vertexIdMap[meshId].find(vertexIdOld);
            if(vertexId_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->vertexIdMap[meshId].end()) {
                return tf_error(E_FAIL, "Could not identify vertex");
            }
            vertices.push_back(mesh->getVertex(vertexId_itr->second));
        }

        *dataElement = (*detype)(vertices);

        if(mesh->add(*dataElement) != S_OK) 
            return tf_error(E_FAIL, "Failed to add to mesh");

        int objIdOld;
        TF_MESH_SURFACEIOFROMEASY(feItr, fileElement.children, metaData, "objId", &objIdOld);
        TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceIdMap[meshId].insert({objIdOld, (*dataElement)->objId});

        feItr = fileElement.children.find("actors");
        if(feItr != fileElement.children.end()) {
            TF_MESH_SURFACEIOFROMEASY(feItr, fileElement.children, metaData, "actors", &(*dataElement)->actors);
        }

        return S_OK;
    }

    template <>
    HRESULT toFile(const TissueForge::models::vertex::SurfaceType &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_MESH_SURFACEIOTOEASY(fe, "id", dataElement.id);

        if(dataElement.actors.size() > 0) {
            std::vector<TissueForge::models::vertex::MeshObjActor*> actors;
            for(auto &a : dataElement.actors) 
                if(a) 
                    actors.push_back(a);
            TF_MESH_SURFACEIOTOEASY(fe, "actors", actors);
        }

        TF_MESH_SURFACEIOTOEASY(fe, "name", dataElement.name);
        TF_MESH_SURFACEIOTOEASY(fe, "style", *dataElement.style);

        fileElement->type = "SurfaceType";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::SurfaceType **dataElement) {
        
        IOChildMap::const_iterator feItr;

        *dataElement = new TissueForge::models::vertex::SurfaceType();

        TF_MESH_SURFACEIOFROMEASY(feItr, fileElement.children, metaData, "name", &(*dataElement)->name);
        if(fileElement.children.find("actors") != fileElement.children.end()) {
            TF_MESH_SURFACEIOFROMEASY(feItr, fileElement.children, metaData, "actors", &(*dataElement)->actors);
        }

        return S_OK;
    }
}

std::string TissueForge::models::vertex::Surface::toString() {
    return TissueForge::io::toString(*this);
}

std::string TissueForge::models::vertex::SurfaceType::toString() {
    return TissueForge::io::toString(*this);
}

SurfaceType *TissueForge::models::vertex::SurfaceType::fromString(const std::string &str) {
    return TissueForge::io::fromString<TissueForge::models::vertex::SurfaceType*>(str);
}
