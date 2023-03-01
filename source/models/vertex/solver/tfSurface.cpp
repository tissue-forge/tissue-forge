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

#include "tfSurface.h"

#include "tfVertex.h"
#include "tfSurface.h"
#include "tfBody.h"
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


#define Surface_GETMESH(name, retval)               \
    Mesh *name = Mesh::get();                       \
    if(!name) {                                     \
        TF_Log(LOG_ERROR) << "Could not get mesh";  \
        return retval;                              \
    }

#define SurfaceHandle_INVALIDHANDLERR { tf_error(E_FAIL, "Invalid handle"); }

#define SurfaceHandle_GETOBJ(name, retval)                              \
    Surface *name;                                                      \
    if(this->id < 0 || !(name = Mesh::get()->getSurface(this->id))) {   \
        SurfaceHandle_INVALIDHANDLERR; return retval; }


/////////////
// Surface //
/////////////


FVector3 triNorm(const FVector3 &p1, const FVector3 &p2, const FVector3 &p3) {
    return Magnum::Math::cross(p1 - p2, p3 - p2);
}

Surface::Surface() : 
    b1{NULL}, 
    b2{NULL}, 
    area{0.f}, 
    perimeter{0.f}, 
    _volumeContr{0.f}, 
    typeId{-1},
    species1{NULL}, 
    species2{NULL}, 
    style{NULL}, 
    density{0.f}
{
    MESHOBJ_INITOBJ
}

Surface::~Surface() {
    MESHOBJ_DELOBJ
}

static HRESULT Surface_fromVertices(Surface *s, std::vector<Vertex*> vertices) {
    Surface_GETMESH(mesh, E_FAIL);

    if(vertices.size() >= 3) {
        for(auto &v : vertices) {
            s->add(v);
            v->add(s);
        }
    } 
    else {
        return tf_error(E_FAIL, "Surfaces require at least 3 vertices");
    }

    for(auto &v : vertices) 
        v->updateConnectedVertices();

    return S_OK;
}

static Surface *Surface_create(std::vector<Vertex*> _vertices) {
    Surface_GETMESH(mesh, NULL);

    Surface *result;
    if(mesh->create(&result) != S_OK) {
        TF_Log(LOG_ERROR);
        return NULL;
    }
    if(Surface_fromVertices(result, _vertices) != S_OK) {
        TF_Log(LOG_ERROR);
        return NULL;
    }
    result->positionChanged();
    return result;
}

static Surface *Surface_create(const std::vector<VertexHandle> &_vertices) {
    Surface_GETMESH(mesh, NULL);

    std::vector<Vertex*> _vertices_objs;
    _vertices_objs.reserve(_vertices.size());
    for(auto &v : _vertices) {
        Vertex *_v = v.vertex();
        if(!_v) {
            TF_Log(LOG_ERROR);
            return NULL;
        }
        _vertices_objs.push_back(_v);
    }

    return Surface_create(_vertices_objs);
}

SurfaceHandle Surface::create(const std::vector<VertexHandle> &_vertices) {
    Surface *s = Surface_create(_vertices);
    return s ? SurfaceHandle(s->_objId) : SurfaceHandle();
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

SurfaceHandle Surface::create(TissueForge::io::ThreeDFFaceData *face) {
    std::vector<TissueForge::io::ThreeDFVertexData*> vverts;
    Surface_GETMESH(mesh, NULL);
    bool failed = false;
    if(Surface_order3DFFaceVertices(face, vverts) != S_OK || mesh->ensureAvailableVertices(vverts.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return NULL;
    }

    std::vector<VertexHandle> _vertices;
    _vertices.reserve(vverts.size());
    for(auto &vv : vverts) {
        VertexHandle v = Vertex::create(vv);
        if(!v) {
            failed = true;
            break;
        }
        _vertices.push_back(v);
    }

    if(failed) {
        TF_Log(LOG_ERROR);
        for(size_t i = 0; i < _vertices.size(); i++) {
            Vertex *v = _vertices[i].vertex();
            v->destroy();
        }
        return NULL;
    }

    return create(_vertices);
}

bool Surface::defines(const Body *obj) const { MESHBOJ_DEFINES_DEF(getSurfaces) }

bool Surface::definedBy(const Vertex *obj) const { MESHOBJ_DEFINEDBY_DEF }

std::string Surface::str() const {
    std::stringstream ss;

    ss << "Surface(";
    if(this->objectId() >= 0) {
        ss << "id=" << this->objectId() << ", typeId=" << this->typeId;
    }
    ss << ")";

    return ss.str();
}

#define SURFACE_RND_IDX(vec_size, idx) {    \
while(idx < 0) idx += vec_size;             \
while(idx >= vec_size) idx -= vec_size;     \
}

HRESULT Surface::add(Vertex *v) {
    if(std::find(vertices.begin(), vertices.end(), v) != vertices.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    vertices.push_back(v);
    return S_OK;
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
    auto itr = std::find(vertices.begin(), vertices.end(), v);
    if(itr == vertices.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    vertices.erase(itr);
    return S_OK;
}

HRESULT Surface::replace(Vertex *toInsert, const int &idx) {
    int _idx = idx;
    SURFACE_RND_IDX(this->vertices.size(), _idx);
    std::replace(this->vertices.begin(), this->vertices.end(), this->vertices[_idx], toInsert);
    return S_OK;
}

HRESULT Surface::replace(Vertex *toInsert, Vertex *toRemove) {
    std::replace(this->vertices.begin(), this->vertices.end(), toRemove, toInsert);
    return toInsert->defines(this) ? S_OK : E_FAIL;
}

HRESULT Surface::add(Body *b) {
    if(b1) {
        if(b1->objectId() == b->objectId()) {
            TF_Log(LOG_ERROR);
            return E_FAIL;
        }

        if(b2) {
            if(b2->objectId() == b->objectId()) {
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

HRESULT Surface::remove(Body *b) {
    if(b1 && b1->objectId() == b->objectId()) 
        b1 = NULL;
    else if(b2 && b2->objectId() == b->objectId()) 
        b2 = NULL;
    else {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
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
    if(this->b1 && this->b1->objectId() == toRemove->objectId()) {
        this->b1 = toInsert;
        return S_OK;
    } 
    else if(this->b2 && this->b2->objectId() == toRemove->objectId()) {
        this->b2 = toInsert;
        return S_OK;
    } 
    return E_FAIL;
}

HRESULT Surface::destroy() {
    if((b1 && b1->destroy() != S_OK) || (b2 && b2->destroy() != S_OK)) 
        return E_FAIL;
    if(this->typeId >= 0 && this->type()->remove(SurfaceHandle(this->_objId)) != S_OK) 
        return E_FAIL;
    std::vector<Vertex*> affectedVertices = getVertices();
    if(this->_objId >= 0) 
        Mesh::get()->remove(this);
    for(auto &v : affectedVertices) 
        v->updateConnectedVertices();
    return S_OK;
}

HRESULT Surface::destroy(Surface *target) {
    auto vertices = target->getVertices();

    std::unordered_set<Vertex*> affectedVertices;
    for(auto &v : vertices) 
        for(auto &nv : v->connectedVertices()) 
            if(!nv->defines(target)) 
                affectedVertices.insert(nv);

    if(target->destroy() != S_OK) 
        return E_FAIL;

    for(auto &v : vertices) 
        if(v->getSurfaces().size() == 0) 
            v->destroy();
    for(auto &v : affectedVertices) 
        v->updateConnectedVertices();

    return S_OK;
}

HRESULT Surface::destroy(SurfaceHandle &target) {
    Surface *_target = target.surface();
    if(!_target) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    HRESULT res = Surface::destroy(_target);
    if(res == S_OK) 
        target.id = -1;
    return res;
}

bool Surface::validate() {
    if(vertices.size() < 3) 
        return false;

    for(auto &v : vertices) 
        if(!v->defines(this) || !this->definedBy(v)) 
            return false;

    return true;
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
                bi = b2;
            }
            else 
                bo = b2;
        }
        else {
            if(bi) {
                TF_Log(LOG_ERROR) << "Two bodies registered on the same side (inside)";
                bo = b2;
            }
            else 
                bi = b2;
        }
    }

    b1 = bo;
    b2 = bi;

    return S_OK;
}

SurfaceType *Surface::type() const {
    if(typeId < 0) 
        return NULL;
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getSurfaceType(typeId);
}

HRESULT Surface::become(SurfaceType *stype) {
    if(this->typeId >= 0 && this->type()->remove(SurfaceHandle(this->_objId)) != S_OK) {
        return tf_error(E_FAIL, "Failed to become");
    }
    return stype->add(SurfaceHandle(this->_objId));
}

HRESULT Surface::insert(Vertex *toInsert, Vertex *v1, Vertex *v2) {
    // Validate input
    if(std::find(vertices.begin(), vertices.end(), toInsert) != vertices.end()) {
        TF_Log(LOG_DEBUG);
        return E_FAIL;
    }

    // Handle wrap
    Vertex *ve = *vertices.rbegin();
    if(((*vertices.begin())->objectId() == v1->objectId() && (*vertices.rbegin())->objectId() == v2->objectId()) || 
        ((*vertices.begin())->objectId() == v2->objectId() && (*vertices.rbegin())->objectId() == v1->objectId())) {
        vertices.insert(vertices.begin(), toInsert);
        return S_OK;
    }

    for(std::vector<Vertex*>::iterator itr = vertices.begin(); itr != vertices.end(); itr++) {
        if((*itr)->objectId() == v1->objectId() || (*itr)->objectId() == v2->objectId()) {
            vertices.insert(itr + 1, toInsert);
            return S_OK;
        }
    }
    TF_Log(LOG_ERROR) << "Vertices not found.";
    return E_FAIL;
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
            result.insert(s);
    result.erase(result.find(const_cast<Surface*>(this)));
    return std::vector<Surface*>(result.begin(), result.end());
}

std::vector<Surface*> Surface::connectedSurfaces() const {
    return connectedSurfaces(vertices);
}

std::vector<Vertex*> Surface::connectingVertices(const Surface *other) const {
    std::vector<Vertex*> result;
    result.reserve(vertices.size());
    for(auto &v : vertices) 
        if(v->defines(other)) 
            result.push_back(v);
    return result;
}

std::vector<unsigned int> Surface::contiguousVertexLabels(const Surface *other) const {
    std::vector<bool> sharedVertices(vertices.size(), false);
    for(unsigned int i = 0; i < sharedVertices.size(); i++) 
        if(vertices[i]->defines(other)) 
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

unsigned int Surface::numSharedContiguousVertexSets(const Surface *other) const {
    unsigned int result = 0;
    for(auto &i : contiguousVertexLabels(other)) 
        result = std::max(result, i);
    return result;
}

std::vector<Vertex*> Surface::sharedContiguousVertices(const Surface *other, const unsigned int &edgeLabel) const {
    std::vector<unsigned int> labs = contiguousVertexLabels(other);
    std::vector<Vertex*> result;
    result.reserve(vertices.size());
    for(unsigned int i = 0; i < labs.size(); i++) 
        if(labs[i] == edgeLabel) 
            result.push_back(vertices[i]);
    return result;
}

FVector3 Surface::getNormal() const { return normal.normalized(); }

FloatP_t Surface::volumeSense(const Body *body) const {
    if(b1 && body->objectId() == b1->objectId()) 
        return 1.f;
    else if(b2 && body->objectId() == b2->objectId()) 
        return -1.f;
    return 0.f;
}

FVector3 Surface::getOutwardNormal(const Body *body) const {
    if(b1 && body->objectId() == b1->objectId()) 
        return getNormal();
    else if(b2 && body->objectId() == b2->objectId()) 
        return getNormal() * -1;
    return FVector3(0);
}

FloatP_t Surface::getVertexArea(const Vertex *v) const {
    FloatP_t result = 0.f;
    
    for(unsigned int i = 0; i < vertices.size(); i++) {
        Vertex *vc = vertices[i];
        Vertex *vn = vertices[Surface_VERTEXINDEX(vertices, i + 1)];

        if(vc->objectId() == v->objectId() || vn->objectId() == v->objectId()) {
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
    return FVector4::planeEquation(getNormal(), centroid).distance(pos);
}

bool Surface::isOutside(const FVector3 &pos) const {
    return normalDistance(pos) > 0;
}

bool Surface::contains(const FVector3 &pos, Vertex **v0, Vertex **v1) const {
    if(std::abs(normalDistance(pos)) > std::numeric_limits<FloatP_t>::min()) 
        return false;

    //  Find the nearest vertex
    Vertex *va = findVertex(pos - centroid);

    //  Find the relevant edges
    Vertex *vb, *vc;
    std::tie(vb, vc) = neighborVertices(va);

    //  Test for penetration

    const FVector3 va_pos = va->getPosition();
    const FVector3 vb_pos = vb->getPosition();

    const FVector3 va_pos_rel = metrics::relativePosition(va_pos, centroid);
    const FVector3 pos_rel = metrics::relativePosition(pos, centroid);

    const FVector3 vb_pos_rel = metrics::relativePosition(vb_pos, centroid);
    const FloatP_t a_va_nb = Magnum::Math::cross(va_pos_rel, pos_rel).length();
    const FVector3 va_pos_rel_nb = metrics::relativePosition(va_pos, pos);

    FloatP_t area, areaTest;
    area = Magnum::Math::cross(va_pos_rel, vb_pos_rel).length();
    areaTest = 
        a_va_nb + 
        Magnum::Math::cross(vb_pos_rel, pos_rel).length() + 
        Magnum::Math::cross(va_pos_rel_nb, metrics::relativePosition(vb_pos, pos)).length()
    ;

    if(areaTest > 0 && abs(area / areaTest - 1) < 1E-6) {
        *v0 = va;
        *v1 = vb;
        return true;
    }

    const FVector3 vc_pos = vc->getPosition();

    const FVector3 vc_pos_rel = metrics::relativePosition(vc_pos, centroid);

    area = Magnum::Math::cross(va_pos_rel, vc_pos_rel).length();
    areaTest = 
        a_va_nb + 
        Magnum::Math::cross(vc_pos_rel, pos_rel).length() + 
        Magnum::Math::cross(va_pos_rel_nb, metrics::relativePosition(vc_pos, pos)).length()
    ;

    if(areaTest > 0 && abs(area / areaTest - 1) < 1E-6) {
        *v0 = va;
        *v1 = vc;
        return true;
    }

    return false;
}

bool Surface::contains(const FVector3 &pos) const {
    Vertex *v0, *v1;
    return contains(pos, &v0, &v1);
}

HRESULT Surface::positionChanged() {
    normal = FVector3(0.f);
    centroid = FVector3(0.f);
    velocity = FVector3(0.f);
    area = 0.f;
    perimeter = 0.f;
    _volumeContr = 0.f;

    for(auto &v : vertices) {
        centroid += v->getPosition();
        velocity += v->getVelocity();
    }
    centroid /= (FloatP_t)vertices.size();
    velocity /= (FloatP_t)vertices.size();

    for(unsigned int i = 0; i < vertices.size(); i++) {
        const FVector3 posc = vertices[i]->getPosition();
        const FVector3 posp = vertices[Surface_VERTEXINDEX(vertices, i + 1)]->getPosition();
        perimeter += (posp - posc).length();

        const FVector3 triNormal = triNorm(posc, centroid, posp);
        _volumeContr += triNormal.dot(centroid);
        area += triNormal.length();
        normal += triNormal;
    }

    area /= 2.f;
    _volumeContr /= 6.f;

    // Handle small surfaces
    // If the normal cannot be determined, 
    // then choose an arbitrary normal and hope for the best
    FVector3 _normal = normal.normalized();
    if(Magnum::Math::isNan(_normal)) {
        if(normal.length() == 0) {
            TF_Log(LOG_DEBUG) << "Zero normal";
            normal = {1.0, 0.0, 0.0};
        }
    } 

    return S_OK;
}

HRESULT Surface::sew(Surface *s1, Surface *s2, const FloatP_t &distCf) {
    if(s1->objectId() == s2->objectId()) 
        return S_OK;

    for(auto &v : s1->vertices) 
        if(v->positionChanged() != S_OK) 
            return E_FAIL;
    for(auto &v : s2->vertices) 
        if(v->positionChanged() != S_OK) 
            return E_FAIL;
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
    std::vector<FloatP_t> dist2Matched(s1->vertices.size(), distCrit2);
    size_t numMatched = 0;
    for(int i = 0; i < s1->vertices.size(); i++) {
        Vertex *vi = s1->vertices[i];
        if(vi->defines(s2)) 
            continue;
        
        auto posi = vi->getPosition();
        FloatP_t minDist2 = distCrit2;
        int matchedIdx = -1;
        for(int j = 0; j < s2->vertices.size(); j++) { 
            Vertex *vj = s2->vertices[j];
            auto dist2 = (s2_positions[j] - posi).dot();
            if(dist2 < minDist2 && !vj->defines(s1)) {
                minDist2 = dist2;
                matchedIdx = j;
            }
        }

        // Prevent duplicates
        if(matchedIdx >= 0) {
            for(int k = 0; k < i; k++) {
                if(indicesMatched[k] == matchedIdx) {
                    if(minDist2 < dist2Matched[k]) {
                        indicesMatched[k] = -1;
                        dist2Matched[k] = distCrit2;
                    }
                    else {
                        matchedIdx = -1;
                        minDist2 = distCrit2;
                    }
                }
            }
        }

        indicesMatched[i] = matchedIdx;
        dist2Matched[i] = minDist2;
        if(matchedIdx >= 0) 
            numMatched++;
    }

    if(numMatched == 0) 
        return S_OK;

    // Merge matched vertices
    std::vector<std::pair<Vertex*, Vertex*> > toMerge;
    for(int i = 0; i < indicesMatched.size(); i++) {
        int vj_idx = indicesMatched[i];
        if(vj_idx >= 0) 
            toMerge.push_back({s1->vertices[i], s2->vertices[vj_idx]});
    }
    for(auto &p : toMerge) {
        Vertex *vi, *vj;
        std::tie(vi, vj) = p;
        vi->merge(vj);
    }

    MeshSolver::log(MeshLogEventType::Create, {s1->_objId, s2->_objId}, {s1->objType(), s2->objType()}, "sew");

    return S_OK;
}

HRESULT Surface::sew(const SurfaceHandle &s1, const SurfaceHandle &s2, const FloatP_t &distCf) {
    Surface *_s1 = s1.surface();
    Surface *_s2 = s2.surface();
    if(!_s1 || !_s2) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return Surface::sew(_s1, _s2, distCf);
}

HRESULT Surface::sew(std::vector<Surface*> _surfaces, const FloatP_t &distCf) {
    for(std::vector<Surface*>::iterator itri = _surfaces.begin(); itri != _surfaces.end() - 1; itri++) 
        for(std::vector<Surface*>::iterator itrj = itri + 1; itrj != _surfaces.end(); itrj++) 
            if(*itri != *itrj && sew(*itri, *itrj, distCf) != S_OK) 
                return E_FAIL;

    return S_OK;
}

HRESULT Surface::sew(std::vector<SurfaceHandle> _surfaces, const FloatP_t &distCf) {
    std::vector<Surface*> surfaces;
    surfaces.reserve(_surfaces.size());
    for(auto &_s : _surfaces) {
        Surface *s = _s.surface();
        if(!s) {
            SurfaceHandle_INVALIDHANDLERR;
            return E_FAIL;
        }
        surfaces.push_back(s);
    }
    return Surface::sew(surfaces, distCf);
}

HRESULT Surface::merge(Surface *toRemove, const std::vector<FloatP_t> &lenCfs) {
    if(vertices.size() != toRemove->vertices.size()) {
        TF_Log(LOG_ERROR) << "Surfaces must have the same number of vertices to merge";
        return E_FAIL;
    }

    // Find vertices that are not shared
    std::vector<Vertex*> toKeepExcl;
    for(auto &v : vertices) 
        if(!v->defines(toRemove)) 
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
                if(!rv->defines(s)) {
                    TF_Log(LOG_ERROR) << "Something went wrong during surface merge";
                    return E_FAIL;
                }
                s->replace(kv, rv);
                kv->add(s);
            }
    }

    // Replace surface in child bodies
    for(auto &b : toRemove->getBodies()) {
        if(!defines(b)) {
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
        if(v->setPosition(newPos, false) != S_OK) 
            return E_FAIL;
    }

    MeshSolver::log(MeshLogEventType::Create, {_objId, toRemove->_objId}, {objType(), toRemove->objType()}, "merge");
    
    // Remove surface and vertices that are not shared
    if(toRemove->destroy() != S_OK) 
        return E_FAIL;
    for(auto &v : toRemoveOrdered) 
        if(v->destroy() != S_OK) 
            return E_FAIL;

    // Update connectivity
    for(auto &v : vertices) {
        v->updateConnectedVertices();
        for(auto &nv : v->connectedVertices()) 
            if(!nv->defines(this)) 
                nv->updateConnectedVertices();
    }
    
    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

Surface *Surface::extend(const unsigned int &vertIdxStart, const FVector3 &pos) {
    // Validate indices
    if(vertIdxStart >= vertices.size()) {
        TF_Log(LOG_ERROR) << "Invalid vertex indices (" << vertIdxStart << ", " << vertices.size() << ")";
        return NULL;
    }
    // Validate state
    if(typeId < 0) {
        TF_Log(LOG_ERROR) << "No type";
        return NULL;
    }

    // Construct new vertex at specified position
    VertexHandle vert = Vertex::create(pos);
    if(!vert) {
        TF_Log(LOG_ERROR) << "Failed to create vertex";
        return NULL;
    }

    // Construct new surface, add new parts and return
    Surface *s = (*type())({
        VertexHandle(vertices[vertIdxStart]->_objId), 
        VertexHandle(vertices[vertIdxStart == vertices.size() - 1 ? 0 : vertIdxStart + 1]->_objId), 
        vert
    }).surface();
    if(!s) {
        vert.destroy();
        tf_error(E_FAIL, "Failed to create surface");
        return NULL;
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {_objId, s->_objId}, {objType(), s->objType()}, "extend");

    return s;
}

Surface *Surface::extrude(const unsigned int &vertIdxStart, const FloatP_t &normLen) {
    Surface_GETMESH(mesh, NULL);

    // Validate indices
    if(vertIdxStart >= vertices.size()) {
        TF_Log(LOG_ERROR) << "Invalid vertex indices (" << vertIdxStart << ", " << vertices.size() << ")";
        return NULL;
    }
    // Validate state
    if(typeId < 0) {
        TF_Log(LOG_ERROR) << "No type";
        return NULL;
    }
    if(mesh->ensureAvailableVertices(2) != S_OK) {
        TF_Log(LOG_ERROR);
        return NULL;
    }

    // Get base vertices
    VertexHandle v0(vertices[vertIdxStart]->_objId);
    VertexHandle v1(vertices[vertIdxStart == vertices.size() - 1 ? 0 : vertIdxStart + 1]->_objId);

    // Construct new vertices
    FVector3 disp = getNormal() * normLen;
    VertexHandle v2 = Vertex::create(v0.getPosition() + disp);
    VertexHandle v3 = Vertex::create(v1.getPosition() + disp);
    if(!v2 || !v3) {
        TF_Log(LOG_ERROR) << "Failed to create a vertex";
        return NULL;
    }

    // Construct new surface, add new parts and return
    Surface *s = (*type())({v0, v1, v2, v3}).surface();
    if(!s) {
        TF_Log(LOG_ERROR) << "Failed to create surface";
        v2.destroy();
        v3.destroy();
        return NULL;
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {_objId, s->_objId}, {objType(), s->objType()}, "extrude");

    return s;
}

Surface *Surface::split(Vertex *v1, Vertex *v2) { 
    // Verify that vertices are in surface
    if(!v1->defines(this) || !v2->defines(this)) { 
        tf_error(E_FAIL, "Vertices are not part of the splitting surface");
        return NULL;
    }

    // Verify that vertices are not adjacent
    const std::vector<Vertex*> v1_nbs = v1->connectedVertices();
    if(std::find(v1_nbs.begin(), v1_nbs.end(), v2) != v1_nbs.end()) {
        tf_error(E_FAIL, "Vertices are adjacent");
        return NULL;
    }
    // Validate state
    if(typeId < 0) {
        TF_Log(LOG_ERROR) << "No type";
        return NULL;
    }

    // Extract vertices for new surface
    std::vector<VertexHandle> v_new_surf;
    v_new_surf.reserve(vertices.size());
    v_new_surf.emplace_back(v1->_objId);
    std::vector<Vertex*>::iterator v_itr = std::find(vertices.begin(), vertices.end(), v1);
    while(true) {
        v_itr++;
        if(v_itr == vertices.end()) 
            v_itr = vertices.begin();
        if((*v_itr)->objectId() == v2->objectId()) 
            break;
        v_new_surf.emplace_back((*v_itr)->_objId);
    }
    v_new_surf.emplace_back(v2->_objId);
    for(auto v_itr = v_new_surf.begin() + 1; v_itr != v_new_surf.end() - 1; v_itr++) {
        Vertex *v = (*v_itr).vertex();
        remove(v);
        v->remove(this);
    }

    // Build new surface
    Surface *s_new = (*type())(v_new_surf).surface();
    if(!s_new) {
        return NULL;
    }

    // Continue hierarchy
    for(auto &b : getBodies()) {
        s_new->add(b);
        b->add(s_new);
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(
        MeshLogEventType::Create, 
        {_objId, s_new->_objId, v1->_objId, v2->_objId}, 
        {objType(), s_new->objType(), v1->objType(), v2->objType()}, 
        "split"
    );

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
    for(int i = 0; i < s_vertices.size(); i++) {
        if(s_vertices[i]->objectId() == (*v_new_start)->objectId()) 
            *v_old_start = i == 0 ? s_vertices.back() : s_vertices[i - 1];
        if(s_vertices[i]->objectId() == (*v_new_end)->objectId()) 
            *v_old_end = i + 1 == s_vertices.size() ? s_vertices.front() : s_vertices[i + 1];
    }

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
    Surface_GETMESH(mesh, NULL);

    if(cp_norm.isZero()) {
        tf_error(E_FAIL, "Zero normal");
        return 0;
    }
    if(mesh->ensureAvailableVertices(2) != S_OK) {
        TF_Log(LOG_ERROR);
        return 0;
    }

    FVector4 planeEq = FVector4::planeEquation(cp_norm.normalized(), cp_pos);

    FVector3 pos_start, pos_end; 
    Vertex *_v_new_start, *_v_old_start, *_v_new_end, *_v_old_end;
    if(Surface_SurfaceCutPlanePointsPairs(this, planeEq, pos_start, pos_end, &_v_new_start, &_v_old_start, &_v_new_end, &_v_old_end) != S_OK) 
        return NULL;

    VertexHandle v_new_start(_v_new_start->_objId);
    VertexHandle v_old_start(_v_old_start->_objId);
    VertexHandle v_new_end(_v_new_end->_objId);
    VertexHandle v_old_end(_v_old_end->_objId);

    // Create and insert new vertices
    VertexHandle v_start = Vertex::create(pos_start);
    VertexHandle v_end   = Vertex::create(pos_end);
    if(!v_start || !v_end || v_start.insert(v_old_start, v_new_start) != S_OK || v_end.insert(v_old_end, v_new_end) != S_OK) {
        v_start.destroy();
        v_end.destroy();
        return NULL;
    }

    // Create new surface
    return split(v_start.vertex(), v_end.vertex());
}


///////////////////
// SurfaceHandle //
///////////////////


SurfaceHandle::SurfaceHandle(const int &_id) : id{_id} {}

Surface *SurfaceHandle::surface() const {
    Surface *o = this->id >= 0 ? Mesh::get()->getSurface(this->id) : NULL;
    if(!o) {
        TF_Log(LOG_ERROR) << "Invalid handle";
    }
    return o;
}

bool SurfaceHandle::defines(const BodyHandle &b) const {
    SurfaceHandle_GETOBJ(o, false);
    Body *_b = b.body();
    if(!_b) {
        SurfaceHandle_INVALIDHANDLERR;
        return false;
    }
    return o->defines(_b);
}

bool SurfaceHandle::definedBy(const VertexHandle &v) const {
    SurfaceHandle_GETOBJ(o, false);
    Vertex *_v = v.vertex();
    if(!_v) {
        SurfaceHandle_INVALIDHANDLERR;
        return false;
    }
    return o->definedBy(_v);
}

HRESULT SurfaceHandle::destroy() {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    HRESULT res = o->destroy();
    if(res == S_OK) 
        this->id = -1;
    return res;
}

bool SurfaceHandle::validate() {
    SurfaceHandle_GETOBJ(o, false);
    return o->validate();
}

HRESULT SurfaceHandle::positionChanged() {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    return o->positionChanged();
}

std::string SurfaceHandle::str() const {
    std::stringstream ss;

    ss << "SurfaceHandle(";
    Surface *_s = this->surface();
    if(_s) {
        ss << "id=" << this->id << ", typeId=" << _s->typeId;
    }
    ss << ")";

    return ss.str();
}

std::string SurfaceHandle::toString() {
    return TissueForge::io::toString(*this);
}

SurfaceHandle SurfaceHandle::fromString(const std::string &s) {
    return TissueForge::io::fromString<SurfaceHandle>(s);
}

HRESULT SurfaceHandle::add(const VertexHandle &v) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Vertex *_v = v.vertex();
    if(!_v) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->add(_v);
}

HRESULT SurfaceHandle::insert(const VertexHandle &v, const int &idx) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Vertex *_v = v.vertex();
    if(!_v) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->insert(_v, idx);
}

HRESULT SurfaceHandle::insert(const VertexHandle &v, const VertexHandle &before) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Vertex *_v = v.vertex();
    Vertex *_before = before.vertex();
    if(!_v || !_before) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->insert(_v, _before);
}

HRESULT SurfaceHandle::insert(const VertexHandle &toInsert, const VertexHandle &v1, const VertexHandle &v2) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Vertex *_toInsert = toInsert.vertex();
    Vertex *_v1 = v1.vertex();
    Vertex *_v2 = v2.vertex();
    if(!_toInsert || !_v1 || !_v2) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->insert(_toInsert, _v1, _v2);
}

HRESULT SurfaceHandle::remove(const VertexHandle &v) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Vertex *_v = v.vertex();
    if(!_v) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->remove(_v);
}

HRESULT SurfaceHandle::replace(const VertexHandle &toInsert, const int &idx) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Vertex *_toInsert = toInsert.vertex();
    if(!_toInsert) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toInsert, idx);
}

HRESULT SurfaceHandle::replace(const VertexHandle &toInsert, const VertexHandle &toRemove) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Vertex *_toInsert = toInsert.vertex();
    Vertex *_toRemove = toRemove.vertex();
    if(!_toInsert || !_toRemove) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toInsert, _toRemove);
}

HRESULT SurfaceHandle::add(const BodyHandle &b) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Body *_b = b.body();
    if(!_b) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->add(_b);
}

HRESULT SurfaceHandle::remove(const BodyHandle &b) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Body *_b = b.body();
    if(!_b) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->remove(_b);
}

HRESULT SurfaceHandle::replace(const BodyHandle &toInsert, const int &idx) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Body *_toInsert = toInsert.body();
    if(!_toInsert) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toInsert, idx);
}

HRESULT SurfaceHandle::replace(const BodyHandle &toInsert, const BodyHandle &toRemove) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Body *_toInsert = toInsert.body();
    Body *_toRemove = toRemove.body();
    if(!_toInsert || !_toRemove) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    return o->replace(_toInsert, _toRemove);
}

HRESULT SurfaceHandle::refreshBodies() {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    return o->refreshBodies();
}

SurfaceType *SurfaceHandle::type() const {
    SurfaceHandle_GETOBJ(o, NULL);
    return o->type();
}

HRESULT SurfaceHandle::become(SurfaceType *stype) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    return o->become(stype);
}

std::vector<BodyHandle> SurfaceHandle::getBodies() const {
    SurfaceHandle_GETOBJ(o, {});
    std::vector<Body*> _result = o->getBodies();
    std::vector<BodyHandle> result;
    result.reserve(_result.size());
    for(auto &_b : _result) 
        result.push_back(_b ? BodyHandle(_b->objectId()) : BodyHandle());
    return result;
}

std::vector<VertexHandle> SurfaceHandle::getVertices() const {
    SurfaceHandle_GETOBJ(o, {});
    std::vector<Vertex*> _result = o->getVertices();
    std::vector<VertexHandle> result;
    result.reserve(_result.size());
    for(auto &_v : _result) 
        result.push_back(_v ? VertexHandle(_v->objectId()) : VertexHandle());
    return result;
}

VertexHandle SurfaceHandle::findVertex(const FVector3 &dir) const {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Vertex *v = o->findVertex(dir);
    return v ? VertexHandle(v->objectId()) : VertexHandle();
}

BodyHandle SurfaceHandle::findBody(const FVector3 &dir) const {
    SurfaceHandle_GETOBJ(o, NULL);
    Body *_b = o->findBody(dir);
    return _b ? BodyHandle(_b->objectId()) : BodyHandle();
}

std::tuple<VertexHandle, VertexHandle> SurfaceHandle::neighborVertices(const VertexHandle &v) const {
    SurfaceHandle_GETOBJ(o, std::make_tuple(VertexHandle(), VertexHandle()));
    Vertex *_v = v.vertex();
    if(!_v) {
        SurfaceHandle_INVALIDHANDLERR;
        return std::make_tuple(VertexHandle(), VertexHandle());
    }
    Vertex *_va, *_vb;
    std::tie(_va, _vb) = o->neighborVertices(_v);
    VertexHandle va = _va ? VertexHandle(_va->objectId()) : VertexHandle();
    VertexHandle vb = _vb ? VertexHandle(_vb->objectId()) : VertexHandle();
    return std::make_tuple(va, vb);
}

std::vector<SurfaceHandle> SurfaceHandle::neighborSurfaces() const {
    SurfaceHandle_GETOBJ(o, {});
    std::vector<Surface*> _result = o->neighborSurfaces();
    std::vector<SurfaceHandle> result;
    result.reserve(_result.size());
    for(auto &_s : _result) 
        result.push_back(_s ? SurfaceHandle(_s->objectId()) : SurfaceHandle());
    return result;
}

std::vector<SurfaceHandle> SurfaceHandle::connectedSurfaces(const std::vector<VertexHandle> &verts) const {
    SurfaceHandle_GETOBJ(o, {});
    std::vector<Vertex*> _verts;
    _verts.reserve(verts.size());
    for(auto &v : verts) {
        Vertex *_v = v.vertex();
        if(!_v) {
            SurfaceHandle_INVALIDHANDLERR;
            return {};
        }
        _verts.push_back(_v);
    }
    std::vector<Surface*> _result = o->connectedSurfaces(_verts);
    std::vector<SurfaceHandle> result;
    result.reserve(_result.size());
    for(auto &_s : _result) 
        result.push_back(_s ? SurfaceHandle(_s->objectId()) : SurfaceHandle());
    return result;
}

std::vector<SurfaceHandle> SurfaceHandle::connectedSurfaces() const {
    SurfaceHandle_GETOBJ(o, {});
    std::vector<Surface*> _result = o->connectedSurfaces();
    std::vector<SurfaceHandle> result;
    result.reserve(_result.size());
    for(auto &_s : _result) 
        result.push_back(_s ? SurfaceHandle(_s->objectId()) : SurfaceHandle());
    return result;
}

std::vector<VertexHandle> SurfaceHandle::connectingVertices(const SurfaceHandle &other) const {
    SurfaceHandle_GETOBJ(o, {});
    Surface *_other = other.surface();
    if(!_other) {
        SurfaceHandle_INVALIDHANDLERR;
        return {};
    }
    std::vector<Vertex*> _result = o->connectingVertices(_other);
    std::vector<VertexHandle> result;
    result.reserve(_result.size());
    for(auto &_s : _result) 
        result.push_back(_s ? VertexHandle(_s->objectId()) : VertexHandle());
    return result;
}

std::vector<unsigned int> SurfaceHandle::contiguousVertexLabels(const SurfaceHandle &other) const {
    SurfaceHandle_GETOBJ(o, {});
    Surface *_other = other.surface();
    if(!_other) {
        SurfaceHandle_INVALIDHANDLERR;
        return {};
    }
    return o->contiguousVertexLabels(_other);
}

unsigned int SurfaceHandle::numSharedContiguousVertexSets(const SurfaceHandle &other) const {
    SurfaceHandle_GETOBJ(o, 0);
    Surface *_other = other.surface();
    if(!_other) {
        SurfaceHandle_INVALIDHANDLERR;
        return {};
    }
    return o->numSharedContiguousVertexSets(_other);
}

std::vector<VertexHandle> SurfaceHandle::sharedContiguousVertices(const SurfaceHandle &other, const unsigned int &edgeLabel) const {
    SurfaceHandle_GETOBJ(o, {});
    Surface *_other = other.surface();
    if(!_other) {
        SurfaceHandle_INVALIDHANDLERR;
        return {};
    }
    std::vector<Vertex*> _result = o->sharedContiguousVertices(_other, edgeLabel);
    std::vector<VertexHandle> result;
    result.reserve(_result.size());
    for(auto &_s : _result) 
        result.push_back(_s ? VertexHandle(_s->objectId()) : VertexHandle());
    return result;
}

FVector3 SurfaceHandle::getNormal() const {
    SurfaceHandle_GETOBJ(o, FVector3());
    return o->getNormal();
}

FVector3 SurfaceHandle::getCentroid() const {
    SurfaceHandle_GETOBJ(o, FVector3());
    return o->getCentroid();
}

FVector3 SurfaceHandle::getUnnormalizedNormal() const {
    SurfaceHandle_GETOBJ(o, FVector3());
    return o->getUnnormalizedNormal();
}

FVector3 SurfaceHandle::getVelocity() const {
    SurfaceHandle_GETOBJ(o, FVector3());
    return o->getVelocity();
}

FloatP_t SurfaceHandle::getArea() const {
    SurfaceHandle_GETOBJ(o, 0);
    return o->getArea();
}

FloatP_t SurfaceHandle::getPerimeter() const {
    SurfaceHandle_GETOBJ(o, 0);
    return o->getPerimeter();
}

FloatP_t SurfaceHandle::getDensity() const {
    SurfaceHandle_GETOBJ(o, 0);
    return o->getDensity();
}

void SurfaceHandle::setDensity(const FloatP_t &_density) const {
    SurfaceHandle_GETOBJ(o, );
    return o->setDensity(_density);
}

FloatP_t SurfaceHandle::getMass() const {
    SurfaceHandle_GETOBJ(o, 0);
    return o->getMass();
}

FloatP_t SurfaceHandle::volumeSense(const BodyHandle &body) const {
    SurfaceHandle_GETOBJ(o, 0);
    Body *_body = body.body();
    if(!_body) {
        SurfaceHandle_INVALIDHANDLERR;
        return 0;
    }
    return o->volumeSense(_body);
}

FloatP_t SurfaceHandle::getVolumeContr(const BodyHandle &body) const {
    SurfaceHandle_GETOBJ(o, 0);
    Body *_body = body.body();
    if(!_body) {
        SurfaceHandle_INVALIDHANDLERR;
        return 0;
    }
    return o->getVolumeContr(_body);
}

FVector3 SurfaceHandle::getOutwardNormal(const BodyHandle &body) const {
    SurfaceHandle_GETOBJ(o, FVector3());
    Body *_body = body.body();
    if(!_body) {
        SurfaceHandle_INVALIDHANDLERR;
        return FVector3();
    }
    return o->getOutwardNormal(_body);
}

FloatP_t SurfaceHandle::getVertexArea(const VertexHandle &v) const {
    SurfaceHandle_GETOBJ(o, 0);
    Vertex *_v = v.vertex();
    if(!_v) {
        SurfaceHandle_INVALIDHANDLERR;
        return 0;
    }
    return o->getVertexArea(_v);
}

FloatP_t SurfaceHandle::getVertexMass(const VertexHandle &v) const {
    SurfaceHandle_GETOBJ(o, 0);
    Vertex *_v = v.vertex();
    if(!_v) {
        SurfaceHandle_INVALIDHANDLERR;
        return 0;
    }
    return o->getVertexMass(_v);
}

state::StateVector *SurfaceHandle::getSpeciesOutward() const {
    SurfaceHandle_GETOBJ(o, 0);
    return o->species1;
}

HRESULT SurfaceHandle::setSpeciesOutward(state::StateVector *s) const {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    o->species1 = s;
    return S_OK;
}

state::StateVector *SurfaceHandle::getSpeciesInward() const {
    SurfaceHandle_GETOBJ(o, 0);
    return o->species2;
}

HRESULT SurfaceHandle::setSpeciesInward(state::StateVector *s) const {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    o->species2 = s;
    return S_OK;
}

rendering::Style *SurfaceHandle::getStyle() const {
    SurfaceHandle_GETOBJ(o, 0);
    return o->style;
}

HRESULT SurfaceHandle::setStyle(rendering::Style *s) const {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    o->style = s;
    return S_OK;
}

FVector3 SurfaceHandle::triangleNormal(const unsigned int &idx) const {
    SurfaceHandle_GETOBJ(o, FVector3());
    return o->triangleNormal(idx);
}

FloatP_t SurfaceHandle::normalDistance(const FVector3 &pos) const {
    SurfaceHandle_GETOBJ(o, 0);
    return o->normalDistance(pos);
}

bool SurfaceHandle::isOutside(const FVector3 &pos) const {
    SurfaceHandle_GETOBJ(o, false);
    return o->isOutside(pos);
}

bool SurfaceHandle::contains(const FVector3 &pos, VertexHandle &v0, VertexHandle &v1) const {
    SurfaceHandle_GETOBJ(o, false);
    Vertex *_v0, *_v1;
    bool result = o->contains(pos, &_v0, &_v1);
    if(result) {
        v0.id = _v0->objectId();
        v1.id = _v1->objectId();
    }
    return result;
}

bool SurfaceHandle::contains(const FVector3 &pos) const {
    SurfaceHandle_GETOBJ(o, false);
    return o->contains(pos);
}

HRESULT SurfaceHandle::merge(SurfaceHandle &toRemove, const std::vector<FloatP_t> &lenCfs) {
    SurfaceHandle_GETOBJ(o, E_FAIL);
    Surface *_toRemove = toRemove.surface();
    if(!_toRemove) {
        SurfaceHandle_INVALIDHANDLERR;
        return E_FAIL;
    }
    HRESULT res = o->merge(_toRemove, lenCfs);
    if(res == S_OK) 
        toRemove.id = -1;
    return res;
}

SurfaceHandle SurfaceHandle::extend(const unsigned int &vertIdxStart, const FVector3 &pos) {
    Surface_GETMESH(mesh, SurfaceHandle());
    if(mesh->ensureAvailableSurfaces(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return SurfaceHandle();
    }
    SurfaceHandle_GETOBJ(o, SurfaceHandle());
    Surface *s = o->extend(vertIdxStart, pos);
    return s ? SurfaceHandle(s->objectId()) : SurfaceHandle();
}

SurfaceHandle SurfaceHandle::extrude(const unsigned int &vertIdxStart, const FloatP_t &normLen) {
    Surface_GETMESH(mesh, SurfaceHandle());
    if(mesh->ensureAvailableSurfaces(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return SurfaceHandle();
    }
    SurfaceHandle_GETOBJ(o, SurfaceHandle());
    Surface *s = o->extrude(vertIdxStart, normLen);
    return s ? SurfaceHandle(s->objectId()) : SurfaceHandle();
}

SurfaceHandle SurfaceHandle::split(const VertexHandle &v1, const VertexHandle &v2) {
    Surface_GETMESH(mesh, SurfaceHandle());
    if(mesh->ensureAvailableSurfaces(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return SurfaceHandle();
    }
    SurfaceHandle_GETOBJ(o, SurfaceHandle());
    Vertex *_v1 = v1.vertex();
    Vertex *_v2 = v2.vertex();
    if(!_v1 || !_v2) {
        SurfaceHandle_INVALIDHANDLERR;
        return SurfaceHandle();
    }
    Surface *s = o->split(_v1, _v2);
    return s ? SurfaceHandle(s->objectId()) : SurfaceHandle();
}

SurfaceHandle SurfaceHandle::split(const FVector3 &cp_pos, const FVector3 &cp_norm) {
    Surface_GETMESH(mesh, SurfaceHandle());
    if(mesh->ensureAvailableSurfaces(1) != S_OK) {
        TF_Log(LOG_ERROR);
        return SurfaceHandle();
    }
    SurfaceHandle_GETOBJ(o, SurfaceHandle());
    Surface *s = o->split(cp_pos, cp_norm);
    return s ? SurfaceHandle(s->objectId()) : SurfaceHandle();
}


/////////////////
// SurfaceType //
/////////////////


SurfaceType::SurfaceType(const FloatP_t &flatLam, const FloatP_t &convexLam, const bool &noReg) : 
    MeshObjType() 
{
    name = "Surface";
    density = 0.f;

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

HRESULT SurfaceType::add(const SurfaceHandle &i) {
    if(!i) 
        return tf_error(E_FAIL, "Invalid object");
    Surface *_i = i.surface();
    if(!_i) 
        return tf_error(E_FAIL, "Object not registered");

    SurfaceType *iType = _i->type();
    if(iType) 
        iType->remove(i);

    _i->typeId = this->id;
    this->_instanceIds.push_back(i.id);
    return S_OK;
}

HRESULT SurfaceType::remove(const SurfaceHandle &i) {
    if(!i) 
        return tf_error(E_FAIL, "Invalid object");
    Surface *_i = i.surface();
    if(!_i) 
        return tf_error(E_FAIL, "Object not registered");

    auto itr = std::find(this->_instanceIds.begin(), this->_instanceIds.end(), i.id);
    if(itr == this->_instanceIds.end()) 
        return tf_error(E_FAIL, "Instance not of this type");

    this->_instanceIds.erase(itr);
    _i->typeId = -1;
    return S_OK;
}

std::vector<SurfaceHandle> SurfaceType::getInstances() {
    std::vector<SurfaceHandle> result;

    Mesh *m = Mesh::get();
    if(m) { 
        result.reserve(_instanceIds.size());
        for(size_t i = 0; i < m->sizeSurfaces(); i++) {
            Surface *s = m->getSurface(i);
            if(s && s->typeId == this->id) 
                result.emplace_back(s->objectId());
        }
    }

    return result;
}

unsigned int SurfaceType::getNumInstances() {
    return getInstances().size();
}

static SurfaceHandle SurfaceType_fromVertices(SurfaceType *stype, const std::vector<VertexHandle> &vertices) {
    SurfaceHandle s = Surface::create(vertices);
    Surface *_s = s.surface();
    if(!_s || stype->add(s) != S_OK) {
        TF_Log(LOG_ERROR) << "Failed to create instance";
        if(_s) 
            s.destroy();
        return SurfaceHandle();
    }
    _s->setDensity(stype->density);
    return s;
}

SurfaceHandle SurfaceType::operator() (const std::vector<VertexHandle> &_vertices) {
    return SurfaceType_fromVertices(this, _vertices);
}

SurfaceHandle SurfaceType::operator() (const std::vector<FVector3> &_positions) {
    Surface_GETMESH(mesh, NULL);
    if(mesh->ensureAvailableVertices(_positions.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return NULL;
    }

    std::vector<VertexHandle> _vertices;
    bool failed = false;
    for(auto &p : _positions) {
        VertexHandle v = Vertex::create(p);
        if(!v) {
            TF_Log(LOG_ERROR) << "Failed to create a vertex";
            failed = true;
            break;
        }
        _vertices.push_back(v);
    }

    if(failed) {
        for(auto &v : _vertices) {
            v.vertex()->destroy();
        }
        return NULL;
    }
    
    TF_Log(LOG_DEBUG) << "Created " << _vertices.size() << " vertices";
    
    return (*this)(_vertices);
}

SurfaceHandle SurfaceType::operator() (TissueForge::io::ThreeDFFaceData *face) {
    std::vector<TissueForge::io::ThreeDFVertexData*> vverts;
    std::vector<FVector3> _positions;
    if(Surface_order3DFFaceVertices(face, vverts) == S_OK) 
        for(auto &vv : vverts) 
            _positions.push_back(vv->position);
    
    return (*this)(_positions);
}

SurfaceHandle SurfaceType::nPolygon(const unsigned int &n, const FVector3 &center, const FloatP_t &radius, const FVector3 &ax1, const FVector3 &ax2) {
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

SurfaceHandle SurfaceType::replace(VertexHandle &toReplace, std::vector<FloatP_t> lenCfs) {
    Surface_GETMESH(mesh, SurfaceHandle());

    if(!toReplace) {
        SurfaceHandle_INVALIDHANDLERR;
        return SurfaceHandle();
    }

    std::vector<VertexHandle> neighbors = toReplace.connectedVertices();
    if(lenCfs.size() != neighbors.size()) {
        TF_Log(LOG_ERROR) << "Length coefficients are inconsistent with connectivity";
        return SurfaceHandle();
    } 

    for(auto &cf : lenCfs) 
        if(cf <= 0.f || cf >= 1.f) {
            TF_Log(LOG_ERROR) << "Length coefficients must be in (0, 1)";
            return SurfaceHandle();
        }

    if(mesh->ensureAvailableVertices(neighbors.size()) != S_OK) {
        TF_Log(LOG_ERROR);
        return SurfaceHandle();
    }

    // Insert new vertices
    Vertex *_toReplace = toReplace.vertex();
    FVector3 pos0 = _toReplace->getPosition();
    std::vector<VertexHandle> insertedVertices;
    for(unsigned int i = 0; i < neighbors.size(); i++) {
        FloatP_t cf = lenCfs[i];
        if(cf <= 0.f || cf >= 1.f) {
            TF_Log(LOG_ERROR) << "Length coefficients must be in (0, 1)";
            return SurfaceHandle();
        }

        VertexHandle v = neighbors[i];
        FVector3 pos1 = v.getPosition();
        FVector3 pos = pos0 + (pos1 - pos0) * cf;
        VertexHandle vInserted = Vertex::create(pos);
        if(vInserted.insert(toReplace, v) != S_OK) 
            return SurfaceHandle();
        insertedVertices.push_back(vInserted);
    }

    // Disconnect replaced vertex from all surfaces
    _toReplace = toReplace.vertex();
    std::vector<Surface*> toReplaceSurfaces = _toReplace->getSurfaces();
    std::vector<Vertex*> affectedVertices = _toReplace->connectedVertices();
    for(auto &s : toReplaceSurfaces) {
        s->remove(_toReplace);
        _toReplace->remove(s);
    }
    for(auto &v : affectedVertices) 
        v->updateConnectedVertices();

    // Create new surface; its constructor should handle internal connections
    SurfaceHandle inserted = (*this)(insertedVertices);
    if(!inserted) {
        return SurfaceHandle();
    }

    // Remove replaced vertex from the mesh and add inserted surface to the mesh

    MeshSolver::log(MeshLogEventType::Create, {inserted.id, toReplace.id}, {inserted.objType(), toReplace.objType()}, "replace"); 
    toReplace.destroy();
    toReplace.id = -1;

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    return inserted;
}


////////
// io //
////////


namespace TissueForge::io {


    template <>
    HRESULT toFile(TissueForge::models::vertex::Surface *dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "objId", dataElement->objectId());
        TF_IOTOEASY(fileElement, metaData, "typeId", dataElement->typeId);

        if(dataElement->actors.size() > 0) {
            std::vector<TissueForge::models::vertex::MeshObjActor*> actors;
            for(auto &a : dataElement->actors) 
                if(a) 
                    actors.push_back(a);
            TF_IOTOEASY(fileElement, metaData, "actors", actors);
        }

        std::vector<int> vertices;
        for(auto &v : dataElement->getVertices()) 
            vertices.push_back(v->objectId());
        TF_IOTOEASY(fileElement, metaData, "vertices", vertices);

        std::vector<int> bodies;
        for(auto &b : dataElement->getBodies()) 
            bodies.push_back(b->objectId());
        TF_IOTOEASY(fileElement, metaData, "bodies", bodies);

        TF_IOTOEASY(fileElement, metaData, "normal", dataElement->getNormal());
        TF_IOTOEASY(fileElement, metaData, "centroid", dataElement->getCentroid());
        TF_IOTOEASY(fileElement, metaData, "velocity", dataElement->getVelocity());
        TF_IOTOEASY(fileElement, metaData, "area", dataElement->getArea());

        TF_IOTOEASY(fileElement, metaData, "typeId", dataElement->typeId);

        if(dataElement->species1) {
            TF_IOTOEASY(fileElement, metaData, "species_outward", *dataElement->species1);
        }
        if(dataElement->species2) {
            TF_IOTOEASY(fileElement, metaData, "species_inward", *dataElement->species2);
        }
        if(dataElement->style) {
            TF_IOTOEASY(fileElement, metaData, "style", *dataElement->style);
        }

        fileElement.get()->type = "Surface";

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
        Surface_GETMESH(mesh, E_FAIL);

        int typeIdOld;
        TF_IOFROMEASY(fileElement, metaData, "typeId", &typeIdOld);
        auto typeId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.find(typeIdOld);
        if(typeId_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.end()) {
            return tf_error(E_FAIL, "Could not identify type");
        }
        TissueForge::models::vertex::SurfaceType *detype = solver->getSurfaceType(typeId_itr->second);

        std::vector<TissueForge::models::vertex::VertexHandle> vertices;
        std::vector<int> verticesIds;
        TF_IOFROMEASY(fileElement, metaData, "vertices", &verticesIds);
        for(auto &vertexIdOld : verticesIds) {
            auto vertexId_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->vertexIdMap.find(vertexIdOld);
            if(vertexId_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->vertexIdMap.end()) {
                return tf_error(E_FAIL, "Could not identify vertex");
            }
            vertices.emplace_back(vertexId_itr->second);
        }

        *dataElement = (*detype)(vertices).surface();

        if(!(*dataElement)) {
            return tf_error(E_FAIL, "Failed to add surface");
        }

        int objIdOld;
        TF_IOFROMEASY(fileElement, metaData, "objId", &objIdOld);
        TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceIdMap.insert({objIdOld, (*dataElement)->objectId()});

        IOChildMap fec = IOElement::children(fileElement);
        auto feItr = fec.find("actors");
        if(feItr != fec.end()) {
            TF_IOFROMEASY(fileElement, metaData, "actors", &(*dataElement)->actors);
        }

        return S_OK;
    }

    template <>
    HRESULT toFile(const TissueForge::models::vertex::SurfaceHandle &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);

        fileElement.get()->type = "SurfaceHandle";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::SurfaceHandle *dataElement) {

        IOChildMap::const_iterator feItr;

        int id;
        TF_IOFROMEASY(fileElement, metaData, "id", &id);

        *dataElement = TissueForge::models::vertex::SurfaceHandle(id);

        return S_OK;
    }

    template <>
    HRESULT toFile(const TissueForge::models::vertex::SurfaceType &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);

        if(dataElement.actors.size() > 0) {
            std::vector<TissueForge::models::vertex::MeshObjActor*> actors;
            for(auto &a : dataElement.actors) 
                if(a) 
                    actors.push_back(a);
            TF_IOTOEASY(fileElement, metaData, "actors", actors);
        }

        TF_IOTOEASY(fileElement, metaData, "name", dataElement.name);
        TF_IOTOEASY(fileElement, metaData, "style", *dataElement.style);

        fileElement.get()->type = "SurfaceType";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::SurfaceType **dataElement) {
        
        *dataElement = new TissueForge::models::vertex::SurfaceType();
        (*dataElement)->actors.clear();

        TF_IOFROMEASY(fileElement, metaData, "name", &(*dataElement)->name);
        IOChildMap fec = IOElement::children(fileElement);
        if(fec.find("actors") != fec.end()) {
            TF_IOFROMEASY(fileElement, metaData, "actors", &(*dataElement)->actors);
        }

        return S_OK;
    }
}

std::string TissueForge::models::vertex::Surface::toString() {
    TissueForge::io::IOElement el = TissueForge::io::IOElement::create();
    std::string result;
    if(TissueForge::io::toFile(this, TissueForge::io::MetaData(), el) == S_OK) 
        result = TissueForge::io::toStr(el);
    else 
        result = "";
    return result;
}

std::string TissueForge::models::vertex::SurfaceType::toString() {
    return TissueForge::io::toString(*this);
}

SurfaceType *TissueForge::models::vertex::SurfaceType::fromString(const std::string &str) {
    return TissueForge::io::fromString<TissueForge::models::vertex::SurfaceType*>(str);
}
