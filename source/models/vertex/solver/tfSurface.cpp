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

#include <Magnum/Math/Math.h>

#include <tfLogger.h>
#include <tf_util.h>
#include <tf_metrics.h>

#include <io/tfThreeDFVertexData.h>
#include <io/tfThreeDFEdgeData.h>


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
    style{NULL}
{}

Surface::Surface(std::vector<Vertex*> _vertices) : 
    Surface()
{
    if(_vertices.size() >= 3) {
        for(auto &v : _vertices) {
            addParent(v);
            v->addChild(this);
        }
    } 
    else {
        TF_Log(LOG_ERROR) << "Surfaces require at least 3 vertices (" << _vertices.size() << " given)";
    }
}

static HRESULT Surface_order3DFFaceVertices(io::ThreeDFFaceData *face, std::vector<io::ThreeDFVertexData*> &result) {
    auto vedges = face->getEdges();
    auto vverts = face->getVertices();
    result.clear();
    std::vector<int> edgesLeft;
    for(int i = 1; i < vedges.size(); edgesLeft.push_back(i), i++) {}
    
    io::ThreeDFVertexData *currentVertex;
    io::ThreeDFEdgeData *edge = vedges[0];
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

Surface::Surface(io::ThreeDFFaceData *face) : 
    Surface()
{
    std::vector<io::ThreeDFVertexData*> vverts;
    if(Surface_order3DFFaceVertices(face, vverts) == S_OK) {
        for(auto &vv : vverts) {
            Vertex *v = new Vertex(vv);
            addParent(v);
            v->addChild(this);
        }
    }
}

std::vector<MeshObj*> Surface::children() {
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
        if(n.dot(normal) > 0) 
            bo = b2;
        else 
            bi = b2;
    }

    b1 = bo;
    b2 = bi;

    return S_OK;
}

SurfaceType *Surface::type() {
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getSurfaceType(typeId);
}

HRESULT Surface::insert(Vertex *toInsert, Vertex *v1, Vertex *v2) {
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

std::vector<Structure*> Surface::getStructures() {
    std::vector<Structure*> result;
    if(b1) 
        for(auto &s : b1->getStructures()) 
            result.push_back(s);
    if(b2) 
        for(auto &s : b2->getStructures()) 
            result.push_back(s);
    return util::unique(result);
}

std::vector<Body*> Surface::getBodies() {
    std::vector<Body*> result;
    if(b1) 
        result.push_back(b1);
    if(b2) 
        result.push_back(b2);
    return result;
}

Vertex *Surface::findVertex(const FVector3 &dir) {
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

Body *Surface::findBody(const FVector3 &dir) {
    Body *result = 0;

    FVector3 pta = centroid;
    FVector3 ptb = pta + dir;
    FloatP_t bestDist2 = 0;

    for(auto &b : getBodies()) {
        FVector3 pt = b->getCentroid();
        FloatP_t dist2 = Magnum::Math::Distance::linePointSquared(pta, ptb, pt);
        if((!result || dist2 <= bestDist2) && dir.dot(pt - pta) >= 0.f) { 
            result = b;
            bestDist2 = dist2;
        }
    }

    return result;
}

std::tuple<Vertex*, Vertex*> Surface::neighborVertices(Vertex *v) {
    Vertex *vp = NULL;
    Vertex *vn = NULL; 

    Vertex *vo;
    for(auto itr = vertices.begin(); itr != vertices.end(); itr++) {
        if(*itr == v) {
            vp = itr + 1 == vertices.end() ? *vertices.begin()     : *(itr + 1);
            vn = itr == vertices.begin()   ? *(vertices.end() - 1) : *(itr - 1);
            break;
        }
    }

    return {vp, vn};
}

std::vector<Surface*> Surface::neighborSurfaces() { 
    std::vector<Surface*> result;
    if(b1) 
        for(auto &s : b1->neighborSurfaces(this)) 
            result.push_back(s);
    if(b2) 
        for(auto &s : b2->neighborSurfaces(this)) 
            result.push_back(s);
    return util::unique(result);
}

std::vector<unsigned int> Surface::contiguousEdgeLabels(Surface *other) {
    std::vector<Vertex*> overtices = TissueForge::models::vertex::vectorToDerived<Vertex>(other->parents());
    std::vector<bool> sharedVertices(vertices.size(), false);
    for(unsigned int i = 0; i < sharedVertices.size(); i++) 
        if(std::find(overtices.begin(), overtices.end(), vertices[i]) != overtices.end()) 
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

unsigned int Surface::numSharedContiguousEdges(Surface *other) {
    unsigned int result = 0;
    for(auto &i : contiguousEdgeLabels(other)) 
        result = std::max(result, i);
    return result;
}

FloatP_t Surface::volumeSense(Body *body) {
    if(body == b1) 
        return 1.f;
    else if(body == b2) 
        return -1.f;
    return 0.f;
}

FloatP_t Surface::getVertexArea(Vertex *v) {
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

FVector3 Surface::triangleNormal(const unsigned int &idx) {
    return triNorm(vertices[idx]->getPosition(), 
                   centroid, 
                   vertices[Surface_VERTEXINDEX(vertices, idx + 1)]->getPosition());
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

    normal = normal.normalized();
    area /= 2.f;
    _volumeContr /= 6.f;

    return S_OK;
}

HRESULT Surface::sew(Surface *s1, Surface *s2, const FloatP_t &distCf) {
    if(s1 == s2) 
        return S_OK;

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
                        if(!vi->has(s) && vi->addChild(s) != S_OK)
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
        if(vj->mesh && vj->mesh->remove(vj) != S_OK) 
            return E_FAIL;
        delete vj;
    }

    return S_OK;
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

Surface *SurfaceType::operator() (io::ThreeDFFaceData *face) {
    std::vector<io::ThreeDFVertexData*> vverts;
    std::vector<FVector3> _positions;
    if(Surface_order3DFFaceVertices(face, vverts) == S_OK) 
        for(auto &vv : vverts) 
            _positions.push_back(vv->position);
    
    return (*this)(_positions);
}

Surface *SurfaceType::nPolygon(const unsigned int &n, const FVector3 &center, const FloatP_t &radius, const FVector3 &ax1, const FVector3 &ax2) {
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
