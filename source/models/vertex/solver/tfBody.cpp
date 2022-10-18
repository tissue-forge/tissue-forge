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

std::vector<MeshObj*> Body::parents() { return TissueForge::models::vertex::vectorToBase(surfaces); }

std::vector<MeshObj*> Body::children() { return TissueForge::models::vertex::vectorToBase(structures); }

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

BodyType *Body::type() {
    MeshSolver *solver = MeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getBodyType(typeId);
}

std::vector<Structure*> Body::getStructures() {
    std::unordered_set<Structure*> result;
    for(auto &s : structures) {
        result.insert(s);
        for(auto &ss : s->getStructures()) 
            result.insert(ss);
    }
    return std::vector<Structure*>(result.begin(), result.end());
}

std::vector<Vertex*> Body::getVertices() {
    std::unordered_set<Vertex*> result;

    for(auto &s : surfaces) 
        for(auto &v : s->vertices) 
            result.insert(v);

    return std::vector<Vertex*>(result.begin(), result.end());
}

Vertex *Body::findVertex(const FVector3 &dir) {
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

Surface *Body::findSurface(const FVector3 &dir) {
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

std::vector<Body*> Body::neighborBodies() {
    std::unordered_set<Body*> result;
    for(auto &s : surfaces) 
        for(auto &b : s->getBodies()) 
            if(b != this) 
                result.insert(b);
    return std::vector<Body*>(result.begin(), result.end());
}

std::vector<Surface*> Body::neighborSurfaces(Surface *s) {
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

FVector3 Body::getVelocity() {
    FVector3 result;
    for(auto &v : getVertices()) 
        result += v->particle()->getVelocity() * getVertexMass(v);
    return result / getMass();
}

FloatP_t Body::getVertexArea(Vertex *v) {
    FloatP_t result;
    for(auto &s : surfaces) 
        result += s->getVertexArea(v);
    return result;
}

FloatP_t Body::getVertexVolume(Vertex *v) {
    if(area == 0.f) 
        return 0.f;
    return getVertexArea(v) / area * volume;
}

std::vector<Surface*> Body::findInterface(Body *b) {
    std::vector<Surface*> result;
    for(auto &s : surfaces) 
        if(s->in(b)) 
            result.push_back(s);
    return result;
}

FloatP_t Body::contactArea(Body *other) {
    FloatP_t result = 0.f;
    for(auto &s : surfaces) 
        if(s->in(other)) 
            result += s->area;
    return result;
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
