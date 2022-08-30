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

#include "tfVertex.h"

#include "tfSurface.h"
#include "tfBody.h"
#include "tfStructure.h"
#include "tfMeshSolver.h"

#include <tfLogger.h>
#include <tfEngine.h>
#include <tf_util.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


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

std::vector<Vertex*> Vertex::neighborVertices() {
    std::vector<Vertex*> result;

    for(auto &s : surfaces) 
        for(auto &v : s->neighborVertices(this)) 
            result.push_back(v);
    return util::unique(result);
}

std::vector<Surface*> Vertex::sharedSurfaces(Vertex *other) {
    std::vector<Surface*> result;
    for(auto &s : surfaces) 
        if(other->in(s) && std::find(result.begin(), result.end(), s) == result.end()) 
            result.push_back(s);
    return result;
}

float Vertex::getVolume() {
    float result = 0.f;
    for(auto &b : getBodies()) 
        result += b->getVertexVolume(this);
    return result;
}

float Vertex::getMass() {
    float result = 0.f;
    for(auto &b : getBodies()) 
        result += b->getVertexMass(this);
    return result;
}

HRESULT Vertex::positionChanged() {
    return S_OK;
}

HRESULT Vertex::updateProperties() {
    ParticleHandle *p = particle();
    const float vMass = getMass();
    if(p && vMass > 0.f) {
        p->setMass(vMass);
    }
    return S_OK;
}

ParticleHandle *Vertex::particle() {
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

FVector3 Vertex::getPosition() {
    auto p = particle();
    if(!p) { 
        TF_Log(LOG_ERROR) << "No assigned particle.";
        FVector3(-1.f, -1.f, -1.f);
    }
    return p->getPosition();
}

HRESULT Vertex::setPosition(const FVector3 &pos) {
    auto p = particle();
    if(!p) {
        TF_Log(LOG_ERROR) << "No assigned particle.";
        return E_FAIL;
    }
    p->setPosition(pos);

    for(auto &s : surfaces) 
        s->positionChanged();

    return S_OK;
}

Vertex::Vertex() : 
    MeshObj(), 
    pid{-1}
{}

Vertex::Vertex(const unsigned int &_pid) : 
    Vertex() 
{
    pid = (int)_pid;
};

Vertex::Vertex(const FVector3 &position) : 
    Vertex()
{
    MeshParticleType *ptype = MeshParticleType_get();
    if(!ptype) {
        TF_Log(LOG_ERROR) << "Could not instantiate particle type";
        this->pid = -1;
    } 
    else {
        FVector3 _position = position;
        ParticleHandle *ph = (*ptype)(&_position);
        this->pid = ph->id;
    }
}

Vertex::Vertex(io::ThreeDFVertexData *vdata) :
    Vertex(vdata->position)
{}

Vertex::~Vertex() {
    if(this->pid >= 0) 
        this->particle()->destroy();
}

std::vector<MeshObj*> Vertex::children() {
    return TissueForge::models::vertex::vectorToBase(surfaces);
}

HRESULT Vertex::addChild(MeshObj *obj) {
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

HRESULT Vertex::removeChild(MeshObj *obj) {
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

std::vector<Structure*> Vertex::getStructures() {
    std::vector<Structure*> result;
    for(auto &s : surfaces) 
        for(auto &ss : s->getStructures()) 
            result.push_back(ss);
    return util::unique(result);
}

std::vector<Body*> Vertex::getBodies() {
    std::vector<Body*> result;
    for(auto &s : surfaces) 
        for(auto &b : s->getBodies()) 
            result.push_back(b);
    return util::unique(result);
}

Surface *Vertex::findSurface(const FVector3 &dir) {
    Surface *result = 0;

    FVector3 pta = getPosition();
    FVector3 ptb = pta + dir;
    float bestDist2 = 0;

    for(auto &s : getSurfaces()) {
        FVector3 pt = s->getCentroid();
        float dist2 = Magnum::Math::Distance::linePointSquared(pta, ptb, pt);
        if((!result || dist2 <= bestDist2) && dir.dot(pt - pta) >= 0.f) { 
            result = s;
            bestDist2 = dist2;
        }
    }

    return result;
}

Body *Vertex::findBody(const FVector3 &dir) {
    Body *result = 0;

    FVector3 pta = getPosition();
    FVector3 ptb = pta + dir;
    float bestDist2 = 0;

    for(auto &b : getBodies()) {
        FVector3 pt = b->getCentroid();
        float dist2 = Magnum::Math::Distance::linePointSquared(pta, ptb, pt);
        if((!result || dist2 <= bestDist2) && dir.dot(pt - pta) >= 0.f) { 
            result = b;
            bestDist2 = dist2;
        }
    }

    return result;
}