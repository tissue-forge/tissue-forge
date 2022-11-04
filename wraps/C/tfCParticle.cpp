/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

#include "tfCParticle.h"

#include "TissueForge_c_private.h"

#include "tfCCluster.h"
#include "tfCStateVector.h"

#include <tfParticle.h>
#include <tfParticleList.h>
#include <tfParticleTypeList.h>
#include <state/tfSpeciesList.h>
#include <rendering/tfStyle.h>
#include <tfEngine.h>


using namespace TissueForge;


namespace TissueForge { 


    ParticleHandle *castC(struct tfParticleHandleHandle *handle) {
        return castC<ParticleHandle, tfParticleHandleHandle>(handle);
    }

    ParticleType *castC(struct tfParticleTypeHandle *handle) {
        return castC<ParticleType, tfParticleTypeHandle>(handle);
    }

    ParticleList *castC(struct tfParticleListHandle *handle) {
        return castC<ParticleList, tfParticleListHandle>(handle);
    }

    ParticleTypeList *castC(struct tfParticleTypeListHandle *handle) {
        return castC<ParticleTypeList, tfParticleTypeListHandle>(handle);
    }

}

#define TFC_PARTICLEHANDLE_GET(handle) \
    ParticleHandle *phandle = TissueForge::castC<ParticleHandle, tfParticleHandleHandle>(handle); \
    TFC_PTRCHECK(phandle);

#define TFC_PARTICLEHANDLE_GETN(handle, name) \
    ParticleHandle *name = TissueForge::castC<ParticleHandle, tfParticleHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_PTYPEHANDLE_GET(handle) \
    ParticleType *ptype = TissueForge::castC<ParticleType, tfParticleTypeHandle>(handle); \
    TFC_PTRCHECK(ptype);

#define TFC_PARTICLELIST_GET(handle) \
    ParticleList *plist = TissueForge::castC<ParticleList, tfParticleListHandle>(handle); \
    TFC_PTRCHECK(plist);

#define TFC_PARTICLETYPELIST_GET(handle) \
    ParticleTypeList *ptlist = TissueForge::castC<ParticleTypeList, tfParticleTypeListHandle>(handle); \
    TFC_PTRCHECK(ptlist);


////////////////////////
// tfParticleTypeSpec //
////////////////////////


struct tfParticleTypeSpec tfParticleTypeSpec_init() {
    struct tfParticleTypeSpec particleTypeSpec = {
        1.0, 
        0.0, 
        1.0, 
        NULL, 
        0.0, 
        0.0, 
        0.0, 
        0, 
        0, 
        NULL, 
        NULL, 
        NULL, 
        0, 
        NULL
    };
    return particleTypeSpec;
}


/////////////////////////////
// tfParticleTypeStyleSpec //
/////////////////////////////


struct tfParticleTypeStyleSpec tfParticleTypeStyleSpec_init() {
    struct tfParticleTypeStyleSpec particleTypeStyleSpec = {
        NULL, 
        1, 
        NULL, 
        "rainbow", 
        0.0, 
        1.0
    };
    return particleTypeStyleSpec;
}


//////////////////////
// ParticleDynamics //
//////////////////////


HRESULT tfParticleDynamics_init(struct tfParticleDynamicsEnumHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->PARTICLE_NEWTONIAN = PARTICLE_NEWTONIAN;
    handle->PARTICLE_OVERDAMPED = PARTICLE_OVERDAMPED;
    return S_OK;
}


///////////////////
// ParticleFlags //
///////////////////


HRESULT tfParticleFlags_init(struct tfParticleFlagsHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->PARTICLE_NONE = PARTICLE_NONE;
    handle->PARTICLE_GHOST = PARTICLE_GHOST;
    handle->PARTICLE_CLUSTER = PARTICLE_CLUSTER;
    handle->PARTICLE_BOUND = PARTICLE_BOUND;
    handle->PARTICLE_FROZEN_X = PARTICLE_FROZEN_X;
    handle->PARTICLE_FROZEN_Y = PARTICLE_FROZEN_Y;
    handle->PARTICLE_FROZEN_Z = PARTICLE_FROZEN_Z;
    handle->PARTICLE_FROZEN = PARTICLE_FROZEN;
    handle->PARTICLE_LARGE = PARTICLE_LARGE;
    return S_OK;
}


////////////////////
// ParticleHandle //
////////////////////


HRESULT tfParticleHandle_init(struct tfParticleHandleHandle *handle, unsigned int pid) {
    TFC_PTRCHECK(handle);
    if(pid >= _Engine.s.size_parts) 
        return E_FAIL;
    for(unsigned int i = 0; i < _Engine.s.size_parts; i++) {
        Particle *p = _Engine.s.partlist[i];
        if(p && p->id == pid) {
            ParticleHandle *ph = new ParticleHandle(p->id);
            handle->tfObj = (void*)ph;
            return S_OK;
        }
    }
    return E_FAIL;
}

HRESULT tfParticleHandle_destroy(struct tfParticleHandleHandle *handle) {
    return TissueForge::capi::destroyHandle<ParticleHandle, tfParticleHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfParticleHandle_getType(struct tfParticleHandleHandle *handle, struct tfParticleTypeHandle *typeHandle) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(typeHandle);
    ParticleType *ptype = phandle->type();
    TFC_PTRCHECK(ptype);

    typeHandle->tfObj = (void*)ptype;
    return S_OK;
}

HRESULT tfParticleHandle_split(struct tfParticleHandleHandle *handle, struct tfParticleHandleHandle *newParticleHandle) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(newParticleHandle);

    auto nphandle = phandle->fission();
    TFC_PTRCHECK(nphandle);

    newParticleHandle->tfObj = (void*)nphandle;

    return S_OK;
}

HRESULT tfParticleHandle_destroyParticle(struct tfParticleHandleHandle *handle) {
    TFC_PARTICLEHANDLE_GET(handle);
    return phandle->destroy();
}

HRESULT tfParticleHandle_sphericalPosition(struct tfParticleHandleHandle *handle, tfFloatP_t **position) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(position);
    auto sp = phandle->sphericalPosition();
    TFC_VECTOR3_COPYFROM(sp, (*position));
    return S_OK;
}

HRESULT tfParticleHandle_sphericalPositionPoint(struct tfParticleHandleHandle *handle, tfFloatP_t *origin, tfFloatP_t **position) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(origin);
    TFC_PTRCHECK(position);
    FVector3 _origin = FVector3::from(origin);
    auto sp = phandle->sphericalPosition(0, &_origin);
    TFC_VECTOR3_COPYFROM(sp, (*position));
    return S_OK;
}

HRESULT tfParticleHandle_relativePosition(struct tfParticleHandleHandle *handle, tfFloatP_t *origin, tfFloatP_t **position) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(origin);
    TFC_PTRCHECK(position);
    auto p = phandle->relativePosition(FVector3::from(origin));
    TFC_VECTOR3_COPYFROM(p, (*position));
    return S_OK;
}

HRESULT tfParticleHandle_become(struct tfParticleHandleHandle *handle, struct tfParticleTypeHandle *typeHandle) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTYPEHANDLE_GET(typeHandle);
    return phandle->become(ptype);
}

HRESULT package_parts(ParticleList *plist, struct tfParticleHandleHandle **hlist) {
    TFC_PTRCHECK(plist);
    TFC_PTRCHECK(hlist);
    tfParticleHandleHandle *_hlist = (tfParticleHandleHandle*)malloc(plist->nr_parts * sizeof(tfParticleHandleHandle));
    if(!_hlist) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < plist->nr_parts; i++) {
        Particle *p = _Engine.s.partlist[plist->parts[i]];
        TFC_PTRCHECK(p);
        ParticleHandle *ph = new ParticleHandle(p->id);
        _hlist[i].tfObj = (void*)ph;
    }
    *hlist = _hlist;
    return S_OK;
}

HRESULT tfParticleHandle_neighborsD(struct tfParticleHandleHandle *handle, tfFloatP_t distance, struct tfParticleHandleHandle **neighbors, int *numNeighbors) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(neighbors);
    TFC_PTRCHECK(numNeighbors);
    tfFloatP_t _distance = distance;
    auto nbs = phandle->neighbors(&_distance);
    *numNeighbors = nbs->nr_parts;
    return package_parts(nbs, neighbors);
}

HRESULT tfParticleHandle_neighborsT(
    struct tfParticleHandleHandle *handle, 
    struct tfParticleTypeHandle *ptypes, 
    int numTypes, 
    struct tfParticleHandleHandle **neighbors, 
    int *numNeighbors) 
{
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(ptypes);
    TFC_PTRCHECK(neighbors);
    TFC_PTRCHECK(numNeighbors);
    std::vector<ParticleType> _ptypes;
    for(unsigned int i = 0; i < numTypes; i++) {
        tfParticleTypeHandle pth = ptypes[i];
        TFC_PTRCHECK(pth.tfObj);
        _ptypes.push_back(*(ParticleType*)pth.tfObj);
    }
    auto nbs = phandle->neighbors(0, &_ptypes);
    *numNeighbors = nbs->nr_parts;
    return package_parts(nbs, neighbors);
}

HRESULT tfParticleHandle_neighborsDT(
    struct tfParticleHandleHandle *handle, 
    tfFloatP_t distance, 
    struct tfParticleTypeHandle *ptypes, 
    int numTypes, 
    struct tfParticleHandleHandle **neighbors, 
    int *numNeighbors) 
{
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(ptypes);
    TFC_PTRCHECK(neighbors);
    TFC_PTRCHECK(numNeighbors);
    std::vector<ParticleType> _ptypes;
    for(unsigned int i = 0; i < numTypes; i++) {
        tfParticleTypeHandle pth = ptypes[i];
        TFC_PTRCHECK(pth.tfObj);
        _ptypes.push_back(*(ParticleType*)pth.tfObj);
    }
    tfFloatP_t _distance = distance;
    auto nbs = phandle->neighbors(&_distance, &_ptypes);
    *numNeighbors = nbs->nr_parts;
    return package_parts(nbs, neighbors);
}

HRESULT tfParticleHandle_getBondedNeighbors(struct tfParticleHandleHandle *handle, struct tfParticleHandleHandle **neighbors, int *numNeighbors) 
{
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(neighbors);
    TFC_PTRCHECK(numNeighbors);
    auto plist = phandle->getBondedNeighbors();
    *numNeighbors = plist->nr_parts;
    if(plist->nr_parts == 0) 
        return S_OK;
    return package_parts(plist, neighbors);
}

HRESULT tfParticleHandle_distance(struct tfParticleHandleHandle *handle, struct tfParticleHandleHandle *other, tfFloatP_t *distance) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PARTICLEHANDLE_GETN(other, ohandle);
    TFC_PTRCHECK(distance);
    *distance = phandle->distance(ohandle);
    return S_OK;
}

HRESULT tfParticleHandle_getMass(struct tfParticleHandleHandle *handle, tfFloatP_t *mass) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(mass);
    *mass = phandle->getMass();
    return S_OK;
}

HRESULT tfParticleHandle_setMass(struct tfParticleHandleHandle *handle, tfFloatP_t mass) {
    TFC_PARTICLEHANDLE_GET(handle);
    phandle->setMass(mass);
    return S_OK;
}

HRESULT tfParticleHandle_getFrozen(struct tfParticleHandleHandle *handle, bool *frozen) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(frozen);
    *frozen = phandle->getFrozen();
    return S_OK;
}

HRESULT tfParticleHandle_setFrozen(struct tfParticleHandleHandle *handle, bool frozen) {
    TFC_PARTICLEHANDLE_GET(handle);
    phandle->setFrozen(frozen);
    return S_OK;
}

HRESULT tfParticleHandle_getFrozenX(struct tfParticleHandleHandle *handle, bool *frozen) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(frozen);
    *frozen = phandle->getFrozenX();
    return S_OK;
}

HRESULT tfParticleHandle_setFrozenX(struct tfParticleHandleHandle *handle, bool frozen) {
    TFC_PARTICLEHANDLE_GET(handle);
    phandle->setFrozenX(frozen);
    return S_OK;
}

HRESULT tfParticleHandle_getFrozenY(struct tfParticleHandleHandle *handle, bool *frozen) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(frozen);
    *frozen = phandle->getFrozenY();
    return S_OK;
}

HRESULT tfParticleHandle_setFrozenY(struct tfParticleHandleHandle *handle, bool frozen) {
    TFC_PARTICLEHANDLE_GET(handle);
    phandle->setFrozenY(frozen);
    return S_OK;
}

HRESULT tfParticleHandle_getFrozenZ(struct tfParticleHandleHandle *handle, bool *frozen) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(frozen);
    *frozen = phandle->getFrozenZ();
    return S_OK;
}

HRESULT tfParticleHandle_setFrozenZ(struct tfParticleHandleHandle *handle, bool frozen) {
    TFC_PARTICLEHANDLE_GET(handle);
    phandle->setFrozenZ(frozen);
    return S_OK;
}

HRESULT tfParticleHandle_getStyle(struct tfParticleHandleHandle *handle, struct tfRenderingStyleHandle *style) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(style);
    rendering::Style *pstyle = phandle->getStyle();
    TFC_PTRCHECK(pstyle);
    style->tfObj = (void*)pstyle;
    return S_OK;
}

HRESULT tfParticleHandle_hasStyle(struct tfParticleHandleHandle *handle, bool *hasStyle) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(hasStyle);
    *hasStyle = phandle->getStyle() != NULL;
    return S_OK;
}

HRESULT tfParticleHandle_setStyle(struct tfParticleHandleHandle *handle, struct tfRenderingStyleHandle *style) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(style); TFC_PTRCHECK(style->tfObj);
    phandle->setStyle((rendering::Style*)style->tfObj);
    return S_OK;
}

HRESULT tfParticleHandle_getAge(struct tfParticleHandleHandle *handle, tfFloatP_t *age) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(age);
    *age = phandle->getAge();
    return S_OK;
}

HRESULT tfParticleHandle_getRadius(struct tfParticleHandleHandle *handle, tfFloatP_t *radius) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(radius);
    *radius = phandle->getRadius();
    return S_OK;
}

HRESULT tfParticleHandle_setRadius(struct tfParticleHandleHandle *handle, tfFloatP_t radius) {
    TFC_PARTICLEHANDLE_GET(handle);
    phandle->setRadius(radius);
    return S_OK;
}

HRESULT tfParticleHandle_getName(struct tfParticleHandleHandle *handle, char **name, unsigned int *numChars) {
    TFC_PARTICLEHANDLE_GET(handle);
    return TissueForge::capi::str2Char(phandle->getName(), name, numChars);
}

HRESULT tfParticleHandle_getName2(struct tfParticleHandleHandle *handle, char **name, unsigned int *numChars) {
    TFC_PARTICLEHANDLE_GET(handle);
    return TissueForge::capi::str2Char(phandle->getName2(), name, numChars);
}

HRESULT tfParticleHandle_getPosition(struct tfParticleHandleHandle *handle, tfFloatP_t **position) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(position);
    auto p = phandle->getPosition();
    TFC_VECTOR3_COPYFROM(p, (*position));
    return S_OK;
}

HRESULT tfParticleHandle_setPosition(struct tfParticleHandleHandle *handle, tfFloatP_t *position) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(position);
    phandle->setPosition(FVector3::from(position));
    return S_OK;
}

HRESULT tfParticleHandle_getVelocity(struct tfParticleHandleHandle *handle, tfFloatP_t **velocity) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(velocity);
    auto v = phandle->getVelocity();
    TFC_VECTOR3_COPYFROM(v, (*velocity));
    return S_OK;
}

HRESULT tfParticleHandle_setVelocity(struct tfParticleHandleHandle *handle, tfFloatP_t *velocity) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(velocity);
    phandle->setVelocity(FVector3::from(velocity));
    return S_OK;
}

HRESULT tfParticleHandle_getForce(struct tfParticleHandleHandle *handle, tfFloatP_t **force) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(force);
    auto f = phandle->getForce();
    TFC_VECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT tfParticleHandle_getForceInit(struct tfParticleHandleHandle *handle, tfFloatP_t **force) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(force);
    auto f = phandle->getForceInit();
    TFC_VECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT tfParticleHandle_setForceInit(struct tfParticleHandleHandle *handle, tfFloatP_t *force) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(force);
    phandle->setForceInit(FVector3::from(force));
    return S_OK;
}

HRESULT tfParticleHandle_getId(struct tfParticleHandleHandle *handle, int *pid) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(pid);
    *pid = phandle->getId();
    return S_OK;
}

HRESULT tfParticleHandle_getTypeId(struct tfParticleHandleHandle *handle, int *tid) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(tid);
    *tid = phandle->getTypeId();
    return S_OK;
}

HRESULT tfParticleHandle_getClusterId(struct tfParticleHandleHandle *handle, int *cid) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(cid);
    *cid = phandle->getClusterId();
    return S_OK;
}

HRESULT tfParticleHandle_getFlags(struct tfParticleHandleHandle *handle, int *flags) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(flags);
    *flags = phandle->getFlags();
    return S_OK;
}

HRESULT tfParticleHandle_hasSpecies(struct tfParticleHandleHandle *handle, bool *flag) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(flag);
    *flag = phandle->getSpecies() != NULL;
    return S_OK;
}

HRESULT tfParticleHandle_getSpecies(struct tfParticleHandleHandle *handle, struct tfStateStateVectorHandle *svec) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(svec);
    auto _svec = phandle->getSpecies();
    TFC_PTRCHECK(_svec);
    svec->tfObj = (void*)_svec;
    return S_OK;
}

HRESULT tfParticleHandle_setSpecies(struct tfParticleHandleHandle *handle, struct tfStateStateVectorHandle *svec) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(svec); TFC_PTRCHECK(svec->tfObj);
    Particle *p = phandle->part();
    TFC_PTRCHECK(p);
    p->state_vector = (state::StateVector*)svec->tfObj;
    return S_OK;
}

HRESULT tfParticleHandle_toCluster(struct tfParticleHandleHandle *handle, struct tfClusterParticleHandleHandle *chandle) {
    TFC_PARTICLEHANDLE_GET(handle);
    TFC_PTRCHECK(chandle);
    if(phandle->getClusterId() < 0) 
        return E_FAIL;
    chandle->tfObj = (void*)phandle;
    return S_OK;
}

HRESULT tfParticleHandle_toString(struct tfParticleHandleHandle *handle, char **str, unsigned int *numChars) {
    TFC_PARTICLEHANDLE_GET(handle);
    auto p = phandle->part();
    TFC_PTRCHECK(p);
    return TissueForge::capi::str2Char(p->toString(), str, numChars);
}


//////////////////
// ParticleType //
//////////////////


HRESULT tfParticleType_init(struct tfParticleTypeHandle *handle) {
    TFC_PTRCHECK(handle);
    ParticleType *ptype = new ParticleType(true);
    handle->tfObj = (void*)ptype;
    return S_OK;
}

HRESULT tfParticleType_initD(struct tfParticleTypeHandle *handle, struct tfParticleTypeSpec pdef) {
    HRESULT result = tfParticleType_init(handle);
    if(result != S_OK) 
        return result;
    TFC_PTYPEHANDLE_GET(handle);

    ptype->mass = pdef.mass;
    ptype->radius = pdef.radius;
    if(pdef.target_energy) ptype->target_energy = *pdef.target_energy;
    ptype->minimum_radius = pdef.minimum_radius;
    ptype->dynamics = pdef.dynamics;
    ptype->setFrozen(pdef.frozen);
    if(pdef.name) 
        if((result = tfParticleType_setName(handle, pdef.name)) != S_OK) 
            return result;
    if(pdef.numSpecies > 0) {
        ptype->species = new state::SpeciesList();
        for(unsigned int i = 0; i < pdef.numSpecies; i++) 
            if((result = ptype->species->insert(pdef.species[i])) != S_OK) 
                return result;
    }
    if(pdef.style) {
        if(pdef.style->color) 
            ptype->style->setColor(pdef.style->color);
        else if(pdef.style->speciesName) 
            ptype->style->newColorMapper(ptype, pdef.style->speciesName, pdef.style->speciesMapName, pdef.style->speciesMapMin, pdef.style->speciesMapMax);
        ptype->style->setVisible(pdef.style->visible);
    }

    return S_OK;
}

HRESULT tfParticleType_getName(struct tfParticleTypeHandle *handle, char **name, unsigned int *numChars) {
    TFC_PTYPEHANDLE_GET(handle);
    return TissueForge::capi::str2Char(std::string(ptype->name), name, numChars);
}

HRESULT tfParticleType_setName(struct tfParticleTypeHandle *handle, const char *name) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(name);
    std::string ns(name);
    std::strncpy(ptype->name, ns.c_str(), ns.size());
    return S_OK;
}

HRESULT tfParticleType_getId(struct tfParticleTypeHandle *handle, int *id) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(id);
    *id = ptype->id;
    return S_OK;
}

HRESULT tfParticleType_getTypeFlags(struct tfParticleTypeHandle *handle, unsigned int *flags) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(flags);
    *flags = ptype->type_flags;
    return S_OK;
}

HRESULT tfParticleType_setTypeFlags(struct tfParticleTypeHandle *handle, unsigned int flags) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->type_flags = flags;
    return S_OK;
}

HRESULT tfParticleType_getParticleFlags(struct tfParticleTypeHandle *handle, unsigned int *flags) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(flags);
    *flags = ptype->particle_flags;
    return S_OK;
}

HRESULT tfParticleType_setParticleFlags(struct tfParticleTypeHandle *handle, unsigned int flags) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->particle_flags = flags;
    return S_OK;
}

HRESULT tfParticleType_getStyle(struct tfParticleTypeHandle *handle, struct tfRenderingStyleHandle *style) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(style);
    TFC_PTRCHECK(ptype->style);
    style->tfObj = (void*)ptype->style;
    return S_OK;
}

HRESULT tfParticleType_setStyle(struct tfParticleTypeHandle *handle, struct tfRenderingStyleHandle *style) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(style); TFC_PTRCHECK(style->tfObj);
    ptype->style = (rendering::Style*)style->tfObj;
    return S_OK;
}

HRESULT tfParticleType_hasSpecies(struct tfParticleTypeHandle *handle, bool *flag) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(flag);
    *flag = ptype->species != NULL;
    return S_OK;
}

HRESULT tfParticleType_getSpecies(struct tfParticleTypeHandle *handle, struct tfStateSpeciesListHandle *slist) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(ptype->species)
    TFC_PTRCHECK(slist);
    slist->tfObj = (void*)ptype->species;
    return S_OK;
}

HRESULT tfParticleType_setSpecies(struct tfParticleTypeHandle *handle, struct tfStateSpeciesListHandle *slist) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(slist); TFC_PTRCHECK(slist->tfObj);
    ptype->species = (state::SpeciesList*)slist->tfObj;
    return S_OK;
}

HRESULT tfParticleType_getMass(struct tfParticleTypeHandle *handle, tfFloatP_t *mass) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(mass);
    *mass = ptype->mass;
    return S_OK;
}

HRESULT tfParticleType_setMass(struct tfParticleTypeHandle *handle, tfFloatP_t mass) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->mass = mass;
    return S_OK;
}

HRESULT tfParticleType_getRadius(struct tfParticleTypeHandle *handle, tfFloatP_t *radius) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(radius);
    *radius = ptype->radius;
    return S_OK;
}

HRESULT tfParticleType_setRadius(struct tfParticleTypeHandle *handle, tfFloatP_t radius) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->radius = radius;
    return S_OK;
}

HRESULT tfParticleType_getKineticEnergy(struct tfParticleTypeHandle *handle, tfFloatP_t *kinetic_energy) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(kinetic_energy);
    *kinetic_energy = ptype->kinetic_energy;
    return S_OK;
}

HRESULT tfParticleType_getPotentialEnergy(struct tfParticleTypeHandle *handle, tfFloatP_t *potential_energy) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(potential_energy);
    *potential_energy = ptype->potential_energy;
    return S_OK;
}

HRESULT tfParticleType_getTargetEnergy(struct tfParticleTypeHandle *handle, tfFloatP_t *target_energy) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(target_energy);
    *target_energy = ptype->target_energy;
    return S_OK;
}

HRESULT tfParticleType_setTargetEnergy(struct tfParticleTypeHandle *handle, tfFloatP_t target_energy) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->target_energy = target_energy;
    return S_OK;
}

HRESULT tfParticleType_getMinimumRadius(struct tfParticleTypeHandle *handle, tfFloatP_t *minimum_radius) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(minimum_radius);
    *minimum_radius = ptype->minimum_radius;
    return S_OK;
}

HRESULT tfParticleType_setMinimumRadius(struct tfParticleTypeHandle *handle, tfFloatP_t minimum_radius) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->minimum_radius = minimum_radius;
    return S_OK;
}

HRESULT tfParticleType_getDynamics(struct tfParticleTypeHandle *handle, unsigned char *dynamics) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(dynamics);
    *dynamics = ptype->dynamics;
    return S_OK;
}

HRESULT tfParticleType_setDynamics(struct tfParticleTypeHandle *handle, unsigned char dynamics) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->dynamics = dynamics;
    return S_OK;
}

HRESULT tfParticleType_getNumParticles(struct tfParticleTypeHandle *handle, int *numParts) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(numParts);
    *numParts = ptype->parts.nr_parts;
    return S_OK;
}

HRESULT tfParticleType_getParticle(struct tfParticleTypeHandle *handle, int i, struct tfParticleHandleHandle *phandle) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(phandle);
    Particle *p = ptype->particle(i);
    TFC_PTRCHECK(p);
    
    ParticleHandle *ph = new ParticleHandle(p->id);
    phandle->tfObj = (void*)ph;
    return S_OK;
}

HRESULT tfParticleType_isCluster(struct tfParticleTypeHandle *handle, bool *isCluster) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(isCluster);
    *isCluster = ptype->isCluster();
    return S_OK;
}

HRESULT tfParticleType_toCluster(struct tfParticleTypeHandle *handle, struct tfClusterParticleTypeHandle *chandle) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(chandle);
    if(!ptype->isCluster()) 
        return E_FAIL;
    chandle->tfObj = (void*)ptype;
    return S_OK;
}

HRESULT tfParticleType_createParticle(struct tfParticleTypeHandle *handle, int *pid, tfFloatP_t *position, tfFloatP_t *velocity) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(pid);
    FVector3 p, v, *p_ptr = NULL, *v_ptr = NULL;
    if(position) {
        p = FVector3::from(position);
        p_ptr = &p;
    }
    if(velocity) {
        v = FVector3::from(velocity);
        v_ptr = &v;
    }
    ParticleHandle *ph = (*ptype)(p_ptr, v_ptr);
    if(!ph) 
        return E_FAIL;

    *pid = ph->id;
    return S_OK;
}

HRESULT tfParticleType_createParticleS(struct tfParticleTypeHandle *handle, int *pid, const char *str) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(pid);
    TFC_PTRCHECK(str);
    ParticleHandle *ph = (*ptype)(str);
    *pid = ph->id;
    return S_OK;
}

HRESULT tfParticleType_factory(struct tfParticleTypeHandle *handle, int **pids, unsigned int nr_parts, tfFloatP_t *positions, tfFloatP_t *velocities) {
    TFC_PTYPEHANDLE_GET(handle);
    if(nr_parts == 0) 
        return E_FAIL;

    unsigned int nr_parts_ui = (unsigned int)nr_parts;

    std::vector<FVector3> _positions, *_positions_p = NULL;
    std::vector<FVector3> _velocities, *_velocities_p = NULL;

    if(positions) {
        _positions.reserve(nr_parts);
        for(unsigned int i = 0; i < nr_parts; i++) 
            _positions.push_back(FVector3::from(&positions[3 * i]));
        _positions_p = &_positions;
    }
    if(velocities) {
        _velocities.reserve(nr_parts);
        for(unsigned int i = 0; i < nr_parts; i++) 
            _velocities.push_back(FVector3::from(&velocities[3 * i]));
        _velocities_p = &_velocities;
    }

    std::vector<int> _pids_v = ptype->factory(nr_parts_ui, _positions_p, _velocities_p);
    if(pids) 
        std::copy(_pids_v.begin(), _pids_v.end(), *pids);

    return S_OK;
}

HRESULT tfParticleType_newType(struct tfParticleTypeHandle *handle, const char *_name, struct tfParticleTypeHandle *newTypehandle) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(_name);
    TFC_PTRCHECK(newTypehandle);
    ParticleType *ptypeNew = ptype->newType(_name);
    TFC_PTRCHECK(ptypeNew);
    newTypehandle->tfObj = (void*)ptypeNew;
    return S_OK;
}

HRESULT tfParticleType_registerType(struct tfParticleTypeHandle *handle) {
    TFC_PTYPEHANDLE_GET(handle);
    HRESULT result = ptype->registerType();
    if(result != S_OK) 
        return result;
    auto pid = ptype->id;
    handle->tfObj = (void*)&_Engine.types[pid];
    return S_OK;
}

HRESULT tfParticleType_isRegistered(struct tfParticleTypeHandle *handle, bool *isRegistered) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(isRegistered);
    *isRegistered = ptype->isRegistered();
    return S_OK;
}

HRESULT tfParticleType_getFrozen(struct tfParticleTypeHandle *handle, bool *frozen) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(frozen);
    *frozen = ptype->getFrozen();
    return S_OK;
}

HRESULT tfParticleType_setFrozen(struct tfParticleTypeHandle *handle, bool frozen) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->setFrozen(frozen);
    return S_OK;
}

HRESULT tfParticleType_getFrozenX(struct tfParticleTypeHandle *handle, bool *frozen) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(frozen);
    *frozen = ptype->getFrozenX();
    return S_OK;
}

HRESULT tfParticleType_setFrozenX(struct tfParticleTypeHandle *handle, bool frozen) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->setFrozenX(frozen);
    return S_OK;
}

HRESULT tfParticleType_getFrozenY(struct tfParticleTypeHandle *handle, bool *frozen) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(frozen);
    *frozen = ptype->getFrozenY();
    return S_OK;
}

HRESULT tfParticleType_setFrozenY(struct tfParticleTypeHandle *handle, bool frozen) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->setFrozenY(frozen);
    return S_OK;
}

HRESULT tfParticleType_getFrozenZ(struct tfParticleTypeHandle *handle, bool *frozen) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(frozen);
    *frozen = ptype->getFrozenZ();
    return S_OK;
}

HRESULT tfParticleType_setFrozenZ(struct tfParticleTypeHandle *handle, bool frozen) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->setFrozenZ(frozen);
    return S_OK;
}

HRESULT tfParticleType_getTemperature(struct tfParticleTypeHandle *handle, tfFloatP_t *temperature) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(temperature);
    *temperature = ptype->getTemperature();
    return S_OK;
}

HRESULT tfParticleType_getTargetTemperature(struct tfParticleTypeHandle *handle, tfFloatP_t *temperature) {
    TFC_PTYPEHANDLE_GET(handle);
    TFC_PTRCHECK(temperature);
    *temperature = ptype->getTargetTemperature();
    return S_OK;
}

HRESULT tfParticleType_setTargetTemperature(struct tfParticleTypeHandle *handle, tfFloatP_t temperature) {
    TFC_PTYPEHANDLE_GET(handle);
    ptype->setTargetTemperature(temperature);
    return S_OK;
}

HRESULT tfParticleType_toString(struct tfParticleTypeHandle *handle, char **str, unsigned int *numChars) {
    TFC_PTYPEHANDLE_GET(handle);
    return TissueForge::capi::str2Char(ptype->toString(), str, numChars);
}

HRESULT tfParticleType_fromString(struct tfParticleTypeHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    ParticleType *ptype = ParticleType::fromString(str);
    TFC_PTRCHECK(ptype);
    handle->tfObj = (void*)ptype;
    return S_OK;
}


//////////////////
// ParticleList //
//////////////////


HRESULT tfParticleList_init(struct tfParticleListHandle *handle) {
    TFC_PTRCHECK(handle);
    ParticleList *plist = new ParticleList();
    handle->tfObj = (void*)plist;
    return S_OK;
}

HRESULT tfParticleList_initP(struct tfParticleListHandle *handle, struct tfParticleHandleHandle **particles, unsigned int numParts) {
    TFC_PTRCHECK(handle);
    std::vector<ParticleHandle*> _particles;
    for(unsigned int i = 0; i < numParts; i++) {
        tfParticleHandleHandle *phandle = particles[i];
        TFC_PTRCHECK(phandle); TFC_PTRCHECK(phandle->tfObj);
        _particles.push_back((ParticleHandle*)phandle->tfObj);
    }

    ParticleList *plist = new ParticleList(_particles);
    handle->tfObj = (void*)plist;
    return S_OK;
}

HRESULT tfParticleList_initI(struct tfParticleListHandle *handle, int *parts, unsigned int numParts) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(parts);
    ParticleList *plist = new ParticleList(numParts, parts);
    handle->tfObj = (void*)plist;
    return S_OK;
}

HRESULT tfParticleList_copy(struct tfParticleListHandle *source, struct tfParticleListHandle *destination) {
    TFC_PARTICLELIST_GET(source);
    TFC_PTRCHECK(destination);
    ParticleList *_destination = new ParticleList(*plist);
    destination->tfObj = (void*)_destination;
    return S_OK;
}

HRESULT tfParticleList_destroy(struct tfParticleListHandle *handle) {
    return TissueForge::capi::destroyHandle<ParticleList, tfParticleListHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfParticleList_getIds(struct tfParticleListHandle *handle, int **parts) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(parts);
    *parts = plist->parts;
    return S_OK;
}

HRESULT tfParticleList_getNumParts(struct tfParticleListHandle *handle, unsigned int *numParts) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(numParts);
    *numParts = plist->nr_parts;
    return S_OK;
}

HRESULT tfParticleList_free(struct tfParticleListHandle *handle) {
    TFC_PARTICLELIST_GET(handle);
    plist->free();
    return S_OK;
}

HRESULT tfParticleList_insertI(struct tfParticleListHandle *handle, int item, unsigned int *index) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(index);
    *index = plist->insert(item);
    return S_OK;
}

HRESULT tfParticleList_insertP(struct tfParticleListHandle *handle, struct tfParticleHandleHandle *particle, unsigned int *index) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(particle); TFC_PTRCHECK(particle->tfObj);
    TFC_PTRCHECK(index);
    *index = plist->insert((ParticleHandle*)particle->tfObj);
    return S_OK;
}

HRESULT tfParticleList_remove(struct tfParticleListHandle *handle, int id) {
    TFC_PARTICLELIST_GET(handle);
    plist->remove(id);
    return S_OK;
}

HRESULT tfParticleList_extend(struct tfParticleListHandle *handle, struct tfParticleListHandle *other) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(other); TFC_PTRCHECK(other->tfObj);
    plist->extend(*(ParticleList*)other->tfObj);
    return S_OK;
}

HRESULT tfParticleList_item(struct tfParticleListHandle *handle, unsigned int i, struct tfParticleHandleHandle *item) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(item);
    auto *part = plist->item(i);
    TFC_PTRCHECK(part);
    item->tfObj = (void*)part;
    return E_FAIL;
}

HRESULT tfParticleList_getAll(struct tfParticleListHandle *handle) {
    TFC_PTRCHECK(handle);
    ParticleList *plist = ParticleList::all();
    handle->tfObj = (void*)plist;
    return S_OK;
}

HRESULT tfParticleList_getVirial(struct tfParticleListHandle *handle, tfFloatP_t **virial) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(virial);
    auto _virial = plist->getVirial();
    TFC_MATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT tfParticleList_getRadiusOfGyration(struct tfParticleListHandle *handle, tfFloatP_t *rog) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(rog);
    *rog = plist->getRadiusOfGyration();
    return S_OK;
}

HRESULT tfParticleList_getCenterOfMass(struct tfParticleListHandle *handle, tfFloatP_t **com) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(com);
    auto _com = plist->getCenterOfMass();
    TFC_VECTOR3_COPYFROM(_com, (*com));
    return S_OK;
}

HRESULT tfParticleList_getCentroid(struct tfParticleListHandle *handle, tfFloatP_t **cent) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(cent);
    auto _cent = plist->getCentroid();
    TFC_VECTOR3_COPYFROM(_cent, (*cent));
    return S_OK;
}

HRESULT tfParticleList_getMomentOfInertia(struct tfParticleListHandle *handle, tfFloatP_t **moi) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(moi);
    auto _moi = plist->getMomentOfInertia();
    TFC_MATRIX3_COPYFROM(_moi, (*moi));
    return S_OK;
}

HRESULT tfParticleList_getPositions(struct tfParticleListHandle *handle, tfFloatP_t **positions) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(positions);
    return TissueForge::capi::copyVecVecs3_2Arr(plist->getPositions(), positions);
}

HRESULT tfParticleList_getVelocities(struct tfParticleListHandle *handle, tfFloatP_t **velocities) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(velocities);
    return TissueForge::capi::copyVecVecs3_2Arr(plist->getVelocities(), velocities);
}

HRESULT tfParticleList_getForces(struct tfParticleListHandle *handle, tfFloatP_t **forces) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(forces);
    return TissueForge::capi::copyVecVecs3_2Arr(plist->getForces(), forces);
}

HRESULT tfParticleList_sphericalPositions(struct tfParticleListHandle *handle, tfFloatP_t **coordinates) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(coordinates)
    return TissueForge::capi::copyVecVecs3_2Arr(plist->sphericalPositions(), coordinates);
}

HRESULT tfParticleList_sphericalPositionsO(struct tfParticleListHandle *handle, tfFloatP_t *origin, tfFloatP_t **coordinates) {
    TFC_PARTICLELIST_GET(handle);
    TFC_PTRCHECK(origin);
    TFC_PTRCHECK(coordinates);
    FVector3 _origin = FVector3::from(origin);
    return TissueForge::capi::copyVecVecs3_2Arr(plist->sphericalPositions(&_origin), coordinates);
}

HRESULT tfParticleList_toString(struct tfParticleListHandle *handle, char **str, unsigned int *numChars) {
    TFC_PARTICLELIST_GET(handle);
    return TissueForge::capi::str2Char(plist->toString(), str, numChars);
}

HRESULT tfParticleList_fromString(struct tfParticleListHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    handle->tfObj = (void*)ParticleList::fromString(str);
    return S_OK;
}


//////////////////////
// ParticleTypeList //
//////////////////////


HRESULT tfParticleTypeList_init(struct tfParticleTypeListHandle *handle) {
    TFC_PTRCHECK(handle);
    ParticleTypeList *ptlist = new ParticleTypeList();
    handle->tfObj = (void*)ptlist;
    return S_OK;
}

HRESULT tfParticleTypeList_initP(struct tfParticleTypeListHandle *handle, struct tfParticleTypeHandle **parts, unsigned int numParts) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(parts);
    std::vector<ParticleType*> _parts;
    for(unsigned int i = 0; i < numParts; i++) {
        tfParticleTypeHandle *phandle = parts[i];
        TFC_PTRCHECK(phandle); TFC_PTRCHECK(phandle->tfObj);
        _parts.push_back((ParticleType*)phandle->tfObj);
    }
    ParticleTypeList *plist = new ParticleTypeList(_parts);
    handle->tfObj = (void*)plist;
    return S_OK;
}

HRESULT tfParticleTypeList_initI(struct tfParticleTypeListHandle *handle, int *parts, unsigned int numParts) {
    TFC_PTRCHECK(handle);
    ParticleTypeList *ptlist = new ParticleTypeList(numParts, parts);
    handle->tfObj = (void*)ptlist;
    return S_OK;
}

HRESULT tfParticleTypeList_copy(struct tfParticleTypeListHandle *source, struct tfParticleTypeListHandle *destination) {
    TFC_PARTICLETYPELIST_GET(source);
    TFC_PTRCHECK(destination);
    ParticleTypeList *_destination = new ParticleTypeList(*ptlist);
    destination->tfObj = (void*)_destination;
    return S_OK;
}

HRESULT tfParticleTypeList_destroy(struct tfParticleTypeListHandle *handle) {
    return TissueForge::capi::destroyHandle<ParticleTypeList, tfParticleTypeListHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfParticleTypeList_getIds(struct tfParticleTypeListHandle *handle, int **parts) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(parts);
    *parts = ptlist->parts;
    return S_OK;
}

HRESULT tfParticleTypeList_getNumParts(struct tfParticleTypeListHandle *handle, unsigned int *numParts) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(numParts);
    *numParts = ptlist->nr_parts;
    return S_OK;
}

HRESULT tfParticleTypeList_free(struct tfParticleTypeListHandle *handle) {
    TFC_PARTICLETYPELIST_GET(handle);
    ptlist->free();
    return S_OK;
}

HRESULT tfParticleTypeList_insertI(struct tfParticleTypeListHandle *handle, int item, unsigned int *index) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(index);
    *index = ptlist->insert(item);
    return S_OK;
}

HRESULT tfParticleTypeList_insertP(struct tfParticleTypeListHandle *handle, struct tfParticleTypeHandle *ptype, unsigned int *index) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(ptype); TFC_PTRCHECK(ptype->tfObj);
    TFC_PTRCHECK(index);
    *index = ptlist->insert((ParticleType*)ptype->tfObj);
    return S_OK;
}

HRESULT tfParticleTypeList_remove(struct tfParticleTypeListHandle *handle, int id) {
    TFC_PARTICLETYPELIST_GET(handle);
    return ptlist->remove(id);
}

HRESULT tfParticleTypeList_extend(struct tfParticleTypeListHandle *handle, struct tfParticleTypeListHandle *other) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(other); TFC_PTRCHECK(other->tfObj);
    ptlist->extend(*(ParticleTypeList*)other->tfObj);
    return S_OK;
}

HRESULT tfParticleTypeList_item(struct tfParticleTypeListHandle *handle, unsigned int i, struct tfParticleTypeHandle *item) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(item);
    ParticleType *ptype = ptlist->item(i);
    TFC_PTRCHECK(ptype);
    item->tfObj = (void*)ptype;
    return S_OK;
}

HRESULT tfParticleTypeList_getAll(struct tfParticleTypeListHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->tfObj = (void*)ParticleTypeList::all();
    return S_OK;
}

HRESULT tfParticleTypeList_getVirial(struct tfParticleTypeListHandle *handle, tfFloatP_t **virial) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(virial);
    auto _virial = ptlist->getVirial();
    TFC_MATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT tfParticleTypeList_getRadiusOfGyration(struct tfParticleTypeListHandle *handle, tfFloatP_t *rog) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(rog);
    *rog = ptlist->getRadiusOfGyration();
    return S_OK;
}

HRESULT tfParticleTypeList_getCenterOfMass(struct tfParticleTypeListHandle *handle, tfFloatP_t **com) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(com);
    auto _com = ptlist->getCenterOfMass();
    TFC_VECTOR3_COPYFROM(_com, (*com));
    return S_OK;
}

HRESULT tfParticleTypeList_getCentroid(struct tfParticleTypeListHandle *handle, tfFloatP_t **cent) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(cent);
    auto _cent = ptlist->getCentroid();
    TFC_VECTOR3_COPYFROM(_cent, (*cent));
    return S_OK;
}

HRESULT tfParticleTypeList_getMomentOfInertia(struct tfParticleTypeListHandle *handle, tfFloatP_t **moi) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(moi);
    auto _moi = ptlist->getMomentOfInertia();
    TFC_MATRIX3_COPYFROM(_moi, (*moi));
    return S_OK;
}

HRESULT tfParticleTypeList_getPositions(struct tfParticleTypeListHandle *handle, tfFloatP_t **positions) {
    TFC_PARTICLETYPELIST_GET(handle);
    return TissueForge::capi::copyVecVecs3_2Arr(ptlist->getPositions(), positions);
}

HRESULT tfParticleTypeList_getVelocities(struct tfParticleTypeListHandle *handle, tfFloatP_t **velocities) {
    TFC_PARTICLETYPELIST_GET(handle);
    return TissueForge::capi::copyVecVecs3_2Arr(ptlist->getVelocities(), velocities);
}

HRESULT tfParticleTypeList_getForces(struct tfParticleTypeListHandle *handle, tfFloatP_t **forces) {
    TFC_PARTICLETYPELIST_GET(handle);
    return TissueForge::capi::copyVecVecs3_2Arr(ptlist->getForces(), forces);
}

HRESULT tfParticleTypeList_sphericalPositions(struct tfParticleTypeListHandle *handle, tfFloatP_t **coordinates) {
    TFC_PARTICLETYPELIST_GET(handle);
    return TissueForge::capi::copyVecVecs3_2Arr(ptlist->sphericalPositions(), coordinates);
}

HRESULT tfParticleTypeList_sphericalPositionsO(struct tfParticleTypeListHandle *handle, tfFloatP_t *origin, tfFloatP_t **coordinates) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(origin);
    FVector3 _origin = FVector3::from(origin);
    return TissueForge::capi::copyVecVecs3_2Arr(ptlist->sphericalPositions(&_origin), coordinates);
}

HRESULT tfParticleTypeList_getParticles(struct tfParticleTypeListHandle *handle, struct tfParticleListHandle *plist) {
    TFC_PARTICLETYPELIST_GET(handle);
    TFC_PTRCHECK(plist);
    ParticleList *_plist = ptlist->particles();
    TFC_PTRCHECK(_plist);
    plist->tfObj = (void*)_plist;
    return S_OK;
}

HRESULT tfParticleTypeList_toString(struct tfParticleTypeListHandle *handle, char **str, unsigned int *numChars) {
    TFC_PARTICLETYPELIST_GET(handle);
    return TissueForge::capi::str2Char(ptlist->toString(), str, numChars);
}

HRESULT tfParticleTypeList_fromString(struct tfParticleTypeListHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    ParticleTypeList *ptlist = ParticleTypeList::fromString(str);
    handle->tfObj = (void*)ptlist;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfParticle_Verify() {
    return Particle_Verify();
}

HRESULT tfParticleType_FindFromName(struct tfParticleTypeHandle *handle, const char* name) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(name);
    ParticleType *ptype = ParticleType_FindFromName(name);
    TFC_PTRCHECK(ptype);
    handle->tfObj = (void*)ptype;
    return S_OK;
}

/**
 * @brief Get a registered particle type by type id
 * 
 * @param handle handle to populate
 * @param name name of cluster type
 * @return S_OK on success 
 */
HRESULT tfParticleType_getFromId(struct tfParticleTypeHandle *handle, unsigned int pid) {
    TFC_PTRCHECK(handle);
    if(pid >= _Engine.nr_types) 
        return E_FAIL;
    
    ParticleType *ptype = &_Engine.types[pid];
    TFC_PTRCHECK(ptype);
    
    handle->tfObj = (void*)ptype;
    return S_OK;
}

unsigned int *tfParticle_Colors() {
    return Particle_Colors;
}
