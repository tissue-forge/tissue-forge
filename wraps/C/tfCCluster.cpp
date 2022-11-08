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

#include "tfCCluster.h"

#include "TissueForge_c_private.h"
#include "tfCParticle.h"

#include <tfCluster.h>
#include <tfEngine.h>


using namespace TissueForge;


namespace TissueForge { 


    ClusterParticleHandle *castC(struct tfClusterParticleHandleHandle *handle) {
        return castC<ClusterParticleHandle, tfClusterParticleHandleHandle>(handle);
    }

    ClusterParticleType *castC(struct tfClusterParticleTypeHandle *handle) {
        return castC<ClusterParticleType, tfClusterParticleTypeHandle>(handle);
    }

}

#define TFC_CLUSTERHANDLE_GET(handle) \
    ClusterParticleHandle *phandle = TissueForge::castC<ClusterParticleHandle, tfClusterParticleHandleHandle>(handle); \
    TFC_PTRCHECK(phandle);

#define TFC_TYPEHANDLE_GET(handle) \
    ClusterParticleType *ptype = TissueForge::castC<ClusterParticleType, tfClusterParticleTypeHandle>(handle); \
    TFC_PTRCHECK(ptype);


///////////////////////
// tfClusterTypeSpec //
///////////////////////


struct tfClusterTypeSpec tfClusterTypeSpec_init() {
    struct tfClusterTypeSpec clusterTypeSpec = {
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
        0, 
        NULL
    };
    return clusterTypeSpec;
}


///////////////////////////
// ClusterParticleHandle //
///////////////////////////


HRESULT tfClusterParticleHandle_init(struct tfClusterParticleHandleHandle *handle, int id) {
    TFC_PTRCHECK(handle);
    if(id >= _Engine.s.size_parts) 
        return E_FAIL;

    Particle *p = _Engine.s.partlist[id];
    TFC_PTRCHECK(p);

    ClusterParticleHandle *phandle = new ClusterParticleHandle(id);
    handle->tfObj = (void*)phandle;
    return S_OK;
}

HRESULT tfClusterParticleHandle_destroy(struct tfClusterParticleHandleHandle *handle) {
    return TissueForge::capi::destroyHandle<ClusterParticleHandle, tfClusterParticleHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfClusterParticleHandle_createParticle(
    struct tfClusterParticleHandleHandle *handle, 
    struct tfParticleTypeHandle *partTypeHandle, 
    int *pid, 
    tfFloatP_t **position, 
    tfFloatP_t **velocity) 
{
    TFC_CLUSTERHANDLE_GET(handle); 
    TFC_PTRCHECK(partTypeHandle); TFC_PTRCHECK(partTypeHandle->tfObj);
    TFC_PTRCHECK(pid);
    ParticleType *partType = (ParticleType*)partTypeHandle->tfObj;

    FVector3 pos, vel, *pos_p = NULL, *vel_p = NULL;
    if(position) {
        pos = FVector3::from(*position);
        pos_p = &pos;
    }
    if(velocity) {
        vel = FVector3::from(*velocity);
        vel_p = &vel;
    }

    auto p = (*phandle)(partType, pos_p, vel_p);
    TFC_PTRCHECK(p);
    
    *pid = p->id;

    return S_OK;
}

HRESULT tfClusterParticleHandle_createParticleS(
    struct tfClusterParticleHandleHandle *handle, 
    struct tfParticleTypeHandle *partTypeHandle, 
    int *pid, 
    const char *str) 
{
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(partTypeHandle); TFC_PTRCHECK(!partTypeHandle->tfObj);
    TFC_PTRCHECK(pid);
    TFC_PTRCHECK(str);
    ParticleType *partType = (ParticleType*)partTypeHandle->tfObj;
    auto p = (*phandle)(partType, std::string(str));
    TFC_PTRCHECK(p);
    *pid = p->id;
    delete p;
    return S_OK;
}

HRESULT tfClusterParticleHandle_splitAxis(struct tfClusterParticleHandleHandle *handle, int *cid, tfFloatP_t *axis, tfFloatP_t time) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(cid);
    TFC_PTRCHECK(axis);
    FVector3 _axis = FVector3::from(axis);
    auto c = phandle->split(&_axis, 0, 0);
    TFC_PTRCHECK(c);
    *cid = c->id;
    delete c;
    return S_OK;
}

HRESULT tfClusterParticleHandle_splitRand(struct tfClusterParticleHandleHandle *handle, int *cid, tfFloatP_t time) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(cid);
    auto c = phandle->split();
    TFC_PTRCHECK(c);
    *cid = c->id;
    delete c;
    return S_OK;
}

HRESULT tfClusterParticleHandle_split(struct tfClusterParticleHandleHandle *handle, int *cid, tfFloatP_t time, tfFloatP_t *normal, tfFloatP_t *point) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(cid);
    TFC_PTRCHECK(normal);
    TFC_PTRCHECK(point);
    FVector3 _normal = FVector3::from(normal);
    FVector3 _point = FVector3::from(point);
    auto c = phandle->split(0, 0, 0, &_normal, &_point);
    TFC_PTRCHECK(c);
    *cid = c->id;
    delete c;
    return S_OK;
}

HRESULT tfClusterParticleHandle_getNumParts(struct tfClusterParticleHandleHandle *handle, int *numParts) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(numParts);
    Particle *p = phandle->part();
    TFC_PTRCHECK(p);
    *numParts = p->nr_parts;
    return S_OK;
}

HRESULT tfClusterParticleHandle_getParts(struct tfClusterParticleHandleHandle *handle, struct tfParticleListHandle *parts) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(parts);
    Particle *p = phandle->part();
    return tfParticleList_initI(parts, p->parts, p->nr_parts);
}

HRESULT tfClusterParticleHandle_getParticle(struct tfClusterParticleHandleHandle *handle, int i, struct tfParticleHandleHandle *parthandle) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(parthandle);
    Particle *cp = phandle->part();
    TFC_PTRCHECK(cp);
    if(i >= cp->nr_parts) 
        return E_FAIL;
    int pid = cp->parts[i];
    if(pid >= _Engine.s.size_parts) 
        return E_FAIL;
    Particle *p = _Engine.s.partlist[pid];
    TFC_PTRCHECK(p);
    ParticleHandle *ph = new ParticleHandle(p->id);
    parthandle->tfObj = (void*)ph;
    return S_OK;
}

HRESULT tfClusterParticleHandle_getRadiusOfGyration(struct tfClusterParticleHandleHandle *handle, tfFloatP_t *radiusOfGyration) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(radiusOfGyration);
    *radiusOfGyration = phandle->getRadiusOfGyration();
    return S_OK;
}

HRESULT tfClusterParticleHandle_getCenterOfMass(struct tfClusterParticleHandleHandle *handle, tfFloatP_t **com) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(com);
    FVector3 _com = phandle->getCenterOfMass();
    TFC_VECTOR3_COPYFROM(_com, (*com));
    return S_OK;
}

HRESULT tfClusterParticleHandle_getCentroid(struct tfClusterParticleHandleHandle *handle, tfFloatP_t **cent) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(cent);
    FVector3 _cent = phandle->getCentroid();
    TFC_VECTOR3_COPYFROM(_cent, (*cent));
    return S_OK;
}

HRESULT tfClusterParticleHandle_getMomentOfInertia(struct tfClusterParticleHandleHandle *handle, tfFloatP_t **moi) {
    TFC_CLUSTERHANDLE_GET(handle);
    TFC_PTRCHECK(moi);
    FMatrix3 _moi = phandle->getMomentOfInertia();
    TFC_MATRIX3_COPYFROM(_moi, (*moi));
    return S_OK;
}


/////////////////////////
// ClusterParticleType //
/////////////////////////


HRESULT tfClusterParticleType_init(struct tfClusterParticleTypeHandle *handle) {
    TFC_PTRCHECK(handle);
    ClusterParticleType *ptype = new ClusterParticleType(true);
    handle->tfObj = (void*)ptype;
    return S_OK;
}

HRESULT tfClusterParticleType_initD(struct tfClusterParticleTypeHandle *handle, struct tfClusterTypeSpec pdef) {
    HRESULT result = tfClusterParticleType_init(handle);
    if(result != S_OK) 
        return result;
    TFC_TYPEHANDLE_GET(handle);

    ptype->mass = pdef.mass;
    ptype->radius = pdef.radius;
    if(pdef.target_energy) ptype->target_energy = *pdef.target_energy;
    ptype->minimum_radius = pdef.minimum_radius;
    ptype->dynamics = pdef.dynamics;
    ptype->setFrozen(pdef.frozen);
    if(pdef.name) {
        std::string ns(pdef.name);
        std::strncpy(ptype->name, ns.c_str(), ns.size());
    }
    if(pdef.numTypes > 0) {
        tfParticleTypeHandle *pth;
        for(unsigned int i = 0; i < pdef.numTypes; i++) {
            pth = pdef.types[i];
            TFC_PTRCHECK(pth); TFC_PTRCHECK(pth->tfObj);
            ptype->types.insert((ParticleType*)pth->tfObj);
        }
    }

    return S_OK;
}

HRESULT tfClusterParticleType_addType(struct tfClusterParticleTypeHandle *handle, struct tfParticleTypeHandle *phandle) {
    TFC_TYPEHANDLE_GET(handle);
    TFC_PTRCHECK(phandle); TFC_PTRCHECK(phandle->tfObj);
    ParticleType *partType = (ParticleType*)phandle->tfObj;
    ptype->types.insert(partType->id);
    return S_OK;
}

HRESULT tfClusterParticleType_hasType(struct tfClusterParticleTypeHandle *handle, struct tfParticleTypeHandle *phandle, bool *hasType) {
    TFC_TYPEHANDLE_GET(handle);
    TFC_PTRCHECK(phandle); TFC_PTRCHECK(phandle->tfObj);
    TFC_PTRCHECK(hasType);
    ParticleType *partType = (ParticleType*)phandle->tfObj;
    *hasType = ptype->hasType(partType);
    return S_OK;
}

HRESULT tfClusterParticleType_registerType(struct tfClusterParticleTypeHandle *handle) {
    TFC_TYPEHANDLE_GET(handle);
    return ptype->registerType();
}

HRESULT tfClusterParticleType_createParticle(struct tfClusterParticleTypeHandle *handle, int *pid, tfFloatP_t *position, tfFloatP_t *velocity) {
    TFC_TYPEHANDLE_GET(handle);
    tfParticleTypeHandle _handle;
    _handle.tfObj = (void*)ptype;
    return tfParticleType_createParticle(&_handle, pid, position, velocity);
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfClusterParticleType_FindFromName(struct tfClusterParticleTypeHandle *handle, const char* name) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(name);
    ClusterParticleType *ptype = ClusterParticleType_FindFromName(name);
    TFC_PTRCHECK(ptype);
    handle->tfObj = (void*)ptype;
    return S_OK;
}

HRESULT tfClusterParticleType_getFromId(struct tfClusterParticleTypeHandle *handle, unsigned int pid) {
    TFC_PTRCHECK(handle);
    if(pid >= _Engine.nr_types) 
        return E_FAIL;
    
    ParticleType *ptype = &_Engine.types[pid];
    TFC_PTRCHECK(ptype);
    if(!ptype->isCluster()) 
        return E_FAIL;
    
    handle->tfObj = (void*)ptype;
    return S_OK;
}
