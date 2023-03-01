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

#include "tfCBody.h"

#include "tfCVertex.h"
#include "tfCSurface.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tfBody.h>
#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tf_mesh_bind.h>
#include <models/vertex/solver/actors/tfBodyForce.h>
#include <models/vertex/solver/actors/tfSurfaceAreaConstraint.h>
#include <models/vertex/solver/actors/tfVolumeConstraint.h>
#include <models/vertex/solver/actors/tfAdhesion.h>

#include <io/tfThreeDFMeshData.h>
#include <rendering/tfStyle.h>

#include <unordered_map>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    BodyHandle *castC(struct tfVertexSolverBodyHandleHandle *handle) {
        return castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle);
    }

    BodyType *castC(struct tfVertexSolverBodyTypeHandle *handle) {
        return castC<BodyType, tfVertexSolverBodyTypeHandle>(handle);
    }

}

#define TFC_BODYHANDLE_GET(handle) \
    BodyHandle *bhandle = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle); \
    TFC_PTRCHECK(bhandle);

#define TFC_BODYTYPE_GET(handle) \
    BodyType *btype = TissueForge::castC<BodyType, tfVertexSolverBodyTypeHandle>(handle); \
    TFC_PTRCHECK(btype);


////////////////////////////////
// tfVertexSolverBodyTypeSpec //
////////////////////////////////


struct tfVertexSolverBodyTypeSpec tfVertexSolverBodyTypeSpec_init() {
    return {
        NULL,       // tfFloatP_t *density
        NULL,       // tfFloatP_t **bodyForceComps
        NULL,       // tfFloatP_t *surfaceAreaLam
        NULL,       // tfFloatP_t *surfaceAreaVal
        NULL,       // tfFloatP_t *volumeLam
        NULL,       // tfFloatP_t *volumeVal
        NULL,       // char *name
        NULL,       // char **adhesionNames
        NULL,       // tfFloatP_t *adhesionValues
        0,          // unsigned int numAdhesionValues
    };
}


////////////////
// BodyHandle //
////////////////


HRESULT tfVertexSolverBodyHandle_init(struct tfVertexSolverBodyHandleHandle *handle, int id) {
    TFC_PTRCHECK(handle);
    BodyHandle *_tfObj = new BodyHandle(id);
    handle->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_fromString(struct tfVertexSolverBodyHandleHandle *handle, const char *s) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(s);
    BodyHandle *_tfObj = new BodyHandle(BodyHandle::fromString(s));
    handle->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_destroy(struct tfVertexSolverBodyHandleHandle *handle) {
    return TissueForge::capi::destroyHandle<BodyHandle, tfVertexSolverBodyHandleHandle>(handle);
}

HRESULT tfVertexSolverBodyHandle_getId(struct tfVertexSolverBodyHandleHandle *handle, int *objId) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(objId);
    *objId = bhandle->id;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_definedByVertex(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    bool *result
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    TFC_PTRCHECK(result);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    *result = bhandle->definedBy(*_v);
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_definedBySurface(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    bool *result
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    TFC_PTRCHECK(result);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    *result =  bhandle->definedBy(*_s);
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_objType(struct tfVertexSolverBodyHandleHandle *handle, unsigned int *label) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(label);
    *label = bhandle->objType();
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_destroyBody(struct tfVertexSolverBodyHandleHandle *handle) {
    TFC_BODYHANDLE_GET(handle);
    return bhandle->destroy();
}

HRESULT tfVertexSolverBodyHandle_destroyBodyC(struct tfVertexSolverBodyHandleHandle *handle) {
    TFC_BODYHANDLE_GET(handle);
    return Body::destroy(*bhandle);
}

HRESULT tfVertexSolverBodyHandle_validate(struct tfVertexSolverBodyHandleHandle *handle, bool *result) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = bhandle->validate();
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_positionChanged(struct tfVertexSolverBodyHandleHandle *handle) {
    TFC_BODYHANDLE_GET(handle);
    return bhandle->positionChanged();
}

HRESULT tfVertexSolverBodyHandle_str(struct tfVertexSolverBodyHandleHandle *handle, char **str, unsigned int *numChars) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(bhandle->str(), str, numChars);
}

HRESULT tfVertexSolverBodyHandle_toString(struct tfVertexSolverBodyHandleHandle *handle, char **str, unsigned int *numChars) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(bhandle->toString(), str, numChars);
}

HRESULT tfVertexSolverBodyHandle_add(struct tfVertexSolverBodyHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    return bhandle->add(*_s);
}

HRESULT tfVertexSolverBodyHandle_remove(struct tfVertexSolverBodyHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    return bhandle->remove(*_s);
}

HRESULT tfVertexSolverBodyHandle_replace(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toInsert, 
    struct tfVertexSolverSurfaceHandleHandle *toRemove
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(toInsert);
    SurfaceHandle *_toInsert = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(toInsert);
    TFC_PTRCHECK(_toInsert);
    TFC_PTRCHECK(toRemove);
    SurfaceHandle *_toRemove = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(toRemove);
    TFC_PTRCHECK(_toRemove);
    return bhandle->replace(*_toInsert, *_toRemove);
}

HRESULT tfVertexSolverBodyHandle_getType(struct tfVertexSolverBodyHandleHandle *handle, struct tfVertexSolverBodyTypeHandle *btype) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(btype);
    BodyType *_btype = bhandle->type();
    TFC_PTRCHECK(_btype);
    btype->tfObj = (void*)_btype;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_become(struct tfVertexSolverBodyHandleHandle *handle, struct tfVertexSolverBodyTypeHandle *btype) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(btype);
    BodyType *_btype = bhandle->type();
    TFC_PTRCHECK(_btype);
    return bhandle->become(_btype);
}

HRESULT tfVertexSolverBodyHandle_getSurfaces(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<SurfaceHandle> _objs = bhandle->getSurfaces();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getVertices(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<VertexHandle> _objs = bhandle->getVertices();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverVertexHandleHandle*)malloc(sizeof(struct tfVertexSolverVertexHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverVertexHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_findVertex(
    struct tfVertexSolverBodyHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverVertexHandleHandle *v
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(dir);
    TFC_PTRCHECK(v);
    FVector3 _dir = FVector3::from(dir);
    VertexHandle _v = bhandle->findVertex(_dir);
    return tfVertexSolverVertexHandle_init(v, _v.id);
}

HRESULT tfVertexSolverBodyHandle_findSurface(
    struct tfVertexSolverBodyHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverSurfaceHandleHandle *s
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(dir);
    TFC_PTRCHECK(s);
    FVector3 _dir = FVector3::from(dir);
    SurfaceHandle _s = bhandle->findSurface(_dir);
    return tfVertexSolverSurfaceHandle_init(s, _s.id);
}

HRESULT tfVertexSolverBodyHandle_connectedBodies(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<BodyHandle> _objs = bhandle->connectedBodies();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverBodyHandleHandle*)malloc(sizeof(struct tfVertexSolverBodyHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverBodyHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_adjacentBodies(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<BodyHandle> _objs = bhandle->adjacentBodies();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverBodyHandleHandle*)malloc(sizeof(struct tfVertexSolverBodyHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverBodyHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_neighborSurfaces(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    std::vector<SurfaceHandle> _objs = bhandle->neighborSurfaces(*_s);
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getDensity(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t *result) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = bhandle->getDensity();
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_setDensity(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t density) {
    TFC_BODYHANDLE_GET(handle);
    bhandle->setDensity(density);
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getCentroid(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t **result) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    FVector3 _result = bhandle->getCentroid();
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getVelocity(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t **result) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    FVector3 _result = bhandle->getVelocity();
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getArea(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t *result) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = bhandle->getArea();
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getVolume(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t *result) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = bhandle->getVolume();
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getMass(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t *result) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = bhandle->getMass();
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getVertexArea(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    TFC_PTRCHECK(result);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    *result = bhandle->getVertexArea(*_v);
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getVertexVolume(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    TFC_PTRCHECK(result);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    *result = bhandle->getVertexVolume(*_v);
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_getVertexMass(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    TFC_PTRCHECK(result);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    *result = bhandle->getVertexMass(*_v);
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_findInterface(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *b, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(b);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    BodyHandle *_b = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(b);
    TFC_PTRCHECK(_b);
    std::vector<SurfaceHandle> _objs = bhandle->findInterface(*_b);
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_contactArea(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *other, 
    tfFloatP_t *result
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(other);
    TFC_PTRCHECK(result);
    BodyHandle *_other = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(other);
    TFC_PTRCHECK(_other);
    *result = bhandle->contactArea(*_other);
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_sharedVertices(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *other, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(other);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    BodyHandle *_other = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(other);
    TFC_PTRCHECK(_other);
    std::vector<VertexHandle> _objs = bhandle->sharedVertices(*_other);
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverVertexHandleHandle*)malloc(sizeof(struct tfVertexSolverVertexHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverVertexHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
}

HRESULT tfVertexSolverBodyHandle_isOutside(
    struct tfVertexSolverBodyHandleHandle *handle, 
    tfFloatP_t *pos, 
    bool *result
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(result);
    FVector3 _pos = FVector3::from(pos);
    *result = bhandle->isOutside(_pos);
    return S_OK;
}

HRESULT tfVertexSolverBodyHandle_split(
    struct tfVertexSolverBodyHandleHandle *handle, 
    tfFloatP_t *cp_pos, 
    tfFloatP_t *cp_norm, 
    struct tfVertexSolverSurfaceTypeHandle *stype, 
    struct tfVertexSolverBodyHandleHandle *newObj
) {
    TFC_BODYHANDLE_GET(handle);
    TFC_PTRCHECK(cp_pos);
    TFC_PTRCHECK(cp_norm);
    TFC_PTRCHECK(newObj);
    FVector3 _cp_pos = FVector3::from(cp_pos);
    FVector3 _cp_norm = FVector3::from(cp_norm);
    SurfaceType *_stype = stype ? TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(stype) : NULL;
    BodyHandle _newObj = bhandle->split(_cp_pos, _cp_norm, _stype);
    return tfVertexSolverBodyHandle_init(newObj, _newObj.id);
}


//////////////
// BodyType //
//////////////


HRESULT tfVertexSolverBodyType_init(struct tfVertexSolverBodyTypeHandle *handle) {
    TFC_PTRCHECK(handle);
    BodyType *btype = new BodyType(true);
    handle->tfObj = (void*)btype;
    return S_OK;
}

HRESULT tfVertexSolverBodyType_initD(struct tfVertexSolverBodyTypeHandle *handle, struct tfVertexSolverBodyTypeSpec bdef) {
    if(tfVertexSolverBodyType_init(handle) != S_OK) 
        return E_FAIL;
    TFC_PTRCHECK(handle);
    TFC_BODYTYPE_GET(handle);

    if(bdef.density) 
        btype->density = *bdef.density;
    if(bdef.name) 
        btype->name = bdef.name;

    if(bdef.bodyForceComps) {
        FVector3 _bodyForceComps = FVector3::from(*bdef.bodyForceComps);
        if(bind::body(new BodyForce(_bodyForceComps), btype) != S_OK) 
            return E_FAIL;
    }
    if(bdef.surfaceAreaLam && bdef.surfaceAreaVal && 
        bind::body(new SurfaceAreaConstraint(*bdef.surfaceAreaLam, *bdef.surfaceAreaVal), btype) != S_OK) 
        return E_FAIL;
    if(bdef.volumeLam && bdef.volumeVal && bind::body(new VolumeConstraint(*bdef.volumeLam, *bdef.volumeVal), btype) != S_OK) 
        return E_FAIL;

    return S_OK;
}

HRESULT tfVertexSolverBodyType_fromString(struct tfVertexSolverBodyTypeHandle *handle, const char *s) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(s);
    BodyType *_tfObj = BodyType::fromString(s);
    handle->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverBodyType_objType(struct tfVertexSolverBodyTypeHandle *handle, unsigned int *label) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(label);
    *label = btype->objType();
    return S_OK;
}

HRESULT tfVertexSolverBodyType_str(
    struct tfVertexSolverBodyTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(btype->str(), str, numChars);
}

HRESULT tfVertexSolverBodyType_toString(
    struct tfVertexSolverBodyTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(btype->toString(), str, numChars);
}

HRESULT tfVertexSolverBodyType_registerType(struct tfVertexSolverBodyTypeHandle *handle) {
    TFC_BODYTYPE_GET(handle);
    return btype->registerType();
}

HRESULT tfVertexSolverBodyType_isRegistered(struct tfVertexSolverBodyTypeHandle *handle, bool *result) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(result);
    *result = btype->isRegistered();
    return S_OK;
}

HRESULT tfVertexSolverBodyType_getInstances(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<BodyHandle> _objs = btype->getInstances();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverBodyHandleHandle*)malloc(sizeof(struct tfVertexSolverBodyHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverBodyHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverBodyType_getInstanceIds(
    struct tfVertexSolverBodyTypeHandle *handle, 
    int **ids, 
    int *numObjs
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(ids);
    TFC_PTRCHECK(numObjs);
    std::vector<int> _ids = btype->getInstanceIds();
    *numObjs = _ids.size();
    if(*numObjs == 0) 
        return S_OK;
    *ids = (int*)malloc(sizeof(int) * _ids.size());
    memcpy(&(*ids)[0], &_ids.data()[0], sizeof(int) * _ids.size());
    return S_OK;
}

HRESULT tfVertexSolverBodyType_getName(
    struct tfVertexSolverBodyTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(btype->name, str, numChars);
}

HRESULT tfVertexSolverBodyType_setName(struct tfVertexSolverBodyTypeHandle *handle, const char *name) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(name);
    btype->name = name;
    return S_OK;
}

HRESULT tfVertexSolverBodyType_getDensity(struct tfVertexSolverBodyTypeHandle *handle, tfFloatP_t *result) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(result);
    *result = btype->density;
    return S_OK;
}

HRESULT tfVertexSolverBodyType_setDensity(struct tfVertexSolverBodyTypeHandle *handle, tfFloatP_t density) {
    TFC_BODYTYPE_GET(handle);
    btype->density = density;
    return S_OK;
}

HRESULT tfVertexSolverBodyType_createBodyS(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **surfaces, 
    unsigned int numSurfaces, 
    struct tfVertexSolverBodyHandleHandle *newObj
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(surfaces);
    TFC_PTRCHECK(newObj);
    std::vector<SurfaceHandle> _surfaces;
    _surfaces.reserve(numSurfaces);
    for(size_t i = 0; i < numSurfaces; i++) {
        SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(surfaces[i]);
        TFC_PTRCHECK(_s);
        _surfaces.emplace_back(_s->id);
    }
    BodyHandle _newObj = (*btype)(_surfaces);
    return tfVertexSolverBodyHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverBodyType_createBodyIO(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfIoThreeDFMeshDataHandle *ioMesh, 
    struct tfVertexSolverSurfaceTypeHandle *stype, 
    struct tfVertexSolverBodyHandleHandle *newObj
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(ioMesh);
    TFC_PTRCHECK(stype);
    TFC_PTRCHECK(newObj);
    io::ThreeDFMeshData *_ioMesh = TissueForge::castC<io::ThreeDFMeshData, tfIoThreeDFMeshDataHandle>(ioMesh);
    TFC_PTRCHECK(_ioMesh);
    SurfaceType *_stype = TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(stype);
    TFC_PTRCHECK(_stype);
    BodyHandle _newObj = (*btype)(_ioMesh, _stype);
    return tfVertexSolverBodyHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverBodyType_extend(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *base, 
    tfFloatP_t *pos, 
    struct tfVertexSolverBodyHandleHandle *newObj
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(base);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(newObj);
    SurfaceHandle *_base = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(base);
    FVector3 _pos = FVector3::from(pos);
    BodyHandle _newObj = btype->extend(*_base, _pos);
    return tfVertexSolverBodyHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverBodyType_extrude(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *base, 
    tfFloatP_t normLen, 
    struct tfVertexSolverBodyHandleHandle *newObj
) {
    TFC_BODYTYPE_GET(handle);
    TFC_PTRCHECK(base);
    TFC_PTRCHECK(newObj);
    SurfaceHandle *_base = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(base);
    BodyHandle _newObj = btype->extrude(*_base, normLen);
    return tfVertexSolverBodyHandle_init(newObj, _newObj.id);
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolverCreateBodyBySurfaces(
    struct tfVertexSolverSurfaceHandleHandle **surfaces, 
    unsigned int numSurfaces, 
    struct tfVertexSolverBodyHandleHandle *newObj
) {
    TFC_PTRCHECK(surfaces);
    TFC_PTRCHECK(newObj);
    std::vector<SurfaceHandle> _surfaces;
    _surfaces.reserve(numSurfaces);
    for(size_t i = 0; i < numSurfaces; i++) {
        SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(surfaces[i]);
        TFC_PTRCHECK(_s);
        _surfaces.emplace_back(_s->id);
    }
    BodyHandle _newObj = Body::create(_surfaces);
    return tfVertexSolverBodyHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverCreateBodyByIOData(struct tfIoThreeDFMeshDataHandle *ioMesh, struct tfVertexSolverBodyHandleHandle *newObj) {
    TFC_PTRCHECK(ioMesh);
    TFC_PTRCHECK(newObj);
    io::ThreeDFMeshData *_ioMesh = TissueForge::castC<io::ThreeDFMeshData, tfIoThreeDFMeshDataHandle>(ioMesh);
    TFC_PTRCHECK(_ioMesh);
    BodyHandle _newObj = Body::create(_ioMesh);
    return tfVertexSolverBodyHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverFindBodyTypeFromName(const char *name, struct tfVertexSolverBodyTypeHandle *btype) {
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(btype);
    BodyType *_tfObj = BodyType::findFromName(name);
    TFC_PTRCHECK(_tfObj);
    btype->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverBindBodyTypeAdhesion(
    struct tfVertexSolverBodyTypeHandle **btypes, 
    struct tfVertexSolverBodyTypeSpec *bdefs, 
    unsigned int numTypes
) {
    TFC_PTRCHECK(btypes);
    TFC_PTRCHECK(bdefs);

    std::vector<BodyType*> _btypes;
    _btypes.reserve(numTypes);
    for(size_t i = 0; i < numTypes; i++) {
        BodyType *_bt = TissueForge::castC<BodyType, tfVertexSolverBodyTypeHandle>(btypes[i]);
        TFC_PTRCHECK(_bt);
        _btypes.push_back(_bt);
    }

    std::unordered_map<std::string, std::unordered_map<std::string, FloatP_t> > adhesionMap;
    for(auto &bt : _btypes) 
        adhesionMap.insert({bt->name, {}});

    for(size_t i = 0; i < numTypes; i++) {
        tfVertexSolverBodyTypeSpec bdef = bdefs[i];
        if(bdef.adhesionNames && bdef.adhesionValues && bdef.numAdhesionValues > 0) {
            auto itr_i = adhesionMap.find(bdef.name);
            if(itr_i == adhesionMap.end()) 
                continue;
            for(size_t j = 0; j < bdef.numAdhesionValues; j++) {
                auto name = bdef.adhesionNames[j];
                auto itr_j = adhesionMap.find(name);
                if(itr_j != adhesionMap.end()) {
                    auto val = bdef.adhesionValues[j];
                    itr_i->second.insert({name, val});
                    itr_j->second.insert({bdef.name, val});
                }
            }
        }
    }

    for(size_t i = 0; i < _btypes.size(); i++) {
        BodyType *_bt_i = _btypes[i];
        auto itr_i = adhesionMap.find(_bt_i->name);
        for(size_t j = i; i < _btypes.size(); j++) {
            BodyType *_bt_j = _btypes[j];
            // Prevent duplicates
            if(_bt_j->id > _bt_i->id) 
                continue;
            auto itr_j = itr_i->second.find(_bt_j->name);
            if(itr_j == itr_i->second.end()) 
                continue;
            if(bind::types(new Adhesion(itr_j->second), _bt_i, _bt_j) != S_OK) 
                return E_FAIL;
        }
    }

    return S_OK;
}
