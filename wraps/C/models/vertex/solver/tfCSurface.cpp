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

#include "tfCSurface.h"

#include "tfCVertex.h"
#include "tfCBody.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfBody.h>
#include <models/vertex/solver/tf_mesh_bind.h>
#include <models/vertex/solver/actors/tfEdgeTension.h>
#include <models/vertex/solver/actors/tfNormalStress.h>
#include <models/vertex/solver/actors/tfSurfaceAreaConstraint.h>
#include <models/vertex/solver/actors/tfSurfaceTraction.h>
#include <models/vertex/solver/actors/tfAdhesion.h>

#include <io/tfThreeDFFaceData.h>
#include <rendering/tfStyle.h>

#include <unordered_map>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    SurfaceHandle *castC(struct tfVertexSolverSurfaceHandleHandle *handle) {
        return castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle);
    }

    SurfaceType *castC(struct tfVertexSolverSurfaceTypeHandle *handle) {
        return castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle);
    }

}

#define TFC_SURFACEHANDLE_GET(handle) \
    SurfaceHandle *shandle = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle); \
    TFC_PTRCHECK(shandle);

#define TFC_SURFACETYPE_GET(handle) \
    SurfaceType *stype = TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle); \
    TFC_PTRCHECK(stype);


///////////////////////////////////
// tfVertexSolverSurfaceTypeSpec //
///////////////////////////////////


/**
 * @brief Get a default definition
 * 
 */
struct tfVertexSolverSurfaceTypeSpec tfVertexSolverSurfaceTypeSpec_init() {
    return {
        NULL,   // tfFloatP_t *edgeTensionLam
        1,      // unsigned int edgeTensionOrder
        NULL,   // tfFloatP_t *normalStressMag
        NULL,   // tfFloatP_t *surfaceAreaLam
        NULL,   // tfFloatP_t *surfaceAreaVal
        NULL,   // tfFloatP_t *surfaceTractionComps
        NULL,   // char *name
        NULL,   // struct tfVertexSolverSurfaceTypeStyleSpec *style
        NULL,   // char **adhesionNames
        NULL,   // tfFloatP_t *adhesionValues
        0,      // unsigned int numAdhesionValues
    };
}


////////////////////////////////////////
// tfVertexSolverSurfaceTypeStyleSpec //
////////////////////////////////////////


/**
 * @brief Get a default definition
 * 
 */
struct tfVertexSolverSurfaceTypeStyleSpec tfVertexSolverSurfaceTypeStyleSpec_init() {
    return {
        NULL,       // char *color
        1,          // unsigned int visible
    };
}


///////////////////
// SurfaceHandle //
///////////////////


HRESULT tfVertexSolverSurfaceHandle_init(struct tfVertexSolverSurfaceHandleHandle *handle, int id) {
    TFC_PTRCHECK(handle);
    SurfaceHandle *_tfObj = new SurfaceHandle(id);
    handle->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_fromString(struct tfVertexSolverSurfaceHandleHandle *handle, const char *s) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(s);
    SurfaceHandle *_tfObj = new SurfaceHandle(SurfaceHandle::fromString(s));
    handle->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_destroy(struct tfVertexSolverSurfaceHandleHandle *handle) {
    return TissueForge::capi::destroyHandle<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle);
}

HRESULT tfVertexSolverSurfaceHandle_getId(struct tfVertexSolverSurfaceHandleHandle *handle, int *objId) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(objId);
    *objId = shandle->id;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_definesBody(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *b, 
    bool *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(b);
    TFC_PTRCHECK(result);
    BodyHandle *_b = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(b);
    TFC_PTRCHECK(_b);
    return shandle->defines(*_b);
}

HRESULT tfVertexSolverSurfaceHandle_definedByVertex(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    bool *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    TFC_PTRCHECK(result);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    return shandle->definedBy(*_v);
}

HRESULT tfVertexSolverSurfaceHandle_objType(struct tfVertexSolverSurfaceHandleHandle *handle, unsigned int *label) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(label);
    *label = shandle->objType();
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_destroySurface(struct tfVertexSolverSurfaceHandleHandle *handle) {
    TFC_SURFACEHANDLE_GET(handle);
    return shandle->destroy();
}

HRESULT tfVertexSolverSurfaceHandle_destroySurfaceC(struct tfVertexSolverSurfaceHandleHandle *handle) {
    TFC_SURFACEHANDLE_GET(handle);
    return Surface::destroy(*shandle);
}

HRESULT tfVertexSolverSurfaceHandle_validate(struct tfVertexSolverSurfaceHandleHandle *handle, bool *result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = shandle->validate();
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_positionChanged(struct tfVertexSolverSurfaceHandleHandle *handle) {
    TFC_SURFACEHANDLE_GET(handle);
    return shandle->positionChanged();
}

HRESULT tfVertexSolverSurfaceHandle_str(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(shandle->str(), str, numChars);
}

HRESULT tfVertexSolverSurfaceHandle_toString(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(shandle->toString(), str, numChars);
}

HRESULT tfVertexSolverSurfaceHandle_addVertex(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverVertexHandleHandle *v) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    return shandle->add(*_v);
}

HRESULT tfVertexSolverSurfaceHandle_insertVertexAt(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    const int &idx
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    return shandle->insert(*_v, idx);
}

HRESULT tfVertexSolverSurfaceHandle_insertVertexBefore(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    struct tfVertexSolverVertexHandleHandle *before
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    TFC_PTRCHECK(before);
    VertexHandle *_before = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(before);
    TFC_PTRCHECK(_before);
    return shandle->insert(*_v, *_before);
}

HRESULT tfVertexSolverSurfaceHandle_insertVertexBetween(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toInsert, 
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(toInsert);
    VertexHandle *_toInsert = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(toInsert);
    TFC_PTRCHECK(_toInsert);
    TFC_PTRCHECK(v1);
    VertexHandle *_v1 = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v1);
    TFC_PTRCHECK(_v1);
    TFC_PTRCHECK(v2);
    VertexHandle *_v2 = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v2);
    TFC_PTRCHECK(_v2);
    return shandle->insert(*_toInsert, *_v1, *_v2);
}

HRESULT tfVertexSolverSurfaceHandle_removeVertex(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverVertexHandleHandle *v) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    return shandle->remove(*_v);
}

HRESULT tfVertexSolverSurfaceHandle_replaceVertexAt(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toInsert, 
    const int &idx
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(toInsert);
    VertexHandle *_toInsert = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(toInsert);
    TFC_PTRCHECK(_toInsert);
    return shandle->replace(*_toInsert, idx);
}

HRESULT tfVertexSolverSurfaceHandle_replaceVertexWith(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toInsert, 
    struct tfVertexSolverVertexHandleHandle *toRemove
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(toInsert);
    VertexHandle *_toInsert = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(toInsert);
    TFC_PTRCHECK(_toInsert);
    TFC_PTRCHECK(toRemove);
    VertexHandle *_toRemove = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(toRemove);
    TFC_PTRCHECK(_toRemove);
    return shandle->replace(*_toInsert, *_toRemove);
}

HRESULT tfVertexSolverSurfaceHandle_addBody(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverBodyHandleHandle *b) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(b);
    BodyHandle *_b = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(b);
    TFC_PTRCHECK(_b);
    return shandle->add(*_b);
}

HRESULT tfVertexSolverSurfaceHandle_removeBody(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverBodyHandleHandle *b) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(b);
    BodyHandle *_b = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(b);
    TFC_PTRCHECK(_b);
    return shandle->remove(*_b);
}

HRESULT tfVertexSolverSurfaceHandle_replaceBodyAt(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *toInsert, 
    const int &idx
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(toInsert);
    BodyHandle *_toInsert = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(toInsert);
    TFC_PTRCHECK(_toInsert);
    return shandle->replace(*_toInsert, idx);
}

HRESULT tfVertexSolverSurfaceHandle_replaceBodyWith(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *toInsert, 
    struct tfVertexSolverBodyHandleHandle *toRemove
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(toInsert);
    BodyHandle *_toInsert = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(toInsert);
    TFC_PTRCHECK(_toInsert);
    TFC_PTRCHECK(toRemove);
    BodyHandle *_toRemove = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(toRemove);
    TFC_PTRCHECK(_toRemove);
    return shandle->replace(*_toInsert, *_toRemove);
}

HRESULT tfVertexSolverSurfaceHandle_refreshBodies(struct tfVertexSolverSurfaceHandleHandle *handle) {
    TFC_SURFACEHANDLE_GET(handle);
    return shandle->refreshBodies();
}

HRESULT tfVertexSolverSurfaceHandle_getType(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverSurfaceTypeHandle *stype) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(stype);
    SurfaceType *_stype = shandle->type();
    TFC_PTRCHECK(_stype);
    stype->tfObj = (void*)_stype;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_become(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverSurfaceTypeHandle *stype) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(stype);
    SurfaceType *_stype = TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(stype);
    TFC_PTRCHECK(_stype);
    return shandle->become(_stype);
}

HRESULT tfVertexSolverSurfaceHandle_getBodies(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<BodyHandle> _objs = shandle->getBodies();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverBodyHandleHandle*)malloc(sizeof(struct tfVertexSolverBodyHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverBodyHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getVertices(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(numObjs);
    std::vector<VertexHandle> _objs = shandle->getVertices();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverVertexHandleHandle*)malloc(sizeof(struct tfVertexSolverVertexHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverVertexHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_findVertex(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverVertexHandleHandle *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(dir);
    TFC_PTRCHECK(result);
    FVector3 _dir = FVector3::from(dir);
    VertexHandle _result = shandle->findVertex(_dir);
    return tfVertexSolverVertexHandle_init(result, _result.id);
}

HRESULT tfVertexSolverSurfaceHandle_findBody(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverBodyHandleHandle *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(dir);
    TFC_PTRCHECK(result);
    FVector3 _dir = FVector3::from(dir);
    BodyHandle _result = shandle->findBody(_dir);
    return tfVertexSolverBodyHandle_init(result, _result.id);
}

HRESULT tfVertexSolverSurfaceHandle_neighborVertices(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    TFC_PTRCHECK(v1);
    TFC_PTRCHECK(v2);
    VertexHandle _v1, _v2;
    std::tie(_v1, _v2) = shandle->neighborVertices(*_v);
    if(tfVertexSolverVertexHandle_init(v1, _v1.id) != S_OK || tfVertexSolverVertexHandle_init(v1, _v2.id) != S_OK) 
        return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_neighborSurfaces(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<SurfaceHandle> _objs = shandle->neighborSurfaces();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_connectedSurfacesS(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **verts, 
    unsigned int numVerts, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(verts);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);

    std::vector<VertexHandle> _verts;
    _verts.reserve(numVerts);
    for(size_t i = 0; i < numVerts; i++) {
        VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(verts[i]);
        TFC_PTRCHECK(_v);
        _verts.emplace_back(_v->id);
    }
    
    std::vector<SurfaceHandle> _objs = shandle->connectedSurfaces(_verts);
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_connectedSurfaces(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<SurfaceHandle> _objs = shandle->connectedSurfaces();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_connectingVertices(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *other, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
) {
    TFC_SURFACEHANDLE_GET(handle);
    SurfaceHandle *_other = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(other);
    TFC_PTRCHECK(_other);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<VertexHandle> _objs = shandle->connectingVertices(*_other);
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverVertexHandleHandle*)malloc(sizeof(struct tfVertexSolverVertexHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverVertexHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_contiguousVertexLabels(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *other, 
    unsigned int **labels, 
    int *numLabels
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(other);
    SurfaceHandle *_other = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(other);
    TFC_PTRCHECK(_other);
    TFC_PTRCHECK(labels);
    TFC_PTRCHECK(numLabels);
    std::vector<unsigned int> _labels = shandle->contiguousVertexLabels(*_other);
    *numLabels = _labels.size();
    if(*numLabels == 0) 
        return S_OK;
    *labels = (unsigned int*)malloc(sizeof(unsigned int) * _labels.size());
    memcpy(&(*labels)[0], &_labels.data()[0], sizeof(unsigned int) * _labels.size());
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_sharedContiguousVertices(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *other, 
    unsigned int edgeLabel, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
) {
    TFC_SURFACEHANDLE_GET(handle);
    SurfaceHandle *_other = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(other);
    TFC_PTRCHECK(_other);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<VertexHandle> _objs = shandle->sharedContiguousVertices(*_other, edgeLabel);
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverVertexHandleHandle*)malloc(sizeof(struct tfVertexSolverVertexHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverVertexHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getNormal(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t **result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    FVector3 _result = shandle->getNormal();
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getUnnormalizedNormal(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t **result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    FVector3 _result = shandle->getUnnormalizedNormal();
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getCentroid(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t **result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    FVector3 _result = shandle->getCentroid();
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getVelocity(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t **result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    FVector3 _result = shandle->getVelocity();
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getArea(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t *result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = shandle->getArea();
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getPerimeter(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t *result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = shandle->getPerimeter();
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getDensity(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t *result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = shandle->getDensity();
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_setDensity(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t density) {
    TFC_SURFACEHANDLE_GET(handle);
    shandle->setDensity(density);
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getMass(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t *result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    //
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_volumeSense(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *body, 
    tfFloatP_t *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(body);
    TFC_PTRCHECK(result);
    BodyHandle *_body = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(body);
    TFC_PTRCHECK(_body);
    *result = shandle->volumeSense(*_body);
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getVolumeContr(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *body, 
    tfFloatP_t *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(body);
    TFC_PTRCHECK(result);
    BodyHandle *_body = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(body);
    TFC_PTRCHECK(_body);
    *result = shandle->getVolumeContr(*_body);
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getOutwardNormal(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *body, 
    tfFloatP_t **result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(body);
    TFC_PTRCHECK(result);
    BodyHandle *_body = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(body);
    TFC_PTRCHECK(_body);
    FVector3 _result = shandle->getOutwardNormal(*_body);
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getVertexArea(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    TFC_PTRCHECK(result);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    *result = shandle->getVertexArea(*_v);
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getVertexMass(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v);
    TFC_PTRCHECK(result);
    VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v);
    TFC_PTRCHECK(_v);
    *result = shandle->getVertexMass(*_v);
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_hasStyle(struct tfVertexSolverSurfaceHandleHandle *handle, bool *result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = shandle->getStyle() != NULL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_getStyle(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfRenderingStyleHandle *result) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    rendering::Style *_result = shandle->getStyle();
    TFC_PTRCHECK(_result);
    result->tfObj = (void*)_result;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_setStyle(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfRenderingStyleHandle *s) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    rendering::Style *_s = TissueForge::castC<rendering::Style, tfRenderingStyleHandle>(s);
    TFC_PTRCHECK(_s);
    return shandle->setStyle(_s);
}

HRESULT tfVertexSolverSurfaceHandle_normalDistance(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *pos, 
    tfFloatP_t *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(result);
    FVector3 _pos = FVector3::from(pos);
    *result = shandle->normalDistance(_pos);
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_isOutside(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *pos, 
    bool *result
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(result);
    FVector3 _pos = FVector3::from(pos);
    *result = shandle->isOutside(_pos);
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_contains(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *pos, 
    bool *result, 
    struct tfVertexSolverVertexHandleHandle *v0, 
    struct tfVertexSolverVertexHandleHandle *v1
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(result);
    TFC_PTRCHECK(v0);
    TFC_PTRCHECK(v1);
    FVector3 _pos = FVector3::from(pos);
    VertexHandle _v0, _v1;
    *result = shandle->contains(_pos, _v0, _v1);
    if(*result) {
        tfVertexSolverVertexHandle_init(v0, _v0.id);
        tfVertexSolverVertexHandle_init(v1, _v1.id);
    }
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_merge(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toRemove, 
    tfFloatP_t *lenCfs, 
    unsigned int numLenCfs
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(toRemove);
    TFC_PTRCHECK(lenCfs);
    SurfaceHandle *_toRemove = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(toRemove);
    std::vector<FloatP_t> _lenCfs;
    _lenCfs.reserve(numLenCfs);
    for(size_t i = 0; i < numLenCfs; i++) 
        _lenCfs.push_back(lenCfs[i]);
    if(shandle->merge(*_toRemove, _lenCfs) == S_OK && tfVertexSolverSurfaceHandle_destroy(toRemove) != S_OK) 
        return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceHandle_extend(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    unsigned int vertIdxStart, 
    tfFloatP_t *pos, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(newObj);
    FVector3 _pos = FVector3::from(pos);
    SurfaceHandle _newObj = shandle->extend(vertIdxStart, _pos);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverSurfaceHandle_extrude(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    unsigned int vertIdxStart, 
    tfFloatP_t normLen, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(newObj);
    SurfaceHandle _newObj = shandle->extrude(vertIdxStart, normLen);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverSurfaceHandle_splitBy(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(v1);
    VertexHandle *_v1 = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v1);
    TFC_PTRCHECK(_v1);
    TFC_PTRCHECK(v2);
    VertexHandle *_v2 = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v2);
    TFC_PTRCHECK(_v2);
    TFC_PTRCHECK(newObj);
    SurfaceHandle _newObj = shandle->split(*_v1, *_v2);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverSurfaceHandle_splitHow(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *cp_pos, 
    tfFloatP_t *cp_norm, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACEHANDLE_GET(handle);
    TFC_PTRCHECK(cp_pos);
    TFC_PTRCHECK(cp_norm);
    TFC_PTRCHECK(newObj);
    FVector3 _cp_pos = FVector3::from(cp_pos);
    FVector3 _cp_norm = FVector3::from(cp_norm);
    SurfaceHandle _newObj = shandle->split(_cp_pos, _cp_norm);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}


/////////////////
// SurfaceType //
/////////////////


HRESULT tfVertexSolverSurfaceType_init(struct tfVertexSolverSurfaceTypeHandle *handle) {
    TFC_PTRCHECK(handle);
    SurfaceType *_tfObj = new SurfaceType(true);
    handle->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_initD(struct tfVertexSolverSurfaceTypeHandle *handle, struct tfVertexSolverSurfaceTypeSpec sdef) {
    if(tfVertexSolverSurfaceType_init(handle) != S_OK) 
        return E_FAIL;
    TFC_PTRCHECK(handle);
    TFC_SURFACETYPE_GET(handle);

    if(sdef.name) 
        stype->name = sdef.name;
    if(sdef.style) {
        if(sdef.style->color) 
            stype->style->setColor(sdef.style->color);
        stype->style->setVisible(sdef.style->visible);
    }

    if(sdef.edgeTensionLam && bind::surface(new EdgeTension(*sdef.edgeTensionLam, sdef.edgeTensionOrder), stype) != S_OK) 
        return E_FAIL;
    if(sdef.normalStressMag && bind::surface(new NormalStress(*sdef.normalStressMag), stype) != S_OK) 
        return E_FAIL;
    if(sdef.surfaceAreaLam && sdef.surfaceAreaVal && bind::surface(new SurfaceAreaConstraint(*sdef.surfaceAreaLam, *sdef.surfaceAreaVal), stype) != S_OK) 
        return E_FAIL;
    if(sdef.surfaceTractionComps) {
        FVector3 _surfaceTractionComps = FVector3::from(*sdef.surfaceTractionComps);
        if(bind::surface(new SurfaceTraction(_surfaceTractionComps), stype) != S_OK) 
            return E_FAIL;
    }

    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_fromString(struct tfVertexSolverSurfaceTypeHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    SurfaceType *_tfObj = new SurfaceType(SurfaceType::fromString(str));
    handle->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_objType(struct tfVertexSolverSurfaceTypeHandle *handle, unsigned int *label) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(label);
    *label = stype->objType();
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_str(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(stype->str(), str, numChars);
}

HRESULT tfVertexSolverSurfaceType_toString(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(stype->toString(), str, numChars);
}

HRESULT tfVertexSolverSurfaceType_registerType(struct tfVertexSolverSurfaceTypeHandle *handle) {
    TFC_SURFACETYPE_GET(handle);
    return stype->registerType();
}

HRESULT tfVertexSolverSurfaceType_isRegistered(struct tfVertexSolverSurfaceTypeHandle *handle, bool *result) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(result);
    *result = stype->isRegistered();
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_getName(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(stype->name, str, numChars);
}

HRESULT tfVertexSolverSurfaceType_setName(struct tfVertexSolverSurfaceTypeHandle *handle, const char *str) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(str);
    stype->name = str;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_getStyle(struct tfVertexSolverSurfaceTypeHandle *handle, struct tfRenderingStyleHandle *style) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(style);
    style->tfObj = (void*)stype->style;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_getDensity(struct tfVertexSolverSurfaceTypeHandle *handle, tfFloatP_t *result) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(result);
    *result = stype->density;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_setDensity(struct tfVertexSolverSurfaceTypeHandle *handle, tfFloatP_t result) {
    TFC_SURFACETYPE_GET(handle);
    stype->density = result;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_getInstances(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    unsigned int *numObjs
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<SurfaceHandle> _objs = stype->getInstances();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_getInstanceIds(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    int **ids, 
    unsigned int *numObjs
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(ids);
    TFC_PTRCHECK(numObjs);
    std::vector<int> _ids = stype->getInstanceIds();
    *numObjs = _ids.size();
    if(*numObjs == 0) 
        return S_OK;
    *ids = (int*)malloc(sizeof(int) * _ids.size());
    memcpy(&(*ids), &_ids.data()[0], sizeof(int) * _ids.size());
    return S_OK;
}

HRESULT tfVertexSolverSurfaceType_createSurfaceV(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **vertices, 
    unsigned int numVerts, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(vertices);
    TFC_PTRCHECK(newObj);
    std::vector<VertexHandle> _vertices;
    _vertices.reserve(numVerts);
    for(size_t i = 0; i < numVerts; i++) {
        VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(vertices[i]);
        TFC_PTRCHECK(_v);
        _vertices.emplace_back(_v->id);
    }
    SurfaceHandle _newObj = (*stype)(_vertices);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverSurfaceType_createSurfaceP(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    tfFloatP_t **positions, 
    unsigned int numPositions, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(positions);
    TFC_PTRCHECK(newObj);
    std::vector<FVector3> _positions;
    _positions.reserve(numPositions);
    for(size_t i = 0; i < numPositions; i++) 
        _positions.push_back(FVector3::from(positions[i]));
    SurfaceHandle _newObj = (*stype)(_positions);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverSurfaceType_createSurfaceIO(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    struct tfIoThreeDFFaceDataHandle *face, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(face);
    TFC_PTRCHECK(newObj);
    io::ThreeDFFaceData *_face = TissueForge::castC<io::ThreeDFFaceData, tfIoThreeDFFaceDataHandle>(face);
    TFC_PTRCHECK(_face);
    SurfaceHandle _newObj = (*stype)(_face);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverSurfaceType_nPolygon(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    unsigned int n, 
    tfFloatP_t *center, 
    tfFloatP_t radius, 
    tfFloatP_t *ax1, 
    tfFloatP_t *ax2, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(center);
    TFC_PTRCHECK(radius);
    TFC_PTRCHECK(ax1);
    TFC_PTRCHECK(ax2);
    FVector3 _center = FVector3::from(center);
    FVector3 _ax1 = FVector3::from(ax1);
    FVector3 _ax2 = FVector3::from(ax2);
    SurfaceHandle _newObj = stype->nPolygon(n, _center, radius, _ax1, _ax2);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverSurfaceType_replace(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toReplace, 
    tfFloatP_t *lenCfs, 
    unsigned int numLenCfs, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_SURFACETYPE_GET(handle);
    TFC_PTRCHECK(toReplace);
    VertexHandle *_toReplace = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(toReplace);
    TFC_PTRCHECK(_toReplace);
    TFC_PTRCHECK(lenCfs);
    TFC_PTRCHECK(newObj);
    std::vector<FloatP_t> _lenCfs;
    _lenCfs.reserve(numLenCfs);
    for(size_t i = 0; i < numLenCfs; i++) 
        _lenCfs.push_back(lenCfs[i]);
    SurfaceHandle _newObj = stype->replace(*_toReplace, _lenCfs);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolverCreateSurfaceByVertices(
    struct tfVertexSolverVertexHandleHandle **verts, 
    unsigned int numVerts, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_PTRCHECK(verts);
    TFC_PTRCHECK(newObj);

    std::vector<VertexHandle> _verts;
    _verts.reserve(numVerts);
    for(size_t i = 0; i < numVerts; i++) {
        VertexHandle *_v = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(verts[i]);
        TFC_PTRCHECK(_v);
        _verts.emplace_back(_v->id);
    }

    SurfaceHandle _newObj = Surface::create(_verts);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverCreateSurfaceByIOData(
    struct tfIoThreeDFFaceDataHandle *face, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
) {
    TFC_PTRCHECK(face);
    TFC_PTRCHECK(newObj);
    io::ThreeDFFaceData *_face = TissueForge::castC<io::ThreeDFFaceData, tfIoThreeDFFaceDataHandle>(face);
    TFC_PTRCHECK(_face);
    SurfaceHandle _newObj = Surface::create(_face);
    return tfVertexSolverSurfaceHandle_init(newObj, _newObj.id);
}

HRESULT tfVertexSolverFindSurfaceTypeFromName(const char *name, struct tfVertexSolverSurfaceTypeHandle *result) {
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(result);
    SurfaceType *stype = SurfaceType::findFromName(name);
    TFC_PTRCHECK(stype);
    result->tfObj = (void*)stype;
    return S_OK;
}

HRESULT tfVertexSolverBindSurfaceTypeAdhesion(
    struct tfVertexSolverSurfaceTypeHandle **stypes, 
    struct tfVertexSolverSurfaceTypeSpec *sdefs, 
    unsigned int numTypes
) {
    TFC_PTRCHECK(stypes);
    TFC_PTRCHECK(sdefs);

    std::vector<SurfaceType*> _stypes;
    _stypes.reserve(numTypes);
    for(size_t i = 0; i < numTypes; i++) {
        SurfaceType *_st = TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(stypes[i]);
        TFC_PTRCHECK(_st);
        _stypes.push_back(_st);
    }

    std::unordered_map<std::string, std::unordered_map<std::string, FloatP_t> > adhesionMap;
    for(auto &st : _stypes) 
        adhesionMap.insert({st->name, {}});

    for(size_t i = 0; i < numTypes; i++) {
        tfVertexSolverSurfaceTypeSpec sdef = sdefs[i];
        if(sdef.adhesionNames && sdef.adhesionValues && sdef.numAdhesionValues > 0) {
            auto itr_i = adhesionMap.find(sdef.name);
            if(itr_i == adhesionMap.end()) 
                continue;
            for(size_t j = 0; j < sdef.numAdhesionValues; j++) {
                auto name = sdef.adhesionNames[j];
                auto itr_j = adhesionMap.find(name);
                if(itr_j != adhesionMap.end()) {
                    auto val = sdef.adhesionValues[j];
                    itr_i->second.insert({name, val});
                    itr_j->second.insert({sdef.name, val});
                }
            }
        }
    }

    for(size_t i = 0; i < _stypes.size(); i++) {
        SurfaceType *_st_i = _stypes[i];
        auto itr_i = adhesionMap.find(_st_i->name);
        for(size_t j = i; i < _stypes.size(); j++) {
            SurfaceType *_st_j = _stypes[j];
            // Prevent duplicates
            if(_st_j->id > _st_i->id) 
                continue;
            auto itr_j = itr_i->second.find(_st_j->name);
            if(itr_j == itr_i->second.end()) 
                continue;
            if(bind::types(new Adhesion(itr_j->second), _st_i, _st_j) != S_OK) 
                return E_FAIL;
        }
    }

    return S_OK;
}
