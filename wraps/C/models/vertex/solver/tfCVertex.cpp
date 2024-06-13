/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego and Tien Comlekoglu
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

#include "tfCVertex.h"

#include "tfCBody.h"
#include "tfCSurface.h"

#include "TissueForge_c_private.h"

#include <models/vertex/solver/tfVertex.h>

#include <io/tfThreeDFVertexData.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    VertexHandle *castC(struct tfVertexSolverVertexHandleHandle *handle) {
        return castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handle);
    }

}

#define TFC_VERTEXHANDLE_GET(handle) \
    VertexHandle *vhandle = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handle); \
    TFC_PTRCHECK(vhandle);


////////////
// Vertex //
////////////


HRESULT tfVertexSolverVertexHandle_init(struct tfVertexSolverVertexHandleHandle *handle, int id) {
    TFC_PTRCHECK(handle);
    VertexHandle *_tfObj = new VertexHandle(id);
    handle->tfObj = (void*)_tfObj;
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_fromString(struct tfVertexSolverVertexHandleHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    VertexHandle *result = new VertexHandle(VertexHandle::fromString(str));
    handle->tfObj = (void*)result;
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_destroy(struct tfVertexSolverVertexHandleHandle *handle) {
    return TissueForge::capi::destroyHandle<VertexHandle, tfVertexSolverVertexHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfVertexSolverVertexHandle_getId(struct tfVertexSolverVertexHandleHandle *handle, int *objId) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(objId);
    *objId = vhandle->id;
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_definesSurface(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    bool *result
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    TFC_PTRCHECK(result);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    *result = vhandle->defines(*_s);
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_definesBody(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *b, 
    bool *result
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(b);
    TFC_PTRCHECK(result);
    BodyHandle *_b = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(b);
    TFC_PTRCHECK(_b);
    *result = vhandle->defines(*_b);
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_objType(struct tfVertexSolverVertexHandleHandle *handle, int *label) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(label);
    *label = vhandle->objType();
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_destroyVertex(struct tfVertexSolverVertexHandleHandle *handle) {
    TFC_VERTEXHANDLE_GET(handle);
    return vhandle->destroy();
}

HRESULT tfVertexSolverVertexHandle_destroyVertices(struct tfVertexSolverVertexHandleHandle **handles, unsigned int numObjs) {
    TFC_PTRCHECK(handles);
    std::vector<Vertex*> vertices(numObjs);
    for(unsigned int i = 0; i < numObjs; i++) 
        vertices[i] = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handles[i])->vertex();
    return Vertex::destroy(vertices);
}

HRESULT tfVertexSolverVertexHandle_validate(struct tfVertexSolverVertexHandleHandle *handle, bool *result) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = vhandle->validate();
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_positionChanged(struct tfVertexSolverVertexHandleHandle *handle) {
    TFC_VERTEXHANDLE_GET(handle);
    return vhandle->positionChanged();
}

HRESULT tfVertexSolverVertexHandle_str(struct tfVertexSolverVertexHandleHandle *handle, char **str, unsigned int *numChars) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(vhandle->str(), str, numChars);
}

HRESULT tfVertexSolverVertexHandle_toString(struct tfVertexSolverVertexHandleHandle *handle, char **str, unsigned int *numChars) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(vhandle->toString(), str, numChars);
}

HRESULT tfVertexSolverVertexHandle_addSurface(struct tfVertexSolverVertexHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    return vhandle->add(*_s);
}

HRESULT tfVertexSolverVertexHandle_insertSurfaceAt(struct tfVertexSolverVertexHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s, int idx) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    return vhandle->insert(*_s, idx);
}

HRESULT tfVertexSolverVertexHandle_insertSurfaceBefore(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    struct tfVertexSolverSurfaceHandleHandle *before
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    TFC_PTRCHECK(before);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    SurfaceHandle *_before = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(before);
    TFC_PTRCHECK(_before);
    return vhandle->insert(*_s, *_before);
}

HRESULT tfVertexSolverVertexHandle_remove(struct tfVertexSolverVertexHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(s);
    SurfaceHandle *_s = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(s);
    TFC_PTRCHECK(_s);
    return vhandle->remove(*_s);
}

HRESULT tfVertexSolverVertexHandle_replaceSurfaceAt(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toInsert, 
    int idx
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(toInsert);
    SurfaceHandle *_toInsert = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(toInsert);
    TFC_PTRCHECK(_toInsert);
    return vhandle->replace(*_toInsert, idx);
}

HRESULT tfVertexSolverVertexHandle_replaceSurfaceWith(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toInsert, 
    struct tfVertexSolverSurfaceHandleHandle *toRemove
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(toInsert);
    TFC_PTRCHECK(toRemove);
    SurfaceHandle *_toInsert = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(toInsert);
    TFC_PTRCHECK(_toInsert);
    SurfaceHandle *_toRemove = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(toRemove);
    TFC_PTRCHECK(_toRemove);
    return vhandle->replace(*_toInsert, *_toRemove);
}

HRESULT tfVertexSolverVertexHandle_getPartId(struct tfVertexSolverVertexHandleHandle *handle, int *result) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = vhandle->getPartId();
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_getBodies(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<BodyHandle> _objs = vhandle->getBodies();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverBodyHandleHandle*)malloc(sizeof(struct tfVertexSolverBodyHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverBodyHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_getSurfaces(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<SurfaceHandle> _objs = vhandle->getSurfaces();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_findSurface(
    struct tfVertexSolverVertexHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverSurfaceHandleHandle *result
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(dir);
    TFC_PTRCHECK(result);
    FVector3 _dir = FVector3::from(dir);
    SurfaceHandle _result = vhandle->findSurface(_dir);
    return tfVertexSolverSurfaceHandle_init(result, _result.id);
}

HRESULT tfVertexSolverVertexHandle_findBody(
    struct tfVertexSolverVertexHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverBodyHandleHandle *result
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(dir);
    TFC_PTRCHECK(result);
    FVector3 _dir = FVector3::from(dir);
    BodyHandle _result = vhandle->findBody(_dir);
    return tfVertexSolverBodyHandle_init(result, _result.id);
}

HRESULT tfVertexSolverVertexHandle_connectedVertices(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(objs);
    TFC_PTRCHECK(numObjs);
    std::vector<VertexHandle> _objs = vhandle->connectedVertices();
    *numObjs = _objs.size();
    if(*numObjs == 0) 
        return S_OK;
    *objs = (struct tfVertexSolverVertexHandleHandle*)malloc(sizeof(struct tfVertexSolverVertexHandleHandle) * _objs.size());
    for(size_t i = 0; i < _objs.size(); i++) 
        if(tfVertexSolverVertexHandle_init(&(*objs)[i], _objs[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_updateConnectedVertices(struct tfVertexSolverVertexHandleHandle *handle) {
    TFC_VERTEXHANDLE_GET(handle);
    vhandle->updateConnectedVertices();
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_sharedSurfaces(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *other, 
    struct tfVertexSolverSurfaceHandleHandle **result, 
    int *numObjs
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(other);
    TFC_PTRCHECK(result);
    TFC_PTRCHECK(numObjs);
    VertexHandle *_other = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(other);
    TFC_PTRCHECK(_other);
    std::vector<SurfaceHandle> _result = vhandle->sharedSurfaces(*_other);
    *numObjs = _result.size();
    if(*numObjs == 0) 
        return S_OK;
    *result = (struct tfVertexSolverSurfaceHandleHandle *)malloc(sizeof(struct tfVertexSolverSurfaceHandleHandle) * _result.size());
    for(size_t i = 0; i < _result.size(); i++) 
        if(tfVertexSolverSurfaceHandle_init(&(*result)[i], _result[i].id) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_getArea(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t *result) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = vhandle->getArea();
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_getVolume(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t *result) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = vhandle->getVolume();
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_getMass(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t *result) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    *result = vhandle->getMass();
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_updateProperties(struct tfVertexSolverVertexHandleHandle *handle) {
    TFC_VERTEXHANDLE_GET(handle);
    return vhandle->updateProperties();
}

HRESULT tfVertexSolverVertexHandle_particle(struct tfVertexSolverVertexHandleHandle *handle, struct tfParticleHandleHandle *result) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    ParticleHandle *_result = vhandle->particle();
    TFC_PTRCHECK(_result);
    return tfParticleHandle_init(result, _result->id);
}

HRESULT tfVertexSolverVertexHandle_getPosition(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t **result) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    FVector3 _result = vhandle->getPosition();
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_setPosition(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t *pos, bool updateChildren) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(pos);
    FVector3 _pos = FVector3::from(pos);
    return vhandle->setPosition(_pos);
}

HRESULT tfVertexSolverVertexHandle_getVelocity(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t **result) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(result);
    FVector3 _result = vhandle->getVelocity();
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_transferBondsTo(struct tfVertexSolverVertexHandleHandle *handle, struct tfVertexSolverVertexHandleHandle *other) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(other);
    VertexHandle *_other = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(other);
    TFC_PTRCHECK(_other);
    return vhandle->transferBondsTo(*_other);
}

HRESULT tfVertexSolverVertexHandle_replaceSurface(struct tfVertexSolverVertexHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *toReplace) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(toReplace);
    SurfaceHandle *_toReplace = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(toReplace);
    TFC_PTRCHECK(_toReplace);
    return vhandle->replace(*_toReplace);
}

HRESULT tfVertexSolverVertexHandle_replaceBody(struct tfVertexSolverVertexHandleHandle *handle, struct tfVertexSolverBodyHandleHandle *toReplace) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(toReplace);
    BodyHandle *_toReplace = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(toReplace);
    TFC_PTRCHECK(_toReplace);
    return vhandle->replace(*_toReplace);
}

HRESULT tfVertexSolverVertexHandle_merge(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toRemove, 
    tfFloatP_t lenCf
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(toRemove);
    VertexHandle *_toRemove = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(toRemove);
    TFC_PTRCHECK(_toRemove);
    if(vhandle->merge(*_toRemove, lenCf) == S_OK && tfVertexSolverVertexHandle_destroy(toRemove) != S_OK) 
        return E_FAIL;
    return S_OK;
}

HRESULT tfVertexSolverVertexHandle_mergeA(
    struct tfVertexSolverVertexHandleHandle ***handles, 
    unsigned int numMerges, 
    unsigned int *numVertices, 
    tfFloatP_t lenCf
) {
    TFC_PTRCHECK(handles);
    TFC_PTRCHECK(numVertices);
    std::vector<std::vector<Vertex*> > vertices(numMerges);
    for(unsigned int i = 0; i < numMerges; i++) 
        for(unsigned int j = 0; j < numVertices[i]; j++) 
            vertices[i].push_back(TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handles[i][j])->vertex());
    return Vertex::merge(vertices, lenCf);
}

HRESULT tfVertexSolverVertexHandle_insertBetween(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(v1);
    VertexHandle *_v1 = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v1);
    TFC_PTRCHECK(_v1);
    TFC_PTRCHECK(v2);
    VertexHandle *_v2 = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(v2);
    TFC_PTRCHECK(_v2);
    return vhandle->insert(*_v1, *_v2);
}

HRESULT tfVertexSolverVertexHandle_insertBetweenNeighbors(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *vf, 
    struct tfVertexSolverVertexHandleHandle **nbs, 
    int numNbs
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(vf);
    VertexHandle *_vf = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(vf);
    TFC_PTRCHECK(_vf);
    TFC_PTRCHECK(nbs);
    std::vector<VertexHandle> _nbs;
    _nbs.reserve(numNbs);
    for(size_t i = 0; i < numNbs; i++) {
        VertexHandle *_n = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(nbs[i]);
        TFC_PTRCHECK(_n);
        _nbs.push_back(_n->id);
    }
    return vhandle->insert(*_vf, _nbs);
}

HRESULT tfVertexSolverVertexHandle_split(
    struct tfVertexSolverVertexHandleHandle *handle, 
    tfFloatP_t *sep, 
    struct tfVertexSolverVertexHandleHandle *newObj
) {
    TFC_VERTEXHANDLE_GET(handle);
    TFC_PTRCHECK(sep);
    TFC_PTRCHECK(newObj);
    FVector3 _sep = FVector3::from(sep);
    VertexHandle _newObj = vhandle->split(_sep);
    return tfVertexSolverVertexHandle_init(newObj, _newObj.id);
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolverMeshParticleType_get(struct tfParticleTypeHandle *handle) {
    TFC_PTRCHECK(handle);
    MeshParticleType *_handle = MeshParticleType_get();
    TFC_PTRCHECK(_handle);
    handle->tfObj = (void*)_handle;
    return S_OK;
}

HRESULT tfVertexSolverCreateVertexByPartId(unsigned int &pid, int *objId) {
    TFC_PTRCHECK(objId);
    VertexHandle vh = Vertex::create(pid);
    if(vh.id < 0) 
        return E_FAIL;
    *objId = vh.id;
    return S_OK;
}
HRESULT tfVertexSolverCreateVertexByPosition(tfFloatP_t *position, int *objId) {
    TFC_PTRCHECK(objId);
    FVector3 _position = FVector3::from(position);
    VertexHandle vh = Vertex::create(_position);
    if(vh.id < 0) 
        return E_FAIL;
    *objId = vh.id;
    return S_OK;

}
HRESULT tfVertexSolverCreateVertexByIOData(struct tfIoThreeDFVertexDataHandle *vdata, int *objId) {
    TFC_PTRCHECK(vdata);
    io::ThreeDFVertexData *_vdata = TissueForge::castC<io::ThreeDFVertexData, tfIoThreeDFVertexDataHandle>(vdata);
    TFC_PTRCHECK(_vdata);
    TFC_PTRCHECK(objId);
    VertexHandle vh = Vertex::create(_vdata);
    if(vh.id < 0) 
        return E_FAIL;
    *objId = vh.id;
    return S_OK;
}

HRESULT tfVertexSolverCreateVertexByPartIdA(unsigned int *pids, unsigned int numObjs, int **objIds) {
    TFC_PTRCHECK(pids);
    TFC_PTRCHECK(objIds);
    if(numObjs == 0) 
        return S_OK;

    std::vector<unsigned int> _pids(numObjs);
    for(unsigned int i = 0; i < numObjs; i++) 
        _pids[i] = pids[i];

    std::vector<VertexHandle> _newObjs = Vertex::create(_pids);
    for(unsigned int i = 0; i < numObjs; i++) 
        (*objIds)[i] = _newObjs[i].id;
    return S_OK;
}

HRESULT tfVertexSolverCreateVertexByPositionA(tfFloatP_t **positions, unsigned int numObjs, int **objIds) {
    TFC_PTRCHECK(positions);
    TFC_PTRCHECK(objIds);
    if(numObjs == 0) 
        return S_OK;

    std::vector<FVector3> _positions(numObjs);
    for(unsigned int i = 0; i < numObjs; i++) 
        _positions[i] = FVector3::from(positions[i]);

    std::vector<VertexHandle> _newObjs = Vertex::create(_positions);
    for(unsigned int i = 0; i < numObjs; i++) 
        (*objIds)[i] = _newObjs[i].id;
    return S_OK;
}

HRESULT tfVertexSolverCreateVertexByIODataA(struct tfIoThreeDFVertexDataHandle **vdata, unsigned int numObjs, int **objIds) {
    TFC_PTRCHECK(vdata);
    TFC_PTRCHECK(objIds);
    if(numObjs == 0) 
        return S_OK;

    std::vector<io::ThreeDFVertexData*> _vdata(numObjs);
    for(unsigned int i = 0; i < numObjs; i++) 
        _vdata[i] = TissueForge::castC<io::ThreeDFVertexData, tfIoThreeDFVertexDataHandle>(vdata[i]);

    std::vector<VertexHandle> _newObjs = Vertex::create(_vdata);
    for(unsigned int i = 0; i < numObjs; i++) 
        (*objIds)[i] = _newObjs[i].id;
    return S_OK;
}
