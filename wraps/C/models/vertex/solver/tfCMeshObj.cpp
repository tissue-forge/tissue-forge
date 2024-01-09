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

#include "tfCMeshObj.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tfMeshObj.h>
#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfBody.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    MeshObjActor *castC(struct tfVertexSolverMeshObjActorHandle *handle) {
        return castC<MeshObjActor, tfVertexSolverMeshObjActorHandle>(handle);
    }

}

#define TFC_ACTOR_GET(handle, name) \
    MeshObjActor *name = TissueForge::castC<MeshObjActor, tfVertexSolverMeshObjActorHandle>(handle); \
    TFC_PTRCHECK(name);


//////////////////////
// MeshObjTypeLabel //
//////////////////////


HRESULT tfVertexSolverMeshObjTypeLabel_init(struct tfVertexSolverMeshObjTypeLabelHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->NONE = MeshObjTypeLabel::NONE;
    handle->VERTEX = MeshObjTypeLabel::VERTEX;
    handle->SURFACE = MeshObjTypeLabel::SURFACE;
    handle->BODY = MeshObjTypeLabel::BODY;
    return S_OK;
}


//////////////////
// MeshObjActor //
//////////////////


HRESULT tfVertexSolverMeshObjActor_getName(struct tfVertexSolverMeshObjActorHandle *handle) {
    return TissueForge::capi::destroyHandle<MeshObjTypeLabel, tfVertexSolverMeshObjActorHandle>(handle);
}

HRESULT tfVertexSolverMeshObjActor_getName(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_ACTOR_GET(handle, actor);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(actor->name(), str, numChars);
}

HRESULT tfVertexSolverMeshObjActor_toString(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    char **str, 
    unsigned int *numChars
) {
    TFC_ACTOR_GET(handle, actor);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(actor->toString(), str, numChars);
}

HRESULT tfVertexSolverMeshObjActor_getEnergySurface(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *source, 
    struct tfVertexSolverVertexHandleHandle *target, 
    tfFloatP_t *result
) {
    TFC_ACTOR_GET(handle, actor);
    TFC_PTRCHECK(source);
    TFC_PTRCHECK(target);
    TFC_PTRCHECK(result);
    SurfaceHandle *_source = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(source);
    TFC_PTRCHECK(_source);
    VertexHandle *_target = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(target);
    TFC_PTRCHECK(_target);
    Surface *_sourceObj = _source->surface();
    TFC_PTRCHECK(_sourceObj);
    Vertex *_targetObj = _target->vertex();
    TFC_PTRCHECK(_targetObj);
    *result = actor->energy(_sourceObj, _targetObj);
    return S_OK;
}

HRESULT tfVertexSolverMeshObjActor_getForceSurface(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *source, 
    struct tfVertexSolverVertexHandleHandle *target, 
    tfFloatP_t **result
) {
    TFC_ACTOR_GET(handle, actor);
    TFC_PTRCHECK(source);
    TFC_PTRCHECK(target);
    TFC_PTRCHECK(result);
    SurfaceHandle *_source = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(source);
    TFC_PTRCHECK(_source);
    VertexHandle *_target = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(target);
    TFC_PTRCHECK(_target);
    Surface *_sourceObj = _source->surface();
    TFC_PTRCHECK(_sourceObj);
    Vertex *_targetObj = _target->vertex();
    TFC_PTRCHECK(_targetObj);
    FVector3 _result = actor->force(_sourceObj, _targetObj);
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}

HRESULT tfVertexSolverMeshObjActor_getEnergyBody(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *source, 
    struct tfVertexSolverVertexHandleHandle *target, 
    tfFloatP_t *result
) {
    TFC_ACTOR_GET(handle, actor);
    TFC_PTRCHECK(source);
    TFC_PTRCHECK(target);
    TFC_PTRCHECK(result);
    BodyHandle *_source = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(source);
    TFC_PTRCHECK(_source);
    VertexHandle *_target = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(target);
    TFC_PTRCHECK(_target);
    Body *_sourceObj = _source->body();
    TFC_PTRCHECK(_sourceObj);
    Vertex *_targetObj = _target->vertex();
    TFC_PTRCHECK(_targetObj);
    *result = actor->energy(_sourceObj, _targetObj);
    return S_OK;
}

HRESULT tfVertexSolverMeshObjActor_getForceBody(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *source, 
    struct tfVertexSolverVertexHandleHandle *target, 
    tfFloatP_t **result
) {
    TFC_ACTOR_GET(handle, actor);
    TFC_PTRCHECK(source);
    TFC_PTRCHECK(target);
    TFC_PTRCHECK(result);
    BodyHandle *_source = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(source);
    TFC_PTRCHECK(_source);
    VertexHandle *_target = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(target);
    TFC_PTRCHECK(_target);
    Body *_sourceObj = _source->body();
    TFC_PTRCHECK(_sourceObj);
    Vertex *_targetObj = _target->vertex();
    TFC_PTRCHECK(_targetObj);
    FVector3 _result = actor->force(_sourceObj, _targetObj);
    (*result)[0] = _result[0];
    (*result)[1] = _result[1];
    (*result)[2] = _result[2];
    return S_OK;
}


//////////////////////////
// MeshObjTypePairActor //
//////////////////////////


HRESULT tfVertexSolverMeshObjTypePairActor_toBase(
    struct tfVertexSolverMeshObjTypePairActorHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverMeshObjTypePairActor_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverMeshObjTypePairActorHandle *result
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolver_getActorsFromSurfaceByName(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    const char *actorName, 
    struct tfVertexSolverMeshObjActorHandle **actors, 
    unsigned int *numActors
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(actorName);
    SurfaceHandle *_handle = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle);
    TFC_PTRCHECK(_handle);
    Surface *_handleObj = _handle->surface();
    TFC_PTRCHECK(_handleObj);

    std::vector<MeshObjActor*> _actors;
    for(auto &a : _handleObj->actors) 
        if(std::strcmp(a->actorName().c_str(), actorName) == 0) 
            _actors.push_back(a);
    *numActors = _actors.size();
    if(*numActors == 0) 
        return S_OK;

    *actors = (struct tfVertexSolverMeshObjActorHandle*)malloc(sizeof(tfVertexSolverMeshObjActorHandle) * _actors.size());
    for(size_t i = 0; i < _actors.size(); i++) {
        tfVertexSolverMeshObjActorHandle a;
        a.tfObj = (void*)_actors[i];
        (*actors)[i] = a;
    }
    return S_OK;
}

HRESULT tfVertexSolver_getActorsFromBodyByName(
    struct tfVertexSolverBodyHandleHandle *handle, 
    const char *actorName, 
    struct tfVertexSolverMeshObjActorHandle **actors, 
    unsigned int *numActors
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(actorName);
    BodyHandle *_handle = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle);
    TFC_PTRCHECK(_handle);
    Body *_handleObj = _handle->body();
    TFC_PTRCHECK(_handleObj);

    std::vector<MeshObjActor*> _actors;
    for(auto &a : _handleObj->actors) 
        if(std::strcmp(a->actorName().c_str(), actorName) == 0) 
            _actors.push_back(a);
    *numActors = _actors.size();
    if(*numActors == 0) 
        return S_OK;

    *actors = (struct tfVertexSolverMeshObjActorHandle*)malloc(sizeof(tfVertexSolverMeshObjActorHandle) * _actors.size());
    for(size_t i = 0; i < _actors.size(); i++) {
        tfVertexSolverMeshObjActorHandle a;
        a.tfObj = (void*)_actors[i];
        (*actors)[i] = a;
    }
    return S_OK;
}

HRESULT tfVertexSolver_getActorsFromSurfaceTypeByName(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    const char *actorName, 
    struct tfVertexSolverMeshObjActorHandle **actors, 
    unsigned int *numActors
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(actorName);
    SurfaceType *_handle = TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle);
    TFC_PTRCHECK(_handle);

    std::vector<MeshObjActor*> _actors;
    for(auto &a : _handle->actors) 
        if(std::strcmp(a->actorName().c_str(), actorName) == 0) 
            _actors.push_back(a);
    *numActors = _actors.size();
    if(*numActors == 0) 
        return S_OK;

    *actors = (struct tfVertexSolverMeshObjActorHandle*)malloc(sizeof(tfVertexSolverMeshObjActorHandle) * _actors.size());
    for(size_t i = 0; i < _actors.size(); i++) {
        tfVertexSolverMeshObjActorHandle a;
        a.tfObj = (void*)_actors[i];
        (*actors)[i] = a;
    }
    return S_OK;
}

HRESULT tfVertexSolver_getActorsFromBodyTypeByName(
    struct tfVertexSolverBodyTypeHandle *handle, 
    const char *actorName, 
    struct tfVertexSolverMeshObjActorHandle **actors, 
    unsigned int *numActors
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(actorName);
    BodyType *_handle = TissueForge::castC<BodyType, tfVertexSolverBodyTypeHandle>(handle);
    TFC_PTRCHECK(_handle);

    std::vector<MeshObjActor*> _actors;
    for(auto &a : _handle->actors) 
        if(std::strcmp(a->actorName().c_str(), actorName) == 0) 
            _actors.push_back(a);
    *numActors = _actors.size();
    if(*numActors == 0) 
        return S_OK;

    *actors = (struct tfVertexSolverMeshObjActorHandle*)malloc(sizeof(tfVertexSolverMeshObjActorHandle) * _actors.size());
    for(size_t i = 0; i < _actors.size(); i++) {
        tfVertexSolverMeshObjActorHandle a;
        a.tfObj = (void*)_actors[i];
        (*actors)[i] = a;
    }
    return S_OK;
}
