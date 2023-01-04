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

#include "tfCMesh.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tfCVertex.h>
#include <models/vertex/solver/tfCSurface.h>
#include <models/vertex/solver/tfCBody.h>
#include <models/vertex/solver/tfCMeshQuality.h>

#include <models/vertex/solver/tfMesh.h>
#include <models/vertex/solver/tfMeshQuality.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    static VertexHandle *castC(struct tfVertexSolverVertexHandleHandle *handle) {
        return castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handle);
    }

    static SurfaceHandle *castC(struct tfVertexSolverSurfaceHandleHandle *handle) {
        return castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle);
    }

    static BodyHandle *castC(struct tfVertexSolverBodyHandleHandle *handle) {
        return castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle);
    }

    static MeshQuality *castC(struct tfVertexSolverMeshQualityHandle *handle) {
        return castC<MeshQuality, tfVertexSolverMeshQualityHandle>(handle);
    }

}

#define TFC_MESH_GETVERTEXHANDLE(handle, name) \
    VertexHandle *name = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESH_GETSURFACEHANDLE(handle, name) \
    SurfaceHandle *name = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESH_GETBODYHANDLE(handle, name) \
    BodyHandle *name = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESH_GETQUALITY(handle, name) \
    MeshQuality *name = TissueForge::castC<MeshQuality, tfVertexSolverMeshQualityHandle>(handle); \
    TFC_PTRCHECK(name);


#define TFC_MESH_GET(name) \
    Mesh *name = Mesh::get();  \
    TFC_PTRCHECK(name);


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolverMeshHasQuality(bool *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->hasQuality();
    return S_OK;
}

HRESULT tfVertexSolverMeshGetQuality(struct tfVertexSolverMeshQualityHandle *quality) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(quality);
    if(!mesh->hasQuality()) 
        return E_FAIL;
    MeshQuality *_quality = &mesh->getQuality();
    quality->tfObj = (void*)_quality;
    return S_OK;
}

HRESULT tfVertexSolverMeshSetQuality(struct tfVertexSolverMeshQualityHandle *quality) {
    TFC_MESH_GET(mesh);
    TFC_MESH_GETQUALITY(quality, _quality);
    mesh->setQuality(_quality);
    return S_OK;
}

HRESULT tfVertexSolverMeshQualityWorking(bool *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->qualityWorking();
    return S_OK;
}

HRESULT tfVertexSolverMeshEnsureAvailableVertices(unsigned int numAlloc) {
    TFC_MESH_GET(mesh);
    return mesh->ensureAvailableVertices(numAlloc);
}

HRESULT tfVertexSolverMeshEnsureAvailableSurfaces(unsigned int numAlloc) {
    TFC_MESH_GET(mesh);
    return mesh->ensureAvailableSurfaces(numAlloc);
}

HRESULT tfVertexSolverMeshEnsureAvailableBodies(unsigned int numAlloc) {
    TFC_MESH_GET(mesh);
    return mesh->ensureAvailableBodies(numAlloc);
}

HRESULT tfVertexSolverMeshCreateVertex(struct tfVertexSolverVertexHandleHandle *handle, unsigned int pid) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(handle);
    Vertex *v;
    if(mesh->create(&v, pid) != S_OK) 
        return E_FAIL;
    return tfVertexSolverVertexHandle_init(handle, v->objectId());
}

HRESULT tfVertexSolverMeshCreateSurface(struct tfVertexSolverSurfaceHandleHandle *handle) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(handle);
    Surface *s;
    if(mesh->create(&s) != S_OK) 
        return E_FAIL;
    return tfVertexSolverSurfaceHandle_init(handle, s->objectId());
}

HRESULT tfVertexSolverMeshCreateBody(struct tfVertexSolverBodyHandleHandle *handle) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(handle);
    Body *b;
    if(mesh->create(&b) != S_OK) 
        return E_FAIL;
    return tfVertexSolverBodyHandle_init(handle, b->objectId());
}

HRESULT tfVertexSolverMeshLock() {
    TFC_MESH_GET(mesh);
    mesh->lock();
    return S_OK;
}

HRESULT tfVertexSolverMeshUnlock() {
    TFC_MESH_GET(mesh);
    mesh->unlock();
    return S_OK;
}

HRESULT tfVertexSolverMeshFindVertex(tfFloatP_t *pos, tfFloatP_t tol, struct tfVertexSolverVertexHandleHandle *v) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(pos);
    TFC_PTRCHECK(v);
    FVector3 _pos = FVector3::from(pos);
    Vertex *_v = mesh->findVertex(_pos, tol);
    if(!_v) 
        return E_FAIL;
    return tfVertexSolverVertexHandle_init(v, _v->objectId());
}

HRESULT tfVertexSolverMeshGetVertexByPID(unsigned int pid, struct tfVertexSolverVertexHandleHandle *v) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(v);
    Vertex *_v = mesh->getVertexByPID(pid);
    if(!_v) 
        return E_FAIL;
    return tfVertexSolverVertexHandle_init(v, _v->objectId());
}

HRESULT tfVertexSolverMeshGetVertex(unsigned int idx, struct tfVertexSolverVertexHandleHandle *v) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(v);
    Vertex *_v = mesh->getVertex(idx);
    if(!_v) 
        return E_FAIL;
    return tfVertexSolverVertexHandle_init(v, _v->objectId());
}

HRESULT tfVertexSolverMeshGetSurface(unsigned int idx, struct tfVertexSolverSurfaceHandleHandle *s) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(s);
    Surface *_s = mesh->getSurface(idx);
    if(!_s) 
        return E_FAIL;
    return tfVertexSolverSurfaceHandle_init(s, _s->objectId());
}

HRESULT tfVertexSolverMeshGetBody(unsigned int idx, struct tfVertexSolverBodyHandleHandle *b) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(b);
    Body *_b = mesh->getBody(idx);
    if(!_b) 
        return E_FAIL;
    return tfVertexSolverBodyHandle_init(b, _b->objectId());
}

HRESULT tfVertexSolverMeshNumVertices(unsigned int *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->numVertices();
    return S_OK;
}

HRESULT tfVertexSolverMeshNumSurfaces(unsigned int *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->numSurfaces();
    return S_OK;
}

HRESULT tfVertexSolverMeshNumBodies(unsigned int *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->numBodies();
    return S_OK;
}

HRESULT tfVertexSolverMeshSizeVertices(unsigned int *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->sizeVertices();
    return S_OK;
}

HRESULT tfVertexSolverMeshSizeSurfaces(unsigned int *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->sizeSurfaces();
    return S_OK;
}

HRESULT tfVertexSolverMeshSizeBodies(unsigned int *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->sizeBodies();
    return S_OK;
}

HRESULT tfVertexSolverMeshValidate(bool *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->validate();
    return S_OK;
}

HRESULT tfVertexSolverMeshMakeDirty() {
    TFC_MESH_GET(mesh);
    return mesh->makeDirty();;
}

HRESULT tfVertexSolverMeshConnectedVertices(struct tfVertexSolverVertexHandleHandle *v1, struct tfVertexSolverVertexHandleHandle *v2, bool *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(v1);
    TFC_PTRCHECK(v2);
    TFC_PTRCHECK(result);
    TFC_MESH_GETVERTEXHANDLE(v1, _v1);
    TFC_MESH_GETVERTEXHANDLE(v2, _v2);
    Vertex *_v1Obj = _v1->vertex();
    Vertex *_v2Obj = _v2->vertex();
    TFC_PTRCHECK(_v1Obj);
    TFC_PTRCHECK(_v2Obj);
    *result = mesh->connected(_v1Obj, _v2Obj);
    return S_OK;
}

HRESULT tfVertexSolverMeshConnectedSurfaces(struct tfVertexSolverSurfaceHandleHandle *s1, struct tfVertexSolverSurfaceHandleHandle *s2, bool *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(s1);
    TFC_PTRCHECK(s2);
    TFC_PTRCHECK(result);
    TFC_MESH_GETSURFACEHANDLE(s1, _s1);
    TFC_MESH_GETSURFACEHANDLE(s2, _s2);
    Surface *_s1Obj = _s1->surface();
    Surface *_s2Obj = _s2->surface();
    TFC_PTRCHECK(_s1Obj);
    TFC_PTRCHECK(_s2Obj);
    *result = mesh->connected(_s1Obj, _s2Obj);
    return S_OK;
}

HRESULT tfVertexSolverMeshConnectedBodies(struct tfVertexSolverBodyHandleHandle *b1, struct tfVertexSolverBodyHandleHandle *b2, bool *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(b1);
    TFC_PTRCHECK(b2);
    TFC_PTRCHECK(result);
    TFC_MESH_GETBODYHANDLE(b1, _b1);
    TFC_MESH_GETBODYHANDLE(b2, _b2);
    Body *_b1Obj = _b1->body();
    Body *_b2Obj = _b2->body();
    TFC_PTRCHECK(_b1Obj);
    TFC_PTRCHECK(_b2Obj);
    *result = mesh->connected(_b1Obj, _b2Obj);
    return S_OK;
}

HRESULT tfVertexSolverMeshRemoveVertex(struct tfVertexSolverVertexHandleHandle *v) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(v);
    TFC_MESH_GETVERTEXHANDLE(v, _v);
    Vertex *_vObj = _v->vertex();
    TFC_PTRCHECK(_vObj);
    return mesh->remove(_vObj);
}

HRESULT tfVertexSolverMeshRemoveSurface(struct tfVertexSolverSurfaceHandleHandle *s) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(s);
    TFC_MESH_GETSURFACEHANDLE(s, _s);
    Surface *_sObj = _s->surface();
    TFC_PTRCHECK(_sObj);
    return mesh->remove(_sObj);
}

HRESULT tfVertexSolverMeshRemoveBody(struct tfVertexSolverBodyHandleHandle *b) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(b);
    TFC_MESH_GETBODYHANDLE(b, _b);
    Body *_bObj = _b->body();
    TFC_PTRCHECK(_bObj);
    return mesh->remove(_bObj);
}

HRESULT tfVertexSolverMeshIs3D(bool *result) {
    TFC_MESH_GET(mesh);
    TFC_PTRCHECK(result);
    *result = mesh->is3D();
    return S_OK;
}
