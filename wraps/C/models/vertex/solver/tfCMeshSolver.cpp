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

#include "tfCMeshSolver.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tfMeshSolver.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    static VertexHandle *castC(struct tfVertexSolverVertexHandleHandle *handle) {
        return castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handle);
    }

    static SurfaceHandle *castC(struct tfVertexSolverSurfaceHandleHandle *handle) {
        return castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle);
    }

    static SurfaceType *castC(struct tfVertexSolverSurfaceTypeHandle *handle) {
        return castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle);
    }

    static BodyHandle *castC(struct tfVertexSolverBodyHandleHandle *handle) {
        return castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle);
    }

    static BodyType *castC(struct tfVertexSolverBodyTypeHandle *handle) {
        return castC<BodyType, tfVertexSolverBodyTypeHandle>(handle);
    }

}

#define TFC_MESHSOLVER_GETVERTEXHANDLE(handle, name) \
    VertexHandle *name = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHSOLVER_GETSURFACEHANDLE(handle, name) \
    SurfaceHandle *name = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHSOLVER_GETSURFACETYPE(handle, name) \
    SurfaceType *name = TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHSOLVER_GETBODYHANDLE(handle, name) \
    BodyHandle *name = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHSOLVER_GETBODYTYPE(handle, name) \
    BodyType *name = TissueForge::castC<BodyType, tfVertexSolverBodyTypeHandle>(handle); \
    TFC_PTRCHECK(name);


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolverVertexForce(struct tfVertexSolverVertexHandleHandle *v, tfFloatP_t *f) {
    TFC_MESHSOLVER_GETVERTEXHANDLE(v, _v);
    TFC_PTRCHECK(f);
    Vertex *_vObj = _v->vertex();
    TFC_PTRCHECK(_vObj);
    return VertexForce(_vObj, f);
}

HRESULT tfVertexSolverInit() {
    return MeshSolver::init();
}

HRESULT tfVertexSolverCompact() {
    return MeshSolver::compact();
}

HRESULT tfVertexSolverEngineLock() {
    return MeshSolver::engineLock();
}

HRESULT tfVertexSolverEngineUnlock() {
    return MeshSolver::engineUnlock();
}

HRESULT tfVertexSolverIsDirty(bool *result) {
    TFC_PTRCHECK(result);
    *result = MeshSolver::isDirty();
    return S_OK;
}

HRESULT tfVertexSolverSetDirty(bool isDirty) {
    return MeshSolver::setDirty(isDirty);
}

HRESULT tfVertexSolverRegisterBodyType(struct tfVertexSolverBodyTypeHandle *type) {
    TFC_MESHSOLVER_GETBODYTYPE(type, _type);
    return MeshSolver::registerType(_type);
}

HRESULT tfVertexSolverRegisterSurfaceType(struct tfVertexSolverSurfaceTypeHandle *type) {
    TFC_MESHSOLVER_GETSURFACETYPE(type, _type);
    return MeshSolver::registerType(_type);
}

HRESULT tfVertexSolverGetBodyType(const unsigned int &typeId, struct tfVertexSolverBodyTypeHandle *type) {
    TFC_PTRCHECK(type);
    BodyType *_type = MeshSolver::getBodyType(typeId);
    TFC_PTRCHECK(_type);
    type->tfObj = (void*)_type;
    return S_OK;
}

HRESULT tfVertexSolverGetSurfaceType(const unsigned int &typeId, struct tfVertexSolverSurfaceTypeHandle *type) {
    TFC_PTRCHECK(type);
    SurfaceType *_type = MeshSolver::getSurfaceType(typeId);
    TFC_PTRCHECK(_type);
    type->tfObj = (void*)_type;
    return S_OK;
}

HRESULT tfVertexSolverNumBodyTypes(int *numTypes) {
    TFC_PTRCHECK(numTypes);
    *numTypes = MeshSolver::numBodyTypes();
    return S_OK;
}

HRESULT tfVertexSolverNumSurfaceTypes(int *numTypes) {
    TFC_PTRCHECK(numTypes);
    *numTypes = MeshSolver::numSurfaceTypes();
    return S_OK;
}

HRESULT tfVertexSolverPositionChanged() {
    return MeshSolver::positionChanged();
}

HRESULT tfVertexSolverUpdate(bool force) {
    return MeshSolver::update(force);
}
