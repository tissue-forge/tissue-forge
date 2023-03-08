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

#include "tfCSurfaceAreaConstraint.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfSurfaceAreaConstraint.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    SurfaceAreaConstraint *castC(struct tfVertexSolverSurfaceAreaConstraintHandle *handle) {
        return castC<SurfaceAreaConstraint, tfVertexSolverSurfaceAreaConstraintHandle>(handle);
    }

}

#define TFC_SURFACEAREACONSTRAINT_GET(handle, name) \
    SurfaceAreaConstraint *name = TissueForge::castC<SurfaceAreaConstraint, tfVertexSolverSurfaceAreaConstraintHandle>(handle); \
    TFC_PTRCHECK(name);


///////////////////////////
// SurfaceAreaConstraint //
///////////////////////////


HRESULT tfVertexSolverSurfaceAreaConstraint_init(struct tfVertexSolverSurfaceAreaConstraintHandle *handle, tfFloatP_t lam, tfFloatP_t constr) {
    TFC_PTRCHECK(handle);
    SurfaceAreaConstraint *actor = new SurfaceAreaConstraint(lam, constr);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceAreaConstraint_destroy(struct tfVertexSolverSurfaceAreaConstraintHandle *handle) {
    return TissueForge::capi::destroyHandle<SurfaceAreaConstraint, tfVertexSolverSurfaceAreaConstraintHandle>(handle);
}

HRESULT tfVertexSolverSurfaceAreaConstraint_getLam(struct tfVertexSolverSurfaceAreaConstraintHandle *handle, tfFloatP_t *result) {
    TFC_SURFACEAREACONSTRAINT_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->lam;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceAreaConstraint_setLam(struct tfVertexSolverSurfaceAreaConstraintHandle *handle, tfFloatP_t lam) {
    TFC_SURFACEAREACONSTRAINT_GET(handle, actor);
    actor->lam = lam;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceAreaConstraint_getConstr(struct tfVertexSolverSurfaceAreaConstraintHandle *handle, tfFloatP_t *result) {
    TFC_SURFACEAREACONSTRAINT_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->constr;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceAreaConstraint_setConstr(struct tfVertexSolverSurfaceAreaConstraintHandle *handle, tfFloatP_t constr) {
    TFC_SURFACEAREACONSTRAINT_GET(handle, actor);
    actor->constr = constr;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceAreaConstraint_toBase(struct tfVertexSolverSurfaceAreaConstraintHandle *handle, struct tfVertexSolverMeshObjActorHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverSurfaceAreaConstraint_fromBase(struct tfVertexSolverMeshObjActorHandle *handle, struct tfVertexSolverSurfaceAreaConstraintHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}
