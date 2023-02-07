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

#include "tfCFlatSurfaceConstraint.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfFlatSurfaceConstraint.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    FlatSurfaceConstraint *castC(struct tfVertexSolverFlatSurfaceConstraintHandle *handle) {
        return castC<FlatSurfaceConstraint, tfVertexSolverFlatSurfaceConstraintHandle>(handle);
    }

}

#define TFC_FLATSRFCONSTRAINT_GET(handle, name) \
    FlatSurfaceConstraint *name = TissueForge::castC<FlatSurfaceConstraint, tfVertexSolverFlatSurfaceConstraintHandle>(handle); \
    TFC_PTRCHECK(name);


///////////////////////////
// FlatSurfaceConstraint //
///////////////////////////


HRESULT tfVertexSolverFlatSurfaceConstraint_init(struct tfVertexSolverFlatSurfaceConstraintHandle *handle, tfFloatP_t lam) {
    TFC_PTRCHECK(handle);
    FlatSurfaceConstraint *actor = new FlatSurfaceConstraint(lam);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverFlatSurfaceConstraint_destroy(struct tfVertexSolverFlatSurfaceConstraintHandle *handle) {
    return TissueForge::capi::destroyHandle<FlatSurfaceConstraint, tfVertexSolverFlatSurfaceConstraintHandle>(handle);
}

HRESULT tfVertexSolverFlatSurfaceConstraint_getLam(struct tfVertexSolverFlatSurfaceConstraintHandle *handle, tfFloatP_t *result) {
    TFC_FLATSRFCONSTRAINT_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->lam;
    return S_OK;
}

HRESULT tfVertexSolverFlatSurfaceConstraint_setLam(struct tfVertexSolverFlatSurfaceConstraintHandle *handle, tfFloatP_t lam) {
    TFC_FLATSRFCONSTRAINT_GET(handle, actor);
    actor->lam = lam;
    return S_OK;
}

HRESULT tfVertexSolverFlatSurfaceConstraint_toBase(
    struct tfVertexSolverFlatSurfaceConstraintHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverFlatSurfaceConstraint_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverFlatSurfaceConstraintHandle *result
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}
