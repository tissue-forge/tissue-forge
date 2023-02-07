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

#include "tfCConvexPolygonConstraint.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfConvexPolygonConstraint.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    ConvexPolygonConstraint *castC(struct tfVertexSolverConvexPolygonConstraintHandle *handle) {
        return castC<ConvexPolygonConstraint, tfVertexSolverConvexPolygonConstraintHandle>(handle);
    }

}

#define TFC_CVXPOLYCONSTRAINT_GET(handle, name) \
    ConvexPolygonConstraint *name = TissueForge::castC<ConvexPolygonConstraint, tfVertexSolverConvexPolygonConstraintHandle>(handle); \
    TFC_PTRCHECK(name);


/////////////////////////////
// ConvexPolygonConstraint //
/////////////////////////////


HRESULT tfVertexSolverConvexPolygonConstraint_init(struct tfVertexSolverConvexPolygonConstraintHandle *handle, tfFloatP_t lam) {
    TFC_PTRCHECK(handle);
    ConvexPolygonConstraint *actor = new ConvexPolygonConstraint(lam);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverConvexPolygonConstraint_destroy(struct tfVertexSolverConvexPolygonConstraintHandle *handle) {
    return TissueForge::capi::destroyHandle<ConvexPolygonConstraint, tfVertexSolverConvexPolygonConstraintHandle>(handle);
}

HRESULT tfVertexSolverConvexPolygonConstraint_getLam(struct tfVertexSolverConvexPolygonConstraintHandle *handle, tfFloatP_t *result) {
    TFC_CVXPOLYCONSTRAINT_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->lam;
    return S_OK;
}

HRESULT tfVertexSolverConvexPolygonConstraint_setLam(struct tfVertexSolverConvexPolygonConstraintHandle *handle, tfFloatP_t lam) {
    TFC_CVXPOLYCONSTRAINT_GET(handle, actor);
    actor->lam = lam;
    return S_OK;
}

HRESULT tfVertexSolverConvexPolygonConstraint_toBase(
    struct tfVertexSolverConvexPolygonConstraintHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverConvexPolygonConstraint_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverConvexPolygonConstraintHandle *result
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}
