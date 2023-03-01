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

#include "tfCPerimeterConstraint.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfPerimeterConstraint.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    PerimeterConstraint *castC(struct tfVertexSolverPerimeterConstraintHandle *handle) {
        return castC<PerimeterConstraint, tfVertexSolverPerimeterConstraintHandle>(handle);
    }

}

#define TFC_PERIMETERCONSTRAINT_GET(handle, name) \
    PerimeterConstraint *name = TissueForge::castC<PerimeterConstraint, tfVertexSolverPerimeterConstraintHandle>(handle); \
    TFC_PTRCHECK(name);


/////////////////////////
// PerimeterConstraint //
/////////////////////////


HRESULT tfVertexSolverPerimeterConstraint_init(struct tfVertexSolverPerimeterConstraintHandle *handle, tfFloatP_t lam, tfFloatP_t constr) {
    TFC_PTRCHECK(handle);
    PerimeterConstraint *actor = new PerimeterConstraint(lam, constr);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverPerimeterConstraint_destroy(struct tfVertexSolverPerimeterConstraintHandle *handle) {
    return TissueForge::capi::destroyHandle<PerimeterConstraint, tfVertexSolverPerimeterConstraintHandle>(handle);
}

HRESULT tfVertexSolverPerimeterConstraint_getLam(struct tfVertexSolverPerimeterConstraintHandle *handle, tfFloatP_t *result) {
    TFC_PERIMETERCONSTRAINT_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->lam;
    return S_OK;
}

HRESULT tfVertexSolverPerimeterConstraint_setLam(struct tfVertexSolverPerimeterConstraintHandle *handle, tfFloatP_t lam) {
    TFC_PERIMETERCONSTRAINT_GET(handle, actor);
    actor->lam = lam;
    return S_OK;
}

HRESULT tfVertexSolverPerimeterConstraint_getConstr(struct tfVertexSolverPerimeterConstraintHandle *handle, tfFloatP_t *result) {
    TFC_PERIMETERCONSTRAINT_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->constr;
    return S_OK;
}

HRESULT tfVertexSolverPerimeterConstraint_setConstr(struct tfVertexSolverPerimeterConstraintHandle *handle, tfFloatP_t constr) {
    TFC_PERIMETERCONSTRAINT_GET(handle, actor);
    actor->constr = constr;
    return S_OK;
}

HRESULT tfVertexSolverPerimeterConstraint_toBase(struct tfVertexSolverPerimeterConstraintHandle *handle, struct tfVertexSolverMeshObjActorHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverPerimeterConstraint_fromBase(struct tfVertexSolverMeshObjActorHandle *handle, struct tfVertexSolverPerimeterConstraintHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}
