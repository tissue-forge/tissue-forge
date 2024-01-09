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

#include "tfCVolumeConstraint.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfVolumeConstraint.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    VolumeConstraint *castC(struct tfVertexSolverVolumeConstraintHandle *handle) {
        return castC<VolumeConstraint, tfVertexSolverVolumeConstraintHandle>(handle);
    }

}

#define TFC_VOLUMECONSTRAINT_GET(handle, name) \
    VolumeConstraint *name = TissueForge::castC<VolumeConstraint, tfVertexSolverVolumeConstraintHandle>(handle); \
    TFC_PTRCHECK(name);


//////////////////////
// VolumeConstraint //
//////////////////////


HRESULT tfVertexSolverVolumeConstraint_init(struct tfVertexSolverVolumeConstraintHandle *handle, tfFloatP_t lam, tfFloatP_t constr) {
    TFC_PTRCHECK(handle);
    VolumeConstraint *actor = new VolumeConstraint(lam, constr);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverVolumeConstraint_destroy(struct tfVertexSolverVolumeConstraintHandle *handle) {
    return TissueForge::capi::destroyHandle<VolumeConstraint, tfVertexSolverVolumeConstraintHandle>(handle);
}

HRESULT tfVertexSolverVolumeConstraint_getLam(struct tfVertexSolverVolumeConstraintHandle *handle, tfFloatP_t *result) {
    TFC_VOLUMECONSTRAINT_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->lam;
    return S_OK;
}

HRESULT tfVertexSolverVolumeConstraint_setLam(struct tfVertexSolverVolumeConstraintHandle *handle, tfFloatP_t lam) {
    TFC_VOLUMECONSTRAINT_GET(handle, actor);
    actor->lam = lam;
    return S_OK;
}

HRESULT tfVertexSolverVolumeConstraint_getConstr(struct tfVertexSolverVolumeConstraintHandle *handle, tfFloatP_t *result) {
    TFC_VOLUMECONSTRAINT_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->constr;
    return S_OK;
}

HRESULT tfVertexSolverVolumeConstraint_setConstr(struct tfVertexSolverVolumeConstraintHandle *handle, tfFloatP_t constr) {
    TFC_VOLUMECONSTRAINT_GET(handle, actor);
    actor->constr = constr;
    return S_OK;
}

HRESULT tfVertexSolverVolumeConstraint_toBase(struct tfVertexSolverVolumeConstraintHandle *handle, struct tfVertexSolverMeshObjActorHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverVolumeConstraint_fromBase(struct tfVertexSolverMeshObjActorHandle *handle, struct tfVertexSolverVolumeConstraintHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}
