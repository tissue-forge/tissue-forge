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

#include "tfCBodyForce.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfBodyForce.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    BodyForce *castC(struct tfVertexSolverBodyForceHandle *handle) {
        return castC<BodyForce, tfVertexSolverBodyForceHandle>(handle);
    }

}

#define TFC_BODYFORCE_GET(handle, name) \
    BodyForce *name = TissueForge::castC<BodyForce, tfVertexSolverBodyForceHandle>(handle); \
    TFC_PTRCHECK(name);


///////////////
// BodyForce //
///////////////


HRESULT tfVertexSolverBodyForce_init(struct tfVertexSolverBodyForceHandle *handle, tfFloatP_t *comps) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(comps);
    FVector3 _comps = FVector3::from(comps);
    BodyForce *actor = new BodyForce(_comps);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverBodyForce_destroy(struct tfVertexSolverBodyForceHandle *handle) {
    return TissueForge::capi::destroyHandle<BodyForce, tfVertexSolverBodyForceHandle>(handle);
}

HRESULT tfVertexSolverBodyForce_getComps(struct tfVertexSolverBodyForceHandle *handle, tfFloatP_t **comps) {
    TFC_BODYFORCE_GET(handle, actor);
    TFC_PTRCHECK(comps);
    (*comps)[0] = actor->comps[0];
    (*comps)[1] = actor->comps[1];
    (*comps)[2] = actor->comps[2];
    return S_OK;
}

HRESULT tfVertexSolverBodyForce_setComps(struct tfVertexSolverBodyForceHandle *handle, tfFloatP_t *comps) {
    TFC_BODYFORCE_GET(handle, actor);
    TFC_PTRCHECK(comps);
    actor->comps = FVector3::from(comps);
    return S_OK;
}

HRESULT tfVertexSolverBodyForce_toBase(struct tfVertexSolverBodyForceHandle *handle, struct tfVertexSolverMeshObjActorHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    handle->tfObj = result->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverBodyForce_fromBase(struct tfVertexSolverMeshObjActorHandle *handle, struct tfVertexSolverBodyForceHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    handle->tfObj = result->tfObj;
    return S_OK;
}
