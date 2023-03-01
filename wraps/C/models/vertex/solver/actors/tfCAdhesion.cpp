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

#include "tfCAdhesion.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfAdhesion.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    Adhesion *castC(struct tfVertexSolverAdhesionHandle *handle) {
        return castC<Adhesion, tfVertexSolverAdhesionHandle>(handle);
    }

}

#define TFC_ADHESION_GET(handle, name) \
    Adhesion *name = TissueForge::castC<Adhesion, tfVertexSolverAdhesionHandle>(handle); \
    TFC_PTRCHECK(name);


//////////////
// Adhesion //
//////////////


HRESULT tfVertexSolverAdhesion_init(struct tfVertexSolverAdhesionHandle *handle, tfFloatP_t lam) {
    TFC_PTRCHECK(handle);
    Adhesion *actor = new Adhesion(lam);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverAdhesion_destroy(struct tfVertexSolverAdhesionHandle *handle) {
    return TissueForge::capi::destroyHandle<Adhesion, tfVertexSolverAdhesionHandle>(handle);
}

HRESULT tfVertexSolverAdhesion_getLam(struct tfVertexSolverAdhesionHandle *handle, tfFloatP_t *result) {
    TFC_ADHESION_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->lam;
    return S_OK;
}

HRESULT tfVertexSolverAdhesion_setLam(struct tfVertexSolverAdhesionHandle *handle, tfFloatP_t lam) {
    TFC_ADHESION_GET(handle, actor);
    actor->lam = lam;
    return S_OK;
}

HRESULT tfVertexSolverAdhesion_toBase(struct tfVertexSolverAdhesionHandle *handle, struct tfVertexSolverMeshObjTypePairActorHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverAdhesion_fromBase(struct tfVertexSolverMeshObjTypePairActorHandle *handle, struct tfVertexSolverAdhesionHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}
