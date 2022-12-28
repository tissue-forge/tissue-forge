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

#include "tfCNormalStress.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfNormalStress.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    NormalStress *castC(struct tfVertexSolverNormalStressHandle *handle) {
        return castC<NormalStress, tfVertexSolverNormalStressHandle>(handle);
    }

}

#define TFC_NORMALSTRESS_GET(handle, name) \
    NormalStress *name = TissueForge::castC<NormalStress, tfVertexSolverNormalStressHandle>(handle); \
    TFC_PTRCHECK(name);


//////////////////
// NormalStress //
//////////////////


HRESULT tfVertexSolverNormalStress_init(struct tfVertexSolverNormalStressHandle *handle, tfFloatP_t mag) {
    TFC_PTRCHECK(handle);
    NormalStress *actor = new NormalStress(mag);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverNormalStress_destroy(struct tfVertexSolverNormalStressHandle *handle) {
    return TissueForge::capi::destroyHandle<NormalStress, tfVertexSolverNormalStressHandle>(handle);
}

HRESULT tfVertexSolverNormalStress_getMag(struct tfVertexSolverNormalStressHandle *handle, tfFloatP_t *result) {
    TFC_NORMALSTRESS_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->mag;
    return S_OK;
}

HRESULT tfVertexSolverNormalStress_setMag(struct tfVertexSolverNormalStressHandle *handle, tfFloatP_t mag) {
    TFC_NORMALSTRESS_GET(handle, actor);
    actor->mag = mag;
    return S_OK;
}

HRESULT tfVertexSolverNormalStress_toBase(struct tfVertexSolverNormalStressHandle *handle, struct tfVertexSolverMeshObjActorHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverNormalStress_fromBase(struct tfVertexSolverMeshObjActorHandle *handle, struct tfVertexSolverNormalStressHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}
