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

#include "tfCEdgeTension.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/actors/tfEdgeTension.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    EdgeTension *castC(struct tfVertexSolverEdgeTensionHandle *handle) {
        return castC<EdgeTension, tfVertexSolverEdgeTensionHandle>(handle);
    }

}

#define TFC_EDGETENSION_GET(handle, name) \
    EdgeTension *name = TissueForge::castC<EdgeTension, tfVertexSolverEdgeTensionHandle>(handle); \
    TFC_PTRCHECK(name);


/////////////////
// EdgeTension //
/////////////////


HRESULT tfVertexSolverEdgeTension_init(struct tfVertexSolverEdgeTensionHandle *handle, tfFloatP_t lam, unsigned int order) {
    TFC_PTRCHECK(handle);
    EdgeTension *actor = new EdgeTension(lam, order);
    handle->tfObj = (void*)actor;
    return S_OK;
}

HRESULT tfVertexSolverEdgeTension_destroy(struct tfVertexSolverEdgeTensionHandle *handle) {
    return TissueForge::capi::destroyHandle<EdgeTension, tfVertexSolverEdgeTensionHandle>(handle);
}

HRESULT tfVertexSolverEdgeTension_getLam(struct tfVertexSolverEdgeTensionHandle *handle, tfFloatP_t *result) {
    TFC_EDGETENSION_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->lam;
    return S_OK;
}

HRESULT tfVertexSolverEdgeTension_setLam(struct tfVertexSolverEdgeTensionHandle *handle, tfFloatP_t lam) {
    TFC_EDGETENSION_GET(handle, actor);
    actor->lam = lam;
    return S_OK;
}

HRESULT tfVertexSolverEdgeTension_getOrder(struct tfVertexSolverEdgeTensionHandle *handle, unsigned int *result) {
    TFC_EDGETENSION_GET(handle, actor);
    TFC_PTRCHECK(result);
    *result = actor->order;
    return S_OK;
}

HRESULT tfVertexSolverEdgeTension_setOrder(struct tfVertexSolverEdgeTensionHandle *handle, unsigned int order) {
    TFC_EDGETENSION_GET(handle, actor);
    actor->order = order;
    return S_OK;
}

HRESULT tfVertexSolverEdgeTension_toBase(struct tfVertexSolverEdgeTensionHandle *handle, struct tfVertexSolverMeshObjActorHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}

HRESULT tfVertexSolverEdgeTension_fromBase(struct tfVertexSolverMeshObjActorHandle *handle, struct tfVertexSolverEdgeTensionHandle *result) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(result);
    result->tfObj = handle->tfObj;
    return S_OK;
}
