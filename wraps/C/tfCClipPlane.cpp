/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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

#include "tfCClipPlane.h"

#include "TissueForge_c_private.h"

#include <rendering/tfClipPlane.h>


//////////////////
// Module casts //
//////////////////


using namespace TissueForge;


namespace TissueForge { 


    rendering::ClipPlane *castC(struct tfRenderingClipPlaneHandle *handle) {
        return castC<rendering::ClipPlane, tfRenderingClipPlaneHandle>(handle);
    }

}

#define TFC_LIPPLANE_GET(handle, varname) \
    rendering::ClipPlane *varname = TissueForge::castC<rendering::ClipPlane, tfRenderingClipPlaneHandle>(handle); \
    TFC_PTRCHECK(varname);


//////////////////////////
// rendering::ClipPlane //
//////////////////////////


HRESULT tfRenderingClipPlane_getIndex(struct tfRenderingClipPlaneHandle *handle, int *index) {
    TFC_LIPPLANE_GET(handle, cp);
    TFC_PTRCHECK(index);
    *index = cp->index;
    return S_OK;
}

HRESULT tfRenderingClipPlane_getPoint(struct tfRenderingClipPlaneHandle *handle, float **point) {
    TFC_LIPPLANE_GET(handle, cp);
    TFC_PTRCHECK(point);
    auto _point = cp->getPoint();
    TFC_VECTOR3_COPYFROM(_point, (*point));
    return S_OK;
}

HRESULT tfRenderingClipPlane_getNormal(struct tfRenderingClipPlaneHandle *handle, float **normal) {
    TFC_LIPPLANE_GET(handle, cp);
    TFC_PTRCHECK(normal);
    auto _normal = cp->getNormal();
    TFC_VECTOR3_COPYFROM(_normal, (*normal));
    return S_OK;
}

HRESULT tfRenderingClipPlane_getEquation(struct tfRenderingClipPlaneHandle *handle, float **pe) {
    TFC_LIPPLANE_GET(handle, cp);
    TFC_PTRCHECK(pe);
    auto _pe = cp->getEquation();
    TFC_VECTOR4_COPYFROM(_pe, (*pe));
    return S_OK;
}

HRESULT tfRenderingClipPlane_setEquationE(struct tfRenderingClipPlaneHandle *handle, float *pe) {
    TFC_LIPPLANE_GET(handle, cp);
    TFC_PTRCHECK(pe);
    return cp->setEquation(fVector4::from(pe));
}

HRESULT tfRenderingClipPlane_setEquationPN(struct tfRenderingClipPlaneHandle *handle, float *point, float *normal) {
    TFC_LIPPLANE_GET(handle, cp);
    TFC_PTRCHECK(point);
    TFC_PTRCHECK(normal);
    return cp->setEquation(fVector3::from(point), fVector3::from(normal));
}

HRESULT tfRenderingClipPlane_destroyCP(struct tfRenderingClipPlaneHandle *handle) {
    TFC_LIPPLANE_GET(handle, cp);
    return cp->destroy();
}

HRESULT tfRenderingClipPlane_destroy(struct tfRenderingClipPlaneHandle *handle) {
    return TissueForge::capi::destroyHandle<rendering::ClipPlane, tfRenderingClipPlaneHandle>(handle) ? S_OK : E_FAIL;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfRenderingClipPlanes_len(unsigned int *numCPs) {
    TFC_PTRCHECK(numCPs);
    *numCPs = rendering::ClipPlanes::len();
    return S_OK;
}

HRESULT tfRenderingClipPlanes_item(struct tfRenderingClipPlaneHandle *handle, unsigned int index) {
    if(index >= rendering::ClipPlanes::len()) 
        return E_FAIL;
    TFC_PTRCHECK(handle);
    rendering::ClipPlane *cp = new rendering::ClipPlane(rendering::ClipPlanes::item(index));
    handle->tfObj = (void*)cp;
    return S_OK;
}

HRESULT tfRenderingClipPlanes_createE(struct tfRenderingClipPlaneHandle *handle, float *pe) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(pe);
    rendering::ClipPlane *cp = new rendering::ClipPlane(rendering::ClipPlanes::create(fVector4::from(pe)));
    handle->tfObj = (void*)cp;
    return S_OK;
}

HRESULT tfRenderingClipPlanes_createPN(struct tfRenderingClipPlaneHandle *handle, float *point, float *normal) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(point);
    TFC_PTRCHECK(normal);
    rendering::ClipPlane *cp = new rendering::ClipPlane(rendering::ClipPlanes::create(fVector3::from(point), fVector3::from(normal)));
    handle->tfObj = (void*)cp;
    return S_OK;
}
