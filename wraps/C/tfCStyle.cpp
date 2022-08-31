/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

#include "tfCStyle.h"

#include "TissueForge_c_private.h"

#include <Magnum/Math/Color.h>

#include <rendering/tfStyle.h>
#include <tfParticle.h>


using namespace TissueForge;


namespace TissueForge { 


    rendering::Style *castC(struct tfRenderingStyleHandle *handle) {
        return castC<rendering::Style, tfRenderingStyleHandle>(handle);
    }

}

#define TFC_STYLE_GET(handle) \
    rendering::Style *style = TissueForge::castC<rendering::Style, tfRenderingStyleHandle>(handle); \
    TFC_PTRCHECK(style);


//////////////////////
// rendering::Style //
//////////////////////


HRESULT tfRenderingStyle_init(struct tfRenderingStyleHandle *handle) {
    TFC_PTRCHECK(handle);
    rendering::Style *style = new rendering::Style();
    handle->tfObj = (void*)style;
    return S_OK;
}

HRESULT tfRenderingStyle_initC(struct tfRenderingStyleHandle *handle, float *color, bool visible) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(color);
    fVector3 _color = Magnum::Color3::from(color);
    rendering::Style *style = new rendering::Style(&_color, visible);
    handle->tfObj = (void*)style;
    return S_OK;
}

HRESULT tfRenderingStyle_initS(struct tfRenderingStyleHandle *handle, const char *colorName, bool visible) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(colorName);
    rendering::Style *style = new rendering::Style(colorName, visible);
    handle->tfObj = (void*)style;
    return S_OK;
}

HRESULT tfRenderingStyle_destroy(struct tfRenderingStyleHandle *handle) {
    return TissueForge::capi::destroyHandle<rendering::Style, tfRenderingStyleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfRenderingStyle_setColor(struct tfRenderingStyleHandle *handle, const char *colorName) {
    TFC_STYLE_GET(handle);
    TFC_PTRCHECK(colorName);
    return style->setColor(colorName);
}

HRESULT tfRenderingStyle_getVisible(struct tfRenderingStyleHandle *handle, bool *visible) {
    TFC_STYLE_GET(handle);
    TFC_PTRCHECK(visible);
    *visible = style->getVisible();
    return S_OK;
}

HRESULT tfRenderingStyle_setVisible(struct tfRenderingStyleHandle *handle, bool visible) {
    TFC_STYLE_GET(handle);
    style->setVisible(visible);
    return S_OK;
}

HRESULT tfRenderingStyle_newColorMapper(
    struct tfRenderingStyleHandle *handle, 
    struct tfParticleTypeHandle *partType,
    const char *speciesName, 
    const char *name, 
    float min, 
    float max) 
{
    TFC_STYLE_GET(handle);
    TFC_PTRCHECK(partType); TFC_PTRCHECK(partType->tfObj);
    TFC_PTRCHECK(speciesName);
    TFC_PTRCHECK(name);
    style->newColorMapper((ParticleType*)partType->tfObj, speciesName, name, min, max);
    return S_OK;
}

HRESULT tfRenderingStyle_toString(struct tfRenderingStyleHandle *handle, char **str, unsigned int *numChars) {
    TFC_STYLE_GET(handle);
    return TissueForge::capi::str2Char(style->toString(), str, numChars);
}

HRESULT tfRenderingStyle_fromString(struct tfRenderingStyleHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    rendering::Style *style = rendering::Style::fromString(str);
    TFC_PTRCHECK(style);
    handle->tfObj = (void*)style;
    return S_OK;
}
