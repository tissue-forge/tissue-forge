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

#include "tfCStyle.h"

#include "TissueForge_c_private.h"

#include <Magnum/Math/Color.h>

#include <rendering/tfColorMapper.h>
#include <rendering/tfStyle.h>
#include <tfParticle.h>


using namespace TissueForge;


namespace TissueForge { 


    rendering::ColorMapper *castC(struct tfRenderingColorMapperHandle *handle) {
        return castC<rendering::ColorMapper, tfRenderingColorMapperHandle>(handle);
    }

    rendering::Style *castC(struct tfRenderingStyleHandle *handle) {
        return castC<rendering::Style, tfRenderingStyleHandle>(handle);
    }

}

#define TFC_COLORMAPPER_GET(handle) \
    rendering::ColorMapper *cmap = TissueForge::castC<rendering::ColorMapper, tfRenderingColorMapperHandle>(handle); \
    TFC_PTRCHECK(cmap);

#define TFC_STYLE_GET(handle) \
    rendering::Style *style = TissueForge::castC<rendering::Style, tfRenderingStyleHandle>(handle); \
    TFC_PTRCHECK(style);


////////////////////////////
// rendering::ColorMapper //
////////////////////////////


HRESULT tfRenderingColorMapper_init(struct tfRenderingColorMapperHandle* handle, const char* name, float min, float max) {
    TFC_PTRCHECK(handle);
    rendering::ColorMapper *cmap = new rendering::ColorMapper(name, min, max);
    handle->tfObj = (void*)cmap;
    return S_OK;
}

HRESULT tfRenderingColorMapper_destroy(struct tfRenderingColorMapperHandle* handle) {
    return TissueForge::capi::destroyHandle<rendering::ColorMapper, tfRenderingColorMapperHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfRenderingColorMapper_getMinVal(struct tfRenderingColorMapperHandle* handle, float* val) {
    TFC_COLORMAPPER_GET(handle);
    TFC_PTRCHECK(val);
    *val = cmap->min_val;
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMinVal(struct tfRenderingColorMapperHandle* handle, float val) {
    TFC_COLORMAPPER_GET(handle);
    cmap->min_val = val;
    return S_OK;
}

HRESULT tfRenderingColorMapper_getMaxVal(struct tfRenderingColorMapperHandle* handle, float* val) {
    TFC_COLORMAPPER_GET(handle);
    TFC_PTRCHECK(val);
    *val = cmap->max_val;
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMaxVal(struct tfRenderingColorMapperHandle* handle, float val) {
    TFC_COLORMAPPER_GET(handle);
    cmap->max_val = val;
    return S_OK;
}

HRESULT tfRenderingColorMapper_hasMapParticle(struct tfRenderingColorMapperHandle* handle, bool* result) {
    TFC_COLORMAPPER_GET(handle);
    TFC_PTRCHECK(result);
    *result = cmap->hasMapParticle();
    return S_OK;
}

HRESULT tfRenderingColorMapper_hasMapAngle(struct tfRenderingColorMapperHandle* handle, bool* result) {
    TFC_COLORMAPPER_GET(handle);
    TFC_PTRCHECK(result);
    *result = cmap->hasMapAngle();
    return S_OK;
}

HRESULT tfRenderingColorMapper_hasMapBond(struct tfRenderingColorMapperHandle* handle, bool* result) {
    TFC_COLORMAPPER_GET(handle);
    TFC_PTRCHECK(result);
    *result = cmap->hasMapBond();
    return S_OK;
}

HRESULT tfRenderingColorMapper_hasMapDihedral(struct tfRenderingColorMapperHandle* handle, bool* result) {
    TFC_COLORMAPPER_GET(handle);
    TFC_PTRCHECK(result);
    *result = cmap->hasMapDihedral();
    return S_OK;
}

HRESULT tfRenderingColorMapper_clearMapParticle(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->clearMapParticle();
    return S_OK;
}

HRESULT tfRenderingColorMapper_clearMapAngle(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->clearMapAngle();
    return S_OK;
}

HRESULT tfRenderingColorMapper_clearMapBond(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->clearMapBond();
    return S_OK;
}

HRESULT tfRenderingColorMapper_clearMapDihedral(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->clearMapDihedral();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticlePositionX(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticlePositionX();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticlePositionY(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticlePositionY();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticlePositionZ(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticlePositionZ();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticleVelocityX(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticleVelocityX();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticleVelocityY(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticleVelocityY();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticleVelocityZ(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticleVelocityZ();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticleSpeed(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticleSpeed();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticleForceX(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticleForceX();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticleForceY(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticleForceY();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticleForceZ(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapParticleForceZ();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapParticleSpecies(struct tfRenderingColorMapperHandle* handle, struct tfParticleTypeHandle* pType, const char* name) {
    TFC_COLORMAPPER_GET(handle);
    TFC_PTRCHECK(pType);
    TFC_PTRCHECK(name);

    ParticleType *_pType = TissueForge::castC<ParticleType, tfParticleTypeHandle>(pType);
    TFC_PTRCHECK(_pType);

    cmap->setMapParticleSpecies(_pType, name);
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapAngleAngle(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapAngleAngle();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapAngleAngleEq(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapAngleAngleEq();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapBondLength(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapBondLength();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapBondLengthEq(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapBondLengthEq();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapDihedralAngle(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapDihedralAngle();
    return S_OK;
}

HRESULT tfRenderingColorMapper_setMapDihedralAngleEq(struct tfRenderingColorMapperHandle* handle) {
    TFC_COLORMAPPER_GET(handle);
    cmap->setMapDihedralAngleEq();
    return S_OK;
}

HRESULT tfRenderingColorMapper_set_colormap(struct tfRenderingColorMapperHandle* handle, const char* s) {
    TFC_COLORMAPPER_GET(handle);
    cmap->set_colormap(s);
    return S_OK;
}


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

HRESULT tfRenderingStyle_getColorMapper(struct tfRenderingStyleHandle *handle, struct tfRenderingColorMapperHandle* mapper) {
    TFC_STYLE_GET(handle);
    TFC_PTRCHECK(mapper);
    if(style->mapper) {
        mapper->tfObj = (void*)style->mapper;
        return S_OK;
    }
    return E_FAIL;
}

HRESULT tfRenderingStyle_setColorMapper(struct tfRenderingStyleHandle *handle, struct tfRenderingColorMapperHandle* mapper) {
    TFC_STYLE_GET(handle);
    TFC_COLORMAPPER_GET(mapper);
    style->mapper = cmap;
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
