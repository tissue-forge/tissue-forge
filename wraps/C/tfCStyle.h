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

/**
 * @file tfCStyle.h
 * 
 */

#ifndef _WRAPS_C_TFCSTYLE_H_
#define _WRAPS_C_TFCSTYLE_H_

#include "tf_port_c.h"

#include "tfCParticle.h"

// Handles

/**
 * @brief Handle to a @ref rendering::Style instance
 * 
 */
struct CAPI_EXPORT tfRenderingStyleHandle {
    void *tfObj;
};


//////////////////////
// rendering::Style //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_init(struct tfRenderingStyleHandle *handle);

/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param color 3-element RGB color array
 * @param visible visible flag
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_initC(struct tfRenderingStyleHandle *handle, float *color, bool visible);

/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param colorName name of a color
 * @param visible visible flag
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_initS(struct tfRenderingStyleHandle *handle, const char *colorName, bool visible);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_destroy(struct tfRenderingStyleHandle *handle);

/**
 * @brief Set the color by name
 * 
 * @param handle populated handle
 * @param colorName name of color
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_setColor(struct tfRenderingStyleHandle *handle, const char *colorName);

/**
 * @brief Test whether the instance is visible
 * 
 * @param handle populated handle
 * @param visible visible flag
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_getVisible(struct tfRenderingStyleHandle *handle, bool *visible);

/**
 * @brief Set whether the instance is visible
 * 
 * @param handle populated handle
 * @param visible visible flag
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_setVisible(struct tfRenderingStyleHandle *handle, bool visible);

/**
 * @brief Construct and apply a new color map for a particle type and species
 * 
 * @param handle populated handle
 * @param partType particle type
 * @param speciesName name of species
 * @param name name of color map
 * @param min minimum value of map
 * @param max maximum value of map
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_newColorMapper(
    struct tfRenderingStyleHandle *handle, 
    struct tfParticleTypeHandle *partType, 
    const char *speciesName, 
    const char *name, 
    float min, 
    float max
);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_toString(struct tfRenderingStyleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingStyle_fromString(struct tfRenderingStyleHandle *handle, const char *str);

#endif // _WRAPS_C_TFCSTYLE_H_