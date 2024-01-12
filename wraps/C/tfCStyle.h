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

/**
 * @brief Handle to a @ref rendering::ColorMapper instance
 * 
*/
struct CAPI_EXPORT tfRenderingColorMapperHandle {
    void *tfObj;
};


////////////////////////////
// rendering::ColorMapper //
////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param name color map name
 * @param min minimum map value
 * @param max maximum map value
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingColorMapper_init(struct tfRenderingColorMapperHandle* handle, const char* name, float min, float max);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingColorMapper_destroy(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Get the minimum map value
 * 
 * @param handle populated handle
 * @param val value
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_getMinVal(struct tfRenderingColorMapperHandle* handle, float* val);

/**
 * @brief Set the minimum map value
 * 
 * @param handle populated handle
 * @param val value
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMinVal(struct tfRenderingColorMapperHandle* handle, float val);

/**
 * @brief Get the maximum map value
 * 
 * @param handle populated handle
 * @param val value
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_getMaxVal(struct tfRenderingColorMapperHandle* handle, float* val);

/**
 * @brief Set the maximum map value
 * 
 * @param handle populated handle
 * @param val value
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMaxVal(struct tfRenderingColorMapperHandle* handle, float val);

/**
 * @brief Test whether the mapper has a particle map
 * 
 * @param handle populated handle
 * @param result result of test
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_hasMapParticle(struct tfRenderingColorMapperHandle* handle, bool* result);

/**
 * @brief Test whether the mapper has an angle map
 * 
 * @param handle populated handle
 * @param result result of test
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_hasMapAngle(struct tfRenderingColorMapperHandle* handle, bool* result);

/**
 * @brief Test whether the mapper has a bond map
 * 
 * @param handle populated handle
 * @param result result of test
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_hasMapBond(struct tfRenderingColorMapperHandle* handle, bool* result);

/**
 * @brief Test whether the mapper has a dihedral map
 * 
 * @param handle populated handle
 * @param result result of test
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_hasMapDihedral(struct tfRenderingColorMapperHandle* handle, bool* result);

/**
 * @brief Clear the particle map
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_clearMapParticle(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Clear the angle map
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_clearMapAngle(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Clear the bond map
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_clearMapBond(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Clear the dihedral map
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_clearMapDihedral(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to x-coordinate of particle position
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticlePositionX(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to y-coordinate of particle position
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticlePositionY(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to z-coordinate of particle position
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticlePositionZ(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to x-component of particle velocity
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticleVelocityX(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to y-component of particle velocity
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticleVelocityY(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to z-component of particle velocity
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticleVelocityZ(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to particle speed
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticleSpeed(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to x-component of particle force
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticleForceX(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to y-component of particle force
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticleForceY(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to z-component of particle force
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticleForceZ(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the particle map to a species value
 * 
 * @param handle populated handle
 * @param pType particle type
 * @param name species name
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapParticleSpecies(struct tfRenderingColorMapperHandle* handle, struct tfParticleTypeHandle* pType, const char* name);

/**
 * @brief Set the angle map to angle
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapAngleAngle(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the angle map to angle from equilibrium
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapAngleAngleEq(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the bond map to length
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapBondLength(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the bond map to length from equilibrium
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapBondLengthEq(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the dihedral map to angle
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapDihedralAngle(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Set the dihedral map to angle from equilibrium
 * 
 * @param handle populated handle
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingColorMapper_setMapDihedralAngleEq(struct tfRenderingColorMapperHandle* handle);

/**
 * @brief Try to set the colormap. 
 * 
 * If the map doesn't exist, does not do anything and returns false.
 * 
 * @param handle populated handle
 * @param s name of color map
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingColorMapper_set_colormap(struct tfRenderingColorMapperHandle* handle, const char* s);


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
 * @brief Get the color mapper, if any
 * 
 * @param handle populated handle
 * @param mapper color mapper
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingStyle_getColorMapper(struct tfRenderingStyleHandle *handle, struct tfRenderingColorMapperHandle* mapper);

/**
 * @brief Set the color mapper
 * 
 * @param handle populated handle
 * @param mapper color mapper
 * @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfRenderingStyle_setColorMapper(struct tfRenderingStyleHandle *handle, struct tfRenderingColorMapperHandle* mapper);

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