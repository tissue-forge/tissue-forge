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
 * @file tfCClipPlane.h
 * 
 */

#ifndef _WRAPS_C_TFCCLIPPLANE_H_
#define _WRAPS_C_TFCCLIPPLANE_H_

#include "tf_port_c.h"

// Handles

/**
 * @brief Handle to a @ref rendering::ClipPlane instance
 * 
 */
struct CAPI_EXPORT tfRenderingClipPlaneHandle {
    void *tfObj;
};


//////////////////////////
// rendering::ClipPlane //
//////////////////////////


/**
 * @brief Get the index of the clip plane.
 * 
 * @param handle populated handle
 * @param index index of the clip plane; Less than zero if clip plane has been destroyed.
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlane_getIndex(struct tfRenderingClipPlaneHandle *handle, int *index);

/**
 * @brief Get the point of the clip plane
 * 
 * @param handle populated handle
 * @param point point of the clip plane
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlane_getPoint(struct tfRenderingClipPlaneHandle *handle, float **point);

/**
 * @brief Get the normal vector of the clip plane
 * 
 * @param handle populated handle
 * @param normal normal of the clip plane
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlane_getNormal(struct tfRenderingClipPlaneHandle *handle, float **normal);

/**
 * @brief Get the coefficients of the plane equation of the clip plane
 * 
 * @param handle populated handle
 * @param pe plane equation coefficients
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlane_getEquation(struct tfRenderingClipPlaneHandle *handle, float **pe);

/**
 * @brief Set the coefficients of the plane equation of the clip plane
 * 
 * @param handle populated handle
 * @param pe plane equation coefficients
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlane_setEquationE(struct tfRenderingClipPlaneHandle *handle, float *pe);

/**
 * @brief Set the coefficients of the plane equation of the clip plane
 * using a point on the plane and its normal
 * 
 * @param handle populated handle
 * @param point plane point
 * @param normal plane normal vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlane_setEquationPN(struct tfRenderingClipPlaneHandle *handle, float *point, float *normal);

/**
 * @brief Destroy the clip plane
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlane_destroyCP(struct tfRenderingClipPlaneHandle *handle);

/**
 * @brief Destroy the handle instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlane_destroy(struct tfRenderingClipPlaneHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the number of clip planes
 * 
 * @param numCPs number of clip planes
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlanes_len(unsigned int *numCPs);

/**
 * @brief Get a clip plane by index
 * 
 * @param handle handle to populate
 * @param index index of the clip plane
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlanes_item(struct tfRenderingClipPlaneHandle *handle, unsigned int index);

/**
 * @brief Create a clip plane from the coefficients of the equation of the plane
 * 
 * @param handle handle to populate
 * @param pe coefficients of the equation of the plane
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlanes_createE(struct tfRenderingClipPlaneHandle *handle, float *pe);

/**
 * @brief Create a clip plane from a point and normal of the plane
 * 
 * @param handle handle to populate
 * @param point point on the clip plane
 * @param normal normal of the clip plane
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfRenderingClipPlanes_createPN(struct tfRenderingClipPlaneHandle *handle, float *point, float *normal);

#endif // _WRAPS_C_TFCCLIPPLANE_H_