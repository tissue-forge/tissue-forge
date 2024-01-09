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

/**
 * @file tfCSurfaceConstraint.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCSURFACETRACTION_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCSURFACETRACTION_H_

#include <tf_port_c.h>

#include <models/vertex/solver/tfCMeshObj.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::SurfaceTraction instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverSurfaceTractionHandle {
    void *tfObj;
};

/////////////////////
// SurfaceTraction //
/////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param comps force components
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceTraction_init(
    struct tfVertexSolverSurfaceTractionHandle *handle, 
    tfFloatP_t *comps
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceTraction_destroy(struct tfVertexSolverSurfaceTractionHandle *handle);

/**
 * @brief Get the force components
 * 
 * @param handle populated handle
 * @param comps force components
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceTraction_getComps(
    struct tfVertexSolverSurfaceTractionHandle *handle, 
    tfFloatP_t **comps
);

/**
 * @brief Set the force components
 * 
 * @param handle populated handle
 * @param comps force components
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceTraction_setComps(
    struct tfVertexSolverSurfaceTractionHandle *handle, 
    tfFloatP_t *comps
);

/**
 * @brief Cast to a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceTraction_toBase(
    struct tfVertexSolverSurfaceTractionHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
);

/**
 * @brief Cast from a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceTraction_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverSurfaceTractionHandle *result
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCSURFACETRACTION_H_