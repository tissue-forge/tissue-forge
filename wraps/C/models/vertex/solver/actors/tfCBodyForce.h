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

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCBODYFORCE_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCBODYFORCE_H_

#include <tf_port_c.h>

#include <models/vertex/solver/tfCMeshObj.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::BodyForce instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverBodyForceHandle {
    void *tfObj;
};


///////////////
// BodyForce //
///////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param comps force components
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyForce_init(
    struct tfVertexSolverBodyForceHandle *handle, 
    tfFloatP_t *comps
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyForce_destroy(struct tfVertexSolverBodyForceHandle *handle);

/**
 * @brief Get the force components
 * 
 * @param handle populated handle
 * @param comps force components
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyForce_getComps(
    struct tfVertexSolverBodyForceHandle *handle, 
    tfFloatP_t **comps
);

/**
 * @brief Set the force components
 * 
 * @param handle populated handle
 * @param comps force components
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyForce_setComps(
    struct tfVertexSolverBodyForceHandle *handle, 
    tfFloatP_t *comps
);

/**
 * @brief Cast to a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyForce_toBase(
    struct tfVertexSolverBodyForceHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
);

/**
 * @brief Cast from a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyForce_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverBodyForceHandle *result
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCBODYFORCE_H_