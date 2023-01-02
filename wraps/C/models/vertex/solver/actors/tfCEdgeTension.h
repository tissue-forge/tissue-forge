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

/**
 * @file tfCEdgeTension.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCEDGETENSION_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCEDGETENSION_H_

#include <tf_port_c.h>

#include <models/vertex/solver/tfCMeshObj.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::EdgeTension instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverEdgeTensionHandle {
    void *tfObj;
};

/////////////////
// EdgeTension //
/////////////////

/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param lam tension value
 * @param order tension order
 */
CAPI_FUNC(HRESULT) tfVertexSolverEdgeTension_init(
    struct tfVertexSolverEdgeTensionHandle *handle, 
    tfFloatP_t lam, 
    unsigned int order
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverEdgeTension_destroy(struct tfVertexSolverEdgeTensionHandle *handle);

/**
 * @brief Get the tension value
 * 
 * @param handle populated handle
 * @param result tension value
 */
CAPI_FUNC(HRESULT) tfVertexSolverEdgeTension_getLam(
    struct tfVertexSolverEdgeTensionHandle *handle, 
    tfFloatP_t *result
);

/**
 * @brief Set the tension value
 * 
 * @param handle populated handle
 * @param lam tension value
 */
CAPI_FUNC(HRESULT) tfVertexSolverEdgeTension_setLam(
    struct tfVertexSolverEdgeTensionHandle *handle, 
    tfFloatP_t lam
);

/**
 * @brief Get the tension order
 * 
 * @param handle populated handle
 * @param result tension order
 */
CAPI_FUNC(HRESULT) tfVertexSolverEdgeTension_getOrder(
    struct tfVertexSolverEdgeTensionHandle *handle, 
    unsigned int *result
);

/**
 * @brief Set the tension order
 * 
 * @param handle populated handle
 * @param order tension order
 */
CAPI_FUNC(HRESULT) tfVertexSolverEdgeTension_setOrder(
    struct tfVertexSolverEdgeTensionHandle *handle, 
    unsigned int order
);

/**
 * @brief Cast to a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverEdgeTension_toBase(
    struct tfVertexSolverEdgeTensionHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
);

/**
 * @brief Cast from a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverEdgeTension_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverEdgeTensionHandle *result
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCEDGETENSION_H_