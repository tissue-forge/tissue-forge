/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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
 * @file tfCNormalStress.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCNORMALSTRESS_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCNORMALSTRESS_H_

#include <tf_port_c.h>

#include <models/vertex/solver/tfCMeshObj.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::NormalStress instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverNormalStressHandle {
    void *tfObj;
};

//////////////////
// NormalStress //
//////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param mag magnitude
 */
CAPI_FUNC(HRESULT) tfVertexSolverNormalStress_init(
    struct tfVertexSolverNormalStressHandle *handle, 
    tfFloatP_t mag
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverNormalStress_destroy(struct tfVertexSolverNormalStressHandle *handle);

/**
 * @brief Get the magnitude
 * 
 * @param handle populated handle
 * @param result magnitude
 */
CAPI_FUNC(HRESULT) tfVertexSolverNormalStress_getMag(
    struct tfVertexSolverNormalStressHandle *handle, 
    tfFloatP_t *result
);

/**
 * @brief Set the magnitude
 * 
 * @param handle populated handle
 * @param mag NormalStress value
 */
CAPI_FUNC(HRESULT) tfVertexSolverNormalStress_setMag(
    struct tfVertexSolverNormalStressHandle *handle, 
    tfFloatP_t mag
);

/**
 * @brief Cast to a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverNormalStress_toBase(
    struct tfVertexSolverNormalStressHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
);

/**
 * @brief Cast from a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverNormalStress_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverNormalStressHandle *result
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCNORMALSTRESS_H_