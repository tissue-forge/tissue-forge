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
 * @file tfCAdhesion.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCADHESION_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCADHESION_H_

#include <tf_port_c.h>

#include <models/vertex/solver/tfCMeshObj.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::Adhesion instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverAdhesionHandle {
    void *tfObj;
};


//////////////
// Adhesion //
//////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param lam adhesion value
 */
CAPI_FUNC(HRESULT) tfVertexSolverAdhesion_init(
    struct tfVertexSolverAdhesionHandle *handle, 
    tfFloatP_t lam
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverAdhesion_destroy(struct tfVertexSolverAdhesionHandle *handle);

/**
 * @brief Get the adhesion value
 * 
 * @param handle populated handle
 * @param result adhesion value
 */
CAPI_FUNC(HRESULT) tfVertexSolverAdhesion_getLam(
    struct tfVertexSolverAdhesionHandle *handle, 
    tfFloatP_t *result
);

/**
 * @brief Set the adhesion value
 * 
 * @param handle populated handle
 * @param lam adhesion value
 */
CAPI_FUNC(HRESULT) tfVertexSolverAdhesion_setLam(
    struct tfVertexSolverAdhesionHandle *handle, 
    tfFloatP_t lam
);

/**
 * @brief Cast to a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverAdhesion_toBase(
    struct tfVertexSolverAdhesionHandle *handle, 
    struct tfVertexSolverMeshObjTypePairActorHandle *result
);

/**
 * @brief Cast from a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverAdhesion_fromBase(
    struct tfVertexSolverMeshObjTypePairActorHandle *handle, 
    struct tfVertexSolverAdhesionHandle *result
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCADHESION_H_