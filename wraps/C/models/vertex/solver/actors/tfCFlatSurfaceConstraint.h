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
 * @file tfCFlatSurfaceAreaConstraint.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCFLATSURFACECONSTRAINT_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCFLATSURFACECONSTRAINT_H_

#include <tf_port_c.h>

#include <models/vertex/solver/tfCMeshObj.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::FlatSurfaceConstraint instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverFlatSurfaceConstraintHandle {
    void *tfObj;
};

///////////////////////////
// FlatSurfaceConstraint //
///////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param lam constraint value
 */
CAPI_FUNC(HRESULT) tfVertexSolverFlatSurfaceConstraint_init(
    struct tfVertexSolverFlatSurfaceConstraintHandle *handle, 
    tfFloatP_t lam
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverFlatSurfaceConstraint_destroy(struct tfVertexSolverFlatSurfaceConstraintHandle *handle);

/**
 * @brief Get the constraint value
 * 
 * @param handle populated handle
 * @param result constraint value
 */
CAPI_FUNC(HRESULT) tfVertexSolverFlatSurfaceConstraint_getLam(
    struct tfVertexSolverFlatSurfaceConstraintHandle *handle, 
    tfFloatP_t *result
);

/**
 * @brief Set the FlatSurfaceConstraint value
 * 
 * @param handle populated handle
 * @param lam FlatSurfaceConstraint value
 */
CAPI_FUNC(HRESULT) tfVertexSolverFlatSurfaceConstraint_setLam(
    struct tfVertexSolverFlatSurfaceConstraintHandle *handle, 
    tfFloatP_t lam
);

/**
 * @brief Cast to a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverFlatSurfaceConstraint_toBase(
    struct tfVertexSolverFlatSurfaceConstraintHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
);

/**
 * @brief Cast from a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverFlatSurfaceConstraint_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverFlatSurfaceConstraintHandle *result
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCFLATSURFACECONSTRAINT_H_