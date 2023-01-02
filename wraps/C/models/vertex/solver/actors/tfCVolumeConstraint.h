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
 * @file tfCVolumeConstraint.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCVOLUMECONSTRAINT_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCVOLUMECONSTRAINT_H_

#include <tf_port_c.h>

#include <models/vertex/solver/tfCMeshObj.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::VolumeConstraint instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverVolumeConstraintHandle {
    void *tfObj;
};

//////////////////////
// VolumeConstraint //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param lam constraint value
 * @param constr target value
 */
CAPI_FUNC(HRESULT) tfVertexSolverVolumeConstraint_init(
    struct tfVertexSolverVolumeConstraintHandle *handle, 
    tfFloatP_t lam, 
    tfFloatP_t constr
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverVolumeConstraint_destroy(struct tfVertexSolverVolumeConstraintHandle *handle);

/**
 * @brief Get the constraint value
 * 
 * @param handle populated handle
 * @param result constraint value
 */
CAPI_FUNC(HRESULT) tfVertexSolverVolumeConstraint_getLam(
    struct tfVertexSolverVolumeConstraintHandle *handle, 
    tfFloatP_t *result
);

/**
 * @brief Set the constraint value
 * 
 * @param handle populated handle
 * @param lam constraint value
 */
CAPI_FUNC(HRESULT) tfVertexSolverVolumeConstraint_setLam(
    struct tfVertexSolverVolumeConstraintHandle *handle, 
    tfFloatP_t lam
);

/**
 * @brief Get the target value
 * 
 * @param handle populated handle
 * @param result target value
 */
CAPI_FUNC(HRESULT) tfVertexSolverVolumeConstraint_getConstr(
    struct tfVertexSolverVolumeConstraintHandle *handle, 
    tfFloatP_t *result
);

/**
 * @brief Set the target value
 * 
 * @param handle populated handle
 * @param lam target value
 */
CAPI_FUNC(HRESULT) tfVertexSolverVolumeConstraint_setConstr(
    struct tfVertexSolverVolumeConstraintHandle *handle, 
    tfFloatP_t constr
);

/**
 * @brief Cast to a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverVolumeConstraint_toBase(
    struct tfVertexSolverVolumeConstraintHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
);

/**
 * @brief Cast from a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverVolumeConstraint_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverVolumeConstraintHandle *result
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCVOLUMECONSTRAINT_H_