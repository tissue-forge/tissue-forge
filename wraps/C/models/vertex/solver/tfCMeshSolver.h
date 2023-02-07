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
 * @file tfCMeshSolver.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCMESHSOLVER_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCMESHSOLVER_H_

#include <tf_port_c.h>

#include "tfCVertex.h"
#include "tfCSurface.h"
#include "tfCBody.h"


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Calculate the force on a vertex
 * 
 * @param v vertex
 * @param f force
 */
HRESULT tfVertexSolverVertexForce(struct tfVertexSolverVertexHandleHandle *v, tfFloatP_t *f);

/**
 * @brief Initialize the solver
 */
HRESULT tfVertexSolverInit();

/**
 * @brief Reduce internal buffers and storage
 */
HRESULT tfVertexSolverCompact();

/**
 * @brief Locks the engine for thread-safe engine operations
 */
HRESULT tfVertexSolverEngineLock();

/**
 * @brief Unlocks the engine for thread-safe engine operations
 */
HRESULT tfVertexSolverEngineUnlock();

/**
 * @brief Test whether the current mesh state needs updated
 * 
 * @param result result of the test
 */
HRESULT tfVertexSolverIsDirty(bool *result);

/**
 * @brief Set whether the current mesh state needs updated
 * 
 * @param isDirty flag indicating whether the current mesh state needs updated
 */
HRESULT tfVertexSolverSetDirty(bool isDirty);

/**
 * @brief Register a body type
 * 
 * @param type a body type
 */
HRESULT tfVertexSolverRegisterBodyType(struct tfVertexSolverBodyTypeHandle *type);

/**
 * @brief Register a surface type
 * 
 * @param type a surface type
 */
HRESULT tfVertexSolverRegisterSurfaceType(struct tfVertexSolverSurfaceTypeHandle *type);

/**
 * @brief Get the body type by id
 * 
 * @param typeId type id
 * @param type handle to populate
 */
HRESULT tfVertexSolverGetBodyType(const unsigned int &typeId, struct tfVertexSolverBodyTypeHandle *type);

/**
 * @brief Get the surface type by id
 * 
 * @param typeId type id
 * @param type handle to populate
 */
HRESULT tfVertexSolverGetSurfaceType(const unsigned int &typeId, struct tfVertexSolverSurfaceTypeHandle *type);

/**
 * @brief Get the number of registered body types
 * 
 * @param numTypes number of registered body types
 */
HRESULT tfVertexSolverNumBodyTypes(int *numTypes);

/**
 * @brief Get the number of registered surface types
 * 
 * @param numTypes number of registered surface types
 */
HRESULT tfVertexSolverNumSurfaceTypes(int *numTypes);

/**
 * @brief Update internal data due to a change in position
 */
HRESULT tfVertexSolverPositionChanged();

/**
 * @brief Update the solver if dirty
 * 
 * @param force flag to force an update and ignore whether the solver is dirty
 */
HRESULT tfVertexSolverUpdate(bool force);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCMESHSOLVER_H_