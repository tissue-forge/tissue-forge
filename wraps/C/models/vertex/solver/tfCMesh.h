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
 * @file tfCMesh.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCMESH_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCMESH_H_

#include <tf_port_c.h>

#include "tfCMeshQuality.h"


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Test whether this mesh has a mesh quality instance
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshHasQuality(bool *result);

/**
 * @brief Get the mesh quality instance
 * 
 * @param quality mesh quality instance
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshGetQuality(struct tfVertexSolverMeshQualityHandle *quality);

/**
 * @brief Set the mesh quality instance
 * 
 * @param quality mesh quality instance
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshSetQuality(struct tfVertexSolverMeshQualityHandle *quality);

/**
 * @brief Test whether a mesh quality instance is working on the mesh
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQualityWorking(bool *result);

/**
 * @brief Ensure that there are a given number of allocated vertices
 * 
 * @param numAlloc number to ensure allocated
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshEnsureAvailableVertices(unsigned int numAlloc);

/**
 * @brief Ensure that there are a given number of allocated surfaces
 * 
 * @param numAlloc number to ensure allocated
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshEnsureAvailableSurfaces(unsigned int numAlloc);

/**
 * @brief Ensure that there are a given number of allocated bodies
 * 
 * @param numAlloc number to ensure allocated
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshEnsureAvailableBodies(unsigned int numAlloc);

/**
 * @brief Create a vertex
 * 
 * @param handle handle to populate
 * @param pid id of underlying particle
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshCreateVertex(struct tfVertexSolverVertexHandleHandle *handle, unsigned int pid);

/**
 * @brief Create a surface
 * 
 * @param handle handle to populate
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshCreateSurface(struct tfVertexSolverSurfaceHandleHandle *handle);

/**
 * @brief Create a body
 * 
 * @param handle handle to populate
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshCreateBody(struct tfVertexSolverBodyHandleHandle *handle);

/**
 * @brief Locks the mesh for thread-safe operations
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLock();

/**
 * @brief Unlocks the mesh for thread-safe operations
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshUnlock();

/**
 * @brief Find a vertex in this mesh
 * 
 * @param pos position to look
 * @param tol distance tolerance
 * @param v a vertex within the distance tolerance of the position, otherwise NULL
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshFindVertex(
    tfFloatP_t *pos, 
    tfFloatP_t tol, 
    struct tfVertexSolverVertexHandleHandle *v
);

/**
 * @brief Get the vertex for a given particle id
 * 
 * @param pid particle id
 * @param v handle to populate
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshGetVertexByPID(unsigned int pid, struct tfVertexSolverVertexHandleHandle *v);

/**
 * @brief Get the vertex at a location in the list of vertices
 * 
 * @param idx index
 * @param v handle to populate
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshGetVertex(unsigned int idx, struct tfVertexSolverVertexHandleHandle *v);

/**
 * @brief Get a surface at a location in the list of surfaces
 * 
 * @param idx index
 * @param s surface
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshGetSurface(unsigned int idx, struct tfVertexSolverSurfaceHandleHandle *s);

/**
 * @brief Get a body at a location in the list of bodies
 * 
 * @param idx index
 * @param b body
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshGetBody(unsigned int idx, struct tfVertexSolverBodyHandleHandle *b);

/**
 * @brief Get the number of vertices
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshNumVertices(unsigned int *result);

/**
 * @brief Get the number of surfaces
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshNumSurfaces(unsigned int *result);

/**
 * @brief Get the number of bodies
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshNumBodies(unsigned int *result);

/**
 * @brief Get the size of the list of vertices
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshSizeVertices(unsigned int *result);

/**
 * @brief Get the size of the list of surfaces
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshSizeSurfaces(unsigned int *result);

/**
 * @brief Get the size of the list of bodies
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshSizeBodies(unsigned int *result);

/**
 * @brief Validate state of the mesh
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshValidate(bool *result);

/**
 * @brief Manually notify that the mesh has been changed
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshMakeDirty();

/**
 * @brief Check whether two vertices are connected
 * 
 * @param v1 first vertex
 * @param v2 second vertex
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshConnectedVertices(
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2, 
    bool *result
);

/**
 * @brief Check whether two surfaces are connected
 * 
 * @param s1 first surface
 * @param s2 second surface
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshConnectedSurfaces(
    struct tfVertexSolverSurfaceHandleHandle *s1, 
    struct tfVertexSolverSurfaceHandleHandle *s2, 
    bool *result
);

/**
 * @brief Check whether two bodies are connected
 * 
 * @param b1 first body
 * @param b2 second body
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshConnectedBodies(
    struct tfVertexSolverBodyHandleHandle *b1, 
    struct tfVertexSolverBodyHandleHandle *b2, 
    bool *result
);

/**
 * @brief Remove a vertex from the mesh; all connected surfaces and bodies are also removed
 * 
 * @param v vertex to remove
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshRemoveVertex(struct tfVertexSolverVertexHandleHandle *v);

/**
 * @brief Remove a surface from the mesh; all connected bodies are also removed
 * 
 * @param s surface to remove
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshRemoveSurface(struct tfVertexSolverSurfaceHandleHandle *s);

/**
 * @brief Remove a body from the mesh
 * 
 * @param b body to remove
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshRemoveBody(struct tfVertexSolverBodyHandleHandle *b);

/**
 * @brief Test whether the mesh is 3D. 
 * 
 * A 3D mesh has at least one body. 
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshIs3D(bool *result);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCMESH_H_