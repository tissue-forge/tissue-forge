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
 * @file tfCMeshQuality.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCMESHQUALITY_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCMESHQUALITY_H_

#include <tf_port_c.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::MeshQuality instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverMeshQualityHandle {
    void *tfObj;
};


/////////////////
// MeshQuality //
/////////////////


/**
 * @brief Initialize an instance
 * 
 * @param hande handle to populate
 * @param vertexMergeDistCf 
 * @param surfaceDemoteAreaCf 
 * @param bodyDemoteVolumeCf 
 * @param _edgeSplitDistCf 
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_init(
    struct tfVertexSolverMeshQualityHandle *handle, 
    tfFloatP_t vertexMergeDistCf, 
    tfFloatP_t surfaceDemoteAreaCf, 
    tfFloatP_t bodyDemoteVolumeCf, 
    tfFloatP_t _edgeSplitDistCf
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_destroy(struct tfVertexSolverMeshQualityHandle *handle);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str JSON string representation
 * @param numChar number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_toString(
    struct tfVertexSolverMeshQualityHandle *handle, 
    char **str, 
    unsigned int *numChar
);

/**
 * @brief Perform quality operations work
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_doQuality(struct tfVertexSolverMeshQualityHandle *handle);

/**
 * @brief Test whether quality operations are being done
 * 
 * @param handle populated handle
 * @param working result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_working(struct tfVertexSolverMeshQualityHandle *handle, bool *working);

/**
 * @brief Get the distance below which two vertices are scheduled for merging
 * 
 * @param handle populated handle
 * @param val distance below which two vertices are scheduled for merging
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_getVertexMergeDistance(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t *val);

/**
 * @brief Set the distance below which two vertices are scheduled for merging
 * 
 * @param handle populated handle
 * @param val distance below which two vertices are scheduled for merging
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_setVertexMergeDistance(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t val);

/**
 * @brief Get the area below which a surface is scheduled to become a vertex
 * 
 * @param handle populated handle
 * @param val area below which a surface is scheduled to become a vertex
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_getSurfaceDemoteArea(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t *val);

/**
 * @brief Set the area below which a surface is scheduled to become a vertex
 * 
 * @param handle populated handle
 * @param val area below which a surface is scheduled to become a vertex
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_setSurfaceDemoteArea(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t val);

/**
 * @brief Get the volume below which a body is scheduled to become a vertex
 * 
 * @param handle populated handle
 * @param val volume below which a body is scheduled to become a vertex
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_getBodyDemoteVolume(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t *val);

/**
 * @brief Set the volume below which a body is scheduled to become a vertex
 * 
 * @param handle populated handle
 * @param val volume below which a body is scheduled to become a vertex
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_setBodyDemoteVolume(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t val);

/**
 * @brief Get the distance at which two vertices are seperated when a vertex is split
 * 
 * @param handle populated handle
 * @param val distance at which two vertices are seperated when a vertex is split
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_getEdgeSplitDist(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t *val);

/**
 * @brief Set the distance at which two vertices are seperated when a vertex is split
 * 
 * @param handle populated handle
 * @param val distance at which two vertices are seperated when a vertex is split
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_setEdgeSplitDist(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t val);

/**
 * @brief Get whether 2D collisions are implemented
 * 
 * @param handle populated handle
 * @param collision2D whether 2D collisions are implemented
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_getCollision2D(struct tfVertexSolverMeshQualityHandle *handle, bool *collision2D);

/**
 * @brief Set whether 2D collisions are implemented
 * 
 * @param handle populated handle
 * @param collision2D whether 2D collisions are implemented
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_setCollision2D(struct tfVertexSolverMeshQualityHandle *handle, bool collision2D);

/**
 * @brief Exclude a vertex from quality operations
 * 
 * @param handle populated handle
 * @param id vertex id
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_excludeVertex(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id);

/**
 * @brief Exclude a surface from quality operations
 * 
 * @param handle populated handle
 * @param id surface id
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_excludeSurface(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id);

/**
 * @brief Exclude a body from quality operations
 * 
 * @param handle populated handle
 * @param id body id
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_excludeBody(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id);

/**
 * @brief Include a vertex from quality operations. 
 * 
 * Unless otherwise specified, all vertices are subject to operations.
 * 
 * @param handle populated handle
 * @param id vertex id
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_includeVertex(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id);

/**
 * @brief Include a surface from quality operations. 
 * 
 * Unless otherwise specified, all surfaces are subject to operations.
 * 
 * @param handle populated handle
 * @param id surface id
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_includeSurface(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id);

/**
 * @brief Include a body from quality operations. 
 * 
 * Unless otherwise specified, all bodies are subject to operations.
 * 
 * @param handle populated handle
 * @param id body id
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshQuality_includeBody(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCMESHQUALITY_H_