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
 * @file tfCVertexSolverFIO.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCVERTEXSOLVERFIO_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCVERTEXSOLVERFIO_H_

#include <tf_port_c.h>


/**
 * @brief Test whether imported data is available. 
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverFIOHasImport(bool *result);

/**
 * @brief Map a vertex id from currently imported file data to the created vertex on import
 * 
 * @param fid vertex id according to import file
 * @param mapId vertex id according to simulation state
 */
CAPI_FUNC(HRESULT) tfVertexSolverFIOMapImportVertexId(unsigned int fid, unsigned int *mapId);

/**
 * @brief Map a surface id from currently imported file data to the created surface on import
 * 
 * @param fid surface id according to import file
 * @param mapId surface id according to simulation state
 */
CAPI_FUNC(HRESULT) tfVertexSolverFIOMapImportSurfaceId(unsigned int fid, unsigned int *mapId);

/**
 * @brief Map a surface type id from currently imported file data to the created surface type on import
 * 
 * @param fid surface type id according to import file
 * @param mapId surface type id according to simulation state
 */
CAPI_FUNC(HRESULT) tfVertexSolverFIOMapImportSurfaceTypeId(unsigned int fid, unsigned int *mapId);

/**
 * @brief Map a body id from currently imported file data to the created body on import
 * 
 * @param fid body id according to import file
 * @param mapId body id according to simulation state
 */
CAPI_FUNC(HRESULT) tfVertexSolverFIOMapImportBodyId(unsigned int fid, unsigned int *mapId);

/**
 * @brief Map a body type id from currently imported file data to the created body type on import
 * 
 * @param fid body type id according to import file
 * @param mapId body type id according to simulation state
 */
CAPI_FUNC(HRESULT) tfVertexSolverFIOMapImportBodyTypeId(unsigned int fid, unsigned int *mapId);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCVERTEXSOLVERFIO_H_