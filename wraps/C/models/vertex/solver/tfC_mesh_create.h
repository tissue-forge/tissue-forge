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
 * @file tfC_mesh_create.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFC_MESH_CREATE_H_
#define _WRAPS_C_VERTEX_SOLVER_TFC_MESH_CREATE_H_

#include <tf_port_c.h>

#include "tfCSurface.h"
#include "tfCBody.h"


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Populate the mesh with quadrilateral surfaces. 
 * 
 * Requires an initialized solver. 
 * 
 * @param stype surface type
 * @param startPos starting position
 * @param num_1 number of elements in the first direction
 * @param num_2 number of elements in the second direction
 * @param len_1 length of each element in the first direction
 * @param len_2 length of each element in the second direction
 * @param ax_1 axis name of the first direction (e.g., "x")
 * @param ax_2 axis name of the second direction
 * @param result constructed surfaces
 */
CAPI_FUNC(HRESULT) tfVertexSolverCreateQuadMesh(
    struct tfVertexSolverSurfaceTypeHandle *stype,
    tfFloatP_t *startPos, 
    unsigned int num_1, 
    unsigned int num_2, 
    tfFloatP_t len_1,
    tfFloatP_t len_2,
    const char *ax_1, 
    const char *ax_2, 
    struct tfVertexSolverSurfaceHandleHandle **result
);

/**
 * @brief Populate the mesh with parallelepiped bodies.
 * 
 * Requires an initialized solver. 
 * 
 * @param btype body type
 * @param stype surface type
 * @param startPos starting position
 * @param num_1 number of elements in the first direction
 * @param num_2 number of elements in the second direction
 * @param num_3 number of elements in the third direction
 * @param len_1 length of each element in the first direction
 * @param len_2 length of each element in the second direction
 * @param len_3 length of each element in the third direction
 * @param ax_1 axis name of the first direction (e.g., "x")
 * @param ax_2 axis name of the second direction
 * @param result constructed bodies
 */
CAPI_FUNC(HRESULT) tfVertexSolverCreatePLPDMesh(
    struct tfVertexSolverBodyTypeHandle *btype, 
    struct tfVertexSolverSurfaceTypeHandle *stype,
    tfFloatP_t *startPos, 
    unsigned int num_1, 
    unsigned int num_2, 
    unsigned int num_3, 
    tfFloatP_t len_1,
    tfFloatP_t len_2,
    tfFloatP_t len_3,
    const char *ax_1, 
    const char *ax_2, 
    struct tfVertexSolverBodyHandleHandle **result
);

/**
 * @brief Populate the mesh with hexagonal surfaces.
 * 
 * Requires an initialized solver. 
 * 
 * @param stype surface type
 * @param startPos starting position
 * @param num_1 number of elements in the first direction
 * @param num_2 number of elements in the second direction
 * @param hexRad radius of hexagon vertices
 * @param ax_1 axis name of the first direction (e.g., "x")
 * @param ax_2 axis name of the second direction
 * @param result constructed surfaces
 */
CAPI_FUNC(HRESULT) tfVertexSolverCreateHex2DMesh(
    struct tfVertexSolverSurfaceTypeHandle *stype, 
    tfFloatP_t *startPos, 
    unsigned int num_1, 
    unsigned int num_2, 
    tfFloatP_t hexRad,
    const char *ax_1, 
    const char *ax_2, 
    struct tfVertexSolverSurfaceHandleHandle **result
);

/**
 * @brief Populate the mesh with bodies from extruded hexagonal surfaces. 
 * 
 * Requires an initialized solver. 
 * 
 * Surfaces are placed in the plane of the first and second directions, 
 * and extruded along the third direction.
 * 
 * @param btype body type
 * @param stype surface type
 * @param startPos starting position
 * @param num_1 number of elements in the first direction
 * @param num_2 number of elements in the second direction
 * @param num_3 number of elements in the third directionnumber of elements in the third direction
 * @param hexRad radius of hexagon vertices
 * @param hex_height extrusion length per body
 * @param ax_1 axis name of the first direction (e.g., "x")
 * @param ax_2 axis name of the second direction
 * @param result constructed bodies
 */
CAPI_FUNC(HRESULT) tfVertexSolverCreateHex3DMesh(
    struct tfVertexSolverBodyTypeHandle *btype, 
    struct tfVertexSolverSurfaceTypeHandle *stype,
    tfFloatP_t *startPos, 
    unsigned int num_1, 
    unsigned int num_2, 
    unsigned int num_3, 
    tfFloatP_t hexRad,
    tfFloatP_t hex_height,
    const char *ax_1, 
    const char *ax_2, 
    struct tfVertexSolverBodyHandleHandle **result
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFC_MESH_CREATE_H_