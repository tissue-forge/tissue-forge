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

#ifndef _WRAPS_C_VERTEX_SOLVER_TFC_MESH_BIND_H_
#define _WRAPS_C_VERTEX_SOLVER_TFC_MESH_BIND_H_

#include <tf_port_c.h>

#include "tfCMeshObj.h"
#include "tfCSurface.h"
#include "tfCBody.h"


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Bind an actor to a body type
 * 
 * @param a actor
 * @param b body type
 */
HRESULT tfVertexSolverBindBodyType(
    struct tfVertexSolverMeshObjActorHandle *a, 
    struct tfVertexSolverBodyTypeHandle *b
);

/**
 * @brief Bind an actor to a body
 * 
 * @param a actor
 * @param h body
 */
HRESULT tfVertexSolverBindBody(
    struct tfVertexSolverMeshObjActorHandle *a, 
    struct tfVertexSolverBodyHandleHandle *h
);

/**
 * @brief Bind an actor to a surface type
 * 
 * @param a actor
 * @param s surface type
 */
HRESULT tfVertexSolverBindSurfaceType(
    struct tfVertexSolverMeshObjActorHandle *a, 
    struct tfVertexSolverSurfaceTypeHandle *s
);

/**
 * @brief Bind an actor to a surface
 * 
 * @param a actor
 * @param h surface
 */
HRESULT tfVertexSolverBindSurface(
    struct tfVertexSolverMeshObjActorHandle *a, 
    struct tfVertexSolverSurfaceHandleHandle *h
);

/**
 * @brief Bind an actor to a pair of object types
 * 
 * @param a actor
 * @param type1 first type
 * @param type2 second type
 */
HRESULT tfVertexSolverBindTypesSS(
    struct tfVertexSolverMeshObjTypePairActorHandle *a, 
    struct tfVertexSolverSurfaceTypeHandle *type1, 
    struct tfVertexSolverSurfaceTypeHandle *type2
);

/**
 * @brief Bind an actor to a pair of object types
 * 
 * @param a actor
 * @param type1 first type
 * @param type2 second type
 */
HRESULT tfVertexSolverBindTypesSB(
    struct tfVertexSolverMeshObjTypePairActorHandle *a, 
    struct tfVertexSolverSurfaceTypeHandle *type1, 
    struct tfVertexSolverBodyTypeHandle *type2
);

/**
 * @brief Bind an actor to a pair of object types
 * 
 * @param a actor
 * @param type1 first type
 * @param type2 second type
 */
HRESULT tfVertexSolverBindTypesBB(
    struct tfVertexSolverMeshObjTypePairActorHandle *a, 
    struct tfVertexSolverBodyTypeHandle *type1, 
    struct tfVertexSolverBodyTypeHandle *type2
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFC_MESH_BIND_H_