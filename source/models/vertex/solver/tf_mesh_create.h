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

#ifndef _MODELS_VERTEX_SOLVER_TF_MESH_CREATE_H_
#define _MODELS_VERTEX_SOLVER_TF_MESH_CREATE_H_

#include "tfMesh.h"

#include <vector>

namespace TissueForge::models::vertex {

    /**
     * @brief Create a mesh of quadrilateral surfaces
     * 
     * @param mesh parent mesh
     * @param stype surface type
     * @param startPos starting position
     * @param num_1 number of elements in the first direction
     * @param num_2 number of elements in the second direction
     * @param len_1 length of each element in the first direction
     * @param len_2 length of each element in the second direction
     * @param ax_1 axis name of the first direction (e.g., "x")
     * @param ax_2 axis name of the second direction
     * @return constructed surfaces
     */
    CPPAPI_FUNC(std::vector<std::vector<Surface*> >) createQuadMesh(
        Mesh *mesh, 
        SurfaceType *stype,
        const FVector3 &startPos, 
        const unsigned int &num_1, 
        const unsigned int &num_2, 
        const FloatP_t &len_1,
        const FloatP_t &len_2,
        const char *ax_1="x", 
        const char *ax_2="y"
    );

    /**
     * @brief Create a mesh of parallelepiped bodies
     * 
     * @param mesh parent mesh
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
     * @return constructed bodies
     */
    CPPAPI_FUNC(std::vector<std::vector<std::vector<Body*> > >) createPLPDMesh(
        Mesh *mesh, 
        BodyType *btype, 
        SurfaceType *stype,
        const FVector3 &startPos, 
        const unsigned int &num_1, 
        const unsigned int &num_2, 
        const unsigned int &num_3, 
        const FloatP_t &len_1,
        const FloatP_t &len_2,
        const FloatP_t &len_3,
        const char *ax_1="x", 
        const char *ax_2="y"
    );

    /**
     * @brief Create a mesh of hexagonal surfaces
     * 
     * @param mesh parent mesh
     * @param stype surface type
     * @param startPos starting position
     * @param num_1 number of elements in the first direction
     * @param num_2 number of elements in the second direction
     * @param hexRad radius of hexagon vertices
     * @param ax_1 axis name of the first direction (e.g., "x")
     * @param ax_2 axis name of the second direction
     * @return constructed surfaces
     */
    CPPAPI_FUNC(std::vector<std::vector<Surface*> >) createHex2DMesh(
        Mesh *mesh, 
        SurfaceType *stype, 
        const FVector3 &startPos, 
        const unsigned int &num_1, 
        const unsigned int &num_2, 
        const FloatP_t &hexRad,
        const char *ax_1="x", 
        const char *ax_2="y"
    );

    /**
     * @brief Create a mesh of bodies from extruded hexagonal surfaces. 
     * 
     * Surfaces are places in the plane of the first and second directions, 
     * and extruded along the third direction. 
     * 
     * @param mesh parent mesh
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
     * @return constructed bodies
     */
    CPPAPI_FUNC(std::vector<std::vector<std::vector<Body*> > >) createHex3DMesh(
        Mesh *mesh, 
        BodyType *btype, 
        SurfaceType *stype,
        const FVector3 &startPos, 
        const unsigned int &num_1, 
        const unsigned int &num_2, 
        const unsigned int &num_3, 
        const FloatP_t &hexRad,
        const FloatP_t &hex_height,
        const char *ax_1="x", 
        const char *ax_2="y"
    );
    
};

#endif // _MODELS_VERTEX_SOLVER_TF_MESH_CREATE_H_