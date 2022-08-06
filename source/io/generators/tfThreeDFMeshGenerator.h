/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

#ifndef _SOURCE_IO_GENERATORS_THREEDFMESHGENERATOR_H_
#define _SOURCE_IO_GENERATORS_THREEDFMESHGENERATOR_H_

#include <io/tfThreeDFVertexData.h>
#include <io/tfThreeDFEdgeData.h>
#include <io/tfThreeDFFaceData.h>
#include <io/tfThreeDFMeshData.h>


namespace TissueForge::io {


    struct ThreeDFMeshGenerator { 

        ThreeDFMeshGenerator();

        /**
         * @brief Get the mesh of this generator. 
         * 
         * Mesh must first be processed before it is generated. 
         * 
         * @return ThreeDFMeshData* 
         */
        ThreeDFMeshData *getMesh();

        /**
         * @brief Do all instructions to generate mesh. 
         * 
         * @return HRESULT 
         */
        virtual HRESULT process() = 0;

    protected:

        ThreeDFMeshData *mesh;

    };


    // Supprting generator functions

    /**
     * @brief Adds elements of a ball at a point to a mesh
     * 
     * @param mesh mesh to append
     * @param faces generated faces
     * @param edges generated edges
     * @param vertices generated vertices
     * @param normals generated normals
     * @param radius radius of ball
     * @param offset location of ball
     * @param numDivs number of refinements
     * @return HRESULT 
     */
    HRESULT generateBallMesh(
        ThreeDFMeshData *mesh, 
        std::vector<ThreeDFFaceData*> *faces, 
        std::vector<ThreeDFEdgeData*> *edges, 
        std::vector<ThreeDFVertexData*> *vertices, 
        std::vector<FVector3> *normals, 
        const FloatP_t &radius=1.0, 
        const FVector3 &offset={0.f,0.f,0.f}, 
        const unsigned int &numDivs=0
    );

    HRESULT generateCylinderMesh(
        ThreeDFMeshData *mesh, 
        std::vector<ThreeDFFaceData*> *faces, 
        std::vector<ThreeDFEdgeData*> *edges, 
        std::vector<ThreeDFVertexData*> *vertices, 
        std::vector<FVector3> *normals, 
        const FloatP_t &radius=1.0, 
        const FVector3 &startPt={0.f,0.f,0.f}, 
        const FVector3 &endPt={1.f,1.f,1.f}, 
        const unsigned int &numDivs=0
    );

};

#endif // _SOURCE_IO_GENERATORS_THREEDFMESHGENERATOR_H_