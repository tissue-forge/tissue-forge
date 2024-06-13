/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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

#ifndef _SOURCE_IO_TFTHREEDFACEDATA_H_
#define _SOURCE_IO_TFTHREEDFACEDATA_H_

#include <TissueForge_private.h>

#include <vector>


namespace TissueForge::io {


    struct ThreeDFVertexData;
    struct ThreeDFEdgeData;
    struct ThreeDFMeshData;
    struct ThreeDFStructure;


    /**
     * @brief 3D data file face data
     * 
     */
    struct CAPI_EXPORT ThreeDFFaceData {
        
        /** Parent structure */
        ThreeDFStructure *structure = NULL;

        /** Face normal */
        FVector3 normal;

        /** ID, if any. Unique to its structure and type */
        int id = -1;

        /** Constituent edges */
        std::vector<ThreeDFEdgeData*> edges;

        /** Parent meshes */
        std::vector<ThreeDFMeshData*> meshes;

        /**
         * @brief Get all constituent vertices
         * 
         * @return std::vector<ThreeDFVertexData*> 
         */
        std::vector<ThreeDFVertexData*> getVertices();

        /**
         * @brief Get all constituent edges
         * 
         * @return std::vector<ThreeDFEdgeData*> 
         */
        std::vector<ThreeDFEdgeData*> getEdges();

        /**
        * @brief Get all parent meshes
        * 
        * @return std::vector<ThreeDFMeshData*> 
        */
        std::vector<ThreeDFMeshData*> getMeshes();

        /**
        * @brief Get the number of constituent vertices
        * 
        * @return unsigned int 
        */
        unsigned int getNumVertices();

        /**
        * @brief Get the number of constituent edges
        * 
        * @return unsigned int 
        */
        unsigned int getNumEdges();

        /**
        * @brief Get the number of parent meshes
        * 
        * @return unsigned int 
        */
        unsigned int getNumMeshes();

        /**
        * @brief Test whether a vertex is a constituent
        * 
        * @param v vertex to test
        * @return true 
        * @return false 
        */
        bool has(ThreeDFVertexData *v);
        
        /**
        * @brief Test whether an edge is a constituent
        * 
        * @param e edge to test
        * @return true 
        * @return false 
        */
        bool has(ThreeDFEdgeData *e);
        
        /**
        * @brief Test whether in a mesh
        * 
        * @param m mesh to test
        * @return true 
        * @return false 
        */
        bool in(ThreeDFMeshData *m);
        
        /**
        * @brief Test whether in a structure
        * 
        * @param s structure to test
        * @return true 
        * @return false 
        */
        bool in(ThreeDFStructure *s);

    };

};

#endif // _SOURCE_IO_TFTHREEDFACEDATA_H_