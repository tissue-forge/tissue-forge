/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#ifndef _SOURCE_IO_TFTHREEDMESHDATA_H_
#define _SOURCE_IO_TFTHREEDMESHDATA_H_

#include <TissueForge_private.h>

#include <vector>

#include "tfThreeDFRenderData.h"


namespace TissueForge::io {


    struct ThreeDFVertexData;
    struct ThreeDFEdgeData;
    struct ThreeDFFaceData;
    struct ThreeDFStructure;


    /**
     * @brief 3D data file mesh data
     * 
     */
    struct CAPI_EXPORT ThreeDFMeshData {
        
        /** Parent structure */
        ThreeDFStructure *structure = NULL;

        /** ID, if any. Unique to its structure and type */
        int id = -1;

        /** Constituent faces */
        std::vector<ThreeDFFaceData*> faces;

        /** Mesh name */
        std::string name;

        /** Rendering data */
        ThreeDFRenderData *renderData = NULL;

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
         * @brief Get all constituent faces
         * 
         * @return std::vector<ThreeDFFaceData*> 
         */
        std::vector<ThreeDFFaceData*> getFaces();

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
         * @brief Get the number of constituent faces
         * 
         * @return unsigned int 
         */
        unsigned int getNumFaces();

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
         * @brief Test whether a face is a constituent
         * 
         * @param f face to test
         * @return true 
         * @return false 
         */
        bool has(ThreeDFFaceData *f);
        
        /**
         * @brief Test whether in a structure
         * 
         * @param s structure to test
         * @return true 
         * @return false 
         */
        bool in(ThreeDFStructure *s);

        /**
         * @brief Get the centroid of the mesh
         * 
         * @return FVector3 
         */
        FVector3 getCentroid();

        // Transformations

        /**
         * @brief Translate the mesh by a displacement
         * 
         * @param displacement 
         * @return HRESULT 
         */
        HRESULT translate(const FVector3 &displacement);
        
        /**
         * @brief Translate the mesh to a position
         * 
         * @param position 
         * @return HRESULT 
         */
        HRESULT translateTo(const FVector3 &position);

        /**
         * @brief Rotate the mesh about a point
         * 
         * @param rotMat 
         * @param rotPt 
         * @return HRESULT 
         */
        HRESULT rotateAt(const FMatrix3 &rotMat, const FVector3 &rotPt);
        
        /**
         * @brief Rotate the mesh about its centroid
         * 
         * @param rotMat 
         * @return HRESULT 
         */
        HRESULT rotate(const FMatrix3 &rotMat);
        
        /**
         * @brief Scale the mesh about a point
         * 
         * @param scales 
         * @param scalePt 
         * @return HRESULT 
         */
        HRESULT scaleFrom(const FVector3 &scales, const FVector3 &scalePt);
        
        /**
         * @brief Scale the mesh uniformly about a point
         * 
         * @param scale 
         * @param scalePt 
         * @return HRESULT 
         */
        HRESULT scaleFrom(const FloatP_t &scale, const FVector3 &scalePt);
        
        /**
         * @brief Scale the mesh about its centroid
         * 
         * @param scales 
         * @return HRESULT 
         */
        HRESULT scale(const FVector3 &scales);
        
        /**
         * @brief Scale the mesh uniformly about its centroid
         * 
         * @param scale 
         * @return HRESULT 
         */
        HRESULT scale(const FloatP_t &scale);

    };

};

#endif // _SOURCE_IO_TFTHREEDMESHDATA_H_