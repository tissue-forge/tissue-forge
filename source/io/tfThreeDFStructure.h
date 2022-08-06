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

// todo: implement a sewing method on ThreeDFStructure to join same vertices but in different meshes
//  Currently, vertices are allocated by mesh; if a vertex is shared by meshes, they will show up as separate entities in each mesh. 
//  This method should detect such occurrences and make one shared vertex among many meshes

#ifndef _SOURCE_IO_TFTHREEDFSTRUCTURE_H_
#define _SOURCE_IO_TFTHREEDFSTRUCTURE_H_

#include <tf_port.h>

#include "tfThreeDFVertexData.h"
#include "tfThreeDFEdgeData.h"
#include "tfThreeDFFaceData.h"
#include "tfThreeDFMeshData.h"

#include <unordered_map>


namespace TissueForge::io {


    struct CAPI_EXPORT ThreeDFComponentContainer {
        std::vector<ThreeDFVertexData*> vertices;
        std::vector<ThreeDFEdgeData*> edges;
        std::vector<ThreeDFFaceData*> faces;
        std::vector<ThreeDFMeshData*> meshes;
    };


    /**
     * @brief Container for relevant data found in a 3D data file. 
     * 
     * The structure object owns all constituent data. 
     * 
     * Recursively adds/removes constituent data and all child data. 
     * However, the structure enforces no rules on the constituent container data. 
     * For example, when an edge is added, all constituent vertices are added. 
     * However, no assignment is made to ensure that the parent edge 
     * is properly stored in the parent container of the vertices, neither are 
     * parent edges added when a vertex is added. 
     * 
     */
    struct CAPI_EXPORT ThreeDFStructure {

        /** Inventory of structure objects */
        ThreeDFComponentContainer inventory;

        /** Inventory of objects scheduled for deletion */
        ThreeDFComponentContainer queueRemove;

        /** Default radius applied to vertices when generating meshes from point clouds */
        FloatP_t vRadiusDef = 0.1;

        ~ThreeDFStructure();

        // Structure management

        /**
        * @brief Load from file
        * 
        * @param filePath file absolute path
        * @return HRESULT 
        */
        HRESULT fromFile(const std::string &filePath);

        /**
        * @brief Write to file
        * 
        * @param format output format of file
        * @param filePath file absolute path
        * @return HRESULT 
        */
        HRESULT toFile(const std::string &format, const std::string &filePath);

        /**
        * @brief Flush stucture. All scheduled processes are executed. 
        * 
        */
        HRESULT flush();

        // Inventory management

        /**
        * @brief Extend a structure
        * 
        * @param s stucture to extend with
        */
        HRESULT extend(const ThreeDFStructure &s);

        /**
        * @brief Clear all data of the structure
        * 
        * @return HRESULT 
        */
        HRESULT clear();
        
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
        * @brief Get all constituent meshes
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
        * @brief Get the number of constituent faces
        * 
        * @return unsigned int 
        */
        unsigned int getNumFaces();

        /**
        * @brief Get the number of constituent meshes
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
        * @brief Test whether a face is a constituent
        * 
        * @param f face to test
        * @return true 
        * @return false 
        */
        bool has(ThreeDFFaceData *f);

        /**
        * @brief Test whether a mesh is a constituent
        * 
        * @param m mesh to test
        * @return true 
        * @return false 
        */
        bool has(ThreeDFMeshData *m);

        /**
        * @brief Add a vertex
        * 
        * @param v vertex to add
        */
        void add(ThreeDFVertexData *v);

        /**
        * @brief Add an edge and all constituent data
        * 
        * @param e edge to add
        */
        void add(ThreeDFEdgeData *e);

        /**
        * @brief Add a face and all constituent data
        * 
        * @param f face to add
        */
        void add(ThreeDFFaceData *f);

        /**
        * @brief Add a mesh and all constituent data
        * 
        * @param m mesh to add
        */
        void add(ThreeDFMeshData *m);

        /**
        * @brief Remove a vertex
        * 
        * @param v vertex to remove
        */
        void remove(ThreeDFVertexData *v);

        /**
        * @brief Remove a edge and all constituent data
        * 
        * @param e edge to remove
        */
        void remove(ThreeDFEdgeData *e);

        /**
        * @brief Remove a face and all constituent data
        * 
        * @param f face to remove
        */
        void remove(ThreeDFFaceData *f);

        /**
        * @brief Remove a mesh and all constituent data
        * 
        * @param m mesh to remove
        */
        void remove(ThreeDFMeshData *m);

        void onRemoved(ThreeDFVertexData *v);
        void onRemoved(ThreeDFEdgeData *e);
        void onRemoved(ThreeDFFaceData *f);

        // Transformations

        /**
        * @brief Get the centroid of the structure
        * 
        * @return FVector3 
        */
        FVector3 getCentroid();

        /**
        * @brief Translate the structure by a displacement
        * 
        * @param displacement 
        * @return HRESULT 
        */
        HRESULT translate(const FVector3 &displacement);

        /**
        * @brief Translate the structure to a position
        * 
        * @param position 
        * @return HRESULT 
        */
        HRESULT translateTo(const FVector3 &position);

        /**
        * @brief Rotate the structure about a point
        * 
        * @param rotMat 
        * @param rotPt 
        * @return HRESULT 
        */
        HRESULT rotateAt(const FMatrix3 &rotMat, const FVector3 &rotPt);

        /**
        * @brief Rotate the structure about its centroid
        * 
        * @param rotMat 
        * @return HRESULT 
        */
        HRESULT rotate(const FMatrix3 &rotMat);

        /**
        * @brief Scale the structure about a point
        * 
        * @param scales 
        * @param scalePt 
        * @return HRESULT 
        */
        HRESULT scaleFrom(const FVector3 &scales, const FVector3 &scalePt);

        /**
        * @brief Scale the structure uniformly about a point
        * 
        * @param scale 
        * @param scalePt 
        * @return HRESULT 
        */
        HRESULT scaleFrom(const FloatP_t &scale, const FVector3 &scalePt);

        /**
        * @brief Scale the structure about its centroid
        * 
        * @param scales 
        * @return HRESULT 
        */
        HRESULT scale(const FVector3 &scales);

        /**
        * @brief Scale the structure uniformly about its centroid
        * 
        * @param scale 
        * @return HRESULT 
        */
        HRESULT scale(const FloatP_t &scale);

    private:

        int id_vertex = 0;
        int id_edge = 0;
        int id_face = 0;
        int id_mesh = 0;

    };

};

#endif // _SOURCE_IO_TFTHREEDFSTRUCTURE_H_