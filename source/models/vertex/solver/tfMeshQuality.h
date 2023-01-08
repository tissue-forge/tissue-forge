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

/**
 * @file tfMeshQuality.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFMESHQUALITY_H_
#define _MODELS_VERTEX_SOLVER_TFMESHQUALITY_H_


#include "tfMeshObj.h"

#include <tf_port.h>

#include <mutex>
#include <set>
#include <vector>


namespace TissueForge::models::vertex {


    class Mesh;


    /**
     * @brief An operation that modifies the topology of a mesh to improve its quality
     */
    struct MeshQualityOperation {

        enum Flag : unsigned int {
            None    = 0, 
            Active  = 1 << 0, 
            Custom  = 1 << 1
        };

        unsigned int flags;

        /**
         * @brief Target mesh objects.
         * 
         * Used to identify dependencies between operations
         */
        std::vector<int> targets;

        /** Upstream operations, if any */
        std::set<MeshQualityOperation*> prev;

        /** Downstream operations, if any */
        std::set<MeshQualityOperation*> next;

        /** Lock, for safe modification during concurrent work */
        std::mutex lock;

        MeshQualityOperation(Mesh *_mesh);

        virtual ~MeshQualityOperation() {};

        /**
         * @brief Add an operation to the list of next operations
         * 
         * If the operation is already upstream of this operation, 
         * then the call is ignored
         * 
         * @param _next the operation
         */
        HRESULT appendNext(MeshQualityOperation *_next);

        /**
         * @brief Remove an operation to the list of next operations
         * 
         * @param _next the operation
         */
        HRESULT removeNext(MeshQualityOperation *_next);

        /**
         * @brief Compute all upstream operations
         */
        std::set<MeshQualityOperation*> upstreams() const;

        /**
         * @brief Compute all downstream operations
         */
        std::set<MeshQualityOperation*> downstreams() const;

        /**
         * @brief Compute all upstream operations that have no dependencies
         */
        std::set<MeshQualityOperation*> headOperations() const;

        /**
         * @brief Validate this operation
         */
        virtual HRESULT validate() { return S_OK; }

        /**
         * @brief Check whether this operation is still valid
         * 
         * @return true if this operation is still valid
         */
        virtual bool check() { return !(flags & Flag::None); };

        /**
         * @brief Do all prep, checks and planning for this operation
         */
        virtual void prep() {}

        /**
         * @brief Returns how many vertices will be created by this operation
         */
        virtual size_t numNewVertices() const { return 0; };

        /**
         * @brief Returns how many surfaces will be created by this operation
         */
        virtual size_t numNewSurfaces() const { return 0; };

        /**
         * @brief Returns how many bodies will be created by this operation
         */
        virtual size_t numNewBodies() const { return 0; };

        /**
         * @brief Implement this operation
         * 
         * @return ids of affected children, if any
         */
        virtual std::vector<int> implement() { return {}; }

    protected:

        Mesh *mesh;
    };


    /**
     * @brief Custom mesh quality operation.
     * 
     * todo: implement support for custom mesh quality operations
     */
    struct CAPI_EXPORT CustomQualityOperation : MeshQualityOperation {

        typedef bool (*OperationCheck)();
        typedef void (*OperationPrep)();
        typedef std::vector<int> (*OperationFunction)(std::vector<int>);

        OperationCheck *opCheck;
        OperationPrep *opPrep;
        OperationFunction *opFunc;

        CustomQualityOperation(Mesh *_mesh, OperationFunction *_opFunc, OperationCheck *_opCheck=NULL, OperationPrep *_opPrep=NULL) : 
            MeshQualityOperation(_mesh), 
            opFunc{_opFunc}, 
            opCheck{_opCheck}, 
            opPrep{_opPrep}
        {
            flags = Flag::Active | Flag::Custom;
        };

        virtual ~CustomQualityOperation() {
            if(opCheck) {
                delete opCheck;
                opCheck = 0;
            }
            delete opFunc;
            opFunc = 0;
        };

        bool check() override {
            if(opCheck) return (*opCheck)();
            return true;
        }

        void prep() override {
            if(opPrep) (*opPrep)();
        }

        std::vector<int> implement() override { return (*opFunc)(targets); }
    };


    /**
     * @brief An object that schedules topological operations on a mesh to maintain its quality
     * 
     */
    class CAPI_EXPORT MeshQuality {

        /** 
         * Vertex merge criterion.
         * 
         * Two vertices are merged if their distance is less than this value. 
         */
        FloatP_t vertexMergeDist;

        /**
         * Surface demotion criterion.
         * 
         * A surface becomes a vertex if its area is less than this value.
         */
        FloatP_t surfaceDemoteArea;

        /**
         * Body demotion criterion.
         * 
         * A body becomes a vertex if its volume is less than this value.
         */
        FloatP_t bodyDemoteVolume;

        /**
         * Initial length of an edge created by splitting a vertex.
         */
        FloatP_t edgeSplitDist;

        /**
         * Flag for whether doing 2D collisions
         */
        bool collision2D;

        /** Flag for whether currently doing work */
        bool _working;

    public:

        MeshQuality(
            const FloatP_t &vertexMergeDistCf=0.0001, 
            const FloatP_t &surfaceDemoteAreaCf=0.0001, 
            const FloatP_t &bodyDemoteVolumeCf=0.0001, 
            const FloatP_t &_edgeSplitDistCf=2.0
        );

        /**
         * @brief Get a summary string
         */
        std::string str() const;

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Create an instance from a JSON string representation
         * 
         * @param s JSON string representation
         */
        static MeshQuality fromString(const std::string &s);

        /**
         * @brief Perform quality operations work
         */
        HRESULT doQuality();

        /**
         * @brief Test whether quality operations are being done
         * 
         * @return true if quality operations are being done
         */
        const bool working() const { return _working; }

        /**
         * @brief Get the distance below which two vertices are scheduled for merging
         */
        FloatP_t getVertexMergeDistance() const { return vertexMergeDist; };

        /**
         * @brief Set the distance below which two vertices are scheduled for merging
         * 
         * @param _val distance
         */
        HRESULT setVertexMergeDistance(const FloatP_t &_val);

        /**
         * @brief Get the area below which a surface is scheduled to become a vertex
         */
        FloatP_t getSurfaceDemoteArea() const { return surfaceDemoteArea; };

        /**
         * @brief Set the area below which a surface is scheduled to become a vertex
         * 
         * @param _val area
         */
        HRESULT setSurfaceDemoteArea(const FloatP_t &_val);

        /**
         * @brief Get the volume below which a body is scheduled to become a vertex
         */
        FloatP_t getBodyDemoteVolume() const { return bodyDemoteVolume; }

        /**
         * @brief Set the volume below which a body is scheduled to become a vertex
         * 
         * @param _val volume
         */
        HRESULT setBodyDemoteVolume(const FloatP_t &_val);

        /**
         * @brief Get the distance at which two vertices are seperated when a vertex is split
         */
        FloatP_t getEdgeSplitDist() const { return edgeSplitDist; };

        /**
         * @brief Set the distance at which two vertices are seperated when a vertex is split
         * 
         * @param _val distance
         */
        HRESULT setEdgeSplitDist(const FloatP_t &_val);

        /**
         * @brief Get whether 2D collisions are implemented
         * 
         * @return true if 2D collisions are implemented
         */
        bool getCollision2D() const { return collision2D; }

        /**
         * @brief Set whether 2D collisions are implemented
         * 
         * @param _collision2D flag indicating whether 2D collisions are implemented
         */
        HRESULT setCollision2D(const bool &_collision2D);
    };
}


#endif // _MODELS_VERTEX_SOLVER_TFMESHQUALITY_H_