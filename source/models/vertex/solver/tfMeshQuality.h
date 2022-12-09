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

#ifndef _MODELS_VERTEX_SOLVER_TFMESHQUALITY_H_
#define _MODELS_VERTEX_SOLVER_TFMESHQUALITY_H_


#include "tfMeshObj.h"

#include <tf_port.h>

#include <mutex>
#include <set>
#include <vector>


namespace TissueForge::models::vertex {


    class Mesh;


    /** An operation that modifies the topology of a mesh to improve its quality */
    struct MeshQualityOperation {

        enum Flag : unsigned int {
            None    = 0, 
            Active  = 1 << 0, 
            Custom  = 1 << 1
        };

        unsigned int flags;

        /** Source mesh object */
        MeshObj *source;

        /**
         * @brief Target mesh objects.
         * 
         * Used to identify dependencies between operations
         */
        std::vector<MeshObj*> targets;

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
         */
        HRESULT appendNext(MeshQualityOperation *_next);

        /** Remove an operation to the list of next operations */
        HRESULT removeNext(MeshQualityOperation *_next);

        /** Compute all upstream operations */
        std::set<MeshQualityOperation*> upstreams() const;

        /** Compute all downstream operations */
        std::set<MeshQualityOperation*> downstreams() const;

        /** Compute all upstream operations that have no dependencies */
        std::set<MeshQualityOperation*> headOperations() const;

        /** Validate this operation. 
         * 
         * - A source object cannot target an object of the same type and with a lesser object ID
        */
        HRESULT validate() {
            for(auto &t : targets) 
                if(t->objType() == source->objType() && t->objId < source->objId) 
                    return E_FAIL;
            return S_OK;
        }

        /** Check whether this operation is still valid */
        virtual bool check() { return !(flags & Flag::None); };

        /** Implement this operation */
        virtual HRESULT implement() { return S_OK; }

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
        typedef HRESULT (*OperationFunction)(MeshObj*, std::vector<MeshObj*>);

        OperationCheck *opCheck;
        OperationFunction *opFunc;

        CustomQualityOperation(Mesh *_mesh, OperationFunction *_opFunc, OperationCheck *_opCheck=NULL) : 
            MeshQualityOperation(_mesh), 
            opFunc{_opFunc}, 
            opCheck{_opCheck} 
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

        HRESULT implement() override { return (*opFunc)(source, targets); }
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

        /** Borrowed pointer to the owning mesh */
        Mesh *mesh;

        /** Flag for whether currently doing work */
        bool _working;

    public:

        MeshQuality(
            Mesh *_mesh, 
            const FloatP_t &vertexMergeDistCf=0.0001, 
            const FloatP_t &surfaceDemoteAreaCf=0.0001, 
            const FloatP_t &bodyDemoteVolumeCf=0.0001, 
            const FloatP_t &_edgeSplitDistCf=2.0
        );

        /** Get a JSON string representation */
        std::string toString();

        /** Perform quality operations work */
        HRESULT doQuality();

        /** Test whether quality operations are being done */
        const bool working() const { return _working; }

        /** Get the id of the parent mesh */
        const int getMeshId() const;

        /** Get the distance below which two vertices are scheduled for merging */
        FloatP_t getVertexMergeDistance() const { return vertexMergeDist; };

        /** Set the distance below which two vertices are scheduled for merging */
        HRESULT setVertexMergeDistance(const FloatP_t &_val);

        /** Get the area below which a surface is scheduled to become a vertex */
        FloatP_t getSurfaceDemoteArea() const { return surfaceDemoteArea; };

        /** Set the area below which a surface is scheduled to become a vertex */
        HRESULT setSurfaceDemoteArea(const FloatP_t &_val);

        /** Get the volume below which a body is scheduled to become a vertex */
        FloatP_t getBodyDemoteVolume() const { return bodyDemoteVolume; }

        /** Set the volume below which a body is scheduled to become a vertex */
        HRESULT setBodyDemoteVolume(const FloatP_t &_val);

        /** Get the distance at which two vertices are seperated when a vertex is split */
        FloatP_t getEdgeSplitDist() const { return edgeSplitDist; };

        /** Set the distance at which two vertices are seperated when a vertex is split */
        HRESULT setEdgeSplitDist(const FloatP_t &_val);

        /** Get whether 2D collisions are implemented */
        bool getCollision2D() const { return collision2D; }

        /** Set whether 2D collisions are implemented */
        HRESULT setCollision2D(const bool &_collision2D);
    };
}


#endif // _MODELS_VERTEX_SOLVER_TFMESHQUALITY_H_