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


    struct MeshQualityOperation {

        enum Flag : unsigned int {
            None    = 0, 
            Active  = 1 << 0, 
            Custom  = 1 << 1
        };

        unsigned int flags;

        MeshObj *source;

        std::vector<MeshObj*> targets;

        /** Upstream operations, if any */
        std::vector<MeshQualityOperation*> prev;

        /** Downstream operations, if any */
        std::vector<MeshQualityOperation*> next;

        std::mutex lock;

        MeshQualityOperation(Mesh *_mesh);

        virtual ~MeshQualityOperation() {};

        HRESULT appendNext(MeshQualityOperation *_next);

        HRESULT removeNext(MeshQualityOperation *_next);

        std::set<MeshQualityOperation*> upstreams() const;

        std::set<MeshQualityOperation*> downstreams() const;

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
         * Initial length of an edge created by splitting a vertex.
         */
        FloatP_t edgeSplitDist;

        /** Borrowed pointer to the owning mesh */
        Mesh *mesh;

        /** Flag for whether currently doing work */
        bool _working;

    public:

        MeshQuality(
            Mesh *_mesh, 
            const FloatP_t &vertexMergeDistCf=0.0001, 
            const FloatP_t &surfaceDemoteAreaCf=0.0001, 
            const FloatP_t &_edgeSplitDistCf=2.0
        );

        HRESULT doQuality();

        const bool working() const { return _working; }

        FloatP_t getVertexMergeDistance() const { return vertexMergeDist; };

        HRESULT setVertexMergeDistance(const FloatP_t &_val);

        FloatP_t getSurfaceDemoteArea() const { return surfaceDemoteArea; };

        HRESULT setSurfaceDemoteArea(const FloatP_t &_val);

        FloatP_t getEdgeSplitDist() const { return edgeSplitDist; };

        HRESULT setEdgeSplitDist(const FloatP_t &_val);
    };
}


#endif // _MODELS_VERTEX_SOLVER_TFMESHQUALITY_H_