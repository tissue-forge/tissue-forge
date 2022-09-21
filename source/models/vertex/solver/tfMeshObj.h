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

#ifndef _MODELS_VERTEX_SOLVER_TFMESHOBJ_H_
#define _MODELS_VERTEX_SOLVER_TFMESHOBJ_H_

#include <tf_platform.h>

#include <unordered_set>
#include <unordered_map>
#include <vector>


namespace TissueForge::models::vertex { 


    class Mesh;


    struct MeshObj { 

        enum Type : unsigned int {
            NONE        = 0, 
            VERTEX      = 1, 
            SURFACE     = 2, 
            BODY        = 3, 
            STRUCTURE   = 4
        };

        /** The mesh of this object, if any */
        Mesh *mesh;

        /** Object id; unique by type in a mesh */
        int objId;

        /** Object actors */
        std::vector<struct MeshObjActor*> actors;

        MeshObj();
        virtual ~MeshObj() {};

        virtual MeshObj::Type objType() = 0;

        /** Current parent objects. */
        virtual std::vector<MeshObj*> parents() = 0;

        /** Current child objects. Child objects require this object as part of their definition. */
        virtual std::vector<MeshObj*> children() = 0;

        virtual HRESULT addChild(MeshObj *obj) = 0;

        virtual HRESULT addParent(MeshObj *obj) = 0;

        virtual HRESULT removeChild(MeshObj *obj) = 0;

        virtual HRESULT removeParent(MeshObj *obj) = 0;

        /** Validate state of object for deployment in a mesh */
        virtual bool validate() = 0;

        /** Test whether this object is in another object */
        bool in(MeshObj *obj);

        /** Test whether this object has another object */
        bool has(MeshObj *obj);

    };


    struct MeshObjActor { 

        virtual HRESULT energy(MeshObj *source, MeshObj *target, float &e) = 0;

        virtual HRESULT force(MeshObj *source, MeshObj *target, float *f) = 0;

    };


    struct MeshObjType {

        int id = -1;

        /** Object type actors */
        std::vector<MeshObjActor*> actors;

        virtual MeshObj::Type objType() = 0;

    };


    struct MeshObjTypePairActor : MeshObjActor {

        HRESULT registerPair(MeshObjType *type1, MeshObjType *type2);

        bool hasPair(MeshObjType *type1, MeshObjType *type2);

        virtual HRESULT energy(MeshObj *source, MeshObj *target, float &e) = 0;

        virtual HRESULT force(MeshObj *source, MeshObj *target, float *f) = 0;

    protected:

        std::unordered_map<int, std::unordered_set<int> > typePairs;

    };

}

#endif // _MODELS_VERTEX_SOLVER_TFMESHOBJ_H_