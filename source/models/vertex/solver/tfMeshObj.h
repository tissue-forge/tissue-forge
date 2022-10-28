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


    /**
     * @brief Base mesh object definition. 
     * 
     * All objects that can go into a mesh should derive from this class.
     */
    struct MeshObj { 

        /** Mesh object type enum */
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

        /** Get the mesh object type */
        virtual MeshObj::Type objType() = 0;

        /** Current parent objects. */
        virtual std::vector<MeshObj*> parents() = 0;

        /** Current child objects. Child objects require this object as part of their definition. */
        virtual std::vector<MeshObj*> children() = 0;

        /** Add a child object */
        virtual HRESULT addChild(MeshObj *obj) = 0;

        /** Add a parent object */
        virtual HRESULT addParent(MeshObj *obj) = 0;

        /** Remove a child object */
        virtual HRESULT removeChild(MeshObj *obj) = 0;

        /** Remove a parent object */
        virtual HRESULT removeParent(MeshObj *obj) = 0;

        /** Destroy the object */
        virtual HRESULT destroy() = 0;

        /** Validate state of object for deployment in a mesh */
        virtual bool validate() = 0;

        /** Test whether this object is in another object */
        bool in(MeshObj *obj);

        /** Test whether this object has another object */
        bool has(MeshObj *obj);

    };


    /**
     * @brief Base definition of how a mesh object acts on another mesh object
     */
    struct MeshObjActor { 

        /**
         * @brief Calculate the energy of a source object acting on a target object
         * 
         * @param source source object
         * @param target target object
         * @param e energy 
         */
        virtual HRESULT energy(MeshObj *source, MeshObj *target, FloatP_t &e) = 0;

        /**
         * @brief Calculate the force that a source object exerts on a target object
         * 
         * @param source source object
         * @param target target object
         * @param f force
         */
        virtual HRESULT force(MeshObj *source, MeshObj *target, FloatP_t *f) = 0;

    };


    /**
     * @brief Base mesh object type definition. 
     * 
     * The type definition of a mesh object should derive from this class
     */
    struct MeshObjType {

        /** Id of the type. -1 when not registered with the solver. */
        int id = -1;

        /** Object type actors */
        std::vector<MeshObjActor*> actors;

        /** Get the mesh object type */
        virtual MeshObj::Type objType() = 0;

    };


    /**
     * @brief Base definition of how a mesh object of a type acts on another mesh object
     * through interactions with a mesh object of another type
     */
    struct MeshObjTypePairActor : MeshObjActor {

        /** Register a pair of types for this actor */
        HRESULT registerPair(MeshObjType *type1, MeshObjType *type2);

        /** Test whether a pair of types is registered with this actor */
        bool hasPair(MeshObjType *type1, MeshObjType *type2);

        /**
         * @brief Calculate the energy of a source object acting on a target object
         * 
         * @param source source object
         * @param target target object
         * @param e energy 
         */
        virtual HRESULT energy(MeshObj *source, MeshObj *target, FloatP_t &e) = 0;

        /**
         * @brief Calculate the force that a source object exerts on a target object
         * 
         * @param source source object
         * @param target target object
         * @param f force
         */
        virtual HRESULT force(MeshObj *source, MeshObj *target, FloatP_t *f) = 0;

    protected:

        std::unordered_map<int, std::unordered_set<int> > typePairs;

    };

}

#endif // _MODELS_VERTEX_SOLVER_TFMESHOBJ_H_