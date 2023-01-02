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
 * @file tfMeshObj.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFMESHOBJ_H_
#define _MODELS_VERTEX_SOLVER_TFMESHOBJ_H_

#include <tf_platform.h>
#include <types/tf_types.h>

#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>


namespace TissueForge::models::vertex { 


    class Mesh;
    class Vertex;
    class Surface;
    class Body;

    /**
     * @brief Mesh object type enum
     */
    enum MeshObjTypeLabel : unsigned int {
        NONE        = 0, 
        VERTEX      = 1, 
        SURFACE     = 2, 
        BODY        = 3
    };

    #define MESHOBJ_CLASSDEF(oType)                                 \
                                                                    \
        const int objectId() const { return _objId; }               \
                                                                    \
        /** Get the mesh object type */                             \
        MeshObjTypeLabel objType() const { return oType; }          \
                                                                    \
        /** Destroy the body. */                                    \
        HRESULT destroy();                                          \
                                                                    \
        /** Validate the body */                                    \
        bool validate();                                            \
                                                                    \
        /** Update internal data due to a change in position */     \
        HRESULT positionChanged();

    #define MESHOBJ_INITOBJ                         \
        _objId = -1;

    #define MESHOBJ_DELOBJ

    #define MESHBOJ_DEFINES_DECL(childType)         \
        bool defines(const childType *obj) const;   \

    #define MESHBOJ_DEFINES_DEF(parentAccessor)     \
        if(!obj)                                    \
            return false;                           \
        for(auto &p : obj->parentAccessor())        \
            if(p->objectId() == this->objectId())   \
                return true;                        \
        return false;

    #define MESHOBJ_DEFINEDBY_DECL(parentType)          \
        bool definedBy(const parentType *obj) const;

    #define MESHOBJ_DEFINEDBY_DEF                       \
        return obj && obj->defines(this);


    /**
     * @brief Base definition of how a mesh object acts on another mesh object
     */
    struct MeshObjActor { 

        /**
         * @brief Name of the actor
         */
        virtual std::string name() const { return "MeshObjActor"; }

        /**
         * @brief Unique name of the actor
         */
        static std::string actorName() { return "MeshObjActor"; }

        /**
         * @brief Get a list of actors bound to a mesh object
         * 
         * @tparam T type of actor
         * @tparam O mesh object type
         * @param obj mesh object
         */
        template <typename T, typename O> 
        static std::vector<T*> get(O *obj) {
            std::vector<T*> result;
            if(!obj) 
                return result;
            for(auto &a : obj->actors) 
                if(strcmp(a->name().c_str(), T::actorName().c_str()) == 0) 
                    result.push_back((T*)a);
            return result;
        }

        /**
         * @brief Get a list of actors bound to a mesh object type
         * 
         * @tparam T type of actor
         * @param objType mesh object type
         */
        template <typename T> 
        static std::vector<T*> get(struct MeshObjType *objType) {
            std::vector<T*> result;
            if(!objType) 
                return result;
            for(auto &a : objType->actors) 
                if(strcmp(a->name().c_str(), T::actorName().c_str()) == 0) 
                    result.push_back((T*)a);
            return result;
        }

        /**
         * @brief Get a JSON string representation
         */
        virtual std::string toString();

        /**
         * @brief Calculate the energy of a source object acting on a target object
         * 
         * @param source source object
         * @param target target object
         * @param e energy 
         */
        virtual FloatP_t energy(const Surface *source, const Vertex *target) { return 0; }

        /**
         * @brief Calculate the force that a source object exerts on a target object
         * 
         * @param source source object
         * @param target target object
         * @param f force
         */
        virtual FVector3 force(const Surface *source, const Vertex *target) { return FVector3(0); }

        /**
         * @brief Calculate the energy of a source object acting on a target object
         * 
         * @param source source object
         * @param target target object
         * @param e energy 
         */
        virtual FloatP_t energy(const Body *source, const Vertex *target) { return 0; }

        /**
         * @brief Calculate the force that a source object exerts on a target object
         * 
         * @param source source object
         * @param target target object
         * @param f force
         */
        virtual FVector3 force(const Body *source, const Vertex *target) { return FVector3(0); }

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

        /**
         * @brief Get the mesh object type
         */
        virtual MeshObjTypeLabel objType() const = 0;

        /**
         * @brief Get a summary string
         */
        virtual std::string str() const = 0;

    };


    /**
     * @brief Base definition of how a mesh object of a type acts on another mesh object
     * through interactions with a mesh object of another type
     */
    struct MeshObjTypePairActor : MeshObjActor {

        /**
         * @brief Name of the actor
         */
        virtual std::string name() { return "MeshObjTypePairActor"; }

        /**
         * @brief Unique name of the actor
         */
        static std::string actorName() { return "MeshObjTypePairActor"; }

        /**
         * @brief Register a pair of types for this actor
         * 
         * @param type1 first type
         * @param type2 second type
         */
        HRESULT registerPair(MeshObjType *type1, MeshObjType *type2);

        /**
         * @brief Test whether a pair of types is registered with this actor
         * 
         * @param type1 first type
         * @param type2 second type
         * @return true if the pair of types is registered with this actor
         */
        bool hasPair(MeshObjType *type1, MeshObjType *type2);

    protected:

        std::unordered_map<int, std::unordered_set<int> > typePairs;

    };

}

#endif // _MODELS_VERTEX_SOLVER_TFMESHOBJ_H_