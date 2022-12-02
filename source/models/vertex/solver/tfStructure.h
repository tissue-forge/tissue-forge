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

#ifndef _MODELS_VERTEX_SOLVER_TFSTRUCTURE_H_
#define _MODELS_VERTEX_SOLVER_TFSTRUCTURE_H_

#include <tf_port.h>

#include "tf_mesh.h"


namespace TissueForge::models::vertex { 


    class Vertex;
    class Surface;
    class Body;

    struct StructureType;


    class CAPI_EXPORT Structure : public MeshObj {

        std::vector<Structure*> structures_parent;
        std::vector<Structure*> structures_child;
        std::vector<Body*> bodies;

    public:

        /** Id of the type*/
        unsigned int typeId;

        Structure() : MeshObj() {};

        /** Get the mesh object type */
        MeshObj::Type objType() const override { return MeshObj::Type::STRUCTURE; }

        /** Get the parents of the object */
        std::vector<MeshObj*> parents() const override;

        /** Get the children of the object */
        std::vector<MeshObj*> children() const override { return TissueForge::models::vertex::vectorToBase(structures_child); }

        /** Add a child object */
        HRESULT addChild(MeshObj *obj) override;

        /** Add a parent object */
        HRESULT addParent(MeshObj *obj) override;

        /** Remove a child object */
        HRESULT removeChild(MeshObj *obj) override;

        /** Remove a parent object */
        HRESULT removeParent(MeshObj *obj) override;

        /** Get a summary string */
        std::string str() const override;

        /** Get a JSON string representation */
        std::string toString();

        /**
         * Destroy the structure. 
         * 
         * If the structure is in a mesh, then it is removed from the mesh. 
        */
        HRESULT destroy() override;

        /** Validate the structure */
        bool validate() override { return true; }

        /** Get the structure type */
        StructureType *type() const;

        /** Become a different type */
        HRESULT become(StructureType *stype);

        /** Get the structures that this structure defines */
        std::vector<Structure*> getStructures() const { return structures_parent; }

        /** Get the bodies that define the structure */
        std::vector<Body*> getBodies() const;

        /** Get the surfaces that define the structure */
        std::vector<Surface*> getSurfaces() const;

        /** Get the vertices that define the structure */
        std::vector<Vertex*> getVertices() const;

    };


    /**
     * @brief Mesh structure type
     * 
     * Can be used as a factory to create mesh structure instances with 
     * processes and properties that correspond to the type. 
     */
    struct CAPI_EXPORT StructureType : MeshObjType {

        /** Name of this structure type */
        std::string name;

        StructureType(const bool &noReg=false);

        /** Get the mesh object type */
        MeshObj::Type objType() const override { return MeshObj::Type::STRUCTURE; }

        /** Get a summary string */
        virtual std::string str() const override;

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation. 
         * 
         * The returned type is automatically registered with the solver. 
         * 
         * @param str a string, as returned by ``toString``
         */
        static StructureType *fromString(const std::string &str);

        /** Get a registered type by name */
        static StructureType *findFromName(const std::string &_name);

        /**
         * @brief Registers a type with the engine.
         * 
         * Note that this occurs automatically, unless noReg==true in constructor.  
         */
        virtual HRESULT registerType();

        /**
         * @brief A callback for when a type is registered
         */
        virtual void on_register() {}

        /**
         * @brief Tests whether this type is registered
         * 
         * @return true if registered
         */
        bool isRegistered();

        /**
         * @brief Get the type engine instance
         */
        virtual StructureType *get();

        /** list of instances that belong to this type */    
        std::vector<Structure*> getInstances();

        /** list of instances ids that belong to this type */
        std::vector<int> getInstanceIds();

        /** number of instances that belong to this type */
        unsigned int getNumInstances();
        
    };

    inline bool operator< (const TissueForge::models::vertex::Structure& lhs, const TissueForge::models::vertex::Structure& rhs) { return lhs.objId < rhs.objId; }
    inline bool operator> (const TissueForge::models::vertex::Structure& lhs, const TissueForge::models::vertex::Structure& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::Structure& lhs, const TissueForge::models::vertex::Structure& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::Structure& lhs, const TissueForge::models::vertex::Structure& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::Structure& lhs, const TissueForge::models::vertex::Structure& rhs) { return lhs.objId == rhs.objId; }
    inline bool operator!=(const TissueForge::models::vertex::Structure& lhs, const TissueForge::models::vertex::Structure& rhs) { return !(lhs == rhs); }

    inline bool operator< (const TissueForge::models::vertex::StructureType& lhs, const TissueForge::models::vertex::StructureType& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::models::vertex::StructureType& lhs, const TissueForge::models::vertex::StructureType& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::StructureType& lhs, const TissueForge::models::vertex::StructureType& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::StructureType& lhs, const TissueForge::models::vertex::StructureType& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::StructureType& lhs, const TissueForge::models::vertex::StructureType& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::models::vertex::StructureType& lhs, const TissueForge::models::vertex::StructureType& rhs) { return !(lhs == rhs); }

}


inline std::ostream &operator<<(std::ostream& os, const TissueForge::models::vertex::Structure &o)
{
    os << o.str().c_str();
    return os;
}

inline std::ostream &operator<<(std::ostream& os, const TissueForge::models::vertex::StructureType &o)
{
    os << o.str().c_str();
    return os;
}

#endif // _MODELS_VERTEX_SOLVER_TFSTRUCTURE_H_