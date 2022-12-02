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

#ifndef _MODELS_VERTEX_SOLVER_TFBODY_H_
#define _MODELS_VERTEX_SOLVER_TFBODY_H_

#include <tf_port.h>

#include <state/tfStateVector.h>

#include "tf_mesh.h"

#include <io/tfThreeDFMeshData.h>


namespace TissueForge::models::vertex { 


    class Vertex;
    class Surface;
    class Structure;
    class Mesh;

    struct BodyType;
    struct SurfaceType;

    /**
     * @brief The mesh body is a volume-enclosing object of mesh surfaces. 
     * 
     * The mesh body consists of at least four mesh surfaces. 
     * 
     * The mesh body can have a state vector, which represents a uniform amount of substance 
     * enclosed in the volume of the body. 
     * 
     */
    class CAPI_EXPORT Body : public MeshObj { 

        /** Surfaces that define this body */
        std::vector<Surface*> surfaces;

        /** Structures defined by this body */
        std::vector<Structure*> structures;

        /** current centroid */
        FVector3 centroid;

        /** current surface area */
        FloatP_t area;

        /** current volume */
        FloatP_t volume;

        /** mass density */
        FloatP_t density;

        void _updateInternal();

    public:

        /** Id of the type*/
        unsigned int typeId;

        /** Amount of species in the enclosed volume, if any */
        state::StateVector *species;

        Body();

        /** Construct a body from a set of surfaces */
        Body(std::vector<Surface*> _surfaces);

        /** Construct a body from a mesh */
        Body(TissueForge::io::ThreeDFMeshData *ioMesh);

        /** Get the mesh object type */
        MeshObj::Type objType() const override { return MeshObj::Type::BODY; }

        /** Get the parents of the object */
        std::vector<MeshObj*> parents() const override;

        /** Get the children of the object */
        std::vector<MeshObj*> children() const override;

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

        /** Add a surface */
        HRESULT add(Surface *s);

        /** Remove a surface */
        HRESULT remove(Surface *s);

        /** Replace a surface a surface */
        HRESULT replace(Surface *toInsert, Surface *toRemove);

        /** Add a structure */
        HRESULT add(Structure *s);

        /** Remove a structure */
        HRESULT remove(Structure *s);

        /** Replace a structure */
        HRESULT replace(Structure *toInsert, Structure *toRemove);

        /**
         * Destroy the body. 
         * 
         * If the body is in a mesh, then it and any objects it defines are removed from the mesh. 
        */
        HRESULT destroy() override;

        /** Validate the body */
        bool validate() override;

        /** Update internal data due to a change in position */
        HRESULT positionChanged();

        /** Get the body type */
        BodyType *type() const;

        /** Become a different type */
        HRESULT become(BodyType *btype);

        /** Get the structures defined by the body */
        std::vector<Structure*> getStructures() const;

        /** Get the surfaces that define the body */
        std::vector<Surface*> getSurfaces() const { return surfaces; }

        /** Get the vertices that define the body */
        std::vector<Vertex*> getVertices() const;

        /**
         * @brief Find a vertex that defines this body
         * 
         * @param dir direction to look with respect to the centroid
         */
        Vertex *findVertex(const FVector3 &dir) const;

        /**
         * @brief Find a surface that defines this body
         * 
         * @param dir direction to look with respect to the centroid
         */
        Surface *findSurface(const FVector3 &dir) const;

        /**
         * Get the neighboring bodies. 
         * 
         * A body is a neighbor if it shares a surface.
         */
        std::vector<Body*> neighborBodies() const;

        /**
         * Get the neighboring surfaces of a surface on this body.
         * 
         * Two surfaces are a neighbor on this body if they define the body and share a vertex
         */
        std::vector<Surface*> neighborSurfaces(const Surface *s) const;

        /** Get the mass density */
        FloatP_t getDensity() const { return density; }

        /** Set the mass density */
        void setDensity(const FloatP_t &_density) { density = _density; }

        /** Get the centroid */
        FVector3 getCentroid() const { return centroid; }

        /** Get the velocity, calculated as the velocity of the centroid */
        FVector3 getVelocity() const;

        /** Get the surface area */
        FloatP_t getArea() const { return area; }

        /** Get the volume */
        FloatP_t getVolume() const { return volume; }

        /** Get the mass */
        FloatP_t getMass() const { return volume * density; }

        /** Get the surface area contribution of a vertex to this body */
        FloatP_t getVertexArea(const Vertex *v) const;

        /** Get the volume contribution of a vertex to this body */
        FloatP_t getVertexVolume(const Vertex *v) const;

        /** Get the mass contribution of a vertex to this body */
        FloatP_t getVertexMass(const Vertex *v) const { return getVertexVolume(v) * density; }

        /** Get the surfaces that define the interface between this body and another body */
        std::vector<Surface*> findInterface(const Body *b) const;

        /** Get the contacting surface area of this body with another body */
        FloatP_t contactArea(const Body *other) const;

        /** Test whether a point is outside. Test is performed using the nearest surface */
        bool isOutside(const FVector3 &pos) const;

        /**
         * @brief Split into two bodies. The split is defined by a cut plane
         * 
         * @param cp_pos position on the cut plane
         * @param cp_norm cut plane normal
         * @param stype type of newly created surface. taken from connected surfaces if not specified
         */
        Body *split(const FVector3 &cp_pos, const FVector3 &cp_norm, SurfaceType *stype=NULL);

        
        friend Vertex;
        friend Mesh;
        friend BodyType;

    };


    /**
     * @brief Mesh body type
     * 
     * Can be used as a factory to create mesh body instances with 
     * processes and properties that correspond to the type. 
     */
    struct CAPI_EXPORT BodyType : MeshObjType {

        /** Name of this body type */
        std::string name;

        /** Mass density */
        FloatP_t density;

        BodyType(const bool &noReg=false);

        /** Get the mesh object type */
        MeshObj::Type objType() const override { return MeshObj::Type::BODY; }

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
        static BodyType *fromString(const std::string &str);

        /** Get a registered type by name */
        static BodyType *findFromName(const std::string &_name);

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
        virtual BodyType *get();

        /** list of instances that belong to this type */    
        std::vector<Body*> getInstances();

        /** list of instances ids that belong to this type */
        std::vector<int> getInstanceIds();

        /** number of instances that belong to this type */
        unsigned int getNumInstances();

        /** Construct a body of this type from a set of surfaces */
        Body *operator() (std::vector<Surface*> surfaces);

        /** Construct a body of this type from a mesh */
        Body *operator() (TissueForge::io::ThreeDFMeshData* ioMesh, SurfaceType *stype);

        /** Create a body from a surface in the mesh and a position */
        Body *extend(Surface *base, const FVector3 &pos);

        /** Create a body from a surface in a mesh by extruding along the outward-facing normal of the surface
         * 
         * todo: add support for extruding at an angle
        */
        Body *extrude(Surface *base, const FloatP_t &normLen);

    };

    inline bool operator< (const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return lhs.objId < rhs.objId; }
    inline bool operator> (const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return lhs.objId == rhs.objId; }
    inline bool operator!=(const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return !(lhs == rhs); }

    inline bool operator< (const TissueForge::models::vertex::BodyType& lhs, const TissueForge::models::vertex::BodyType& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::models::vertex::BodyType& lhs, const TissueForge::models::vertex::BodyType& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::BodyType& lhs, const TissueForge::models::vertex::BodyType& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::BodyType& lhs, const TissueForge::models::vertex::BodyType& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::BodyType& lhs, const TissueForge::models::vertex::BodyType& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::models::vertex::BodyType& lhs, const TissueForge::models::vertex::BodyType& rhs) { return !(lhs == rhs); }

}


inline std::ostream &operator<<(std::ostream& os, const TissueForge::models::vertex::Body &o)
{
    os << o.str().c_str();
    return os;
}

inline std::ostream &operator<<(std::ostream& os, const TissueForge::models::vertex::BodyType &o)
{
    os << o.str().c_str();
    return os;
}

#endif // _MODELS_VERTEX_SOLVER_TFBODY_H_