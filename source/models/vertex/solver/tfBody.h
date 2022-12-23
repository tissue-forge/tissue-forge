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
    struct VertexHandle;
    class Surface;
    struct SurfaceHandle;
    class Mesh;

    struct BodyHandle;
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
    class CAPI_EXPORT Body { 

        /** Object id; unique by type in a mesh */
        int _objId;

    public:

        /** Id of the type*/
        int typeId;

    private:

        /** Surfaces that define this body */
        std::vector<Surface*> surfaces;

        /** current centroid */
        FVector3 centroid;

        /** current surface area */
        FloatP_t area;

        /** current volume */
        FloatP_t volume;

        /** mass density */
        FloatP_t density;

    public:

        /** Object actors */
        std::vector<MeshObjActor*> actors;

        /** Amount of species in the enclosed volume, if any */
        state::StateVector *species;

        Body();
        ~Body();

        /** Construct a body from a set of surfaces */
        static BodyHandle create(const std::vector<SurfaceHandle> &_surfaces);

        /** Construct a body from a mesh */
        static BodyHandle create(TissueForge::io::ThreeDFMeshData *ioMesh);

        MESHOBJ_DEFINEDBY_DECL(Vertex);
        MESHOBJ_DEFINEDBY_DECL(Surface);
        MESHOBJ_CLASSDEF(MeshObjTypeLabel::BODY)

        /** Get a summary string */
        std::string str() const;

        /** Get a JSON string representation */
        std::string toString();

        /** Update all internal data and parents */
        void updateInternals();

        /** Add a surface */
        HRESULT add(Surface *s);

        /** Remove a surface */
        HRESULT remove(Surface *s);

        /** Replace a surface a surface */
        HRESULT replace(Surface *toInsert, Surface *toRemove);

        /**
         * Destroy a body. 
         * 
         * Any resulting surfaces without a body are also destroyed. 
         */
        static HRESULT destroy(Body *target);

        /**
         * Destroy a body. 
         * 
         * Any resulting surfaces without a body are also destroyed. 
         */
        static HRESULT destroy(BodyHandle &target);

        /** Get the body type */
        BodyType *type() const;

        /** Become a different type */
        HRESULT become(BodyType *btype);

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


        friend BodyHandle;
        friend Vertex;
        friend Mesh;
        friend BodyType;

    };


    struct CAPI_EXPORT BodyHandle {

        int id;

        BodyHandle(const int &_id=-1);

        /** Get the underlying object, if any */
        Body *body() const;

        bool definedBy(const VertexHandle &v) const;

        bool definedBy(const SurfaceHandle &s) const;

        /** Get the mesh object type */
        MeshObjTypeLabel objType() const { return MeshObjTypeLabel::BODY; }

        /** Destroy the body. */
        HRESULT destroy();

        /** Validate the body */
        bool validate();

        /** Update internal data due to a change in position */
        HRESULT positionChanged();

        /** Get a summary string */
        std::string str() const;

        /** Get a JSON string representation */
        std::string toString();

        /** Create an instance from a JSON string representation */
        static BodyHandle fromString(const std::string &s);

        /** Add a surface */
        HRESULT add(const SurfaceHandle &s);

        /** Remove a surface */
        HRESULT remove(const SurfaceHandle &s);

        /** Replace a surface a surface */
        HRESULT replace(const SurfaceHandle &toInsert, const SurfaceHandle &toRemove);

        /** Get the body type */
        BodyType *type() const;

        /** Become a different type */
        HRESULT become(BodyType *btype);

        /** Get the surfaces that define the body */
        std::vector<SurfaceHandle> getSurfaces() const;

        /** Get the vertices that define the body */
        std::vector<VertexHandle> getVertices() const;

        /**
         * @brief Find a vertex that defines this body
         * 
         * @param dir direction to look with respect to the centroid
         */
        VertexHandle findVertex(const FVector3 &dir) const;

        /**
         * @brief Find a surface that defines this body
         * 
         * @param dir direction to look with respect to the centroid
         */
        SurfaceHandle findSurface(const FVector3 &dir) const;

        /**
         * Get the neighboring bodies. 
         * 
         * A body is a neighbor if it shares a surface.
         */
        std::vector<BodyHandle> neighborBodies() const;

        /**
         * Get the neighboring surfaces of a surface on this body.
         * 
         * Two surfaces are a neighbor on this body if they define the body and share a vertex
         */
        std::vector<SurfaceHandle> neighborSurfaces(const SurfaceHandle &s) const;

        /** Get the mass density */
        FloatP_t getDensity() const;

        /** Set the mass density */
        void setDensity(const FloatP_t &_density);

        /** Get the centroid */
        FVector3 getCentroid() const;

        /** Get the velocity, calculated as the velocity of the centroid */
        FVector3 getVelocity() const;

        /** Get the surface area */
        FloatP_t getArea() const;

        /** Get the volume */
        FloatP_t getVolume() const;

        /** Get the mass */
        FloatP_t getMass() const;

        /** Get the surface area contribution of a vertex to this body */
        FloatP_t getVertexArea(const VertexHandle &v) const;

        /** Get the volume contribution of a vertex to this body */
        FloatP_t getVertexVolume(const VertexHandle &v) const;

        /** Get the mass contribution of a vertex to this body */
        FloatP_t getVertexMass(const VertexHandle &v) const;

        /** Get the amount of species in the enclosed volume, if any */
        state::StateVector *getSpecies() const;

        /** Set the amount of species in the enclosed volume */
        HRESULT setSpecies(state::StateVector *s) const;

        /** Get the surfaces that define the interface between this body and another body */
        std::vector<SurfaceHandle> findInterface(const BodyHandle &b) const;

        /** Get the contacting surface area of this body with another body */
        FloatP_t contactArea(const BodyHandle &other) const;

        /** Test whether a point is outside. Test is performed using the nearest surface */
        bool isOutside(const FVector3 &pos) const;

        /**
         * @brief Split into two bodies. The split is defined by a cut plane
         * 
         * @param cp_pos position on the cut plane
         * @param cp_norm cut plane normal
         * @param stype type of newly created surface. taken from connected surfaces if not specified
         */
        BodyHandle split(const FVector3 &cp_pos, const FVector3 &cp_norm, SurfaceType *stype=NULL);

        operator bool() const { return id >= 0; }
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
        MeshObjTypeLabel objType() const override { return MeshObjTypeLabel::BODY; }

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

        /** Add an instance */
        HRESULT add(const BodyHandle &i);

        /** Remove an instance */
        HRESULT remove(const BodyHandle &i);

        /** list of instances that belong to this type */    
        std::vector<BodyHandle> getInstances();

        /** list of instances ids that belong to this type */
        std::vector<int> getInstanceIds() { return _instanceIds; }

        /** number of instances that belong to this type */
        unsigned int getNumInstances();

        /** Construct a body of this type from a set of surfaces */
        BodyHandle operator() (const std::vector<SurfaceHandle> &surfaces);

        /** Construct a body of this type from a mesh */
        BodyHandle operator() (TissueForge::io::ThreeDFMeshData* ioMesh, SurfaceType *stype);

        /** Create a body from a surface in the mesh and a position */
        BodyHandle extend(const SurfaceHandle &base, const FVector3 &pos);

        /** Create a body from a surface in a mesh by extruding along the outward-facing normal of the surface
         * 
         * todo: add support for extruding at an angle
        */
        BodyHandle extrude(const SurfaceHandle &base, const FloatP_t &normLen);

    private:

        std::vector<int> _instanceIds;

    };

    inline bool operator< (const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return lhs.objectId() < rhs.objectId(); }
    inline bool operator> (const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return lhs.objectId() == rhs.objectId(); }
    inline bool operator!=(const TissueForge::models::vertex::Body& lhs, const TissueForge::models::vertex::Body& rhs) { return !(lhs == rhs); }

    inline bool operator< (const TissueForge::models::vertex::BodyHandle& lhs, const TissueForge::models::vertex::BodyHandle& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::models::vertex::BodyHandle& lhs, const TissueForge::models::vertex::BodyHandle& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::BodyHandle& lhs, const TissueForge::models::vertex::BodyHandle& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::BodyHandle& lhs, const TissueForge::models::vertex::BodyHandle& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::BodyHandle& lhs, const TissueForge::models::vertex::BodyHandle& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::models::vertex::BodyHandle& lhs, const TissueForge::models::vertex::BodyHandle& rhs) { return !(lhs == rhs); }

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