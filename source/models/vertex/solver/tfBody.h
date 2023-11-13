/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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
 * @file tfBody.h
 * 
 */

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

        /**
         * @brief Get a summary string
         */
        std::string str() const;

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Update all internal data and parents
         */
        void updateInternals();

        /**
         * @brief Add a surface
         * 
         * @param s a surface
         */
        HRESULT add(Surface *s);

        /**
         * @brief Remove a surface
         * 
         * @param s a surface
         */
        HRESULT remove(Surface *s);

        /**
         * @brief Replace a surface with a surface
         * 
         * @param toInsert surface to insert
         * @param toRemove surface to remove
         */
        HRESULT replace(Surface *toInsert, Surface *toRemove);

        /**
         * @brief Destroy a body. 
         * 
         * Any resulting surfaces without a body are also destroyed. 
         * 
         * @param target a body
         */
        static HRESULT destroy(Body *target);

        /**
         * @brief Destroy a body. 
         * 
         * Any resulting surfaces without a body are also destroyed. 
         * 
         * @param target a handle to a body
         * @return HRESULT 
         */
        static HRESULT destroy(BodyHandle &target);

        /**
         * @brief Destroy bodies. 
         * 
         * Any resulting surfaces without a body are also destroyed. 
         * 
         * @param target handles to a body
         * @return HRESULT 
         */
        static HRESULT destroy(const std::vector<Body*>& toRemove);

        /**
         * @brief Get the body type
         */
        BodyType *type() const;

        /**
         * @brief Become a different type
         * 
         * @param btype the type to become
         */
        HRESULT become(BodyType *btype);

        /**
         * @brief Get the surfaces that define the body
         */
        const std::vector<Surface*>& getSurfaces() const { return surfaces; }

        /**
         * @brief Get the vertices that define the body
         */
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
         * @brief Get the connected bodies. 
         * 
         * A body is connected if it shares a surface.
         */
        std::vector<Body*> connectedBodies() const;

        /**
         * @brief Get the adjacent bodies. 
         * 
         * A body is adjacent if it shares a vertex.
         */
        std::vector<Body*> adjacentBodies() const;

        /**
         * @brief Get the neighboring surfaces of a surface on this body.
         * 
         * Two surfaces are a neighbor on this body if they define the body and share a vertex
         * 
         * @param s a surface of the body
         */
        std::vector<Surface*> neighborSurfaces(const Surface *s) const;

        /**
         * @brief Get the mass density
         */
        const FloatP_t& getDensity() const { return density; }

        /**
         * @brief Set the mass density
         * 
         * @param _density density
         */
        void setDensity(const FloatP_t &_density) { density = _density; }

        /**
         * @brief Get the centroid
         */
        const FVector3& getCentroid() const { return centroid; }

        /**
         * @brief Get the velocity, calculated as the velocity of the centroid
         */
        FVector3 getVelocity() const;

        /**
         * @brief Get the surface area
         */
        const FloatP_t& getArea() const { return area; }

        /**
         * @brief Get the volume
         */
        const FloatP_t& getVolume() const { return volume; }

        /**
         * @brief Get the mass
         */
        FloatP_t getMass() const { return volume * density; }

        /**
         * @brief Get the surface area contribution of a vertex to this body
         * 
         * @param v a vertex
         */
        FloatP_t getVertexArea(const Vertex *v) const;

        /**
         * @brief Get the volume contribution of a vertex to this body
         * 
         * @param v a vertex
         */
        FloatP_t getVertexVolume(const Vertex *v) const;

        /**
         * @brief Get the mass contribution of a vertex to this body
         * 
         * @param v a vertex
         */
        FloatP_t getVertexMass(const Vertex *v) const { return getVertexVolume(v) * density; }

        /**
         * @brief Get the surfaces that define the interface between this body and another body
         * 
         * @param b a body
         */
        std::vector<Surface*> findInterface(const Body *b) const;

        /**
         * @brief Get the contacting surface area of this body with another body
         * 
         * @param other a body
         */
        FloatP_t contactArea(const Body *other) const;

        /**
         * @brief Get the vertices that define both this body and another body
         * 
         * @param other a body
         */
        std::vector<Vertex*> sharedVertices(const Body *other) const;

        /**
         * @brief Test whether a point is outside. Test is performed using the nearest surface
         * 
         * @param pos position
         * @return true if the point is outside
         */
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


    /**
     * @brief A handle to a @ref Body. 
     *
     * The engine allocates @ref Body memory in blocks, and @ref Body
     * values get moved around all the time, so their addresses change.
     * 
     * This is a safe way to work with a @ref Body.
     */
    struct CAPI_EXPORT BodyHandle {

        int id;

        BodyHandle(const int &_id=-1);

        /**
         * @brief Get the underlying object, if any
         */
        Body *body() const;

        /**
         * @brief Test whether defined by a vertex
         * 
         * @param v a vertex
         * @return true if defined by a vertex
         */
        bool definedBy(const VertexHandle &v) const;

        /**
         * @brief Test whether defined by a surface
         * 
         * @param s a surface
         * @return true if defined by a surface
         */
        bool definedBy(const SurfaceHandle &s) const;

        /**
         * @brief Get the mesh object type
         */
        MeshObjTypeLabel objType() const { return MeshObjTypeLabel::BODY; }

        /**
         * @brief Destroy the body.
         */
        HRESULT destroy();

        /**
         * @brief Validate the body
         */
        bool validate();

        /**
         * @brief Update internal data due to a change in position
         */
        HRESULT positionChanged();

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
         * @param s a JSON string
         */
        static BodyHandle fromString(const std::string &s);

        /**
         * @brief Add a surface
         * 
         * @param s a surface
         */
        HRESULT add(const SurfaceHandle &s);

        /**
         * @brief Remove a surface
         * 
         * @param s a surface
         */
        HRESULT remove(const SurfaceHandle &s);

        /**
         * @brief Replace a surface with a surface
         * 
         * @param toInsert surface to insert
         * @param toRemove surface to remove
         */
        HRESULT replace(const SurfaceHandle &toInsert, const SurfaceHandle &toRemove);

        /**
         * @brief Get the body type
         */
        BodyType *type() const;

        /**
         * @brief Become a different type
         * 
         * @param btype the type to become
         */
        HRESULT become(BodyType *btype);

        /**
         * @brief Get the surfaces that define the body
         */
        std::vector<SurfaceHandle> getSurfaces() const;

        /**
         * @brief Get the vertices that define the body
         */
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
         * @brief Get the connected bodies. 
         * 
         * A body is connected if it shares a surface.
         */
        std::vector<BodyHandle> connectedBodies() const;

        /**
         * @brief Get the adjacent bodies. 
         * 
         * A body is adjacent if it shares a vertex.
         */
        std::vector<BodyHandle> adjacentBodies() const;

        /**
         * @brief Get the neighboring surfaces of a surface on this body.
         * 
         * Two surfaces are a neighbor on this body if they define the body and share a vertex
         * 
         * @param s a surface of the body
         */
        std::vector<SurfaceHandle> neighborSurfaces(const SurfaceHandle &s) const;

        /**
         * @brief Get the mass density
         */
        FloatP_t getDensity() const;

        /**
         * @brief Set the mass density
         * 
         * @param _density density
         */
        void setDensity(const FloatP_t &_density);

        /**
         * @brief Get the centroid
         */
        FVector3 getCentroid() const;

        /**
         * @brief Get the velocity, calculated as the velocity of the centroid
         */
        FVector3 getVelocity() const;

        /**
         * @brief Get the surface area
         */
        FloatP_t getArea() const;

        /**
         * @brief Get the volume
         */
        FloatP_t getVolume() const;

        /**
         * @brief Get the mass
         */
        FloatP_t getMass() const;

        /**
         * @brief Get the surface area contribution of a vertex to this body
         * 
         * @param v a vertex
         */
        FloatP_t getVertexArea(const VertexHandle &v) const;

        /**
         * @brief Get the volume contribution of a vertex to this body
         * 
         * @param v a vertex
         */
        FloatP_t getVertexVolume(const VertexHandle &v) const;

        /**
         * @brief Get the mass contribution of a vertex to this body
         * 
         * @param v a vertex
         */
        FloatP_t getVertexMass(const VertexHandle &v) const;

        /**
         * @brief Get the amount of species in the enclosed volume, if any
         */
        state::StateVector *getSpecies() const;

        /**
         * @brief Set the amount of species in the enclosed volume
         * 
         * @param s species in the enclosed volume
         */
        HRESULT setSpecies(state::StateVector *s) const;

        /**
         * @brief Get the surfaces that define the interface between this body and another body
         * 
         * @param b a body
         */
        std::vector<SurfaceHandle> findInterface(const BodyHandle &b) const;

        /**
         * @brief Get the contacting surface area of this body with another body
         * 
         * @param other a body
         */
        FloatP_t contactArea(const BodyHandle &other) const;

        /**
         * @brief Get the vertices that define both this body and another body
         * 
         * @param other a body
         */
        std::vector<VertexHandle> sharedVertices(const BodyHandle &other) const;

        /**
         * @brief Test whether a point is outside. Test is performed using the nearest surface
         * 
         * @param pos position
         * @return true if the point is outside
         */
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

        /**
         * @brief Get the mesh object type
         */
        MeshObjTypeLabel objType() const override { return MeshObjTypeLabel::BODY; }

        /**
         * @brief Get a summary string
         */
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

        /**
         * @brief Add an instance
         * 
         * @param i instance
         */
        HRESULT add(const BodyHandle &i);

        /**
         * @brief Remove an instance
         * 
         * @param i instance
         */
        HRESULT remove(const BodyHandle &i);

        /**
         * @brief Remove instances
         * 
         * @param i instances to remove
         */
        HRESULT remove(const std::vector<BodyHandle>& i);

        /**
         * @brief list of instances that belong to this type
         */
        std::vector<BodyHandle> getInstances();

        /**
         * @brief list of instances ids that belong to this type
         */
        std::vector<int> getInstanceIds() { return _instanceIds; }

        /**
         * @brief number of instances that belong to this type
         */
        unsigned int getNumInstances();

        /**
         * @brief Construct a body of this type from a set of surfaces
         */
        BodyHandle operator() (const std::vector<SurfaceHandle> &surfaces);

        /**
         * @brief Construct a body of this type from a mesh
         */
        BodyHandle operator() (TissueForge::io::ThreeDFMeshData* ioMesh, SurfaceType *stype);

        /**
         * @brief Create a body from a surface in the mesh and a position
         * 
         * @param base surface
         * @param pos position
         */
        BodyHandle extend(const SurfaceHandle &base, const FVector3 &pos);

        /**
         * @brief Create a body from a surface in a mesh by extruding along the outward-facing normal of the surface
         * 
         * todo: add support for extruding at an angle
         * 
         * @param base surface
         * @param normLen length along which to extrude
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

namespace std {

    template <> 
    struct hash<TissueForge::models::vertex::BodyHandle> {
        size_t operator() (const TissueForge::models::vertex::BodyHandle& h) const {
            return hash<int>()(h.id);
        }
    };
}

#endif // _MODELS_VERTEX_SOLVER_TFBODY_H_