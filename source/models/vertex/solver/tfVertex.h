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
 * @file tfVertex.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFVERTEX_H_
#define _MODELS_VERTEX_SOLVER_TFVERTEX_H_

#include <tf_port.h>

#include <tfParticle.h>
#include <rendering/tfStyle.h>

#include "tfMeshObj.h"
#include "tfMesh.h"

#include <io/tfThreeDFVertexData.h>

#include <vector>


namespace TissueForge::models::vertex { 


    class Surface;
    struct SurfaceHandle;
    class Body;
    struct BodyHandle;
    class Mesh;

    struct VertexHandle;


    struct CAPI_EXPORT MeshParticleType : ParticleType { 

        MeshParticleType() : ParticleType(true) {
            std::memcpy(this->name, "MeshParticleType", sizeof("MeshParticleType"));
            style->setVisible(false);
            dynamics = PARTICLE_OVERDAMPED;
            registerType();
        };

    };

    CAPI_FUNC(MeshParticleType*) MeshParticleType_get();


    /**
     * @brief The mesh vertex is a volume of a mesh centered at a point in a space.
     * 
     */
    class CAPI_EXPORT Vertex {

        /** Object id; unique by type in a mesh */
        int _objId;

        /** Particle id. -1 if not assigned */
        int pid;

        /** Connected surfaces */
        std::vector<Surface*> surfaces;

        /** Cached particle data: mass */
        FloatP_t _particleMass;

        /** Cached particle data: position */
        FVector3 _particlePosition;

        /** Cached particle data: velocity */
        FVector3 _particleVelocity;

        /** Cached neighbor vertices */
        std::vector<Vertex*> _neighborVertices;

    public:

        Vertex();
        ~Vertex();

        /**
         * @brief Create a vertex
         * 
         * @param _pid id of underlying particle
         */
        static VertexHandle create(const unsigned int &_pid);

        /**
         * @brief Create a vertex
         * 
         * @param position position of vertex
         */
        static VertexHandle create(const FVector3 &position);

        /**
         * @brief Create a vertex
         * 
         * @param vdata a vertex
         */
        static VertexHandle create(TissueForge::io::ThreeDFVertexData *vdata);

        MESHBOJ_DEFINES_DECL(Surface);
        MESHBOJ_DEFINES_DECL(Body);
        MESHOBJ_CLASSDEF(MeshObjTypeLabel::VERTEX)

        /**
         * @brief Get a summary string
         */
        std::string str() const;

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Add a surface
         * 
         * @param s surface to add
         */
        HRESULT add(Surface *s);

        /**
         * @brief Insert a surface at a location in the list of surfaces
         * 
         * @param s surface to insert
         * @param idx location in the list of surfaces
         */
        HRESULT insert(Surface *s, const int &idx);

        /**
         * @brief Insert a surface before another surface
         * 
         * @param s a surface 
         * @param before surface to insert before
         */
        HRESULT insert(Surface *s, Surface *before);

        /**
         * @brief Remove a surface
         * 
         * @param s surface to remove
         */
        HRESULT remove(Surface *s);

        /**
         * @brief Replace a surface at a location in the list of surfaces
         * 
         * @param toInsert a surface
         * @param idx a location in the list of surfaces
         */
        HRESULT replace(Surface *toInsert, const int &idx);

        /**
         * @brief Replace a surface with another surface
         * 
         * @param toInsert surface to insert
         * @param toRemove surface to remove
         */
        HRESULT replace(Surface *toInsert, Surface *toRemove);

        /** 
         * @brief Get the id of the underlying particle 
         */
        const int getPartId() const { return pid; }

        /** 
         * @brief Get the bodies defined by the vertex 
         */
        std::vector<Body*> getBodies() const;

        /** 
         * @brief Get the surfaces defined by the vertex 
         */
        std::vector<Surface*> getSurfaces() const { return surfaces; }

        /**
         * @brief Find a surface defined by this vertex
         * 
         * @param dir direction to look with respect to the vertex
         */
        Surface *findSurface(const FVector3 &dir) const;

        /**
         * @brief Find a body defined by this vertex
         * 
         * @param dir direction to look with respect to the vertex
         */
        Body *findBody(const FVector3 &dir) const;

        /**
         * @brief Update internal neighbor vertex data
         */
        void updateNeighborVertices();

        /** 
         * @brief Get the neighbor vertices.
         * 
         * A vertex is a neighbor if it defines an edge with this vertex.
         */
        std::vector<Vertex*> neighborVertices() const { return _neighborVertices; }

        /**
         * @brief Get the surfaces that this vertex and another vertex both define
         * 
         * @param other another vertex
         */
        std::vector<Surface*> sharedSurfaces(const Vertex *other) const;

        /**
         * @brief Get the current area
         */
        FloatP_t getArea() const;

        /** 
         * @brief Get the current volume 
         */
        FloatP_t getVolume() const;

        /** 
         * @brief Get the current mass 
         */
        FloatP_t getMass() const;

        /** 
         * @brief Update the properties of the underlying particle 
         */
        HRESULT updateProperties();

        /** 
         * @brief Get a handle to the underlying particle, if any 
         */
        ParticleHandle *particle() const;

        /** 
         * @brief Get the current position 
         */
        FVector3 getPosition() const { return _particlePosition; }

        /**
         * @brief Set the current position
         * 
         * @param pos position
         * @param updateChildren flag indicating whether to update dependent objects
         */
        HRESULT setPosition(const FVector3 &pos, const bool &updateChildren=true);

        /**
         * @brief Get the current velocity
         */
        FVector3 getVelocity() const { return _particleVelocity; }

        /**
         * @brief Get the cached particle mass from the most recent update
         */
        FloatP_t getCachedParticleMass() const { return _particleMass; }

        /**
         * @brief Transfer all bonds to another vertex
         * 
         * @param other another vertex
         */
        HRESULT transferBondsTo(Vertex *other);

        /**
         * @brief Replace a surface
         * 
         * @param toReplace surface to replace
         */
        HRESULT replace(Surface *toReplace);

        /**
         * @brief Create a vertex and replace a surface with it
         * 
         * @param position position of vertex
         * @param toReplace surface to replace
         * @return newly created vertex
         */
        static Vertex *replace(const FVector3 &position, Surface *toReplace);

        /**
         * @brief Create a vertex and replace a surface with it
         * 
         * @param position position of vertex
         * @param toReplace surface to replace
         * @return newly created vertex
         */
        static VertexHandle replace(const FVector3 &position, SurfaceHandle &toReplace);

        /**
         * @brief Replace a body
         * 
         * @param toReplace a body
         */
        HRESULT replace(Body *toReplace);

        /**
         * @brief Create a vertex and replace a body with it
         * 
         * @param position position
         * @param toReplace body to replace
         * @return newly created vertex
         */
        static Vertex *replace(const FVector3 &position, Body *toReplace);

        /**
         * @brief Create a vertex and replace a body with it
         * 
         * @param position position
         * @param toReplace body to replace
         * @return newly created vertex
         */
        static VertexHandle replace(const FVector3 &position, BodyHandle &toReplace);

        /**
         * @brief Merge with a vertex. 
         * 
         * The passed vertex is destroyed.
         * 
         * @param toRemove vertex to remove
         * @param lenCf distance coefficient in [0, 1] for where to place the vertex, from the kept vertex to the removed vertex
         */
        HRESULT merge(Vertex *toRemove, const FloatP_t &lenCf=0.5f);

        /**
         * @brief Inserts a vertex between two vertices
         * 
         * @param v1 first vertex
         * @param v2 second vertex
         */
        HRESULT insert(Vertex *v1, Vertex *v2);

        /**
         * @brief Create a vertex and inserts it between two vertices
         * 
         * @param position position
         * @param v1 first vertex
         * @param v2 second vertex
         * @return newly created vertex
         */
        static Vertex *insert(const FVector3 &position, Vertex *v1, Vertex *v2);

        /**
         * @brief Create a vertex and inserts it between two vertices
         * 
         * @param position position
         * @param v1 first vertex
         * @param v2 second vertex
         * @return newly created vertex
         */
        static VertexHandle insert(const FVector3 &position, const VertexHandle &v1, const VertexHandle &v2);

        /**
         * @brief Insert a vertex between a vertex and each of a set of vertices
         * 
         * @param vf a vertex
         * @param nbs a set of vertices
         */
        HRESULT insert(Vertex *vf, std::vector<Vertex*> nbs);

        /**
         * @brief Create a vertex and insert it between a vertex and each of a set of vertices
         * 
         * @param position position
         * @param vf a vertex
         * @param nbs a set of vertices
         * @return newly created vertex
         */
        static Vertex *insert(const FVector3 &position, Vertex *vf, std::vector<Vertex*> nbs);

        /**
         * @brief Create a vertex and insert it between a vertex and each of a set of vertices
         * 
         * @param position position
         * @param vf a vertex
         * @param nbs a set of vertices
         * @return newly created vertex
         */
        static VertexHandle insert(const FVector3 &position, const VertexHandle &vf, const std::vector<VertexHandle> &nbs);

        /**
         * @brief Calculate the topology of a vertex split without implementing the split
         * 
         * @param sep separation distance
         * @param verts_v vertices that continue to define existing surface
         * @param verts_new_v vertices that define a new surface
         */
        HRESULT splitPlan(const FVector3 &sep, std::vector<Vertex*> &verts_v, std::vector<Vertex*> &verts_new_v);

        /**
         * @brief Implement a pre-calculated vertex split, as determined by splitPlan
         * 
         * @param sep separation distance
         * @param verts_v vertices that continue to define existing surface
         * @param verts_new_v vertices that define a new surface
         */
        Vertex *splitExecute(const FVector3 &sep, const std::vector<Vertex*> &verts_v, const std::vector<Vertex*> &verts_new_v);

        /**
         * @brief Split a vertex into an edge
         * 
         * The vertex must define at least one surface.
         * 
         * New topology is governed by a cut plane at the midpoint of, and orthogonal to, the new edge. 
         * Each first-order neighbor vertex is connected to the vertex of the new edge on the same side of 
         * the cut plane. 
         * 
         * @param sep separation distance
         * @return newly created vertex
         */
        Vertex *split(const FVector3 &sep);


        friend VertexHandle;
        friend Surface;
        friend Body;
        friend Mesh;

    };


    /**
     * @brief A handle to a @ref Vertex. 
     *
     * The engine allocates @ref Vertex memory in blocks, and @ref Vertex
     * values get moved around all the time, so their addresses change.
     * 
     * This is a safe way to work with a @ref Vertex.
     */
    struct CAPI_EXPORT VertexHandle {

        int id;

        VertexHandle(const int &_id=-1);

        /**
         * @brief Get the underlying object, if any
         */
        Vertex *vertex() const;

        /**
         * @brief Test whether defines a surface
         * 
         * @param s a surface
         * @return true if defines a surface
         */
        bool defines(const SurfaceHandle &s) const;

        /**
         * @brief Test whether defines a body
         * 
         * @param b a body
         * @return true if defines a body
         */
        bool defines(const BodyHandle &b) const;

        /**
         * @brief Get the mesh object type
         */
        MeshObjTypeLabel objType() const { return MeshObjTypeLabel::VERTEX; }

        /** 
         * @brief Destroy the vertex 
         */
        HRESULT destroy();

        /** 
         * @brief Validate the vertex
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
        std::string toString() const;

        /**
         * @brief Create an instance from a JSON string representation
         * 
         * @param s JSON string
         */
        static VertexHandle fromString(const std::string &s);

        /**
         * @brief Add a surface
         * 
         * @param s surface to add
         */
        HRESULT add(const SurfaceHandle &s) const;

        /**
         * @brief Insert a surface at a location in the list of surfaces
         * 
         * @param s a surface
         * @param idx location in the list of surfaces
         */
        HRESULT insert(const SurfaceHandle &s, const int &idx) const;

        /**
         * @brief Insert a surface before another surface
         * 
         * @param s a surface
         * @param before surface to insert before
         */
        HRESULT insert(const SurfaceHandle &s, const SurfaceHandle &before) const;

        /**
         * @brief Remove a surface
         * 
         * @param s surface to remove
         */
        HRESULT remove(const SurfaceHandle &s) const;

        /**
         * @brief Replace a surface at a location in the list of surfaces
         * 
         * @param toInsert a surface
         * @param idx location in the list of surfaces
         */
        HRESULT replace(const SurfaceHandle &toInsert, const int &idx) const;

        /**
         * @brief Replace a surface with another surface
         * 
         * @param toInsert a surface
         * @param toRemove another surface
         */
        HRESULT replace(const SurfaceHandle &toInsert, const SurfaceHandle &toRemove) const;

        /**
         * @brief Get the id of the underlying particle
         */
        const int getPartId() const;

        /** 
         * @brief Get the bodies defined by the vertex 
         */
        std::vector<BodyHandle> getBodies() const;

        /** 
         * @brief Get the surfaces defined by the vertex 
         */
        std::vector<SurfaceHandle> getSurfaces() const;

        /**
         * @brief Find a surface defined by this vertex
         * 
         * @param dir direction to look with respect to the vertex
         */
        SurfaceHandle findSurface(const FVector3 &dir) const;

        /**
         * @brief Find a body defined by this vertex
         * 
         * @param dir direction to look with respect to the vertex
         */
        BodyHandle findBody(const FVector3 &dir) const;

        /**
         * @brief Update internal neighbor vertex data
         */
        void updateNeighborVertices() const;

        /** 
         * @brief Get the neighbor vertices.
         * 
         * A vertex is a neighbor if it defines an edge with this vertex.
         */
        std::vector<VertexHandle> neighborVertices() const;

        /**
         * @brief Get the surfaces that this vertex and another vertex both define
         * 
         * @param other another vertex
         */
        std::vector<SurfaceHandle> sharedSurfaces(const VertexHandle &other) const;

        /**
         * @brief Get the current area
         */
        FloatP_t getArea() const;

        /** 
         * @brief Get the current volume 
         */
        FloatP_t getVolume() const;

        /** 
         * @brief Get the current mass 
         */
        FloatP_t getMass() const;

        /** 
         * @brief Update the properties of the underlying particle 
         */
        HRESULT updateProperties() const;

        /** 
         * @brief Get a handle to the underlying particle, if any 
         */
        ParticleHandle *particle() const;

        /** 
         * @brief Get the current position 
         */
        FVector3 getPosition() const;

        /**
         * @brief Set the current position
         * 
         * @param pos position
         * @param updateChildren flag indicating whether to update dependent objects
         */
        HRESULT setPosition(const FVector3 &pos, const bool &updateChildren=true);

        /** 
         * @brief Get the current velocity 
         */
        FVector3 getVelocity() const;

        /**
         * @brief Transfer all bonds to another vertex
         * 
         * @param other another vertex
         */
        HRESULT transferBondsTo(const VertexHandle &other) const;

        /**
         * @brief Replace a surface
         * 
         * @param toReplace surface to replace
         */
        HRESULT replace(SurfaceHandle &toReplace) const;

        /**
         * @brief Replace a body
         * 
         * @param toReplace body to replace
         */
        HRESULT replace(BodyHandle &toReplace);

        /**
         * @brief Merge with a vertex. 
         * 
         * The passed vertex is destroyed.
         * 
         * @param toRemove vertex to remove
         * @param lenCf distance coefficient in [0, 1] for where to place the vertex, from the kept vertex to the removed vertex
         */
        HRESULT merge(VertexHandle &toRemove, const FloatP_t &lenCf=0.5f);

        /**
         * @brief Inserts a vertex between two vertices
         * 
         * @param v1 first vertex
         * @param v2 second vertex
         */
        HRESULT insert(const VertexHandle &v1, const VertexHandle &v2);

        /**
         * @brief Insert a vertex between a vertex and each of a set of vertices
         * 
         * @param vf a vertex
         * @param nbs a set of vertices
         */
        HRESULT insert(const VertexHandle &vf, const std::vector<VertexHandle> &nbs);

        /**
         * @brief Calculate the topology of a vertex split without implementing the split
         * 
         * @param sep separation distance
         * @param verts_v vertices that continue to define existing surface
         * @param verts_new_v vertices that define a new surface
         */
        HRESULT splitPlan(const FVector3 &sep, const std::vector<VertexHandle> &verts_v, const std::vector<VertexHandle> &verts_new_v);

        /**
         * @brief Implement a pre-calculated vertex split, as determined by splitPlan
         * 
         * @param sep separation distance
         * @param verts_v vertices that continue to define existing surface
         * @param verts_new_v vertices that define a new surface
         */
        VertexHandle splitExecute(const FVector3 &sep, const std::vector<VertexHandle> &verts_v, const std::vector<VertexHandle> &verts_new_v);

        /**
         * @brief Split a vertex into an edge
         * 
         * The vertex must define at least one surface.
         * 
         * New topology is governed by a cut plane at the midpoint of, and orthogonal to, the new edge. 
         * Each first-order neighbor vertex is connected to the vertex of the new edge on the same side of 
         * the cut plane. 
         * 
         * @param sep separation distance
         * @return newly created vertex
         */
        VertexHandle split(const FVector3 &sep);

        operator bool() const { return id >= 0; }
    };


    inline bool operator< (const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return lhs.objectId() < rhs.objectId(); }
    inline bool operator> (const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return lhs.objectId() == rhs.objectId(); }
    inline bool operator!=(const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return !(lhs == rhs); }

    inline bool operator< (const TissueForge::models::vertex::VertexHandle& lhs, const TissueForge::models::vertex::VertexHandle& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::models::vertex::VertexHandle& lhs, const TissueForge::models::vertex::VertexHandle& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::VertexHandle& lhs, const TissueForge::models::vertex::VertexHandle& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::VertexHandle& lhs, const TissueForge::models::vertex::VertexHandle& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::VertexHandle& lhs, const TissueForge::models::vertex::VertexHandle& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::models::vertex::VertexHandle& lhs, const TissueForge::models::vertex::VertexHandle& rhs) { return !(lhs == rhs); }

}


inline std::ostream &operator<<(std::ostream& os, const TissueForge::models::vertex::Vertex &o)
{
    os << o.str().c_str();
    return os;
}

#endif // _MODELS_VERTEX_SOLVER_TFVERTEX_H_