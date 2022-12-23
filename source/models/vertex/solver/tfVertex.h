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

    public:

        Vertex();
        ~Vertex();
        static VertexHandle create(const unsigned int &_pid);
        static VertexHandle create(const FVector3 &position);
        static VertexHandle create(TissueForge::io::ThreeDFVertexData *vdata);

        MESHBOJ_DEFINES_DECL(Surface);
        MESHBOJ_DEFINES_DECL(Body);
        MESHOBJ_CLASSDEF(MeshObjTypeLabel::VERTEX)

        /** Get a summary string */
        std::string str() const;

        /** Get a JSON string representation */
        std::string toString();

        /** Add a surface */
        HRESULT add(Surface *s);

        /** Insert a surface at a location in the list of surfaces */
        HRESULT insert(Surface *s, const int &idx);

        /** Insert a surface before another surface */
        HRESULT insert(Surface *s, Surface *before);

        /** Remove a surface */
        HRESULT remove(Surface *s);

        /** Replace a surface at a location in the list of surfaces */
        HRESULT replace(Surface *toInsert, const int &idx);

        /** Replace a surface with another surface */
        HRESULT replace(Surface *toInsert, Surface *toRemove);

        /** Get the id of the underlying particle */
        const int getPartId() const { return pid; }

        /** Get the bodies defined by the vertex */
        std::vector<Body*> getBodies() const;

        /** Get the surfaces defined by the vertex */
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

        /** Get the neighbor vertices.
         * 
         * A vertex is a neighbor if it defines an edge with this vertex.
         */
        std::vector<Vertex*> neighborVertices() const;

        /** Get the surfaces that this vertex and another vertex both define */
        std::vector<Surface*> sharedSurfaces(const Vertex *other) const;

        /** Get the current volume */
        FloatP_t getVolume() const;

        /** Get the current mass */
        FloatP_t getMass() const;

        /** Update the properties of the underlying particle */
        HRESULT updateProperties();

        /** Get a handle to the underlying particle, if any */
        ParticleHandle *particle() const;

        /** Get the current position */
        FVector3 getPosition() const { return _particlePosition; }

        /** Set the current position */
        HRESULT setPosition(const FVector3 &pos, const bool &updateChildren=true);

        /** Get the current velocity */
        FVector3 getVelocity() const { return _particleVelocity; }

        FloatP_t getCachedParticleMass() const { return _particleMass; }

        /** Transfer all bonds to another vertex */
        HRESULT transferBondsTo(Vertex *other);

        /** Replace a surface */
        HRESULT replace(Surface *toReplace);

        /** Create a vertex and replace a surface with it */
        static Vertex *replace(const FVector3 &position, Surface *toReplace);

        /** Create a vertex and replace a surface with it */
        static VertexHandle replace(const FVector3 &position, SurfaceHandle &toReplace);

        /** Replace a body */
        HRESULT replace(Body *toReplace);

        /** Create a vertex and replace a body with it */
        static Vertex *replace(const FVector3 &position, Body *toReplace);

        /** Create a vertex and replace a body with it */
        static VertexHandle replace(const FVector3 &position, BodyHandle &toReplace);

        /** Merge with a vertex. The passed vertex is destroyed. */
        HRESULT merge(Vertex *toRemove, const FloatP_t &lenCf=0.5f);

        /** Inserts a vertex between two vertices */
        HRESULT insert(Vertex *v1, Vertex *v2);

        /** Create a vertex and inserts it between two vertices */
        static Vertex *insert(const FVector3 &position, Vertex *v1, Vertex *v2);

        /** Create a vertex and inserts it between two vertices */
        static VertexHandle insert(const FVector3 &position, const VertexHandle &v1, const VertexHandle &v2);

        /** Insert a vertex between a vertex and each of a set of vertices */
        HRESULT insert(Vertex *vf, std::vector<Vertex*> nbs);

        /** Create a vertex and insert it between a vertex and each of a set of vertices */
        static Vertex *insert(const FVector3 &position, Vertex *vf, std::vector<Vertex*> nbs);

        /** Create a vertex and insert it between a vertex and each of a set of vertices */
        static VertexHandle insert(const FVector3 &position, const VertexHandle &vf, const std::vector<VertexHandle> &nbs);

        /** Calculate the topology of a vertex split without implementing the split */
        HRESULT splitPlan(const FVector3 &sep, std::vector<Vertex*> &verts_v, std::vector<Vertex*> &verts_new_v);

        /** Implement a pre-calculated vertex split, as determined by splitPlan */
        Vertex *splitExecute(const FVector3 &sep, const std::vector<Vertex*> &verts_v, const std::vector<Vertex*> &verts_new_v);

        /** Split a vertex into an edge
         * 
         * The vertex must define at least one surface.
         * 
         * New topology is governed by a cut plane at the midpoint of, and orthogonal to, the new edge. 
         * Each first-order neighbor vertex is connected to the vertex of the new edge on the same side of 
         * the cut plane. 
         */
        Vertex *split(const FVector3 &sep);


        friend VertexHandle;
        friend Surface;
        friend Body;
        friend Mesh;

    };


    struct CAPI_EXPORT VertexHandle {

        int id;

        VertexHandle(const int &_id=-1);

        /** Get the underlying object, if any */
        Vertex *vertex() const;

        bool defines(const SurfaceHandle &s) const;

        bool defines(const BodyHandle &b) const;

        /** Get the mesh object type */
        MeshObjTypeLabel objType() const { return MeshObjTypeLabel::VERTEX; }

        /** Destroy the body. */
        HRESULT destroy();

        /** Validate the body */
        bool validate();

        /** Update internal data due to a change in position */
        HRESULT positionChanged();

        /** Get a summary string */
        std::string str() const;

        /** Get a JSON string representation */
        std::string toString() const;

        /** Create an instance from a JSON string representation */
        static VertexHandle fromString(const std::string &s);

        /** Add a surface */
        HRESULT add(const SurfaceHandle &s) const;

        /** Insert a surface at a location in the list of surfaces */
        HRESULT insert(const SurfaceHandle &s, const int &idx) const;

        /** Insert a surface before another surface */
        HRESULT insert(const SurfaceHandle &s, const SurfaceHandle &before) const;

        /** Remove a surface */
        HRESULT remove(const SurfaceHandle &s) const;

        /** Replace a surface at a location in the list of surfaces */
        HRESULT replace(const SurfaceHandle &toInsert, const int &idx) const;

        /** Replace a surface with another surface */
        HRESULT replace(const SurfaceHandle &toInsert, const SurfaceHandle &toRemove) const;

        /** Get the id of the underlying particle */
        const int getPartId() const;

        /** Get the bodies defined by the vertex */
        std::vector<BodyHandle> getBodies() const;

        /** Get the surfaces defined by the vertex */
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

        /** Get the neighbor vertices.
         * 
         * A vertex is a neighbor if it defines an edge with this vertex.
         */
        std::vector<VertexHandle> neighborVertices() const;

        /** Get the surfaces that this vertex and another vertex both define */
        std::vector<SurfaceHandle> sharedSurfaces(const VertexHandle &other) const;

        /** Get the current volume */
        FloatP_t getVolume() const;

        /** Get the current mass */
        FloatP_t getMass() const;

        /** Update the properties of the underlying particle */
        HRESULT updateProperties() const;

        /** Get a handle to the underlying particle, if any */
        ParticleHandle *particle() const;

        /** Get the current position */
        FVector3 getPosition() const;

        /** Set the current position */
        HRESULT setPosition(const FVector3 &pos, const bool &updateChildren=true);

        /** Get the current velocity */
        FVector3 getVelocity() const;

        /** Transfer all bonds to another vertex */
        HRESULT transferBondsTo(const VertexHandle &other) const;

        /** Replace a surface */
        HRESULT replace(SurfaceHandle &toReplace) const;

        /** Replace a body */
        HRESULT replace(BodyHandle &toReplace);

        /** Merge with a vertex. The passed vertex is destroyed. */
        HRESULT merge(VertexHandle &toRemove, const FloatP_t &lenCf=0.5f);

        /** Inserts a vertex between two vertices */
        HRESULT insert(const VertexHandle &v1, const VertexHandle &v2);

        /** Insert a vertex between a vertex and each of a set of vertices */
        HRESULT insert(const VertexHandle &vf, const std::vector<VertexHandle> &nbs);

        /** Calculate the topology of a vertex split without implementing the split */
        HRESULT splitPlan(const FVector3 &sep, const std::vector<VertexHandle> &verts_v, const std::vector<VertexHandle> &verts_new_v);

        /** Implement a pre-calculated vertex split, as determined by splitPlan */
        VertexHandle splitExecute(const FVector3 &sep, const std::vector<VertexHandle> &verts_v, const std::vector<VertexHandle> &verts_new_v);

        /** Split a vertex into an edge
         * 
         * The vertex must define at least one surface.
         * 
         * New topology is governed by a cut plane at the midpoint of, and orthogonal to, the new edge. 
         * Each first-order neighbor vertex is connected to the vertex of the new edge on the same side of 
         * the cut plane. 
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