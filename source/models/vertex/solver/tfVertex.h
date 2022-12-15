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
    class Body;
    class Structure;
    class Mesh;


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
    class CAPI_EXPORT Vertex : public MeshObj {

        /** Particle id. -1 if not assigned */
        int pid;

        /** Connected surfaces */
        std::vector<Surface*> surfaces;

    public:

        Vertex();
        Vertex(const unsigned int &_pid);
        Vertex(const FVector3 &position);
        Vertex(TissueForge::io::ThreeDFVertexData *vdata);

        /** Get the mesh object type */
        MeshObj::Type objType() const override { return MeshObj::Type::VERTEX; }

        /** Get the parents of the object */
        std::vector<MeshObj*> parents() const override { return std::vector<MeshObj*>(); }

        /** Get the children of the object */
        std::vector<MeshObj*> children() const override;

        /** Add a child object */
        HRESULT addChild(MeshObj *obj) override;

        /** Add a parent object */
        HRESULT addParent(MeshObj *obj) override { return E_FAIL; }

        /** Remove a child object */
        HRESULT removeChild(MeshObj *obj) override;

        /** Remove a parent object */
        HRESULT removeParent(MeshObj *obj) override { return E_FAIL; }

        /** Get a summary string */
        std::string str() const override;

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

        /**
         * Destroy the vertex. 
         * 
         * If the vertex is in a mesh, then it and any objects it defines are removed from the mesh. 
         * 
         * The underlying particle is also destroyed, if any. 
        */
        HRESULT destroy() override;

        /** Validate the vertex */
        bool validate() override { return true; }

        /** Get the id of the underlying particle */
        const int getPartId() const { return pid; }

        /** Get the structures defined by the vertex */
        std::vector<Structure*> getStructures() const;

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

        /** Update internal data due to a change in position */
        HRESULT positionChanged();

        /** Update the properties of the underlying particle */
        HRESULT updateProperties();

        /** Get a handle to the underlying particle, if any */
        ParticleHandle *particle() const;

        /** Get the current position */
        FVector3 getPosition() const;

        /** Set the current position */
        HRESULT setPosition(const FVector3 &pos);

        /** Transfer all bonds to another vertex */
        HRESULT transferBondsTo(Vertex *other);

        /** Replace a surface */
        HRESULT replace(Surface *toReplace);

        /** Create a vertex and replace a surface with it */
        static Vertex *replace(const FVector3 &position, Surface *toReplace);

        /** Replace a body */
        HRESULT replace(Body *toReplace);

        /** Create a vertex and replace a body with it */
        static Vertex *replace(const FVector3 &position, Body *toReplace);

        /** Merge with a vertex. The passed vertex is destroyed. */
        HRESULT merge(Vertex *toRemove, const FloatP_t &lenCf=0.5f);

        /** Inserts a vertex between two vertices */
        HRESULT insert(Vertex *v1, Vertex *v2);

        /** Create a vertex and inserts it between two vertices */
        static Vertex *insert(const FVector3 &position, Vertex *v1, Vertex *v2);

        /** Insert a vertex between a vertex and each of a set of vertices */
        HRESULT insert(Vertex *vf, std::vector<Vertex*> nbs);

        /** Create a vertex and insert it between a vertex and each of a set of vertices */
        static Vertex *insert(const FVector3 &position, Vertex *vf, std::vector<Vertex*> nbs);

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


        friend Surface;
        friend Body;
        friend Mesh;

    };

    inline bool operator< (const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return lhs.objId < rhs.objId; }
    inline bool operator> (const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return lhs.objId == rhs.objId; }
    inline bool operator!=(const TissueForge::models::vertex::Vertex& lhs, const TissueForge::models::vertex::Vertex& rhs) { return !(lhs == rhs); }

}


inline std::ostream &operator<<(std::ostream& os, const TissueForge::models::vertex::Vertex &o)
{
    os << o.str().c_str();
    return os;
}

#endif // _MODELS_VERTEX_SOLVER_TFVERTEX_H_