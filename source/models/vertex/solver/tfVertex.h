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
        Vertex(io::ThreeDFVertexData *vdata);

        /** Get the mesh object type */
        MeshObj::Type objType() { return MeshObj::Type::VERTEX; }

        /** Get the parents of the object */
        std::vector<MeshObj*> parents() { return std::vector<MeshObj*>(); }

        /** Get the children of the object */
        std::vector<MeshObj*> children();

        /** Add a child object */
        HRESULT addChild(MeshObj *obj);

        /** Add a parent object */
        HRESULT addParent(MeshObj *obj) { return E_FAIL; }

        /** Remove a child object */
        HRESULT removeChild(MeshObj *obj);

        /** Remove a parent object */
        HRESULT removeParent(MeshObj *obj) { return E_FAIL; }

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
        HRESULT destroy();

        /** Validate the vertex */
        bool validate() { return true; }

        /** Get the structures defined by the vertex */
        std::vector<Structure*> getStructures();

        /** Get the bodies defined by the vertex */
        std::vector<Body*> getBodies();

        /** Get the surfaces defined by the vertex */
        std::vector<Surface*> getSurfaces() { return surfaces; }

        /**
         * @brief Find a surface defined by this vertex
         * 
         * @param dir direction to look with respect to the vertex
         */
        Surface *findSurface(const FVector3 &dir);

        /**
         * @brief Find a body defined by this vertex
         * 
         * @param dir direction to look with respect to the vertex
         */
        Body *findBody(const FVector3 &dir);

        /** Get the neighbor vertices.
         * 
         * A vertex is a neighbor if it defines an edge with this vertex.
         */
        std::vector<Vertex*> neighborVertices();

        /** Get the surfaces that this vertex and another vertex both define */
        std::vector<Surface*> sharedSurfaces(Vertex *other);

        /** Get the current volume */
        FloatP_t getVolume();

        /** Get the current mass */
        FloatP_t getMass();

        /** Update internal data due to a change in position */
        HRESULT positionChanged();

        /** Update the properties of the underlying particle */
        HRESULT updateProperties();

        /** Get a handle to the underlying particle, if any */
        ParticleHandle *particle();

        /** Get the current position */
        FVector3 getPosition();

        /** Set the current position */
        HRESULT setPosition(const FVector3 &pos);

        /** Transfer all bonds to another vertex */
        HRESULT transferBondsTo(Vertex *other);


        friend Surface;
        friend Body;
        friend Mesh;

    };

}

#endif // _MODELS_VERTEX_SOLVER_TFVERTEX_H_