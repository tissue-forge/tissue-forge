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

#ifndef _MODELS_VERTEX_SOLVER_TFMESH_H_
#define _MODELS_VERTEX_SOLVER_TFMESH_H_

#define TFMESHINV_INCR 100

#include "tfVertex.h"
#include "tfSurface.h"
#include "tfBody.h"
#include "tfStructure.h"
#include "tfMeshQuality.h"

#include <mutex>
#include <set>
#include <vector>


namespace TissueForge::models::vertex {


    struct MeshRenderer;
    struct MeshSolver;


    class CAPI_EXPORT Mesh { 

        std::vector<Vertex*> vertices;
        std::vector<Surface*> surfaces;
        std::vector<Body*> bodies;
        std::vector<Structure*> structures;

        std::set<unsigned int> vertexIdsAvail, surfaceIdsAvail, bodyIdsAvail, structureIdsAvail;
        std::vector<Vertex*> verticesByPID;
        bool isDirty;
        MeshSolver *_solver = NULL;
        MeshQuality *_quality;
        std::mutex meshLock;

    public:

        Mesh();

        ~Mesh();

        /** Get the id of this mesh */
        const int getId() const;

        /** Get a JSON string representation */
        std::string toString();

        /** Test whether this mesh has a mesh quality instance */
        bool hasQuality() const { return _quality; }

        /** Get the mesh quality instance */
        TissueForge::models::vertex::MeshQuality &getQuality() const { return *_quality; }

        /** Set the mesh quality instance */
        HRESULT setQuality(TissueForge::models::vertex::MeshQuality *quality);

        /** Test whether a mesh quality instance is working on the mesh */
        bool qualityWorking() const { return hasQuality() && getQuality().working(); }

        /** Add a vertex */
        HRESULT add(Vertex *obj);

        /** Add a surface */
        HRESULT add(Surface *obj);

        /** Add a body */
        HRESULT add(Body *obj);

        /** Add a structure */
        HRESULT add(Structure *obj);

        /** Remove a mesh object */
        HRESULT removeObj(MeshObj *obj);

        /** Locks the mesh for thread-safe operations */
        void lock() { this->meshLock.lock(); }
        
        /** Unlocks the mesh for thread-safe operations */
        void unlock() { this->meshLock.unlock(); }

        /**
         * @brief Find a vertex in this mesh
         * 
         * @param pos position to look
         * @param tol distance tolerance
         * @return a vertex within the distance tolerance of the position, otherwise NULL
         */
        Vertex *findVertex(const FVector3 &pos, const FloatP_t &tol = 0.0001) const;

        /** Get the vertex for a given particle id */
        Vertex *getVertexByPID(const unsigned int &pid) const;

        /** Get the vertex at a location in the list of vertices */
        Vertex *getVertex(const unsigned int &idx) const;

        /** Get a surface at a location in the list of surfaces */
        Surface *getSurface(const unsigned int &idx) const;

        /** Get a body at a location in the list of bodies */
        Body *getBody(const unsigned int &idx) const;

        /** Get a structure at a location in the list of structures */
        Structure *getStructure(const unsigned int &idx) const;

        /** Get the number of vertices */
        unsigned int numVertices() const { return vertices.size() - vertexIdsAvail.size(); }

        /** Get the number of surfaces */
        unsigned int numSurfaces() const { return surfaces.size() - surfaceIdsAvail.size(); }

        /** Get the number of bodies */
        unsigned int numBodies() const { return bodies.size() - bodyIdsAvail.size(); }

        /** Get the number of structures */
        unsigned int numStructures() const { return structures.size() - structureIdsAvail.size(); }

        /** Get the size of the list of vertices */
        unsigned int sizeVertices() const { return vertices.size(); }

        /** Get the size of the list of surfaces */
        unsigned int sizeSurfaces() const { return surfaces.size(); }

        /** Get the size of the list of bodies */
        unsigned int sizeBodies() const { return bodies.size(); }

        /** Get the size of the list of structures */
        unsigned int sizeStructures() const { return structures.size(); }

        /** Validate state of the mesh */
        bool validate();

        /** Manually notify that the mesh has been changed */
        HRESULT makeDirty();

        /** Check whether two vertices are connected */
        bool connected(const Vertex *v1, const Vertex *v2) const;

        /** Check whether two surfaces are connected */
        bool connected(const Surface *s1, const Surface *s2) const;

        /** Check whether two bodies are connected */
        bool connected(const Body *b1, const Body *b2) const;

        // Mesh editing

        /** Remove a vertex from the mesh; all connected surfaces and bodies are also removed */
        HRESULT remove(Vertex *v);

        /** Remove a surface from the mesh; all connected bodies are also removed */
        HRESULT remove(Surface *s);

        /** Remove a body from the mesh */
        HRESULT remove(Body *b);

        /** Remove a structure from the mesh */
        HRESULT remove(Structure *s);

        friend MeshRenderer;
        friend MeshSolver;

    };

};

#endif // _MODELS_VERTEX_SOLVER_TFMESH_H_