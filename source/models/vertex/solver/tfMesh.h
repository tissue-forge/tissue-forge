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
 * @file tfMesh.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFMESH_H_
#define _MODELS_VERTEX_SOLVER_TFMESH_H_

#define TFMESHINV_INCR 100

#include "tfVertex.h"
#include "tfSurface.h"
#include "tfBody.h"
#include "tfMeshQuality.h"

#include <mutex>
#include <set>
#include <vector>
#include <unordered_map>


namespace TissueForge::models::vertex {


    struct MeshRenderer;
    struct MeshSolver;


    /**
     * @brief Contains all @ref Vertex, @ref Surface and @ref Body instances
     * 
     */
    class CAPI_EXPORT Mesh { 

        std::vector<Vertex> *vertices;
        size_t nr_vertices;
        std::vector<Surface> *surfaces;
        size_t nr_surfaces;
        std::vector<Body> *bodies;
        size_t nr_bodies;

        std::set<unsigned int> vertexIdsAvail, surfaceIdsAvail, bodyIdsAvail;
        std::unordered_map<int, Vertex*> verticesByPID;
        bool isDirty;
        MeshSolver *_solver = NULL;
        MeshQuality *_quality;
        std::mutex meshLock;

        HRESULT incrementVertices(const size_t &numIncr=TFMESHINV_INCR);
        HRESULT incrementSurfaces(const size_t &numIncr=TFMESHINV_INCR);
        HRESULT incrementBodies(const size_t &numIncr=TFMESHINV_INCR);
        HRESULT allocateVertex(Vertex **obj);
        HRESULT allocateSurface(Surface **obj);
        HRESULT allocateBody(Body **obj);

    public:

        Mesh();

        ~Mesh();

        /**
         * @brief Get a summary string
         */
        std::string str() const;

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Test whether this mesh has a mesh quality instance
         * 
         * @return true if this mesh has a mesh quality instance
         */
        bool hasQuality() const { return _quality; }

        /**
         * @brief Get the mesh quality instance
         */
        TissueForge::models::vertex::MeshQuality &getQuality() const { return *_quality; }

        /**
         * @brief Set the mesh quality instance
         * 
         * @param quality the mesh quality instance
         */
        HRESULT setQuality(TissueForge::models::vertex::MeshQuality *quality);

        /**
         * @brief Test whether a mesh quality instance is working on the mesh
         * 
         * @return true if a mesh quality instance is working on the mesh
         */
        bool qualityWorking() const { return hasQuality() && getQuality().working(); }

        /**
         * @brief Ensure that there are a given number of allocated vertices
         * 
         * @param numAlloc a given number of allocated vertices
         */
        HRESULT ensureAvailableVertices(const size_t &numAlloc);

        /**
         * @brief Ensure that there are a given number of allocated surfaces
         * 
         * @param numAlloc a given number of allocated surfaces
         */
        HRESULT ensureAvailableSurfaces(const size_t &numAlloc);

        /**
         * @brief Ensure that there are a given number of allocated bodies
         * 
         * @param numAlloc a given number of allocated bodies
         */
        HRESULT ensureAvailableBodies(const size_t &numAlloc);

        /**
         * @brief Create a vertex
         * 
         * @param obj a vertex to populate
         * @param pid the id of the underlying particle
         */
        HRESULT create(Vertex **obj, const unsigned int &pid);

        /**
         * @brief Create a surface
         * 
         * @param obj a surface to populate
         */
        HRESULT create(Surface **obj);

        /**
         * @brief Create a body
         * 
         * @param obj a body to populate
         */
        HRESULT create(Body **obj);

        /**
         * @brief Get the mesh
         */
        static Mesh *get();

        /**
         * @brief Locks the mesh for thread-safe operations
         */
        void lock() { this->meshLock.lock(); }
        
        /**
         * @brief Unlocks the mesh for thread-safe operations
         */
        void unlock() { this->meshLock.unlock(); }

        /**
         * @brief Find a vertex in this mesh
         * 
         * @param pos position to look
         * @param tol distance tolerance
         * @return a vertex within the distance tolerance of the position, otherwise NULL
         */
        Vertex *findVertex(const FVector3 &pos, const FloatP_t &tol = 0.0001);

        /**
         * @brief Get the vertex for a given particle id
         * 
         * @param pid particle id
         */
        Vertex *getVertexByPID(const unsigned int &pid) const;

        /**
         * @brief Get the vertex at a location in the list of vertices
         * 
         * @param idx location in the list
         */
        Vertex *getVertex(const unsigned int &idx);

        /**
         * @brief Get the surface at a location in the list of surfaces
         * 
         * @param idx location in the list
         */
        Surface *getSurface(const unsigned int &idx);

        /**
         * @brief Get the body at a location in the list of bodies
         * 
         * @param idx location in the list
         */
        Body *getBody(const unsigned int &idx);

        /**
         * @brief Get the number of vertices
         */
        unsigned int numVertices() const { return nr_vertices; }

        /**
         * @brief Get the number of surfaces
         */
        unsigned int numSurfaces() const { return nr_surfaces; }

        /**
         * @brief Get the number of bodies
         */
        unsigned int numBodies() const { return nr_bodies; }

        /**
         * @brief Get the size of the list of vertices
         */
        unsigned int sizeVertices() const { return vertices->size(); }

        /**
         * @brief Get the size of the list of surfaces
         */
        unsigned int sizeSurfaces() const { return surfaces->size(); }

        /**
         * @brief Get the size of the list of bodies
         */
        unsigned int sizeBodies() const { return bodies->size(); }

        /**
         * @brief Validate state of the mesh
         * 
         * @return true if in a valid state
         */
        bool validate();

        /**
         * @brief Manually notify that the mesh has been changed
         */
        HRESULT makeDirty();

        /**
         * @brief Check whether two vertices are connected
         * 
         * @param v1 first vertex
         * @param v2 second vertex
         * @return true if the two vertices are connected
         */
        bool connected(const Vertex *v1, const Vertex *v2) const;

        /**
         * @brief Check whether two surfaces are connected
         * 
         * @param v1 first surface
         * @param v2 second surface
         * @return true if the two surfaces are connected
         */
        bool connected(const Surface *s1, const Surface *s2) const;

        /**
         * @brief Check whether two bodies are connected
         * 
         * @param v1 first body
         * @param v2 second body
         * @return true if the two bodies are connected
         */
        bool connected(const Body *b1, const Body *b2) const;

        /**
         * @brief Remove a vertex from the mesh; all dependent surfaces and bodies are also removed
         * 
         * @param v a vertex
         */
        HRESULT remove(Vertex *v);

        /**
         * @brief Remove a surface from the mesh; all dependent bodies are also removed
         * 
         * @param s a surface
         */
        HRESULT remove(Surface *s);

        /**
         * @brief Remove a body from the mesh
         * 
         * @param b a body
         */
        HRESULT remove(Body *b);

        friend MeshRenderer;
        friend MeshSolver;

    };

};

#endif // _MODELS_VERTEX_SOLVER_TFMESH_H_