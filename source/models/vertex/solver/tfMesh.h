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
        bool isDirty;
        MeshSolver *_solver = NULL;
        MeshQuality *_quality;
        std::mutex meshLock;

    public:

        Mesh();

        ~Mesh();

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

        /** Inserts a vertex between two vertices */
        HRESULT insert(Vertex *toInsert, Vertex *v1, Vertex *v2);

        /** Insert a vertex between a vertex and each of a set of vertices */
        HRESULT insert(Vertex *toInsert, Vertex *vf, std::vector<Vertex*> nbs);
        
        /** Replace a surface with a vertex */
        HRESULT replace(Vertex *toInsert, Surface *toReplace);

        /** Replace a body with a vertex */
        HRESULT replace(Vertex *toInsert, Body *toReplace);

        /** Replace a vertex with a surface. Vertices are created for the surface along every destroyed edge. */
        Surface *replace(SurfaceType *toInsert, Vertex *toReplace, std::vector<FloatP_t> lenCfs);

        /** Merge two vertices. */
        HRESULT merge(Vertex *toKeep, Vertex *toRemove, const FloatP_t &lenCf=0.5f);

        /** Merge two surfaces. 
         * 
         * Surfaces must have the same number of vertices. Vertices are paired by nearest distance.
        */
        HRESULT merge(Surface *toKeep, Surface *toRemove, const std::vector<FloatP_t> &lenCfs);

        /** Create a surface from two vertices of a surface in the mesh and a position */
        Surface *extend(Surface *base, const unsigned int &vertIdxStart, const FVector3 &pos);

        /** Create a body from a surface in the mesh and a position */
        Body *extend(Surface *base, BodyType *btype, const FVector3 &pos);

        /** Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface
         * 
         * todo: add support for extruding at an angle w.r.t. the center of the edge and centroid of the base surface
        */
        Surface *extrude(Surface *base, const unsigned int &vertIdxStart, const FloatP_t &normLen);

        /** Create a body from a surface in a mesh by extruding along the outward-facing normal of the surface
         * 
         * todo: add support for extruding at an angle
        */
        Body *extrude(Surface *base, BodyType *btype, const FloatP_t &normLen);

        /** Sew two surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        HRESULT sew(Surface *s1, Surface *s2, const FloatP_t &distCf=0.01);

        /** Sew a set of surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        HRESULT sew(std::vector<Surface*> _surfaces, const FloatP_t &distCf=0.01);

        /** Calculate the topology of a vertex split without implementing the split */
        HRESULT splitPlan(Vertex *v, const FVector3 &sep, std::vector<Vertex*> &verts_v, std::vector<Vertex*> &verts_new_v);

        /** Implement a pre-calculated vertex split, as determined by splitPlan */
        Vertex *splitExecute(Vertex *v, const FVector3 &sep, const std::vector<Vertex*> &verts_v, const std::vector<Vertex*> &verts_new_v);

        /** Split a vertex into an edge
         * 
         * The vertex must define at least one surface.
         * 
         * New topology is governed by a cut plane at the midpoint of, and orthogonal to, the new edge. 
         * Each first-order neighbor vertex is connected to the vertex of the new edge on the same side of 
         * the cut plane. 
         */
        Vertex *split(Vertex *v, const FVector3 &sep);

        /** Split a surface into two surfaces
         * 
         * Both vertices must already be in the surface and not adjacent
         * 
         * Vertices in the winding from from vertex to second go to newly created surface
         * 
         * Requires updated surface members (e.g., centroid)
        */
        Surface *split(Surface *s, Vertex *v1, Vertex *v2);

        /** Split a surface into two surfaces
         * 
         * Requires updated surface members (e.g., centroid)
        */
        Surface *split(Surface *s, const FVector3 &cp_pos, const FVector3 &cp_norm);

        /** Split a body into two bodies */
        Body *split(Body *b, const FVector3 &cp_pos, const FVector3 &cp_norm, SurfaceType *stype=NULL);

        friend MeshRenderer;
        friend MeshSolver;

    };

};

#endif // _MODELS_VERTEX_SOLVER_TFMESH_H_