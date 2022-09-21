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

        bool hasQuality() { return _quality; }
        TissueForge::models::vertex::MeshQuality &getQuality() { return *_quality; }
        HRESULT setQuality(TissueForge::models::vertex::MeshQuality *quality);

        HRESULT add(Vertex *obj);
        HRESULT add(Surface *obj);
        HRESULT add(Body *obj);
        HRESULT add(Structure *obj);
        HRESULT removeObj(MeshObj *obj);

        /** Locks the mesh for thread-safe operations */
        void lock() { this->meshLock.lock(); }
        
        /** Unlocks the mesh for thread-safe operations */
        void unlock() { this->meshLock.unlock(); }

        Vertex *findVertex(const FVector3 &pos, const float &tol = 0.0001);

        Vertex *getVertex(const unsigned int &idx);
        Surface *getSurface(const unsigned int &idx);
        Body *getBody(const unsigned int &idx);
        Structure *getStructure(const unsigned int &idx);

        unsigned int numVertices() const { return vertices.size() - vertexIdsAvail.size(); }
        unsigned int numSurfaces() const { return surfaces.size() - surfaceIdsAvail.size(); }
        unsigned int numBodies() const { return bodies.size() - bodyIdsAvail.size(); }
        unsigned int numStructures() const { return structures.size() - structureIdsAvail.size(); }

        unsigned int sizeVertices() const { return vertices.size(); }
        unsigned int sizeSurfaces() const { return surfaces.size(); }
        unsigned int sizeBodies() const { return bodies.size(); }
        unsigned int sizeStructures() const { return structures.size(); }

        /** Validate state of the mesh */
        bool validate();

        /** Manually notify that the mesh has been changed */
        HRESULT makeDirty();

        /** Check whether two vertices are connected */
        bool connected(Vertex *v1, Vertex *v2);

        /** Check whether two surfaces are connected */
        bool connected(Surface *s1, Surface *s2);

        /** Check whether two bodies are connected */
        bool connected(Body *b1, Body *b2);

        // Mesh editing

        /** Remove a vertex from the mesh; all connected surfaces and bodies are also removed */
        HRESULT remove(Vertex *v);

        /** Remove a surface from the mesh; all connected bodies are also removed */
        HRESULT remove(Surface *s);

        /** Remove a body from the mesh */
        HRESULT remove(Body *b);

        /** Inserts a vertex between two vertices */
        HRESULT insert(Vertex *toInsert, Vertex *v1, Vertex *v2);

        /** Insert a vertex between a vertex and each of a set of vertices */
        HRESULT insert(Vertex *toInsert, Vertex *vf, std::vector<Vertex*> nbs);
        
        /** Replace a surface with a vertex */
        HRESULT replace(Vertex *toInsert, Surface *toReplace);

        /** Replace a vertex with a surface. Vertices are created for the surface along every destroyed edge. */
        Surface *replace(SurfaceType *toInsert, Vertex *toReplace, std::vector<float> lenCfs);

        /** Merge two vertices. */
        HRESULT merge(Vertex *toKeep, Vertex *toRemove, const float &lenCf=0.5f);

        /** Merge two surfaces. 
         * 
         * Surfaces must have the same number of vertices. Vertices are paired by nearest distance.
        */
        HRESULT merge(Surface *toKeep, Surface *toRemove, const std::vector<float> &lenCfs);

        /** Create a surface from two vertices of a surface in the mesh and a position */
        Surface *extend(Surface *base, const unsigned int &vertIdxStart, const FVector3 &pos);

        /** Create a body from a surface in the mesh and a position */
        Body *extend(Surface *base, BodyType *btype, const FVector3 &pos);

        /** Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface
         * 
         * todo: add support for extruding at an angle w.r.t. the center of the edge and centroid of the base surface
        */
        Surface *extrude(Surface *base, const unsigned int &vertIdxStart, const float &normLen);

        /** Create a body from a surface in a mesh by extruding along the outward-facing normal of the surface
         * 
         * todo: add support for extruding at an angle
        */
        Body *extrude(Surface *base, BodyType *btype, const float &normLen);

        /** Sew two surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        HRESULT sew(Surface *s1, Surface *s2, const float &distCf=0.01);

        /** Sew a set of surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        HRESULT sew(std::vector<Surface*> _surfaces, const float &distCf=0.01);

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

        friend MeshRenderer;
        friend MeshSolver;

    };

};

#endif // _MODELS_VERTEX_SOLVER_TFMESH_H_