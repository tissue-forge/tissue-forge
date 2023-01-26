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
 * @file tfSurface.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFSURFACE_H_
#define _MODELS_VERTEX_SOLVER_TFSURFACE_H_

#include <tf_port.h>

#include <state/tfStateVector.h>
#include <rendering/tfStyle.h>

#include "tf_mesh.h"

#include <io/tfThreeDFFaceData.h>

#include <tuple>


namespace TissueForge::models::vertex { 


    class Vertex;
    struct VertexHandle;
    class Body;
    struct BodyHandle;
    class Mesh;

    struct SurfaceHandle;
    struct SurfaceType;
    struct BodyType;


    /**
     * @brief The mesh surface is an area-enclosed object of implicit mesh edges defined by mesh vertices. 
     * 
     * The mesh surface consists of at least three mesh vertices. 
     * 
     * The mesh surface is always flat. 
     * 
     * The mesh surface can have a state vector, which represents a uniform amount of substance 
     * attached to the surface. 
     * 
     */
    class CAPI_EXPORT Surface {

        /** Object id; unique by type in a mesh */
        int _objId;

    public:

        /** Id of the type*/
        int typeId;

    private:

        /** Connected body, if any, where the surface normal is outward-facing */
        Body *b1;
        
        /** Connected body, if any, where the surface normal is inward-facing */
        Body *b2;

        /** Vertices that define the surface */
        std::vector<Vertex*> vertices;

        /** Surface normal */
        FVector3 normal;

        /** Surface centroid */
        FVector3 centroid;

        /** Surface velocity */
        FVector3 velocity;

        /** Surface area */
        FloatP_t area;

        /** Surface perimeter*/
        FloatP_t perimeter;

        /** Volume contributed by this surface to its child bodies */
        FloatP_t _volumeContr;

        /** Mass density; only used in 2D simulation */
        FloatP_t density;

    public:

        /** Object actors */
        std::vector<MeshObjActor*> actors;

        /** Species on outward-facing side of the surface, if any; not currently supported */
        state::StateVector *species1;

        /** Species on inward-facing side of the surface, if any; not currently supported */
        state::StateVector *species2;

        /** Surface style, if any */
        rendering::Style *style;

        Surface();
        ~Surface();

        /**
         * @brief Construct a surface from a set of vertices
         * 
         * @param _vertices a set of vertices
         */
        static SurfaceHandle create(const std::vector<VertexHandle> &_vertices);

        /**
         * @brief Construct a surface from a face
         * 
         * @param face a face
         */
        static SurfaceHandle create(TissueForge::io::ThreeDFFaceData *face);

        MESHBOJ_DEFINES_DECL(Body);
        MESHOBJ_DEFINEDBY_DECL(Vertex);
        MESHOBJ_CLASSDEF(MeshObjTypeLabel::SURFACE)

        /**
         * @brief Get a summary string
         */
        std::string str() const;

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Add a vertex
         * 
         * @param v vertex to add
         */
        HRESULT add(Vertex *v);

        /**
         * @brief Insert a vertex at a location in the list of vertices
         * 
         * @param v vertex to insert
         * @param idx location
         */
        HRESULT insert(Vertex *v, const int &idx);

        /**
         * @brief Insert a vertex before another vertex
         * 
         * @param v vertex to insert
         * @param before vertex to insert before
         */
        HRESULT insert(Vertex *v, Vertex *before);

        /**
         * @brief Insert a vertex between two vertices
         * 
         * @param toInsert vertex to insert
         * @param v1 first vertex
         * @param v2 second vertex
         */
        HRESULT insert(Vertex *toInsert, Vertex *v1, Vertex *v2);

        /**
         * @brief Remove a vertex
         * 
         * @param v vertex to remove
         */
        HRESULT remove(Vertex *v);

        /**
         * @brief Replace a vertex at a location in the list of vertices
         * 
         * @param toInsert vertex to insert
         * @param idx location of vertex to replace
         */
        HRESULT replace(Vertex *toInsert, const int &idx);

        /**
         * @brief Replace a vertex with another vertex
         * 
         * @param toInsert vertex to insert
         * @param toRemove vertex to remove
         */
        HRESULT replace(Vertex *toInsert, Vertex *toRemove);

        /**
         * @brief Add a body
         * 
         * @param b body to add
         */
        HRESULT add(Body *b);

        /**
         * @brief Remove a body
         * 
         * @param b body to remove
         */
        HRESULT remove(Body *b);

        /**
         * @brief Replace a body at a location in the list of bodies
         * 
         * @param toInsert body to insert
         * @param idx location of body to remve
         */
        HRESULT replace(Body *toInsert, const int &idx);

        /**
         * @brief Replace a body with another body
         * 
         * @param toInsert body to insert
         * @param toRemove body to remove
         */
        HRESULT replace(Body *toInsert, Body *toRemove);

        /**
         * @brief Destroy a surface. 
         * 
         * Any resulting vertices without a surface are also destroyed. 
         * 
         * @param target surface to destroy
         */
        static HRESULT destroy(Surface *target);

        /**
         * @brief Destroy a surface. 
         * 
         * Any resulting vertices without a surface are also destroyed. 
         * 
         * @param target handle to the surface to destroy
         */
        static HRESULT destroy(SurfaceHandle &target);

        /**
         * @brief Refresh internal ordering of defined bodies
         */
        HRESULT refreshBodies();

        /**
         * @brief Get the surface type
         */
        SurfaceType *type() const;

        /**
         * @brief Become a different type
         * 
         * @param stype type to become
         */
        HRESULT become(SurfaceType *stype);

        /**
         * @brief Get the bodies defined by the surface
         */
        std::vector<Body*> getBodies() const;

        /**
         * @brief Get the vertices that define the surface
         */
        std::vector<Vertex*> getVertices() const { return vertices; }

        /**
         * @brief Find a vertex that defines this surface
         * 
         * @param dir direction to look with respect to the centroid
         */
        Vertex *findVertex(const FVector3 &dir) const;

        /**
         * @brief Find a body that this surface defines
         * 
         * @param dir direction to look with respect to the centroid
         */
        Body *findBody(const FVector3 &dir) const;

        /**
         * @brief Connected vertices on the same surface
         * 
         * @param v a vertex
         */
        std::tuple<Vertex*, Vertex*> neighborVertices(const Vertex *v) const;

        /**
         * @brief Connected surfaces on the same body
         */
        std::vector<Surface*> neighborSurfaces() const;

        /**
         * @brief Surfaces that share at least one vertex in a set of vertices
         * 
         * @param verts vertices
         */
        std::vector<Surface*> connectedSurfaces(const std::vector<Vertex*> &verts) const;

        /**
         * @brief Surfaces that share at least one vertex
         */
        std::vector<Surface*> connectedSurfaces() const;

        /**
         * @brief Vertices defining this and another surface
         * 
         * @param other another surface
         */
        std::vector<Vertex*> connectingVertices(const Surface *other) const;

        /**
         * @brief Get the integer labels of the contiguous vertices that this surface shares with another surface
         * 
         * @param other another surface
         */
        std::vector<unsigned int> contiguousVertexLabels(const Surface *other) const;

        /**
         * @brief Get the number of contiguous vertex sets that this surface shares with another surface
         * 
         * @param other another surface
         */
        unsigned int numSharedContiguousVertexSets(const Surface *other) const;

        /**
         * @brief Get the vertices of a contiguous shared vertex set with another surface. 
         * 
         * Vertices are labeled in increasing order starting with "1". A requested set that does not exist returns empty. 
         * 
         * A requested edge with label "0" returns all vertices not shared with another surface
         * 
         * @param other another surface
         * @param edgeLabel edge label
         */
        std::vector<Vertex*> sharedContiguousVertices(const Surface *other, const unsigned int &edgeLabel) const;

        /**
         * @brief Get the surface normal
         */
        FVector3 getNormal() const;

        /**
         * @brief Get the surface unnormalized normal
         */
        FVector3 getUnnormalizedNormal() const { return normal; }

        /**
         * @brief Get the centroid
         */
        FVector3 getCentroid() const { return centroid; }

        /**
         * @brief Get the velocity, calculated as the velocity of the centroid
         */
        FVector3 getVelocity() const { return velocity; }

        /**
         * @brief Get the area
         */
        FloatP_t getArea() const { return area; }

        /**
         * @brief Get the perimeter
         */
        FloatP_t getPerimeter() const { return perimeter; }

        /**
         * @brief Get the mass density; only used in 2D simulation
         */
        FloatP_t getDensity() const { return density; }

        /**
         * @brief Set the mass density; only used in 2D simulation
         * 
         * @param _density density
         */
        void setDensity(const FloatP_t &_density) { density = _density; }

        /**
         * @brief Get the mass; only used in 2D simulation
         */
        FloatP_t getMass() const { return area * density; }

        /**
         * @brief Get the sign of the volume contribution to a body that this surface contributes
         * 
         * @param body a body
         */
        FloatP_t volumeSense(const Body *body) const;

        /**
         * @brief Get the volume that this surface contributes to a body
         * 
         * @param body a body
         */
        FloatP_t getVolumeContr(const Body *body) const { return _volumeContr * volumeSense(body); }

        /**
         * @brief Get the outward facing normal w.r.t. a body
         * 
         * @param body a body
         */
        FVector3 getOutwardNormal(const Body *body) const;

        /**
         * @brief Get the area that a vertex contributes to this surface
         * 
         * @param v a vertex
         */
        FloatP_t getVertexArea(const Vertex *v) const;

        /**
         * @brief Get the mass contribution of a vertex to this surface; only used in 2D simulation
         * 
         * @param v a vertex
         */
        FloatP_t getVertexMass(const Vertex *v) const { return getVertexArea(v) * density; }

        /**
         * @brief Get the normal of a triangle
         * 
         * @param idx location of first vertex
         */
        FVector3 triangleNormal(const unsigned int &idx) const;

        /**
         * @brief Get the normal distance to a point. 
         * 
         * A negative distance means that the point is on the inner side
         * 
         * @param pos position
         */
        FloatP_t normalDistance(const FVector3 &pos) const;

        /**
         * @brief Test whether a point is on the outer side
         * 
         * @param pos position
         * @return true if the point is on the outer side
         */
        bool isOutside(const FVector3 &pos) const;

        /**
         * @brief Sew two surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
         * 
         * @param s1 first surface
         * @param s2 second surface
         * @param distCf distance criterion coefficient
         */
        static HRESULT sew(Surface *s1, Surface *s2, const FloatP_t &distCf=0.01);

        /**
         * @brief Sew two surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
         * 
         * @param s1 first surface
         * @param s2 second surface
         * @param distCf distance criterion coefficient
         */
        static HRESULT sew(const SurfaceHandle &s1, const SurfaceHandle &s2, const FloatP_t &distCf=0.01);

        /**
         * @brief Sew a set of surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
         * 
         * @param _surfaces a set of surfaces 
         * @param distCf distance criterion coefficient
         */
        static HRESULT sew(std::vector<Surface*> _surfaces, const FloatP_t &distCf=0.01);

        /**
         * @brief Sew a set of surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
         * 
         * @param _surfaces a set of surfaces
         * @param distCf distance criterion coefficient
         * @return HRESULT 
         */
        static HRESULT sew(std::vector<SurfaceHandle> _surfaces, const FloatP_t &distCf=0.01);

        /**
         * @brief Merge with a surface. The passed surface is destroyed. 
         * 
         * Surfaces must have the same number of vertices. Vertices are paired by nearest distance.
         * 
         * @param toRemove surface to remove
         * @param lenCfs distance coefficients in [0, 1] for where to place the merged vertex, from each kept vertex to each removed vertex
         */
        HRESULT merge(Surface *toRemove, const std::vector<FloatP_t> &lenCfs);

        /**
         * @brief Create a surface from two vertices and a position
         * 
         * @param vertIdxStart index of first vertex
         * @param pos position
         */
        Surface *extend(const unsigned int &vertIdxStart, const FVector3 &pos);

        /**
         * @brief Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface
         * 
         * todo: add support for extruding at an angle w.r.t. the center of the edge and centroid of the base surface
         * 
         * @param vertIdxStart index of first vertex
         * @param normLen length along surface normal by which to extrude
         */
        Surface *extrude(const unsigned int &vertIdxStart, const FloatP_t &normLen);

        /**
         * @brief Split into two surfaces
         * 
         * Both vertices must already be in the surface and not adjacent
         * 
         * Vertices in the winding from from vertex to second go to newly created surface
         * 
         * Requires updated surface members (e.g., centroid)
         * 
         * @param v1 fist vertex defining the split
         * @param v2 second vertex defining the split
         * @return Surface* 
         */
        Surface *split(Vertex *v1, Vertex *v2);

        /**
         * @brief Split into two surfaces
         * 
         * Requires updated surface members (e.g., centroid)
         * 
         * @param cp_pos point on the cut plane
         * @param cp_norm normal of the cut plane
         */
        Surface *split(const FVector3 &cp_pos, const FVector3 &cp_norm);


        friend SurfaceHandle;
        friend Vertex;
        friend Body;
        friend BodyType;
        friend Mesh;

    };


    /**
     * @brief A handle to a @ref Surface. 
     *
     * The engine allocates @ref Surface memory in blocks, and @ref Surface
     * values get moved around all the time, so their addresses change.
     * 
     * This is a safe way to work with a @ref Surface.
     */
    struct CAPI_EXPORT SurfaceHandle {

        int id;

        SurfaceHandle(const int &_id=-1);

        /**
         * @brief Get the underlying object, if any
         */
        Surface *surface() const;

        /**
         * @brief Test whether defines a body
         * 
         * @param b a body
         * @return true if defines a body
         */
        bool defines(const BodyHandle &b) const;

        /**
         * @brief Test whether defined by a vertex
         * 
         * @param v a vertex
         * @return true if defined by a vertex
         */
        bool definedBy(const VertexHandle &v) const;

        /**
         * @brief Get the mesh object type
         */
        MeshObjTypeLabel objType() const { return MeshObjTypeLabel::SURFACE; }

        /**
         * @brief Destroy the surface
         */
        HRESULT destroy();

        /**
         * @brief Validate the surface
         * 
         * @return true if valid
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
         * @param s JSON string
         */
        static SurfaceHandle fromString(const std::string &s);

        /**
         * @brief Add a vertex
         * 
         * @param v vertex to add
         */
        HRESULT add(const VertexHandle &v);

        /**
         * @brief Insert a vertex at a location in the list of vertices
         * 
         * @param v vertex to insert
         * @param idx location of insertion
         */
        HRESULT insert(const VertexHandle &v, const int &idx);

        /**
         * @brief Insert a vertex before another vertex
         * 
         * @param v vertex to insert
         * @param before vertex to insert before
         */
        HRESULT insert(const VertexHandle &v, const VertexHandle &before);

        /**
         * @brief Insert a vertex between two vertices
         * 
         * @param toInsert vertex to insert
         * @param v1 first vertex
         * @param v2 second vertex
         */
        HRESULT insert(const VertexHandle &toInsert, const VertexHandle &v1, const VertexHandle &v2);

        /**
         * @brief Remove a vertex
         * 
         * @param v vertex to remove
         */
        HRESULT remove(const VertexHandle &v);

        /**
         * @brief Replace a vertex at a location in the list of vertices
         * 
         * @param toInsert vertex to insert
         * @param idx location of vertex to remove
         */
        HRESULT replace(const VertexHandle &toInsert, const int &idx);

        /**
         * @brief Replace a vertex with another vertex
         * 
         * @param toInsert vertex to insert
         * @param toRemove vertex to remove
         */
        HRESULT replace(const VertexHandle &toInsert, const VertexHandle &toRemove);

        /**
         * @brief Add a body
         * 
         * @param b body to add
         */
        HRESULT add(const BodyHandle &b);

        /**
         * @brief Remove a body
         * 
         * @param b body to remove
         */
        HRESULT remove(const BodyHandle &b);

        /**
         * @brief Replace a body at a location in the list of bodies
         * 
         * @param toInsert body to insert
         * @param idx location of body to remove
         */
        HRESULT replace(const BodyHandle &toInsert, const int &idx);

        /**
         * @brief Replace a body with another body
         * 
         * @param toInsert body to insert
         * @param toRemove body to remove
         */
        HRESULT replace(const BodyHandle &toInsert, const BodyHandle &toRemove);

        /**
         * @brief Refresh internal ordering of defined bodies
         */
        HRESULT refreshBodies();

        /**
         * @brief Get the surface type
         */
        SurfaceType *type() const;

        /**
         * @brief Become a different type
         * 
         * @param stype type to become
         */
        HRESULT become(SurfaceType *stype);

        /**
         * @brief Get the bodies defined by the surface
         */
        std::vector<BodyHandle> getBodies() const;

        /**
         * @brief Get the vertices that define the surface
         */
        std::vector<VertexHandle> getVertices() const;

        /**
         * @brief Find a vertex that defines this surface
         * 
         * @param dir direction to look with respect to the centroid
         */
        VertexHandle findVertex(const FVector3 &dir) const;

        /**
         * @brief Find a body that this surface defines
         * 
         * @param dir direction to look with respect to the centroid
         */
        BodyHandle findBody(const FVector3 &dir) const;

        /**
         * @brief Connected vertices on the same surface
         * 
         * @param v a vertex
         */
        std::tuple<VertexHandle, VertexHandle> neighborVertices(const VertexHandle &v) const;

        /**
         * @brief Connected surfaces on the same body
         */
        std::vector<SurfaceHandle> neighborSurfaces() const;

        /**
         * @brief Surfaces that share at least one vertex in a set of vertices
         * 
         * @param verts a set of vertices
         */
        std::vector<SurfaceHandle> connectedSurfaces(const std::vector<VertexHandle> &verts) const;

        /**
         * @brief Surfaces that share at least one vertex
         */
        std::vector<SurfaceHandle> connectedSurfaces() const;

        /**
         * @brief Vertices defining this and another surface
         * 
         * @param other another surface
         */
        std::vector<VertexHandle> connectingVertices(const SurfaceHandle &other) const;

        /**
         * @brief Get the integer labels of the contiguous edges that this surface shares with another surface
         * 
         * @param other another surface
         */
        std::vector<unsigned int> contiguousVertexLabels(const SurfaceHandle &other) const;

        /**
         * @brief Get the number of contiguous edges that this surface shares with another surface
         * 
         * @param other another surface
         */
        unsigned int numSharedContiguousVertexSets(const SurfaceHandle &other) const;

        /**
         * @brief Get the vertices of a contiguous shared edge with another surface. 
         * 
         * Edges are labeled in increasing order starting with "1". A requested edge that does not exist returns empty. 
         * 
         * A requested edge with label "0" returns all vertices not shared with another surface
         * 
         * @param other another surface
         * @param edgeLabel edge label
         */
        std::vector<VertexHandle> sharedContiguousVertices(const SurfaceHandle &other, const unsigned int &edgeLabel) const;

        /**
         * @brief Get the normal
         */
        FVector3 getNormal() const;

        /**
         * @brief Get the surface unnormalized normal
         */
        FVector3 getUnnormalizedNormal() const;

        /**
         * @brief Get the centroid
         */
        FVector3 getCentroid() const;

        /**
         * @brief Get the velocity, calculated as the velocity of the centroid
         */
        FVector3 getVelocity() const;

        /** 
         * @brief Get the area 
         */
        FloatP_t getArea() const;

        /**
         * @brief Get the perimeter
         */
        FloatP_t getPerimeter() const;

        /**
         * @brief Get the mass density; only used in 2D simulation
         */
        FloatP_t getDensity() const;

        /**
         * @brief Set the mass density; only used in 2D simulation
         * 
         * @param _density density
         */
        void setDensity(const FloatP_t &_density) const;

        /**
         * @brief Get the mass; only used in 2D simulation
         */
        FloatP_t getMass() const;

        /** 
         * @brief Get the sign of the volume contribution to a body that this surface contributes 
         * 
         * @param body a body
         */
        FloatP_t volumeSense(const BodyHandle &body) const;

        /** 
         * @brief Get the volume that this surface contributes to a body 
         * 
         * @param body a body
         */
        FloatP_t getVolumeContr(const BodyHandle &body) const;

        /** 
         * @brief Get the outward facing normal w.r.t. a body 
         * 
         * @param body a body
         */
        FVector3 getOutwardNormal(const BodyHandle &body) const;

        /**
         * @brief Get the area that a vertex contributes to this surface
         * 
         * @param v a vertex
         */
        FloatP_t getVertexArea(const VertexHandle &v) const;

        /**
         * @brief Get the mass contribution of a vertex to this surface; only used in 2D simulation
         * 
         * @param v a vertex
         */
        FloatP_t getVertexMass(const VertexHandle &v) const;

        /** 
         * @brief Get the species on outward-facing side of the surface, if any 
         */
        state::StateVector *getSpeciesOutward() const;

        /** 
         * @brief Set the species on outward-facing side of the surface 
         * 
         * @param s species
         */
        HRESULT setSpeciesOutward(state::StateVector *s) const;

        /** 
         * @brief Get the species on inward-facing side of the surface, if any 
         */
        state::StateVector *getSpeciesInward() const;

        /** 
         * @brief Set the species on inward-facing side of the surface 
         * 
         * @param s species
         */
        HRESULT setSpeciesInward(state::StateVector *s) const;

        /** 
         * @brief Get the surface style, if any 
         */
        rendering::Style *getStyle() const;

        /** 
         * @brief Set the surface style 
         * 
         * @param s style
         */
        HRESULT setStyle(rendering::Style *s) const;

        /** 
         * @brief Get the normal of a triangle 
         * 
         * @param idx index of first triangle vertex
         */
        FVector3 triangleNormal(const unsigned int &idx) const;

        /** 
         * @brief Get the normal distance to a point.
         * 
         * A negative distance means that the point is on the inner side 
         * 
         * @param pos position
         */
        FloatP_t normalDistance(const FVector3 &pos) const;

        /**
         * @brief Test whether a point is on the outer side
         * 
         * @param pos position
         * @return true if on the outer side
         */
        bool isOutside(const FVector3 &pos) const;

        /**
         * @brief Merge with a surface. The passed surface is destroyed. 
         * 
         * Surfaces must have the same number of vertices. Vertices are paired by nearest distance.
         * 
         * @param toRemove surface to remove 
         * @param lenCfs distance coefficients in [0, 1] for where to place the merged vertex, from each kept vertex to each removed vertex
         */
        HRESULT merge(SurfaceHandle &toRemove, const std::vector<FloatP_t> &lenCfs);

        /**
         * @brief Create a surface from two vertices and a position
         * 
         * @param vertIdxStart index of first vertex
         * @param pos position
         */
        SurfaceHandle extend(const unsigned int &vertIdxStart, const FVector3 &pos);

        /**
         * @brief Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface
         * 
         * todo: add support for extruding at an angle w.r.t. the center of the edge and centroid of the base surface
         * 
         * @param vertIdxStart index of first vertex
         * @param normLen length along surface normal by which to extrude
         */
        SurfaceHandle extrude(const unsigned int &vertIdxStart, const FloatP_t &normLen);

        /**
         * @brief Split into two surfaces
         * 
         * Both vertices must already be in the surface and not adjacent
         * 
         * Vertices in the winding from from vertex to second go to newly created surface
         * 
         * Requires updated surface members (e.g., centroid)
         * 
         * @param v1 first vertex
         * @param v2 second vertex
         */
        SurfaceHandle split(const VertexHandle &v1, const VertexHandle &v2);

        /**
         * @brief Split into two surfaces
         * 
         * Requires updated surface members (e.g., centroid)
         * 
         * @param cp_pos point on the cut plane
         * @param cp_norm normal of the cut plane
         */
        SurfaceHandle split(const FVector3 &cp_pos, const FVector3 &cp_norm);

        operator bool() const { return id >= 0; }
    };


    /**
     * @brief Mesh surface type. 
     * 
     * Can be used as a factory to create mesh surface instances with 
     * processes and properties that correspond to the type. 
     */
    struct CAPI_EXPORT SurfaceType : MeshObjType {

        /** Name of this surface type */
        std::string name;

        /** The style of the surface type */
        rendering::Style *style;

        /** Mass density; only used in 2D simulation */
        FloatP_t density;

        /**
         * @brief Construct a new surface type
         * 
         * @param flatLam parameter for flat surface constraint
         * @param convexLam parameter for convex surface constraint
         */
        SurfaceType(const FloatP_t &flatLam, const FloatP_t &convexLam, const bool &noReg=false);

        /** Construct a new surface type */
        SurfaceType(const bool &noReg=false) : SurfaceType(0.1, 0.1, noReg) {};

        /**
         * @brief Get the mesh object type
         */
        MeshObjTypeLabel objType() const override { return MeshObjTypeLabel::SURFACE; }

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
        static SurfaceType *fromString(const std::string &str);

        /**
         * @brief Get a registered type by name
         * 
         * @param _name type name
         */
        static SurfaceType *findFromName(const std::string &_name);

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
        virtual SurfaceType *get();

        /**
         * @brief Add an instance
         * 
         * @param i instance to add
         */
        HRESULT add(const SurfaceHandle &i);

        /**
         * @brief Remove an instance
         * 
         * @param i instance to remove
         */
        HRESULT remove(const SurfaceHandle &i);

        /**
         * @brief list of instances that belong to this type
         */
        std::vector<SurfaceHandle> getInstances();

        /**
         * @brief list of instances ids that belong to this type
         */
        std::vector<int> getInstanceIds() { return _instanceIds; }

        /**
         * @brief number of instances that belong to this type
         */
        unsigned int getNumInstances();

        /**
         * @brief Construct a surface of this type from a set of vertices
         * 
         * @param _vertices a set of vertices
         */
        SurfaceHandle operator() (const std::vector<VertexHandle> &_vertices);

        /**
         * @brief Construct a surface of this type from a set of positions
         * 
         * @param _positions a set of positions
         */
        SurfaceHandle operator() (const std::vector<FVector3> &_positions);

        /**
         * @brief Construct a surface of this type from a face
         * 
         * @param face a face
         */
        SurfaceHandle operator() (TissueForge::io::ThreeDFFaceData *face);

        /**
         * @brief Construct a polygon with n vertices circumscribed on a circle
         * 
         * @param n number of vertices
         * @param center center of circle
         * @param radius radius of circle
         * @param ax1 first axis defining the orientation of the circle
         * @param ax2 second axis defining the orientation of the circle
         */
        SurfaceHandle nPolygon(const unsigned int &n, const FVector3 &center, const FloatP_t &radius, const FVector3 &ax1, const FVector3 &ax2);

        /**
         * @brief Replace a vertex with a surface. 
         * 
         * Vertices are created for the surface along every destroyed edge.
         * 
         * @param toReplace vertex to replace
         * @param lenCfs distance coefficients in [0, 1] defining where to create a new vertex along each edge
         */
        SurfaceHandle replace(VertexHandle &toReplace, std::vector<FloatP_t> lenCfs);

    private:

        std::vector<int> _instanceIds;

    };

    inline bool operator< (const TissueForge::models::vertex::Surface& lhs, const TissueForge::models::vertex::Surface& rhs) { return lhs.objectId() < rhs.objectId(); }
    inline bool operator> (const TissueForge::models::vertex::Surface& lhs, const TissueForge::models::vertex::Surface& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::Surface& lhs, const TissueForge::models::vertex::Surface& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::Surface& lhs, const TissueForge::models::vertex::Surface& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::Surface& lhs, const TissueForge::models::vertex::Surface& rhs) { return lhs.objectId() == rhs.objectId(); }
    inline bool operator!=(const TissueForge::models::vertex::Surface& lhs, const TissueForge::models::vertex::Surface& rhs) { return !(lhs == rhs); }

    inline bool operator< (const TissueForge::models::vertex::SurfaceHandle& lhs, const TissueForge::models::vertex::SurfaceHandle& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::models::vertex::SurfaceHandle& lhs, const TissueForge::models::vertex::SurfaceHandle& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::SurfaceHandle& lhs, const TissueForge::models::vertex::SurfaceHandle& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::SurfaceHandle& lhs, const TissueForge::models::vertex::SurfaceHandle& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::SurfaceHandle& lhs, const TissueForge::models::vertex::SurfaceHandle& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::models::vertex::SurfaceHandle& lhs, const TissueForge::models::vertex::SurfaceHandle& rhs) { return !(lhs == rhs); }

    inline bool operator< (const TissueForge::models::vertex::SurfaceType& lhs, const TissueForge::models::vertex::SurfaceType& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::models::vertex::SurfaceType& lhs, const TissueForge::models::vertex::SurfaceType& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::models::vertex::SurfaceType& lhs, const TissueForge::models::vertex::SurfaceType& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::models::vertex::SurfaceType& lhs, const TissueForge::models::vertex::SurfaceType& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::models::vertex::SurfaceType& lhs, const TissueForge::models::vertex::SurfaceType& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::models::vertex::SurfaceType& lhs, const TissueForge::models::vertex::SurfaceType& rhs) { return !(lhs == rhs); }

}


inline std::ostream &operator<<(std::ostream& os, const TissueForge::models::vertex::Surface &o)
{
    os << o.str().c_str();
    return os;
}

inline std::ostream &operator<<(std::ostream& os, const TissueForge::models::vertex::SurfaceType &o)
{
    os << o.str().c_str();
    return os;
}

#endif // _MODELS_VERTEX_SOLVER_TFSURFACE_H_