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

        std::vector<Vertex*> vertices;

        FVector3 normal;

        FVector3 centroid;

        FVector3 velocity;

        FloatP_t area;

        /** Volume contributed by this surface to its child bodies */
        FloatP_t _volumeContr;

    public:

        /** Object actors */
        std::vector<MeshObjActor*> actors;

        /** Species on outward-facing side of the surface, if any */
        state::StateVector *species1;

        /** Species on inward-facing side of the surface, if any */
        state::StateVector *species2;

        /** Surface style, if any */
        rendering::Style *style;

        Surface();
        ~Surface();

        /** Construct a surface from a set of vertices */
        static SurfaceHandle create(const std::vector<VertexHandle> &_vertices);

        /** Construct a surface from a face */
        static SurfaceHandle create(TissueForge::io::ThreeDFFaceData *face);

        MESHBOJ_DEFINES_DECL(Body);
        MESHOBJ_DEFINEDBY_DECL(Vertex);
        MESHOBJ_CLASSDEF(MeshObjTypeLabel::SURFACE)

        /** Get a summary string */
        std::string str() const;

        /** Get a JSON string representation */
        std::string toString();

        /** Add a vertex */
        HRESULT add(Vertex *v);

        /** Insert a vertex at a location in the list of vertices */
        HRESULT insert(Vertex *v, const int &idx);

        /** Insert a vertex before another vertex */
        HRESULT insert(Vertex *v, Vertex *before);

        /** Insert a vertex between two vertices */
        HRESULT insert(Vertex *toInsert, Vertex *v1, Vertex *v2);

        /** Remove a vertex */
        HRESULT remove(Vertex *v);

        /** Replace a vertex at a location in the list of vertices */
        HRESULT replace(Vertex *toInsert, const int &idx);

        /** Replace a vertex with another vertex */
        HRESULT replace(Vertex *toInsert, Vertex *toRemove);

        /** Add a body */
        HRESULT add(Body *b);

        /** Remove a body */
        HRESULT remove(Body *b);

        /** Replace a body at a location in the list of bodies */
        HRESULT replace(Body *toInsert, const int &idx);

        /** Replace a body with another body */
        HRESULT replace(Body *toInsert, Body *toRemove);

        /**
         * Destroy a surface. 
         * 
         * Any resulting vertices without a surface are also destroyed. 
         */
        static HRESULT destroy(Surface *target);

        /**
         * Destroy a surface. 
         * 
         * Any resulting vertices without a surface are also destroyed. 
         */
        static HRESULT destroy(SurfaceHandle &target);

        /** Refresh internal ordering of defined bodies */
        HRESULT refreshBodies();

        /** Get the surface type */
        SurfaceType *type() const;

        /** Become a different type */
        HRESULT become(SurfaceType *stype);

        /** Get the bodies defined by the surface */
        std::vector<Body*> getBodies() const;

        /** Get the vertices that define the surface */
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

        /** Connected vertices on the same surface. */
        std::tuple<Vertex*, Vertex*> neighborVertices(const Vertex *v) const;

        /** Connected surfaces on the same body. */
        std::vector<Surface*> neighborSurfaces() const;

        /** Surfaces that share at least one vertex in a set of vertices. */
        std::vector<Surface*> connectedSurfaces(const std::vector<Vertex*> &verts) const;

        /** Surfaces that share at least one vertex. */
        std::vector<Surface*> connectedSurfaces() const;

        /** Get the integer labels of the contiguous edges that this surface shares with another surface */
        std::vector<unsigned int> contiguousEdgeLabels(const Surface *other) const;

        /** Get the number of contiguous edges that this surface shares with another surface */
        unsigned int numSharedContiguousEdges(const Surface *other) const;

        /** Get the surface normal */
        FVector3 getNormal() const { return normal; }

        /** Get the centroid */
        FVector3 getCentroid() const { return centroid; }

        /**
         * Get the velocity, calculated as the velocity of the centroid
        */
        FVector3 getVelocity() const { return velocity; }

        /** Get the area */
        FloatP_t getArea() const { return area; }

        /** Get the sign of the volume contribution to a body that this surface contributes */
        FloatP_t volumeSense(const Body *body) const;

        /** Get the volume that this surface contributes to a body */
        FloatP_t getVolumeContr(const Body *body) const { return _volumeContr * volumeSense(body); }

        /** Get the outward facing normal w.r.t. a body */
        FVector3 getOutwardNormal(const Body *body) const;

        /** Get the area that a vertex contributes to this surface */
        FloatP_t getVertexArea(const Vertex *v) const;

        /** Get the normal of a triangle */
        FVector3 triangleNormal(const unsigned int &idx) const;

        /** Get the normal distance to a point; negative distance means that the point is on the inner side */
        FloatP_t normalDistance(const FVector3 &pos) const;

        /** Test whether a point is on the outer side */
        bool isOutside(const FVector3 &pos) const;

        /** Sew two surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        static HRESULT sew(Surface *s1, Surface *s2, const FloatP_t &distCf=0.01);

        /** Sew two surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        static HRESULT sew(const SurfaceHandle &s1, const SurfaceHandle &s2, const FloatP_t &distCf=0.01);

        /** Sew a set of surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        static HRESULT sew(std::vector<Surface*> _surfaces, const FloatP_t &distCf=0.01);

        /** Sew a set of surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        static HRESULT sew(std::vector<SurfaceHandle> _surfaces, const FloatP_t &distCf=0.01);

        /** Merge with a surface. The passed surface is destroyed. 
         * 
         * Surfaces must have the same number of vertices. Vertices are paired by nearest distance.
        */
        HRESULT merge(Surface *toRemove, const std::vector<FloatP_t> &lenCfs);

        /** Create a surface from two vertices and a position */
        Surface *extend(const unsigned int &vertIdxStart, const FVector3 &pos);

        /** Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface
         * 
         * todo: add support for extruding at an angle w.r.t. the center of the edge and centroid of the base surface
        */
        Surface *extrude(const unsigned int &vertIdxStart, const FloatP_t &normLen);

        /** Split into two surfaces
         * 
         * Both vertices must already be in the surface and not adjacent
         * 
         * Vertices in the winding from from vertex to second go to newly created surface
         * 
         * Requires updated surface members (e.g., centroid)
        */
        Surface *split(Vertex *v1, Vertex *v2);

        /** Split into two surfaces
         * 
         * Requires updated surface members (e.g., centroid)
        */
        Surface *split(const FVector3 &cp_pos, const FVector3 &cp_norm);


        friend SurfaceHandle;
        friend Vertex;
        friend Body;
        friend BodyType;
        friend Mesh;

    };


    struct CAPI_EXPORT SurfaceHandle {

        int id;

        SurfaceHandle(const int &_id=-1);

        /** Get the underlying object, if any */
        Surface *surface() const;

        bool defines(const BodyHandle &b) const;

        bool definedBy(const VertexHandle &v) const;

        /** Get the mesh object type */
        MeshObjTypeLabel objType() const { return MeshObjTypeLabel::SURFACE; }

        /** Destroy the body. */
        HRESULT destroy();

        /** Validate the body */
        bool validate();

        /** Update internal data due to a change in position */
        HRESULT positionChanged();

        /** Get a summary string */
        std::string str() const;

        /** Get a JSON string representation */
        std::string toString();

        /** Create an instance from a JSON string representation */
        static SurfaceHandle fromString(const std::string &s);

        /** Add a vertex */
        HRESULT add(const VertexHandle &v);

        /** Insert a vertex at a location in the list of vertices */
        HRESULT insert(const VertexHandle &v, const int &idx);

        /** Insert a vertex before another vertex */
        HRESULT insert(const VertexHandle &v, const VertexHandle &before);

        /** Insert a vertex between two vertices */
        HRESULT insert(const VertexHandle &toInsert, const VertexHandle &v1, const VertexHandle &v2);

        /** Remove a vertex */
        HRESULT remove(const VertexHandle &v);

        /** Replace a vertex at a location in the list of vertices */
        HRESULT replace(const VertexHandle &toInsert, const int &idx);

        /** Replace a vertex with another vertex */
        HRESULT replace(const VertexHandle &toInsert, const VertexHandle &toRemove);

        /** Add a body */
        HRESULT add(const BodyHandle &b);

        /** Remove a body */
        HRESULT remove(const BodyHandle &b);

        /** Replace a body at a location in the list of bodies */
        HRESULT replace(const BodyHandle &toInsert, const int &idx);

        /** Replace a body with another body */
        HRESULT replace(const BodyHandle &toInsert, const BodyHandle &toRemove);

        /** Refresh internal ordering of defined bodies */
        HRESULT refreshBodies();

        /** Get the surface type */
        SurfaceType *type() const;

        /** Become a different type */
        HRESULT become(SurfaceType *stype);

        /** Get the bodies defined by the surface */
        std::vector<BodyHandle> getBodies() const;

        /** Get the vertices that define the surface */
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

        /** Connected vertices on the same surface. */
        std::tuple<VertexHandle, VertexHandle> neighborVertices(const VertexHandle &v) const;

        /** Connected surfaces on the same body. */
        std::vector<SurfaceHandle> neighborSurfaces() const;

        /** Surfaces that share at least one vertex in a set of vertices. */
        std::vector<SurfaceHandle> connectedSurfaces(const std::vector<VertexHandle> &verts) const;

        /** Surfaces that share at least one vertex. */
        std::vector<SurfaceHandle> connectedSurfaces() const;

        /** Get the integer labels of the contiguous edges that this surface shares with another surface */
        std::vector<unsigned int> contiguousEdgeLabels(const SurfaceHandle &other) const;

        /** Get the number of contiguous edges that this surface shares with another surface */
        unsigned int numSharedContiguousEdges(const SurfaceHandle &other) const;

        /** Get the surface normal */
        FVector3 getNormal() const;

        /** Get the centroid */
        FVector3 getCentroid() const;

        /**
         * Get the velocity, calculated as the velocity of the centroid
        */
        FVector3 getVelocity() const;

        /** Get the area */
        FloatP_t getArea() const;

        /** Get the sign of the volume contribution to a body that this surface contributes */
        FloatP_t volumeSense(const BodyHandle &body) const;

        /** Get the volume that this surface contributes to a body */
        FloatP_t getVolumeContr(const BodyHandle &body) const;

        /** Get the outward facing normal w.r.t. a body */
        FVector3 getOutwardNormal(const BodyHandle &body) const;

        /** Get the area that a vertex contributes to this surface */
        FloatP_t getVertexArea(const VertexHandle &v) const;

        /** Get the species on outward-facing side of the surface, if any */
        state::StateVector *getSpeciesOutward() const;

        /** Set the species on outward-facing side of the surface */
        HRESULT setSpeciesOutward(state::StateVector *s) const;

        /** Get the species on inward-facing side of the surface, if any */
        state::StateVector *getSpeciesInward() const;

        /** Set the species on inward-facing side of the surface */
        HRESULT setSpeciesInward(state::StateVector *s) const;

        /** Get the surface style, if any */
        rendering::Style *getStyle() const;

        /** Set the surface style */
        HRESULT setStyle(rendering::Style *s) const;

        /** Get the normal of a triangle */
        FVector3 triangleNormal(const unsigned int &idx) const;

        /** Get the normal distance to a point; negative distance means that the point is on the inner side */
        FloatP_t normalDistance(const FVector3 &pos) const;

        /** Test whether a point is on the outer side */
        bool isOutside(const FVector3 &pos) const;

        /** Merge with a surface. The passed surface is destroyed. 
         * 
         * Surfaces must have the same number of vertices. Vertices are paired by nearest distance.
        */
        HRESULT merge(SurfaceHandle &toRemove, const std::vector<FloatP_t> &lenCfs);

        /** Create a surface from two vertices and a position */
        SurfaceHandle extend(const unsigned int &vertIdxStart, const FVector3 &pos);

        /** Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface
         * 
         * todo: add support for extruding at an angle w.r.t. the center of the edge and centroid of the base surface
        */
        SurfaceHandle extrude(const unsigned int &vertIdxStart, const FloatP_t &normLen);

        /** Split into two surfaces
         * 
         * Both vertices must already be in the surface and not adjacent
         * 
         * Vertices in the winding from from vertex to second go to newly created surface
         * 
         * Requires updated surface members (e.g., centroid)
        */
        SurfaceHandle split(const VertexHandle &v1, const VertexHandle &v2);

        /** Split into two surfaces
         * 
         * Requires updated surface members (e.g., centroid)
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

        /**
         * @brief Construct a new surface type
         * 
         * @param flatLam parameter for flat surface constraint
         * @param convexLam parameter for convex surface constraint
         */
        SurfaceType(const FloatP_t &flatLam, const FloatP_t &convexLam, const bool &noReg=false);

        /** Construct a new surface type */
        SurfaceType(const bool &noReg=false) : SurfaceType(0.1, 0.1, noReg) {};

        /** Get the mesh object type */
        MeshObjTypeLabel objType() const override { return MeshObjTypeLabel::SURFACE; }

        /** Get a summary string */
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

        /** Get a registered type by name */
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

        /** Add an instance */
        HRESULT add(const SurfaceHandle &i);

        /** Remove an instance */
        HRESULT remove(const SurfaceHandle &i);

        /** list of instances that belong to this type */    
        std::vector<SurfaceHandle> getInstances();

        /** list of instances ids that belong to this type */
        std::vector<int> getInstanceIds() { return _instanceIds; }

        /** number of instances that belong to this type */
        unsigned int getNumInstances();

        /** Construct a surface of this type from a set of vertices */
        SurfaceHandle operator() (const std::vector<VertexHandle> &_vertices);

        /** Construct a surface of this type from a set of positions */
        SurfaceHandle operator() (const std::vector<FVector3> &_positions);

        /** Construct a surface of this type from a face */
        SurfaceHandle operator() (TissueForge::io::ThreeDFFaceData *face);

        /** Construct a polygon with n vertices circumscribed on a circle */
        SurfaceHandle nPolygon(const unsigned int &n, const FVector3 &center, const FloatP_t &radius, const FVector3 &ax1, const FVector3 &ax2);

        /** Replace a vertex with a surface. Vertices are created for the surface along every destroyed edge. */
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