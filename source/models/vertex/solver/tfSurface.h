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
    class Body;
    class Structure;
    class Mesh;

    struct SurfaceType;


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
    class CAPI_EXPORT Surface : public MeshObj {

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

        /** Id of the type*/
        unsigned int typeId;

        /** Species on outward-facing side of the surface, if any */
        state::StateVector *species1;

        /** Species on inward-facing side of the surface, if any */
        state::StateVector *species2;

        /** Surface style, if any */
        rendering::Style *style;

        Surface();

        /** Construct a surface from a set of vertices */
        Surface(std::vector<Vertex*> _vertices);

        /** Construct a surface from a face */
        Surface(io::ThreeDFFaceData *face);

        /** Get the mesh object type */
        MeshObj::Type objType() const override { return MeshObj::Type::SURFACE; }

        /** Get the parents of the object */
        std::vector<MeshObj*> parents() const override { return TissueForge::models::vertex::vectorToBase(vertices); }

        /** Get the children of the object */
        std::vector<MeshObj*> children() const override;

        /** Add a child object */
        HRESULT addChild(MeshObj *obj) override;

        /** Add a parent object */
        HRESULT addParent(MeshObj *obj) override;

        /** Remove a child object */
        HRESULT removeChild(MeshObj *obj) override;

        /** Remove a parent object */
        HRESULT removeParent(MeshObj *obj) override;

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
         * Destroy the surface. 
         * 
         * If the surface is in a mesh, then it and any objects it defines are removed from the mesh. 
        */
        HRESULT destroy() override;

        /** Validate the surface */
        bool validate() override;

        /** Refresh internal ordering of defined bodies */
        HRESULT refreshBodies();

        /** Get the surface type */
        SurfaceType *type() const;

        /** Become a different type */
        HRESULT become(SurfaceType *stype);

        /** Get the structures defined by the surface */
        std::vector<Structure*> getStructures() const;

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

        /** Update internal data due to a change in position */
        HRESULT positionChanged();

        /** Sew two surfaces 
         * 
         * All vertices are merged that are a distance apart less than a distance criterion. 
         * 
         * The distance criterion is the square root of the average of the two surface areas, multiplied by a coefficient. 
        */
        static HRESULT sew(Surface *s1, Surface *s2, const FloatP_t &distCf=0.01);


        friend Body;
        friend Mesh;

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
        MeshObj::Type objType() const override { return MeshObj::Type::SURFACE; }

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

        /** Construct a surface of this type from a set of vertices */
        Surface *operator() (std::vector<Vertex*> _vertices);

        /** Construct a surface of this type from a set of positions */
        Surface *operator() (const std::vector<FVector3> &_positions);

        /** Construct a surface of this type from a face */
        Surface *operator() (io::ThreeDFFaceData *face);

        /** Construct a polygon with n vertices circumscribed on a circle */
        Surface *nPolygon(const unsigned int &n, const FVector3 &center, const FloatP_t &radius, const FVector3 &ax1, const FVector3 &ax2);

    };

}

#endif // _MODELS_VERTEX_SOLVER_TFSURFACE_H_