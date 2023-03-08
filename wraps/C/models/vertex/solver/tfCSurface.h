/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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
 * @file tfCSurface.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCSURFACE_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCSURFACE_H_

#include <tf_port_c.h>

#include <tfC_io.h>
#include <tfCStyle.h>


/**
 * @brief Surface type style definition in Tissue Forge C
 * 
 */
struct CAPI_EXPORT tfVertexSolverSurfaceTypeStyleSpec {
    char *color;
    unsigned int visible;
};

/**
 * @brief Surface type definition in Tissue Forge C
 * 
 */
struct CAPI_EXPORT tfVertexSolverSurfaceTypeSpec {
    tfFloatP_t *edgeTensionLam;
    unsigned int edgeTensionOrder;
    tfFloatP_t *normalStressMag;
    tfFloatP_t *surfaceAreaLam;
    tfFloatP_t *surfaceAreaVal;
    tfFloatP_t **surfaceTractionComps;
    char *name;
    struct tfVertexSolverSurfaceTypeStyleSpec *style;
    char **adhesionNames;
    tfFloatP_t *adhesionValues;
    unsigned int numAdhesionValues;
};


// Handles

/**
 * @brief Handle to a @ref models::vertex::SurfaceHandle instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverSurfaceHandleHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref models::vertex::SurfaceType instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverSurfaceTypeHandle {
    void *tfObj;
};


///////////////////////////////////
// tfVertexSolverSurfaceTypeSpec //
///////////////////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct tfVertexSolverSurfaceTypeSpec) tfVertexSolverSurfaceTypeSpec_init();


////////////////////////////////////////
// tfVertexSolverSurfaceTypeStyleSpec //
////////////////////////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct tfVertexSolverSurfaceTypeStyleSpec) tfVertexSolverSurfaceTypeStyleSpec_init();


///////////////////
// SurfaceHandle //
///////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param id object id
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_init(struct tfVertexSolverSurfaceHandleHandle *handle, int id);

/**
 * @brief Create an instance from a JSON string representation
 * 
 * @param handle handle to populate
 * @param s JSON string
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_fromString(struct tfVertexSolverSurfaceHandleHandle *handle, const char *s);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_destroy(struct tfVertexSolverSurfaceHandleHandle *handle);

/**
 * @brief Get the id
 * 
 * @param handle populated handle
 * @param objId object id
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getId(struct tfVertexSolverSurfaceHandleHandle *handle, int *objId);

/**
 * @brief Test whether a surface defines a body
 * 
 * @param handle populated handle
 * @param b body
 * @param result result of test
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_definesBody(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *b, 
    bool *result
);

/**
 * @brief Test whether a surface is defined by a vertex
 * 
 * @param handle populated handle
 * @param v vertex
 * @param result result of test
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_definedByVertex(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    bool *result
);

/**
 * @brief Get the mesh object type
 * 
 * @param handle populated handle
 * @param label type label
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_objType(struct tfVertexSolverSurfaceHandleHandle *handle, unsigned int *label);

/**
 * @brief Destroy the surface
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_destroySurface(struct tfVertexSolverSurfaceHandleHandle *handle);

/**
 * @brief Destroy an instance. 
 * 
 * Any resulting vertices without a surface are also destroyed. 
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_destroySurfaceC(struct tfVertexSolverSurfaceHandleHandle *handle);

/**
 * @brief Validate the body
 * 
 * @param handle populated handle
 * @param result result of validation
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_validate(struct tfVertexSolverSurfaceHandleHandle *handle, bool *result);

/**
 * @brief Update internal data due to a change in position
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_positionChanged(struct tfVertexSolverSurfaceHandleHandle *handle);

/**
 * @brief Get a summary string
 * 
 * @param handle populated handle
 * @param str summary string
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_str(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str JSON string
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_toString(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Add a vertex
 * 
 * @param handle populated handle
 * @param v vertex to add
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_addVertex(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverVertexHandleHandle *v);

/**
 * @brief Insert a vertex at a location in the list of vertices
 * 
 * @param handle populated handle
 * @param v vertex to insert
 * @param idx location for insertion
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_insertVertexAt(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    const int &idx
);

/**
 * @brief Insert a vertex before another vertex
 * 
 * @param handle populated handle
 * @param v vertex to insert
 * @param before vertex to insert before
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_insertVertexBefore(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    struct tfVertexSolverVertexHandleHandle *before
);

/**
 * @brief Insert a vertex between two vertices
 * 
 * @param handle populated handle
 * @param toInsert vertex to insert
 * @param v1 first vertex
 * @param v2 second vertex
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_insertVertexBetween(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toInsert, 
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2
);

/**
 * @brief Remove a vertex
 * 
 * @param handle populated handle
 * @param v vertex to remove
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_removeVertex(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverVertexHandleHandle *v);

/**
 * @brief Replace a vertex at a location in the list of vertices
 * 
 * @param handle populated handle
 * @param toInsert vertex to insert
 * @param idx location of vertex to replace
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_replaceVertexAt(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toInsert, 
    const int &idx
);

/**
 * @brief Replace a vertex with another vertex
 * 
 * @param handle populated handle
 * @param toInsert vertex to insert
 * @param toRemove vertex to replace
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_replaceVertexWith(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toInsert, 
    struct tfVertexSolverVertexHandleHandle *toRemove
);

/**
 * @brief Add a body
 * 
 * @param handle populated handle
 * @param b body to add
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_addBody(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverBodyHandleHandle *b);

/**
 * @brief Remove a body
 * 
 * @param handle populated handle
 * @param b body to remove
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_removeBody(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverBodyHandleHandle *b);

/**
 * @brief Replace a body at a location in the list of bodies
 * 
 * @param handle populated handle
 * @param toInsert body to insert
 * @param idx location of body to replace
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_replaceBodyAt(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *toInsert, 
    const int &idx
);

/**
 * @brief Replace a body with another body
 * 
 * @param handle populated handle
 * @param toInsert body to insert
 * @param toRemove body to replace
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_replaceBodyWith(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *toInsert, 
    struct tfVertexSolverBodyHandleHandle *toRemove
);

/**
 * @brief Refresh internal ordering of defined bodies
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_refreshBodies(struct tfVertexSolverSurfaceHandleHandle *handle);

/**
 * @brief Get the surface type
 * 
 * @param handle populated handle
 * @param stype surface type
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getType(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverSurfaceTypeHandle *stype);

/**
 * @brief Become a different type
 * 
 * @param handle populated handle
 * @param stype type to become
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_become(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfVertexSolverSurfaceTypeHandle *stype);

/**
 * @brief Get the bodies defined by the surface
 * 
 * @param handle populated handle
 * @param objs objects
 * @param numObjs number of objects
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getBodies(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get the vertices that define the surface
 * 
 * @param handle populated handle
 * @param objs objects
 * @param numObjs number of objects
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getVertices(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Find a vertex that defines this surface
 * 
 * @param handle populated handle
 * @param dir direction to look with respect to the centroid
 * @param result vertex
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_findVertex(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverVertexHandleHandle *result
);

/**
 * @brief Find a body that this surface defines
 * 
 * @param handle
 * @param dir direction to look with respect to the centroid
 * @param result body
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_findBody(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverBodyHandleHandle *result
);

/**
 * @brief Connected vertices on the same surface
 * 
 * @param handle populated handle
 * @param v vertex of interest
 * @param v1 first connected vertex
 * @param v2 second connected vertex
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_neighborVertices(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2
);

/**
 * @brief Connected surfaces on the same body
 * 
 * @param handle populated handle
 * @param objs surfaces
 * @param numObjs number of surfaces
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_neighborSurfaces(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Surfaces that share at least one vertex in a set of vertices
 * 
 * @param handle populated handle
 * @param verts vertices
 * @param numVerts number of vertices
 * @param objs objects
 * @param numObjs number of objects
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_connectedSurfacesS(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **verts, 
    unsigned int numVerts, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Surfaces that share at least one vertex
 * 
 * @param handle populated handle
 * @param objs objects
 * @param numObjs number of objects
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_connectedSurfaces(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Vertices defining this and another surface
 * 
 * @param handle
 * @param other another surface
 * @param objs objects
 * @param numObjs number of objects
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_connectingVertices(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *other, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get the integer labels of the contiguous vertices that this surface shares with another surface
 * 
 * @param handle populated handle
 * @param other another surface
 * @param labels edge labels
 * @param numLabels number of edge labels
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_contiguousVertexLabels(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *other, 
    unsigned int **labels, 
    int *numLabels
);

/**
 * @brief Get the vertices of a contiguous shared edge with another surface. 
 * 
 * Edges are labeled in increasing order starting with "1". A requested edge that does not exist returns empty. 
 * 
 * A requested edge with label "0" returns all vertices not shared with another surface
 * 
 * @param handle populated handle
 * @param other another surface
 * @param edgeLabel edge label
 * @param objs objects
 * @param numObjs number of objects
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_sharedContiguousVertices(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *other, 
    unsigned int edgeLabel, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get the surface normal
 * 
 * @param handle populated handle
 * @param result surface normal
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getNormal(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t **result);

/**
 * @brief Get the surface unnormalized normal
 * 
 * @param handle populated handle
 * @param result surface unnormalized normal
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getUnnormalizedNormal(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t **result);

/**
 * @brief Get the centroid
 * 
 * @param handle populated handle
 * @param result centroid
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getCentroid(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t **result);

/**
 * @brief Get the velocity, calculated as the velocity of the centroid
 * 
 * @param handle populated handle
 * @param result velocity
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getVelocity(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t **result);

/**
 * @brief Get the area
 * 
 * @param handle populated handle
 * @param result area
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getArea(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Get the perimeter
 * 
 * @param handle populated handle
 * @param result perimeter
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getPerimeter(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Get the mass density; only used in 2D simulation
 * 
 * @param handle populated handle
 * @param result mass density
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getDensity(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Set the mass density; only used in 2D simulation
 * 
 * @param handle populated handle
 * @param density density
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_setDensity(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t density);

/**
 * @brief Get the mass; only used in 2D simulation
 * 
 * @param handle populated handle
 * @param result mass
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getMass(struct tfVertexSolverSurfaceHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Get the sign of the volume contribution to a body that this surface contributes
 * 
 * @param handle populated handle
 * @param body a body
 * @param result sign of the volume contribution
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_volumeSense(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *body, 
    tfFloatP_t *result
);

/**
 * @brief Get the volume that this surface contributes to a body
 * 
 * @param handle populated handle
 * @param body a body
 * @param result volume contribution
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getVolumeContr(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *body, 
    tfFloatP_t *result
);

/**
 * @brief Get the outward facing normal w.r.t. a body
 * 
 * @param handle populated handle
 * @param body a body
 * @param result normal
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getOutwardNormal(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *body, 
    tfFloatP_t **result
);

/**
 * @brief Get the area that a vertex contributes to this surface
 * 
 * @param handle populated handle
 * @param v a vertex
 * @param result area contribution
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getVertexArea(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
);

/**
 * @brief Get the mass contribution of a vertex to this surface; only used in 2D simulation
 * 
 * @param populated handle
 * @param v a vertex
 * @param result mass contribution
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getVertexMass(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
);

/**
 * @brief Test whether the surface has a style
 * 
 * @param handle populated handle
 * @param result result of test
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_hasStyle(struct tfVertexSolverSurfaceHandleHandle *handle, bool *result);

/**
 * @brief Get the surface style
 * 
 * @param handle populated handle
 * @param result surface style
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_getStyle(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfRenderingStyleHandle *result);

/**
 * @brief Set the surface style
 * 
 * @param handle populated handle
 * @param s surface style
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_setStyle(struct tfVertexSolverSurfaceHandleHandle *handle, struct tfRenderingStyleHandle *s);

/**
 * @brief Get the normal distance to a point; negative distance means that the point is on the inner side
 * 
 * @param handle populated handle
 * @param pos position
 * @param result normal distance
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_normalDistance(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *pos, 
    tfFloatP_t *result
);

/**
 * @brief Test whether a point is on the outer side
 * 
 * @param handle populated handle
 * @param pos position
 * @param result result of test
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_isOutside(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *pos, 
    bool *result
);

/**
 * @brief Test whether the surface contains a point
 * 
 * @param handle populated handle
 * @param pos position of the point
 * @param result result of the test
 * @param v0 a vertex of the nearest edge
 * @param v1 a vertex of the nearest edge
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_contains(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *pos, 
    bool *result, 
    struct tfVertexSolverVertexHandleHandle *v0, 
    struct tfVertexSolverVertexHandleHandle *v1
);

/**
 * @brief Merge with a surface. The passed surface is destroyed. 
 * 
 * Surfaces must have the same number of vertices. Vertices are paired by nearest distance.
 * 
 * @param handle populated handle
 * @param toRemove surface to remove
 * @param lenCfs distance coefficients in [0, 1] for where to place the merged vertex, from each kept vertex to each removed vertex
 * @param numLenCfs number of distance coefficients
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_merge(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toRemove, 
    tfFloatP_t *lenCfs, 
    unsigned int numLenCfs
);

/**
 * @brief Create a surface from two vertices and a position
 * 
 * @param handle populated handle
 * @param vertIdxStart index of first vertex
 * @param pos position of new vertex
 * @param newObj newly created surface
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_extend(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    unsigned int vertIdxStart, 
    tfFloatP_t *pos, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Create a surface from two vertices of a surface in a mesh by extruding along the normal of the surface
 * 
 * @param handle populated handle
 * @param vertIdxStart index of first vertex
 * @param normLen length along surface normal by which to extrude
 * @param newObj newly created surface
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_extrude(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    unsigned int vertIdxStart, 
    tfFloatP_t normLen, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Split into two surfaces
 * 
 * Both vertices must already be in the surface and not adjacent
 * 
 * Vertices in the winding from from vertex to second go to newly created surface
 * 
 * Requires updated surface members (e.g., centroid)
 * 
 * @param handle populated handle
 * @param v1 fist vertex defining the split
 * @param v2 second vertex defining the split
 * @param newObj newly created surface
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_splitBy(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Split into two surfaces
 * 
 * Requires updated surface members (e.g., centroid)
 * 
 * @param handle populated handle
 * @param cp_pos point on the cut plane
 * @param cp_norm normal of the cut plane
 * @param newObj newly created surface
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceHandle_splitHow(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    tfFloatP_t *cp_pos, 
    tfFloatP_t *cp_norm, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);


/////////////////
// SurfaceType //
/////////////////


/**
 * @brief Initialize a new instance
 * 
 * @param handle handle to populate
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_init(struct tfVertexSolverSurfaceTypeHandle *handle);

/**
 * @brief Initialize an instance from a definition
 * 
 * @param handle handle to populate
 * @param sdef definition
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_initD(struct tfVertexSolverSurfaceTypeHandle *handle, struct tfVertexSolverSurfaceTypeSpec sdef);

/**
 * @brief Create from a JSON string representation. 
 * 
 * The returned type is automatically registered with the solver. 
 * 
 * @param handle
 * @param str a string, as returned by ``toString``
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_fromString(struct tfVertexSolverSurfaceTypeHandle *handle, const char *str);

/**
 * @brief Get the mesh object type
 * 
 * @param handle populated handle
 * @param label mesh object type label
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_objType(struct tfVertexSolverSurfaceTypeHandle *handle, unsigned int *label);

/**
 * @brief Get a summary string
 * 
 * @param handle populated handle
 * @param str summary string
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_str(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Get a JSON string representation
 * 
 * @param 
 * @param str summary string
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_toString(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Registers a type with the engine.
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_registerType(struct tfVertexSolverSurfaceTypeHandle *handle);

/**
 * @brief Tests whether this type is registered
 * 
 * @param handle
 * @param result result of test
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_isRegistered(struct tfVertexSolverSurfaceTypeHandle *handle, bool *result);

/**
 * @brief Name of this surface type
 * 
 * @param handle populated handle
 * @param str name
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_getName(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Set the name of this surface type
 * 
 * @param handle populated handle
 * @param str name
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_setName(struct tfVertexSolverSurfaceTypeHandle *handle, const char *str);

/**
 * @brief Get the style of the surface type
 * 
 * @param handle populated handle
 * @param style style of the surface type
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_getStyle(struct tfVertexSolverSurfaceTypeHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Get the density of the surface type; only used in 2D simulation
 * 
 * @param handle populated handle
 * @param result density
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_getDensity(struct tfVertexSolverSurfaceTypeHandle *handle, tfFloatP_t *result);

/**
 * @brief Set the density of the surface type; only used in 2D simulation
 * 
 * @param handle populated handle
 * @param result density
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_setDensity(struct tfVertexSolverSurfaceTypeHandle *handle, tfFloatP_t result);

/**
 * @brief Get the list of instances that belong to this type
 * 
 * @param handle populated handle
 * @param objs instances
 * @param numObjs number of instances
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_getInstances(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    unsigned int *numObjs
);

/**
 * @brief Get the list of instances ids that belong to this type
 * 
 * @param handle populated handle
 * @param ids instance ids
 * @param numObjs number of instances
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_getInstanceIds(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    int **ids, 
    unsigned int *numObjs
);

/**
 * @brief Construct a surface of this type from a set of vertices
 * 
 * @param handle populated handle
 * @param vertices set of vertices
 * @param numVerts number of vertices
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_createSurfaceV(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **vertices, 
    unsigned int numVerts, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Construct a surface of this type from a set of positions
 * 
 * @param handle populated handle
 * @param positions set of positions
 * @param numPositions number of positions
 * @param newObj newly created objects
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_createSurfaceP(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    tfFloatP_t **positions, 
    unsigned int numPositions, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Construct a surface of this type from a face
 * 
 * @param handle populated handle
 * @param face face data
 * @param newObj newly created objects
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_createSurfaceIO(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    struct tfIoThreeDFFaceDataHandle *face, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Construct a polygon with n vertices circumscribed on a circle
 * 
 * @param handle populated handle
 * @param n number of points
 * @param center center of the circle
 * @param radius radius of the circle
 * @param ax1 first axis defining the orientation of the circle
 * @param ax2 second axis defining the orientation of the circle
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_nPolygon(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    unsigned int n, 
    tfFloatP_t *center, 
    tfFloatP_t radius, 
    tfFloatP_t *ax1, 
    tfFloatP_t *ax2, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Replace a vertex with a surface. Vertices are created for the surface along every destroyed edge
 * 
 * @param handle populated handle
 * @param toReplace vertex to replace
 * @param lenCfs distance coefficients in [0, 1] defining where to create a new vertex along each edge
 * @param numLenCfs number of distance coefficients
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverSurfaceType_replace(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toReplace, 
    tfFloatP_t *lenCfs, 
    unsigned int numLenCfs, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Construct a surface from a set of vertices
 * 
 * @param verts set of vertices
 * @param numVerts number of vertices
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverCreateSurfaceByVertices(
    struct tfVertexSolverVertexHandleHandle **verts, 
    unsigned int numVerts, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Construct a surface from a face
 * 
 * @param face face data
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverCreateSurfaceByIOData(
    struct tfIoThreeDFFaceDataHandle *face, 
    struct tfVertexSolverSurfaceHandleHandle *newObj
);

/**
 * @brief Get a registered type by name
 * 
 * @param name name of type
 * @param result type
 */
CAPI_FUNC(HRESULT) tfVertexSolverFindSurfaceTypeFromName(const char *name, struct tfVertexSolverSurfaceTypeHandle *result);

/**
 * @brief Bind adhesion for all types with matching specification
 * 
 * @param stypes surface types to bind
 * @param sdefs surface type definitions
 * @param numTypes number of types
 */
CAPI_FUNC(HRESULT) tfVertexSolverBindSurfaceTypeAdhesion(
    struct tfVertexSolverSurfaceTypeHandle **stypes, 
    struct tfVertexSolverSurfaceTypeSpec *sdefs, 
    unsigned int numTypes
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCSURFACE_H_