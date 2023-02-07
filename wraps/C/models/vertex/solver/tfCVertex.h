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
 * @file tfCVertex.h
 * 
 */

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCVERTEX_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCVERTEX_H_

#include <tf_port_c.h>

#include <tfCParticle.h>
#include <tfC_io.h>

// Handles

/**
 * @brief Handle to a @ref models::vertex::VertexHandle instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverVertexHandleHandle {
    void *tfObj;
};


////////////
// Vertex //
////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param id object id
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_init(struct tfVertexSolverVertexHandleHandle *handle, int id);

/**
 * @brief Create an instance from a JSON string representation
 * 
 * @param handle handle to populate
 * @param str JSON string
 * @param numChars number of string chars
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_fromString(struct tfVertexSolverVertexHandleHandle *handle, const char *str);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_destroy(struct tfVertexSolverVertexHandleHandle *handle);

/**
 * @brief Get the id of an instance
 * 
 * @param handle populated handle
 * @param objId instance id
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getId(struct tfVertexSolverVertexHandleHandle *handle, int *objId);

/**
 * @brief Test whether a vertex defines a surface
 * 
 * @param handle populated handle
 * @param s surface
 * @param result test result
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_definesSurface(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    bool *result
);

/**
 * @brief Test whether a vertex defines a body
 * 
 * @param handle populated handle
 * @param b body
 * @param result test result
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_definesBody(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *b, 
    bool *result
);

/**
 * @brief Get the mesh object type
 * 
 * @param handle populated handle
 * @param label mesh object type
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_objType(struct tfVertexSolverVertexHandleHandle *handle, int *label);

/**
 * @brief Destroy the vertex
 * 
 * @param handle populated handle
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_destroyVertex(struct tfVertexSolverVertexHandleHandle *handle);

/**
 * @brief Validate the vertex
 * 
 * @param handle populated handle
 * @param result test result
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_validate(struct tfVertexSolverVertexHandleHandle *handle, bool *result);

/**
 * @brief Update internal data due to a change in position
 * 
 * @param handle populated handle
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_positionChanged(struct tfVertexSolverVertexHandleHandle *handle);

/**
 * @brief Get a summary string
 * 
 * @param handle populated handle
 * @param str summary string
 * @param numChars number of chars
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_str(struct tfVertexSolverVertexHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str JSON string
 * @param numChars number of chars
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_toString(struct tfVertexSolverVertexHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Add a surface
 * 
 * @param handle populated handle
 * @param s surface to add
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_addSurface(struct tfVertexSolverVertexHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s);

/**
 * @brief Insert a surface at a location in the list of surfaces
 * 
 * @param handle populated handle
 * @param s surface to insert
 * @param idx index of insertion
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_insertSurfaceAt(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    int idx
);

/**
 * @brief Insert a surface before another surface
 * 
 * @param handle populated handle
 * @param s surface to insert
 * @param before surface to precede
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_insertSurfaceBefore(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    struct tfVertexSolverSurfaceHandleHandle *before
);

/**
 * @brief Remove a surface
 * 
 * @param handle populated handle
 * @param s surface to remove
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_remove(struct tfVertexSolverVertexHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s);

/**
 * @brief Replace a surface at a location in the list of surfaces
 * 
 * @param handle populated handle
 * @param toInsert surface to insert
 * @param idx location of replacement
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_replaceSurfaceAt(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toInsert, 
    int idx
);

/**
 * @brief Replace a surface with another surface
 * 
 * @param handle populated handle
 * @param toInsert surface to insert
 * @param toRemove surface to remove
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_replaceSurfaceWith(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toInsert, 
    struct tfVertexSolverSurfaceHandleHandle *toRemove
);

/**
 * @brief Get the id of the underlying particle
 * 
 * @param handle populated handle
 * @param result particle id, if any (-1 if none)
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getPartId(struct tfVertexSolverVertexHandleHandle *handle, int *result);

/**
 * @brief Get the bodies defined by the vertex
 * 
 * @param handle populated handle
 * @param objs bodies
 * @param numObjs number of bodies
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getBodies(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get the surfaces defined by the vertex
 * 
 * @param handle populated handle
 * @param objs surfaces
 * @param numObjs number of surfaces
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getSurfaces(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Find a surface defined by this vertex
 * 
 * @param handle populated handle
 * @param dir direction to look with respect to the vertex
 * @param result surface
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_findSurface(
    struct tfVertexSolverVertexHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverSurfaceHandleHandle *result
);

/**
 * @brief Find a body defined by this vertex
 * 
 * @param handle populated handle
 * @param dir direction to look with respect to the vertex
 * @param result body
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_findBody(
    struct tfVertexSolverVertexHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverBodyHandleHandle *result
);

/**
 * @brief Get the connected vertices. 
 * 
 * A vertex is connected if it defines an edge with this vertex.
 * 
 * @param handle populated handle
 * @param objs connected vertices
 * @param numObjs number of connected vertices
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_connectedVertices(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Update internal connected vertex data
 * 
 * @param handle populated handle
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_updateConnectedVertices(struct tfVertexSolverVertexHandleHandle *handle);

/**
 * @brief Get the surfaces that this vertex and another vertex both define
 * 
 * @param handle populated handle
 * @param other another vertex
 * @param result surfaces
 * @param numObjs number of surfaces
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_sharedSurfaces(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *other, 
    struct tfVertexSolverSurfaceHandleHandle **result, 
    int *numObjs
);

/**
 * @brief Get the current area
 * 
 * @param handle populated handle
 * @param result current area
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getArea(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Get the current volume
 * 
 * @param handle populated handle
 * @param result current volume
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getVolume(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Get the current mass
 * 
 * @param handle populated handle
 * @param result current mass
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getMass(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Update the properties of the underlying particle
 * 
 * @param handle populated handle
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_updateProperties(struct tfVertexSolverVertexHandleHandle *handle);

/**
 * @brief Get a handle to the underlying particle, if any
 * 
 * @param handle populated handle
 * @param result particle handle
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_particle(struct tfVertexSolverVertexHandleHandle *handle, struct tfParticleHandleHandle *result);

/**
 * @brief Get the current position
 * 
 * @param handle populated handle
 * @param result current position
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getPosition(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t **result);

/**
 * @brief Set the current position
 * 
 * @param handle populated handle
 * @param pos current position
 * @param updateChildren flag for whether to update children
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_setPosition(
    struct tfVertexSolverVertexHandleHandle *handle, 
    tfFloatP_t *pos, 
    bool updateChildren
);

/**
 * @brief Get the current velocity
 * 
 * @param handle populated handle
 * @param result current velocity
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_getVelocity(struct tfVertexSolverVertexHandleHandle *handle, tfFloatP_t **result);

/**
 * @brief Transfer all bonds to another vertex
 * 
 * @param handle populated handle
 * @param other another vertex
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_transferBondsTo(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *other
);

/**
 * @brief Replace a surface
 * 
 * @param handle populated handle
 * @param toReplace surface to replace
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_replaceSurface(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toReplace
);

/**
 * @brief Replace a body
 * 
 * @param handle populated handle
 * @param toReplace body to replace
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_replaceBody(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *toReplace
);

/**
 * @brief Merge with a vertex. The passed vertex is destroyed.
 * 
 * @param handle populated handle
 * @param toRemove vertex to remove
 * @param lenCf distance coefficient in [0, 1] for where to place the vertex, from the kept vertex to the removed vertex
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_merge(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *toRemove, 
    tfFloatP_t lenCf
);

/**
 * @brief Inserts a vertex between two vertices
 * 
 * @param handle populated handle
 * @param v1 first vertex
 * @param v2 second vertex
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_insertBetween(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v1, 
    struct tfVertexSolverVertexHandleHandle *v2
);

/**
 * @brief Insert a vertex between a vertex and each of a set of vertices
 * 
 * @param handle populated handle
 * @param vf another vertex
 * @param nbs a set of vertices
 * @param numNbs number of vertices
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_insertBetweenNeighbors(
    struct tfVertexSolverVertexHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *vf, 
    struct tfVertexSolverVertexHandleHandle **nbs, 
    int numNbs
);

/**
 * @brief Split a vertex into an edge
 * 
 * The vertex must define at least one surface.
 * 
 * New topology is governed by a cut plane at the midpoint of, and orthogonal to, the new edge. 
 * Each first-order neighbor vertex is connected to the vertex of the new edge on the same side of 
 * the cut plane. 
 * 
 * @param handle populated handle
 * @param sep separation distance
 * @param newObj newly created vertex
 */
CAPI_DATA(HRESULT) tfVertexSolverVertexHandle_split(
    struct tfVertexSolverVertexHandleHandle *handle, 
    tfFloatP_t *sep, 
    struct tfVertexSolverVertexHandleHandle *newObj
);


//////////////////////
// Module functions //
//////////////////////

/**
 * @brief Get the particle type of the solver
 * 
 * @param handle handle to populate
 */
CAPI_DATA(HRESULT) tfVertexSolverMeshParticleType_get(struct tfParticleTypeHandle *handle);

/**
 * @brief Create a vertex using the id of an existing particle
 * 
 * @param pid particle id
 * @param objId id of new vertex
 */
CAPI_DATA(HRESULT) tfVertexSolverCreateVertexByPartId(unsigned int &pid, int *objId);

/**
 * @brief Create a vertex at a position
 * 
 * @param position position to create a new vertex
 * @param objId id of new vertex
 */
CAPI_DATA(HRESULT) tfVertexSolverCreateVertexByPosition(tfFloatP_t *position, int *objId);

/**
 * @brief Create a vertex using I/O data
 * 
 * @param vdata I/O data
 * @param objId id of new vertex
 */
CAPI_DATA(HRESULT) tfVertexSolverCreateVertexByIOData(struct tfIoThreeDFVertexDataHandle *vdata, int *objId);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCVERTEX_H_