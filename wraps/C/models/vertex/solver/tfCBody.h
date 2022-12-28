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

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCBODY_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCBODY_H_

#include <tf_port_c.h>

#include <tfC_io.h>


/**
 * @brief Body type definition in Tissue Forge C
 * 
 */
struct CAPI_EXPORT tfVertexSolverBodyTypeSpec {
    tfFloatP_t *density;
    tfFloatP_t **bodyForceComps;
    tfFloatP_t *surfaceAreaLam;
    tfFloatP_t *surfaceAreaVal;
    tfFloatP_t *volumeLam;
    tfFloatP_t *volumeVal;
    char *name;
    char **adhesionNames;
    tfFloatP_t *adhesionValues;
    unsigned int numAdhesionValues;
};


// Handles

/**
 * @brief Handle to a @ref models::vertex::BodyHandle instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverBodyHandleHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref models::vertex::BodyType instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverBodyTypeHandle {
    void *tfObj;
};


////////////////////////////////
// tfVertexSolverBodyTypeSpec //
////////////////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct tfVertexSolverBodyTypeSpec) tfVertexSolverBodyTypeSpec_init();


////////////////
// BodyHandle //
////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param id object id
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_init(struct tfVertexSolverBodyHandleHandle *handle, int id);

/**
 * @brief Create an instance from a JSON string representation
 * 
 * @param handle handle to populate
 * @param s JSON string
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_fromString(struct tfVertexSolverBodyHandleHandle *handle, const char *s);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_destroy(struct tfVertexSolverBodyHandleHandle *handle);

/**
 * @brief Get the object id
 * 
 * @param handle populated handle
 * @param objId object id
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getId(struct tfVertexSolverBodyHandleHandle *handle, int *objId);

/**
 * @brief Test whether a body is defined by a vertex
 * 
 * @param handle populated handle
 * @param v vertex
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_definedByVertex(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    bool *result
);

/**
 * @brief Test whether a body is defined by a surface
 * 
 * @param handle populated handle
 * @param s surface
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_definedBySurface(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    bool *result
);

/**
 * @brief Get the mesh object type
 * 
 * @param handle populated handle
 * @param label object type label
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_objType(struct tfVertexSolverBodyHandleHandle *handle, unsigned int *label);

/**
 * @brief Destroy the body
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_destroyBody(struct tfVertexSolverBodyHandleHandle *handle);

/**
 * @brief Destroy the body. 
 * 
 * Any resulting surfaces without a body are also destroyed. 
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_destroyBodyC(struct tfVertexSolverBodyHandleHandle *handle);

/**
 * @brief Validate the body
 * 
 * @param handle populated handle
 * @param result result of validation
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_validate(struct tfVertexSolverBodyHandleHandle *handle, bool *result);

/**
 * @brief Update internal data due to a change in position
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_positionChanged(struct tfVertexSolverBodyHandleHandle *handle);

/**
 * @brief Get a summary string
 * 
 * @param handle populated handle
 * @param str summary string
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_str(struct tfVertexSolverBodyHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str JSON string
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_toString(struct tfVertexSolverBodyHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Add a surface
 * 
 * @param handle populated handle
 * @param s surface to add
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_add(struct tfVertexSolverBodyHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s);

/**
 * @brief Remove a surface
 * 
 * @param handle populated handle
 * @param s surface to remove
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_remove(struct tfVertexSolverBodyHandleHandle *handle, struct tfVertexSolverSurfaceHandleHandle *s);

/**
 * @brief Replace a surface a surface
 * 
 * @param handle populated handle
 * @param toInsert surface to insert
 * @param toRemove surface to remove
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_replace(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *toInsert, 
    struct tfVertexSolverSurfaceHandleHandle *toRemove
);

/**
 * @brief Get the body type
 * 
 * @param handle populated handle
 * @param btype body type
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getType(struct tfVertexSolverBodyHandleHandle *handle, struct tfVertexSolverBodyTypeHandle *btype);

/**
 * @brief Become a different type
 * 
 * @param handle populated handle
 * @param btype type to become
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_become(struct tfVertexSolverBodyHandleHandle *handle, struct tfVertexSolverBodyTypeHandle *btype);

/**
 * @brief Get the surfaces that define the body
 * 
 * @param handle populated handle
 * @param objs surfaces
 * @param numObjs number of surface
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getSurfaces(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get the vertices that define the body
 * 
 * @param handle populated handle
 * @param objs vertices
 * @param numObjs number of vertices
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getVertices(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Find a vertex that defines this body
 * 
 * @param handle populated handle
 * @param dir direction to look with respect to the centroid
 * @param v vertex
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_findVertex(
    struct tfVertexSolverBodyHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverVertexHandleHandle *v
);

/**
 * @brief Find a surface that defines this body
 * 
 * @param handle populated handle
 * @param dir direction to look with respect to the centroid
 * @param s surface
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_findSurface(
    struct tfVertexSolverBodyHandleHandle *handle, 
    tfFloatP_t *dir, 
    struct tfVertexSolverSurfaceHandleHandle *s
);

/**
 * @brief Get the neighboring bodies. 
 * 
 * A body is a neighbor if it shares a surface.
 * 
 * @param handle populated handle
 * @param objs neighboring bodies
 * @param numObjs number of neighboring bodies
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_neighborBodies(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get the neighboring surfaces of a surface on this body.
 * 
 * Two surfaces are a neighbor on this body if they define the body and share a vertex
 * 
 * @param handle populated handle
 * @param s a surface
 * @param objs neighboring surfaces
 * @param numObjs number of neighboring surfaces
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_neighborSurfaces(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *s, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get the mass density
 * 
 * @param handle populated handle
 * @param result mass density
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getDensity(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Set the mass density
 * 
 * @param handle populated handle
 * @param density mass density
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_setDensity(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t density);

/**
 * @brief Get the centroid
 * 
 * @param handle populated handle
 * @param result centroid
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getCentroid(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t **result);

/**
 * @brief Get the velocity, calculated as the velocity of the centroid
 * 
 * @param handle populated handle
 * @param result velocity
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getVelocity(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t **result);

/**
 * @brief Get the surface area
 * 
 * @param handle populated handle
 * @param result surface area
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getArea(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Get the volume
 * 
 * @param handle populated handle
 * @param result volume
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getVolume(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Get the mass
 * 
 * @param handle populated handle
 * @param result mass
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getMass(struct tfVertexSolverBodyHandleHandle *handle, tfFloatP_t *result);

/**
 * @brief Get the surface area contribution of a vertex to this body
 * 
 * @param handle populated handle
 * @param v a vertex
 * @param result surface area contribution
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getVertexArea(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
);

/**
 * @brief Get the volume contribution of a vertex to this body
 * 
 * @param handle populated handle
 * @param v a vertex
 * @param result volume contribution
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getVertexVolume(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
);

/**
 * @brief Get the mass contribution of a vertex to this body
 * 
 * @param handle populated handle
 * @param v a vertex
 * @param result mass contribution
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_getVertexMass(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverVertexHandleHandle *v, 
    tfFloatP_t *result
);

/**
 * @brief Get the surfaces that define the interface between this body and another body
 * 
 * @param handle populated handle
 * @param b a body
 * @param objs interface surfaces
 * @param numObjs number of interface surfaces
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_findInterface(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *b, 
    struct tfVertexSolverSurfaceHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get the contacting surface area of this body with another body
 * 
 * @param handle populated handle
 * @param other another body
 * @param result contacting surface area
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_contactArea(
    struct tfVertexSolverBodyHandleHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *other, 
    tfFloatP_t *result
);

/**
 * @brief Test whether a point is outside. Test is performed using the nearest surface
 * 
 * @param handle populated handle
 * @param pos position
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_isOutside(
    struct tfVertexSolverBodyHandleHandle *handle, 
    tfFloatP_t *pos, 
    bool *result
);

/**
 * @brief Split into two bodies. The split is defined by a cut plane
 * 
 * @param handle populated handle
 * @param cp_pos position on the cut plane
 * @param cp_norm cut plane normal
 * @param stype type of newly created surface. taken from connected surfaces if specified as 0
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyHandle_split(
    struct tfVertexSolverBodyHandleHandle *handle, 
    tfFloatP_t *cp_pos, 
    tfFloatP_t *cp_norm, 
    struct tfVertexSolverSurfaceTypeHandle *stype, 
    struct tfVertexSolverBodyHandleHandle *newObj
);


//////////////
// BodyType //
//////////////


/**
 * @brief Initialize a new instance
 * 
 * @param handle handle to populate
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_init(struct tfVertexSolverBodyTypeHandle *handle);

/**
 * @brief Initialize an instance from a definition
 * 
 * @param handle handle to populate
 * @param bdef type definition
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_initD(struct tfVertexSolverBodyTypeHandle *handle, struct tfVertexSolverBodyTypeSpec bdef);

/**
 * @brief Create an instance from a JSON string representation
 * 
 * @param handle handle to populate
 * @param s JSON string
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_fromString(struct tfVertexSolverBodyTypeHandle *handle, const char *s);

/**
 * @brief Get the mesh object type
 * 
 * @param handle populated handle
 * @param label object type label
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_objType(struct tfVertexSolverBodyTypeHandle *handle, unsigned int *label);

/**
 * @brief Get a summary string
 * 
 * @param handle populated handle
 * @param str summary string
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_str(
    struct tfVertexSolverBodyTypeHandle *handle, 
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
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_toString(
    struct tfVertexSolverBodyTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Registers a type with the engine.
 * 
 * Note that this occurs automatically, unless noReg==true in constructor.  
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_registerType(struct tfVertexSolverBodyTypeHandle *handle);

/**
 * @brief Tests whether this type is registered
 * 
 * @param result result of the test
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_isRegistered(struct tfVertexSolverBodyTypeHandle *handle, bool *result);

/**
 * @brief Get a list of instances that belong to this type
 * 
 * @param handle populated handle
 * @param objs list of instances
 * @param numObjs number of instances
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_getInstances(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfVertexSolverBodyHandleHandle **objs, 
    int *numObjs
);

/**
 * @brief Get a list of instances ids that belong to this type
 * 
 * @param handle populated handle
 * @param ids list of instances ids
 * @param numObjs number of instances ids
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_getInstanceIds(
    struct tfVertexSolverBodyTypeHandle *handle, 
    int **ids, 
    int *numObjs
);

/**
 * @brief Name of this body type
 * 
 * @param handle populated handle
 * @param str name
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_getName(
    struct tfVertexSolverBodyTypeHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Set the name of this body type
 * 
 * @param handle populated handle
 * @param name name
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_setName(struct tfVertexSolverBodyTypeHandle *handle, const char *name);

/**
 * @brief Get the mass density
 * 
 * @param handle populated handle
 * @param result mass density
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_getDensity(struct tfVertexSolverBodyTypeHandle *handle, tfFloatP_t *result);

/**
 * @brief Set the mass density
 * 
 * @param handle populated handle
 * @param density mass density
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_setDensity(struct tfVertexSolverBodyTypeHandle *handle, tfFloatP_t density);

/**
 * @brief Construct a body of this type from a set of surfaces
 * 
 * @param handle populated handle
 * @param surfaces set of surfaces
 * @param numSurfaces number of surfaces
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_createBodyS(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle **surfaces, 
    unsigned int numSurfaces, 
    struct tfVertexSolverBodyHandleHandle *newObj
);

/**
 * @brief Construct a body of this type from a mesh
 * 
 * @param handle populated handle
 * @param ioMesh mesh
 * @param stype type of newly created surfaces
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_createBodyIO(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfIoThreeDFMeshDataHandle *ioMesh, 
    struct tfVertexSolverSurfaceTypeHandle *stype, 
    struct tfVertexSolverBodyHandleHandle *newObj
);

/**
 * @brief Create a body from a surface in the mesh and a position
 * 
 * @param handle populated handle
 * @param base base surface
 * @param pos position
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_extend(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *base, 
    tfFloatP_t *pos, 
    struct tfVertexSolverBodyHandleHandle *newObj
);

/**
 * @brief Create a body from a surface in a mesh by extruding along the outward-facing normal of the surface
 * 
 * @param handle populated handle
 * @param base base surface
 * @param normLen length along normal direction by which to extrude
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverBodyType_extrude(
    struct tfVertexSolverBodyTypeHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *base, 
    tfFloatP_t normLen, 
    struct tfVertexSolverBodyHandleHandle *newObj
);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Construct a body from a set of surfaces
 * 
 * @param surfaces set of surfaces
 * @param numSurfaces number of surfaces
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverCreateBodyBySurfaces(
    struct tfVertexSolverSurfaceHandleHandle **surfaces, 
    unsigned int numSurfaces, 
    struct tfVertexSolverBodyHandleHandle *newObj
);

/**
 * @brief Construct a body from a mesh
 * 
 * @param ioMesh mesh
 * @param newObj newly created object
 */
CAPI_FUNC(HRESULT) tfVertexSolverCreateBodyByIOData(struct tfIoThreeDFMeshDataHandle *ioMesh, struct tfVertexSolverBodyHandleHandle *newObj);

/**
 * @brief Get a registered type by name
 * 
 * @param name type name
 * @param btype type
 */
CAPI_FUNC(HRESULT) tfVertexSolverFindBodyTypeFromName(const char *name, struct tfVertexSolverBodyTypeHandle *btype);

/**
 * @brief Bind adhesion for all types with matching specification
 * 
 * @param btypes body types to bind
 * @param bdefs body type definitions
 * @param numTypes number of types
 */
CAPI_FUNC(HRESULT) tfVertexSolverBindBodyTypeAdhesion(
    struct tfVertexSolverBodyTypeHandle **btypes, 
    struct tfVertexSolverBodyTypeSpec *bdefs, 
    unsigned int numTypes
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCBODY_H_