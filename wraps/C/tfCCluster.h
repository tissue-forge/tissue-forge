/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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
 * @file tfCCluster.h
 * 
 */

#ifndef _WRAPS_C_TFCCLUSTER_H_
#define _WRAPS_C_TFCCLUSTER_H_

#include "tf_port_c.h"

#include "tfCParticle.h"

/**
 * @brief Cluster type definition in Tissue Forge C
 * 
 */
struct CAPI_EXPORT tfClusterTypeSpec {
    tfFloatP_t mass;
    tfFloatP_t charge;
    tfFloatP_t radius;
    tfFloatP_t *target_energy;
    tfFloatP_t minimum_radius;
    tfFloatP_t eps;
    tfFloatP_t rmin;
    unsigned char dynamics;
    unsigned int frozen;
    char *name;
    char *name2;
    unsigned int numTypes;
    struct tfParticleTypeHandle **types;
};

// Handles

/**
 * @brief Handle to a @ref ClusterParticleHandle instance
 * 
 */
struct CAPI_EXPORT tfClusterParticleHandleHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref ClusterParticleType instance
 * 
 */
struct CAPI_EXPORT tfClusterParticleTypeHandle {
    void *tfObj;
};


///////////////////////
// tfClusterTypeSpec //
///////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct tfClusterTypeSpec) tfClusterTypeSpec_init();


///////////////////////////
// ClusterParticleHandle //
///////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param id particle id
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_init(struct tfClusterParticleHandleHandle *handle, int id);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_destroy(struct tfClusterParticleHandleHandle *handle);

/**
 * @brief Constituent particle constructor. 
 * 
 * The created particle will belong to this cluster. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @param handle populated handle
 * @param partTypeHandle type of particle to create
 * @param pid id of created particle
 * @param position pointer to 3-element array, or NULL for a random position
 * @param velocity pointer to 3-element array, or NULL for a random velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_createParticle(
    struct tfClusterParticleHandleHandle *handle, 
    struct tfParticleTypeHandle *partTypeHandle, 
    int *pid, 
    tfFloatP_t **position, 
    tfFloatP_t **velocity
);

/**
 * @brief Constituent particle constructor. 
 * 
 * The created particle will belong to this cluster. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @param handle populated handle
 * @param partTypeHandle type of particle to create
 * @param pid id of created particle
 * @param str JSON string defining a particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_createParticleS(
    struct tfClusterParticleHandleHandle *handle, 
    struct tfParticleTypeHandle *partTypeHandle, 
    int *pid, 
    const char *str
);

/**
 * @brief Test whether the cluster has an id
 * 
 * @param handle populated handle
 * @param pid particle id to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_hasPartId(struct tfClusterParticleHandleHandle *handle, int pid, bool *result);

/**
 * @brief Test whether the cluster has a particle
 * 
 * @param handle populated handle
 * @param part particle to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_hasPart(struct tfClusterParticleHandleHandle *handle, struct tfParticleHandleHandle *part, bool *result);

/**
 * @brief Split the cluster along an axis. 
 * 
 * @param handle populated handle
 * @param cid id of created cluster
 * @param axis 3-component allocated axis of split
 * @param time time at which to implement the split; currently not supported
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_splitAxis(struct tfClusterParticleHandleHandle *handle, int *cid, tfFloatP_t *axis, tfFloatP_t time);

/**
 * @brief Split the cluster randomly. 
 * 
 * @param handle populated handle
 * @param cid id of created cluster
 * @param time time at which to implement the split; currently not supported
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_splitRand(struct tfClusterParticleHandleHandle *handle, int *cid, tfFloatP_t time);

/**
 * @brief Split the cluster along a point and normal. 
 * 
 * @param handle populated handle
 * @param cid id of created cluster
 * @param time time at which to implement the split; currently not supported
 * @param normal 3-component normal vector of cleavage plane
 * @param point 3-component point on cleavage plane
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_split(struct tfClusterParticleHandleHandle *handle, int *cid, tfFloatP_t time, tfFloatP_t *normal, tfFloatP_t *point);

/**
 * @brief Get the number of particles that are a member of this cluster.
 * 
 * @param handle populated handle
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_getNumParts(struct tfClusterParticleHandleHandle *handle, int *numParts);

/**
 * @brief Get the particles that are a member of this cluster
 * 
 * @param handle populated handle
 * @param parts particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_getParts(struct tfClusterParticleHandleHandle *handle, struct tfParticleListHandle *parts);

/**
 * @brief Get the i'th particle that's a member of this cluster.
 * 
 * @param handle populated handle
 * @param i index of particle to get
 * @param parthandle particle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_getParticle(struct tfClusterParticleHandleHandle *handle, int i, struct tfParticleHandleHandle *parthandle);

/**
 * @brief Get a summary string of the cluster
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_str(struct tfClusterParticleHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Get the radius of gyration
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_getRadiusOfGyration(struct tfClusterParticleHandleHandle *handle, tfFloatP_t *radiusOfGyration);

/**
 * @brief Get the center of mass
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_getCenterOfMass(struct tfClusterParticleHandleHandle *handle, tfFloatP_t **com);

/**
 * @brief Get the centroid
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_getCentroid(struct tfClusterParticleHandleHandle *handle, tfFloatP_t **cent);

/**
 * @brief Get the moment of inertia
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleHandle_getMomentOfInertia(struct tfClusterParticleHandleHandle *handle, tfFloatP_t **moi);


/////////////////////////
// ClusterParticleType //
/////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_init(struct tfClusterParticleTypeHandle *handle);

/**
 * @brief Initialize an instance from a definition
 * 
 * @param handle handle to populate
 * @param pdef definition
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_initD(struct tfClusterParticleTypeHandle *handle, struct tfClusterTypeSpec pdef);

/**
 * @brief Add a particle type to the types of a cluster
 * 
 * @param handle populated handle
 * @param phandle handle to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_addType(struct tfClusterParticleTypeHandle *handle, struct tfParticleTypeHandle *phandle);

/**
 * @brief Get a summary string of the type
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_str(struct tfClusterParticleTypeHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Tests where this cluster has a particle type.
 * 
 * Only tests for immediate ownership and ignores multi-level clusters.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_hasType(struct tfClusterParticleTypeHandle *handle, struct tfParticleTypeHandle *phandle, bool *hasType);

/**
 * @brief Test whether the type has a type id
 * 
 * @param handle populated handle
 * @param pid id to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_hasTypeId(struct tfClusterParticleTypeHandle *handle, int pid, bool *result);

/**
 * @brief Test whether the type has a type
 * 
 * @param handle populated handle
 * @param ptype type to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_hasTypeML(struct tfClusterParticleTypeHandle *handle, struct tfParticleTypeHandle *ptype, bool *result);

/**
 * @brief Test whether the type has a particle
 * 
 * @param handle populated handle
 * @param part particle to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_hasPart(struct tfClusterParticleTypeHandle *handle, struct tfParticleHandleHandle *part, bool *result);

/**
 * @brief Registers a type with the engine. 
 * 
 * Also registers all unregistered constituent types. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_registerType(struct tfClusterParticleTypeHandle *handle);

/**
 * @brief Cluster particle constructor.
 * 
 * @param handle populated handle
 * @param pid id of created particle
 * @param position pointer to 3-element array, or NULL for a random position
 * @param velocity pointer to 3-element array, or NULL for a random velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_createParticle(struct tfClusterParticleTypeHandle *handle, int *pid, tfFloatP_t *position, tfFloatP_t *velocity);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get a registered cluster type by type name
 * 
 * @param handle handle to populate
 * @param name name of cluster type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_FindFromName(struct tfClusterParticleTypeHandle *handle, const char* name);

/**
 * @brief Get a registered cluster type by id
 * 
 * @param handle handle to populate
 * @param pid id of type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfClusterParticleType_getFromId(struct tfClusterParticleTypeHandle *handle, unsigned int pid);

#endif // _WRAPS_C_TFCCLUSTER_H_