/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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
 * @file tfCParticle.h
 * 
 */

#ifndef _WRAPS_C_TFCPARTICLE_H_
#define _WRAPS_C_TFCPARTICLE_H_

#include "tf_port_c.h"

#include "tfCStyle.h"
#include "tfCStateVector.h"
#include "tfCSpecies.h"

/**
 * @brief Mapped particle data enums
 * 
*/
struct CAPI_EXPORT tfMappedParticleDataEnum {
    int MAPPEDPARTICLEDATA_NONE;
    int MAPPEDPARTICLEDATA_POSITION_X;
    int MAPPEDPARTICLEDATA_POSITION_Y;
    int MAPPEDPARTICLEDATA_POSITION_Z;
    int MAPPEDPARTICLEDATA_VELOCITY_X;
    int MAPPEDPARTICLEDATA_VELOCITY_Y;
    int MAPPEDPARTICLEDATA_VELOCITY_Z;
    int MAPPEDPARTICLEDATA_VELOCITY_SPEED;
    int MAPPEDPARTICLEDATA_SPECIES;
    int MAPPEDPARTICLEDATA_FORCE_X;
    int MAPPEDPARTICLEDATA_FORCE_Y;
    int MAPPEDPARTICLEDATA_FORCE_Z;
};

/**
 * @brief Particle type style definition in Tissue Forge C
 * 
 */
struct CAPI_EXPORT tfParticleTypeStyleSpec {
    char *color;
    unsigned int visible;
    char *speciesName;
    char *mapName;
    tfFloatP_t mapMin;
    tfFloatP_t mapMax;
    int mappedParticleData;
};

/**
 * @brief Particle type definition in Tissue Forge C
 * 
 */
struct CAPI_EXPORT tfParticleTypeSpec {
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
    struct tfParticleTypeStyleSpec *style;
    unsigned int numSpecies;
    char **species;
};

// Handles

struct CAPI_EXPORT tfParticleDynamicsEnumHandle {
    unsigned char PARTICLE_NEWTONIAN;
    unsigned char PARTICLE_OVERDAMPED;
};

struct CAPI_EXPORT tfParticleFlagsHandle {
    int PARTICLE_NONE;
    int PARTICLE_GHOST;
    int PARTICLE_CLUSTER;
    int PARTICLE_BOUND;
    int PARTICLE_FROZEN_X;
    int PARTICLE_FROZEN_Y;
    int PARTICLE_FROZEN_Z;
    int PARTICLE_FROZEN;
    int PARTICLE_LARGE;
};

/**
 * @brief Handle to a @ref ParticleHandle instance
 * 
 */
struct CAPI_EXPORT tfParticleHandleHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref ParticleType instance
 * 
 */
struct CAPI_EXPORT tfParticleTypeHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref ParticleList instance
 * 
 */
struct CAPI_EXPORT tfParticleListHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref ParticleTypeList instance
 * 
 */
struct CAPI_EXPORT tfParticleTypeListHandle {
    void *tfObj;
};


//////////////////////////////
// tfMappedParticleDataEnum //
//////////////////////////////


/**
 * @brief Initialize an instance
*/
CAPI_FUNC(struct tfMappedParticleDataEnum) tfMappedParticleDataEnum_init();


////////////////////////
// tfParticleTypeSpec //
////////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct tfParticleTypeSpec) tfParticleTypeSpec_init();


/////////////////////////////
// tfParticleTypeStyleSpec //
/////////////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct tfParticleTypeStyleSpec) tfParticleTypeStyleSpec_init();


//////////////////////
// ParticleDynamics //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfParticleDynamics_init(struct tfParticleDynamicsEnumHandle *handle);


///////////////////
// ParticleFlags //
///////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfParticleFlags_init(struct tfParticleFlagsHandle *handle);


////////////////////
// ParticleHandle //
////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param pid particle id
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_init(struct tfParticleHandleHandle *handle, unsigned int pid);

/**
 * @brief Destroys the handle instance
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_destroy(struct tfParticleHandleHandle *handle);

/**
 * @brief Gets the particle type of this handle. 
 * 
 * @param handle populated handle
 * @param typeHandle type handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getType(struct tfParticleHandleHandle *handle, struct tfParticleTypeHandle *typeHandle);

/**
 * @brief Get a summary string of the particle
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_str(struct tfParticleHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Splits a single particle into two. 
 * 
 * The new particle is placed along a randomly selected orientation. 
 * 
 * The two resulting particles have the same mass and volume. 
 * 
 * If species are attached to the split particle, then the amount of species 
 * is allocated to the two resulting particles such that their species concentrations 
 * are the same as the split particle. Species are conserved. 
 * 
 * The two resulting particles have the type of the split particle. 
 * 
 * The two resulting particles are placed in contact. 
 * 
 * The center of mass of the two resulting particles is the same as that of the split particle. 
 * 
 * The combined mass and volume of the two resulting particles are the same as those of the split particle. 
 * 
 * @param handle populated handle
 * @param newParticleHandle new particle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_split(struct tfParticleHandleHandle *handle, struct tfParticleHandleHandle *newParticleHandle);

/**
 * @brief Splits a single particle into two
 * and optionally with different resulting particle types. 
 * 
 * The new particle is placed along a randomly selected orientation. 
 * 
 * The two resulting particles have the same mass and volume. 
 * 
 * If species are attached to the split particle, then the amount of species 
 * is allocated to the two resulting particles such that their species concentrations 
 * are the same as the split particle. Species are conserved. 
 * 
 * The two resulting particles have the type of the split particle unless otherwise specified. 
 * 
 * The two resulting particles are placed in contact. 
 * 
 * The center of mass of the two resulting particles is the same as that of the split particle. 
 * 
 * The combined mass and volume of the two resulting particles are the same as those of the split particle. 
 * 
 * @param handle populated handle
 * @param newParticleHandle new particle handle to populate
 * @param parentTypeHandle optional type of the split particle after the split (NULL specifies default)
 * @param childTypeHandle optional type of the new particle (NULL specifies default)
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_splitTypes(
    struct tfParticleHandleHandle* handle, 
    struct tfParticleHandleHandle* newParticleHandle,
    struct tfParticleTypeHandle* parentTypeHandle, 
    struct tfParticleTypeHandle* childTypeHandle
);

/**
 * @brief Destroys the particle, and removes it from inventory. 
 * 
 * Subsequent references to a destroyed particle result in an error.
 * @param handle 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_destroyParticle(struct tfParticleHandleHandle *handle);

/**
 * @brief Calculates the particle's coordinates in spherical coordinates relative to the center of the universe. 
 * 
 * @param handle populated handle
 * @param position 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_sphericalPosition(struct tfParticleHandleHandle *handle, tfFloatP_t **position);

/**
 * @brief Calculates the particle's coordinates in spherical coordinates relative to a point
 * 
 * @param handle populated handle
 * @param origin relative point
 * @param position 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_sphericalPositionPoint(struct tfParticleHandleHandle *handle, tfFloatP_t *origin, tfFloatP_t **position);

/**
 * @brief Computes the relative position with respect to an origin while. 
 * 
 * @param handle populated handle
 * @param origin relative point
 * @param position 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_relativePosition(struct tfParticleHandleHandle *handle, tfFloatP_t *origin, tfFloatP_t **position);

/**
 * @brief Dynamically changes the *type* of an object. We can change the type of a 
 * ParticleType-derived object to anyther pre-existing ParticleType-derived 
 * type. What this means is that if we have an object of say type 
 * *A*, we can change it to another type, say *B*, and and all of the forces 
 * and processes that acted on objects of type A stip and the forces and 
 * processes defined for type B now take over. 
 * 
 * @param handle populated handle
 * @param typeHandle new particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_become(struct tfParticleHandleHandle *handle, struct tfParticleTypeHandle *typeHandle);

/**
 * @brief Get the particles within a distance of a particle
 * 
 * @param handle populated handle
 * @param distance distance
 * @param neighbors neighbors
 * @param numNeighbors number of neighbors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_neighborsD(
    struct tfParticleHandleHandle *handle, 
    tfFloatP_t distance, 
    struct tfParticleHandleHandle **neighbors, 
    int *numNeighbors
);

/**
 * @brief Get the particles of a set of types within the global cutoff distance
 * 
 * @param handle populated handle
 * @param ptypes particle types
 * @param numTypes number of particle types
 * @param neighbors neighbors
 * @param numNeighbors number of neighbors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_neighborsT(
    struct tfParticleHandleHandle *handle, 
    struct tfParticleTypeHandle *ptypes, 
    int numTypes, 
    struct tfParticleHandleHandle **neighbors, 
    int *numNeighbors
);

/**
 * @brief Get the particles of a set of types within a distance of a particle
 * 
 * @param handle populated handle
 * @param distance distance
 * @param ptypes particle types
 * @param numTypes number of particle types
 * @param neighbors neighbors
 * @param numNeighbors number of neighbors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_neighborsDT(
    struct tfParticleHandleHandle *handle, 
    tfFloatP_t distance, 
    struct tfParticleTypeHandle *ptypes, 
    int numTypes, 
    struct tfParticleHandleHandle **neighbors, 
    int *numNeighbors
);

/**
 * @brief Get a list of all bonded neighbors. 
 * 
 * @param handle populated handle
 * @param neighbors neighbors
 * @param numNeighbors number of neighbors
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getBondedNeighbors(
    struct tfParticleHandleHandle *handle, 
    struct tfParticleHandleHandle **neighbors, 
    int *numNeighbors
);

/**
 * @brief Calculates the distance to another particle
 * 
 * @param handle populated handle
 * @param other populated handle of another particle
 * @param distance distance
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_distance(struct tfParticleHandleHandle *handle, struct tfParticleHandleHandle *other, tfFloatP_t *distance);

/**
 * @brief Get the particle mass
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getMass(struct tfParticleHandleHandle *handle, tfFloatP_t *mass);

/**
 * @brief Set the particle mass
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setMass(struct tfParticleHandleHandle *handle, tfFloatP_t mass);

/**
 * @brief Test whether the particle is frozen
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getFrozen(struct tfParticleHandleHandle *handle, bool *frozen);

/**
 * @brief Set whether the particle is frozen
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setFrozen(struct tfParticleHandleHandle *handle, bool frozen);

/**
 * @brief Test whether the particle is frozen along the x-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getFrozenX(struct tfParticleHandleHandle *handle, bool *frozen);

/**
 * @brief Set whether the particle is frozen along the x-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setFrozenX(struct tfParticleHandleHandle *handle, bool frozen);

/**
 * @brief Test whether the particle is frozen along the y-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getFrozenY(struct tfParticleHandleHandle *handle, bool *frozen);

/**
 * @brief Set whether the particle is frozen along the y-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setFrozenY(struct tfParticleHandleHandle *handle, bool frozen);

/**
 * @brief Test whether the particle is frozen along the z-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getFrozenZ(struct tfParticleHandleHandle *handle, bool *frozen);

/**
 * @brief Set whether the particle is frozen along the z-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setFrozenZ(struct tfParticleHandleHandle *handle, bool frozen);

/**
 * @brief Get the particle style. Fails if no style is set.
 * 
 * @param handle populated handle
 * @param style handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getStyle(struct tfParticleHandleHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Test whether the particle has a style
 * 
 * @param handle populated handle
 * @param hasStyle flag signifying whether the particle has a style
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_hasStyle(struct tfParticleHandleHandle *handle, bool *hasStyle);

/**
 * @brief Set the particle style
 * 
 * @param handle populated handle
 * @param style style
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setStyle(struct tfParticleHandleHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Get the particle age
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getAge(struct tfParticleHandleHandle *handle, tfFloatP_t *age);

/**
 * @brief Get the particle radius
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getRadius(struct tfParticleHandleHandle *handle, tfFloatP_t *radius);

/**
 * @brief Set the particle radius
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setRadius(struct tfParticleHandleHandle *handle, tfFloatP_t radius);

/**
 * @brief Get the particle name
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getName(struct tfParticleHandleHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Get the particle second name
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getName2(struct tfParticleHandleHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Get the particle position
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getPosition(struct tfParticleHandleHandle *handle, tfFloatP_t **position);

/**
 * @brief Set the particle position
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setPosition(struct tfParticleHandleHandle *handle, tfFloatP_t *position);

/**
 * @brief Get the particle velocity
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getVelocity(struct tfParticleHandleHandle *handle, tfFloatP_t **velocity);

/**
 * @brief Set the particle velocity
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setVelocity(struct tfParticleHandleHandle *handle, tfFloatP_t *velocity);

/**
 * @brief Get the particle force
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getForce(struct tfParticleHandleHandle *handle, tfFloatP_t **force);

/**
 * @brief Get the particle initial force
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getForceInit(struct tfParticleHandleHandle *handle, tfFloatP_t **force);

/**
 * @brief Set the particle initial force
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setForceInit(struct tfParticleHandleHandle *handle, tfFloatP_t *force);

/**
 * @brief Get the particle id
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getId(struct tfParticleHandleHandle *handle, int *pid);

/**
 * @brief Get the particle type id
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getTypeId(struct tfParticleHandleHandle *handle, int *tid);

/**
 * @brief Get the particle cluster id. -1 if particle is not a cluster
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getClusterId(struct tfParticleHandleHandle *handle, int *cid);

/**
 * @brief Get the particle flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getFlags(struct tfParticleHandleHandle *handle, int *flags);

/**
 * @brief Test whether a particle has species
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_hasSpecies(struct tfParticleHandleHandle *handle, bool *flag);

/**
 * @brief Get the state vector. Fails if the particle does not have a state vector
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_getSpecies(struct tfParticleHandleHandle *handle, struct tfStateStateVectorHandle *svec);

/**
 * @brief Set the state vector. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_setSpecies(struct tfParticleHandleHandle *handle, struct tfStateStateVectorHandle *svec);

/**
 * @brief Convert the particle to a cluster; fails if particle is not a cluster
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_toCluster(struct tfParticleHandleHandle *handle, struct tfClusterParticleHandleHandle *chandle);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleHandle_toString(struct tfParticleHandleHandle *handle, char **str, unsigned int *numChars);

/** Test whether lhs < rhs */
CAPI_FUNC(HRESULT) tfParticleHandle_lt(struct tfParticleHandleHandle *lhs, struct tfParticleHandleHandle *rhs, bool *result);

/** Test whether lhs > rhs */
CAPI_FUNC(HRESULT) tfParticleHandle_gt(struct tfParticleHandleHandle *lhs, struct tfParticleHandleHandle *rhs, bool *result);

/** Test whether lhs <= rhs */
CAPI_FUNC(HRESULT) tfParticleHandle_le(struct tfParticleHandleHandle *lhs, struct tfParticleHandleHandle *rhs, bool *result);

/** Test whether lhs >= rhs */
CAPI_FUNC(HRESULT) tfParticleHandle_ge(struct tfParticleHandleHandle *lhs, struct tfParticleHandleHandle *rhs, bool *result);

/** Test whether lhs == rhs */
CAPI_FUNC(HRESULT) tfParticleHandle_eq(struct tfParticleHandleHandle *lhs, struct tfParticleHandleHandle *rhs, bool *result);

/** Test whether lhs != rhs */
CAPI_FUNC(HRESULT) tfParticleHandle_ne(struct tfParticleHandleHandle *lhs, struct tfParticleHandleHandle *rhs, bool *result);


//////////////////
// ParticleType //
//////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_init(struct tfParticleTypeHandle *handle);

/**
 * @brief Initialize an instance from a definition
 * 
 * @param handle handle to populate
 * @param pdef definition
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfParticleType_initD(struct tfParticleTypeHandle *handle, struct tfParticleTypeSpec pdef);

/**
 * @brief Get the type name.
 * 
 * @param handle populated handle
 * @param name type name
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getName(struct tfParticleTypeHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Set the type name. Throws an error if the type is already registered.
 * 
 * @param handle populated handle
 * @param name type name
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setName(struct tfParticleTypeHandle *handle, const char *name);

/**
 * @brief Get the type id. -1 if not registered.
 * 
 * @param handle populated handle
 * @param id type id
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getId(struct tfParticleTypeHandle *handle, int *id);

/**
 * @brief Get the type flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getTypeFlags(struct tfParticleTypeHandle *handle, unsigned int *flags);

/**
 * @brief Set the type flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setTypeFlags(struct tfParticleTypeHandle *handle, unsigned int flags);

/**
 * @brief Get the particle flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getParticleFlags(struct tfParticleTypeHandle *handle, unsigned int *flags);

/**
 * @brief Set the particle flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setParticleFlags(struct tfParticleTypeHandle *handle, unsigned int flags);

/**
 * @brief Get the type style. 
 * 
 * @param handle populated handle
 * @param style handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getStyle(struct tfParticleTypeHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Set the type style
 * 
 * @param handle populated handle
 * @param style style
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setStyle(struct tfParticleTypeHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Test whether the type has species
 * 
 * @param handle populated handle
 * @param flag flag signifying whether the type has species
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_hasSpecies(struct tfParticleTypeHandle *handle, bool *flag);

/**
 * @brief Get the type species. Fails if the type does not have species
 * 
 * @param handle populated handle
 * @param slist species list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getSpecies(struct tfParticleTypeHandle *handle, struct tfStateSpeciesListHandle *slist);

/**
 * @brief Set the type species. 
 * 
 * @param handle populated handle
 * @param slist species list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setSpecies(struct tfParticleTypeHandle *handle, struct tfStateSpeciesListHandle *slist);

/**
 * @brief Get the type mass
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getMass(struct tfParticleTypeHandle *handle, tfFloatP_t *mass);

/**
 * @brief Set the type mass
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setMass(struct tfParticleTypeHandle *handle, tfFloatP_t mass);

/**
 * @brief Get the type radius
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getRadius(struct tfParticleTypeHandle *handle, tfFloatP_t *radius);

/**
 * @brief Set the type radius
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setRadius(struct tfParticleTypeHandle *handle, tfFloatP_t radius);

/**
 * @brief Get the kinetic energy of all particles of this type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getKineticEnergy(struct tfParticleTypeHandle *handle, tfFloatP_t *kinetic_energy);

/**
 * @brief Get the potential energy of all particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getPotentialEnergy(struct tfParticleTypeHandle *handle, tfFloatP_t *potential_energy);

/**
 * @brief Get the target energy of all particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getTargetEnergy(struct tfParticleTypeHandle *handle, tfFloatP_t *target_energy);

/**
 * @brief Set the target energy of all particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setTargetEnergy(struct tfParticleTypeHandle *handle, tfFloatP_t target_energy);

/**
 * @brief Get the default minimum radius of this type. 
 * 
 * If a split event occurs, resulting particles will have a radius 
 * at least as great as this value. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getMinimumRadius(struct tfParticleTypeHandle *handle, tfFloatP_t *minimum_radius);

/**
 * @brief Set the default minimum radius of this type. 
 * 
 * If a split event occurs, resulting particles will have a radius 
 * at least as great as this value. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setMinimumRadius(struct tfParticleTypeHandle *handle, tfFloatP_t minimum_radius);

/**
 * @brief Get the default dynamics of particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getDynamics(struct tfParticleTypeHandle *handle, unsigned char *dynamics);

/**
 * @brief Set the default dynamics of particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setDynamics(struct tfParticleTypeHandle *handle, unsigned char dynamics);

/**
 * @brief Get the number of particles that are a member of this type.
 * 
 * @param handle populated handle
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getNumParts(struct tfParticleTypeHandle *handle, int *numParts);

/**
 * @brief Get the i'th particle that's a member of this type.
 * 
 * @param handle populated handle
 * @param i index of particle to get
 * @param phandle particle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getParticle(struct tfParticleTypeHandle *handle, int i, struct tfParticleHandleHandle *phandle);

/**
 * @brief Test whether this type is a cluster type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_isCluster(struct tfParticleTypeHandle *handle, bool *isCluster);

/**
 * @brief Convert the type to a cluster; fails if the type is not a cluster
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_toCluster(struct tfParticleTypeHandle *handle, struct tfClusterParticleTypeHandle *chandle);

/**
 * @brief Particle constructor.
 * 
 * @param handle populated handle
 * @param pid id of created particle
 * @param position pointer to 3-element array, or NULL for a random position
 * @param velocity pointer to 3-element array, or NULL for a random velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_createParticle(struct tfParticleTypeHandle *handle, int *pid, tfFloatP_t *position, tfFloatP_t *velocity);

/**
 * @brief Particle constructor.
 * 
 * @param handle populated handle
 * @param pid id of created particle
 * @param str JSON string defining a particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_createParticleS(struct tfParticleTypeHandle *handle, int *pid, const char *str);

/**
 * @brief Particle factory constructor, for making lots of particles quickly. 
 * 
 * At minimum, arguments must specify the number of particles to create, whether 
 * specified explicitly or through one or more vector arguments.
 * 
 * @param handle populated handle
 * @param pids ids of created particle, optional
 * @param nr_parts number of particles to create
 * @param positions initial particle positions, optional
 * @param velocities initial particle velocities, optional
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_factory(struct tfParticleTypeHandle *handle, int **pids, unsigned int nr_parts, tfFloatP_t *positions, tfFloatP_t *velocities);

/**
 * @brief Particle type constructor. 
 * 
 * New type is constructed from the definition of the calling type. 
 * 
 * @param handle populated handle
 * @param _name name of the new type
 * @param newTypehandle handle to populate with new type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfParticleType_newType(struct tfParticleTypeHandle *handle, const char *_name, struct tfParticleTypeHandle *newTypehandle);

/**
 * @brief Test whether the type has an id
 * 
 * @param handle populated handle
 * @param pid particle id to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_hasPartId(struct tfParticleTypeHandle *handle, int pid, bool *result);

/**
 * @brief Test whether the type has a particle
 * 
 * @param handle populated handle
 * @param part particle to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_hasPart(struct tfParticleTypeHandle *handle, struct tfParticleHandleHandle *part, bool *result);

/**
 * @brief Registers a type with the engine.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_registerType(struct tfParticleTypeHandle *handle);

/**
 * @brief Tests whether this type is registered
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_isRegistered(struct tfParticleTypeHandle *handle, bool *isRegistered);

/**
 * @brief Get a summary string of the type
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_str(struct tfParticleTypeHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Test whether this type is frozen
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getFrozen(struct tfParticleTypeHandle *handle, bool *frozen);

/**
 * @brief Set whether this type is frozen
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setFrozen(struct tfParticleTypeHandle *handle, bool frozen);

/**
 * @brief Test whether this type is frozen along the x-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getFrozenX(struct tfParticleTypeHandle *handle, bool *frozen);

/**
 * @brief Set whether this type is frozen along the x-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setFrozenX(struct tfParticleTypeHandle *handle, bool frozen);

/**
 * @brief Test whether this type is frozen along the y-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getFrozenY(struct tfParticleTypeHandle *handle, bool *frozen);

/**
 * @brief Set whether this type is frozen along the y-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setFrozenY(struct tfParticleTypeHandle *handle, bool frozen);

/**
 * @brief Test whether this type is frozen along the z-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getFrozenZ(struct tfParticleTypeHandle *handle, bool *frozen);

/**
 * @brief Set whether this type is frozen along the z-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setFrozenZ(struct tfParticleTypeHandle *handle, bool frozen);

/**
 * @brief Get the temperature of this type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getTemperature(struct tfParticleTypeHandle *handle, tfFloatP_t *temperature);

/**
 * @brief Get the target temperature of this type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getTargetTemperature(struct tfParticleTypeHandle *handle, tfFloatP_t *temperature);

/**
 * @brief Set the target temperature of this type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_setTargetTemperature(struct tfParticleTypeHandle *handle, tfFloatP_t temperature);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_toString(struct tfParticleTypeHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * The returned type is automatically registered with the engine. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_fromString(struct tfParticleTypeHandle *handle, const char *str);

/** Test whether lhs < rhs */
CAPI_FUNC(HRESULT) tfParticleType_lt(struct tfParticleTypeHandle *lhs, struct tfParticleTypeHandle *rhs, bool *result);

/** Test whether lhs > rhs */
CAPI_FUNC(HRESULT) tfParticleType_gt(struct tfParticleTypeHandle *lhs, struct tfParticleTypeHandle *rhs, bool *result);

/** Test whether lhs <= rhs */
CAPI_FUNC(HRESULT) tfParticleType_le(struct tfParticleTypeHandle *lhs, struct tfParticleTypeHandle *rhs, bool *result);

/** Test whether lhs >= rhs */
CAPI_FUNC(HRESULT) tfParticleType_ge(struct tfParticleTypeHandle *lhs, struct tfParticleTypeHandle *rhs, bool *result);

/** Test whether lhs == rhs */
CAPI_FUNC(HRESULT) tfParticleType_eq(struct tfParticleTypeHandle *lhs, struct tfParticleTypeHandle *rhs, bool *result);

/** Test whether lhs != rhs */
CAPI_FUNC(HRESULT) tfParticleType_ne(struct tfParticleTypeHandle *lhs, struct tfParticleTypeHandle *rhs, bool *result);


//////////////////
// ParticleList //
//////////////////


/**
 * @brief Initialize an empty instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_init(struct tfParticleListHandle *handle);

/**
 * @brief Initialize an instance with an array of particles
 * 
 * @param handle handle to populate
 * @param particles particles to put in the list
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_initP(struct tfParticleListHandle *handle, struct tfParticleHandleHandle **particles, unsigned int numParts);

/**
 * @brief Initialize an instance with an array of particle ids
 * 
 * @param handle handle to populate
 * @param parts particle ids to put in the list
 * @param numParts number of particle ids
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_initI(struct tfParticleListHandle *handle, int *parts, unsigned int numParts);

/**
 * @brief Copy an instance
 * 
 * @param source list to copy
 * @param destination handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_copy(struct tfParticleListHandle *source, struct tfParticleListHandle *destination);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_destroy(struct tfParticleListHandle *handle);

/**
 * @brief Get the particle ids in the list
 * 
 * @param handle populated handle
 * @param parts particle id array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getIds(struct tfParticleListHandle *handle, int **parts);

/**
 * @brief Get the number of particles
 * 
 * @param handle populated handle
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getNumParts(struct tfParticleListHandle *handle, unsigned int *numParts);

/**
 * @brief Free the memory associated with the parts list.
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_free(struct tfParticleListHandle *handle);

/**
 * @brief Insert the given id into the list, returns the index of the item. 
 * 
 * @param handle populated handle
 * @param item id to insert
 * @param index index of the particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_insertI(struct tfParticleListHandle *handle, int item, unsigned int *index);

/**
 * @brief Inserts the given particle into the list, returns the index of the particle. 
 * 
 * @param handle populated handle
 * @param particle particle to insert
 * @param index index of the particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_insertP(struct tfParticleListHandle *handle, struct tfParticleHandleHandle *particle, unsigned int *index);

/**
 * @brief Looks for the item with the given id and deletes it from the list
 * 
 * @param handle populated handle
 * @param id id to remove
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_remove(struct tfParticleListHandle *handle, int id);

/**
 * @brief inserts the contents of another list
 * 
 * @param handle populated handle
 * @param other another list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_extend(struct tfParticleListHandle *handle, struct tfParticleListHandle *other);

/**
 * @brief Test whether the list has an id
 * 
 * @param handle populated handle
 * @param pid id to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_hasId(struct tfParticleListHandle *handle, int pid, bool *result);

/**
 * @brief Test whether the list has a particle
 * 
 * @param handle populated handle
 * @param part particle to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_hasPart(struct tfParticleListHandle *handle, struct tfParticleHandleHandle *part, bool *result);

/**
 * @brief looks for the item at the given index and returns it if found, otherwise returns NULL
 * 
 * @param handle populated handle
 * @param i index of lookup
 * @param item returned item if found
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_item(struct tfParticleListHandle *handle, unsigned int i, struct tfParticleHandleHandle *item);

/**
 * @brief Initialize an instance populated with all current particles
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getAll(struct tfParticleListHandle *handle);

/**
 * @brief Get the virial tensor of the particles
 * 
 * @param handle populated handle
 * @param virial 9-element allocated array, virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getVirial(struct tfParticleListHandle *handle, tfFloatP_t **virial);

/**
 * @brief Get the radius of gyration of the particles
 * 
 * @param handle populated handle
 * @param rog radius of gyration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getRadiusOfGyration(struct tfParticleListHandle *handle, tfFloatP_t *rog);

/**
 * @brief Get the center of mass of the particles
 * 
 * @param handle populated handle
 * @param com 3-element allocated array, center of mass
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getCenterOfMass(struct tfParticleListHandle *handle, tfFloatP_t **com);

/**
 * @brief Get the centroid of the particles
 * 
 * @param handle populated handle
 * @param cent 3-element allocated array, centroid
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getCentroid(struct tfParticleListHandle *handle, tfFloatP_t **cent);

/**
 * @brief Get the moment of inertia of the particles
 * 
 * @param handle populated handle
 * @param moi 9-element allocated array, moment of inertia
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getMomentOfInertia(struct tfParticleListHandle *handle, tfFloatP_t **moi);

/**
 * @brief Get the particle positions
 * 
 * @param handle populated handle
 * @param positions array of 3-element arrays, positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getPositions(struct tfParticleListHandle *handle, tfFloatP_t **positions);

/**
 * @brief Get the particle velocities
 * 
 * @param handle populated handle
 * @param velocities array of 3-element arrays, velocities; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getVelocities(struct tfParticleListHandle *handle, tfFloatP_t **velocities);

/**
 * @brief Get the forces acting on the particles
 * 
 * @param handle populated handle
 * @param forces array of 3-element arrays, forces; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_getForces(struct tfParticleListHandle *handle, tfFloatP_t **forces);

/**
 * @brief Get the spherical coordinates of each particle for a coordinate system at the center of the universe
 * 
 * @param handle populated handle
 * @param coordinates array of 3-element arrays, spherical positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_sphericalPositions(struct tfParticleListHandle *handle, tfFloatP_t **coordinates);

/**
 * @brief Get the spherical coordinates of each particle for a coordinate system with a specified origin
 * 
 * @param handle populated handle
 * @param origin optional origin of coordinates; default is center of universe
 * @param coordinates array of 3-element arrays, spherical positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_sphericalPositionsO(struct tfParticleListHandle *handle, tfFloatP_t *origin, tfFloatP_t **coordinates);

/**
 * @brief Get whether the list owns its data
 * 
 * @param handle populated handle
 * @param result flag signifying whether the list owns its data
*/
CAPI_FUNC(HRESULT) tfParticleList_getOwnsData(struct tfParticleListHandle *handle, bool* result);

/**
 * @brief Set whether the list owns its data
 * 
 * @param handle populated handle
 * @param flag flag signifying whether the list owns its data
*/
CAPI_FUNC(HRESULT) tfParticleList_setOwnsData(struct tfParticleListHandle *handle, bool flag);

/**
 * @brief Get whether the list is mutable
 * 
 * @param handle populated handle
 * @param result flag signifying whether the list is mutable
*/
CAPI_FUNC(HRESULT) tfParticleList_getMutable(struct tfParticleListHandle *handle, bool* result);

/**
 * @brief Set whether the list is mutable
 * 
 * @param handle populated handle
 * @param flag flag signifying whether the list is mutable
*/
CAPI_FUNC(HRESULT) tfParticleList_setMutable(struct tfParticleListHandle *handle, bool flag);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleList_toString(struct tfParticleListHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfParticleList_fromString(struct tfParticleListHandle *handle, const char *str);


//////////////////////
// ParticleTypeList //
//////////////////////


/**
 * @brief Initialize an empty instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_init(struct tfParticleTypeListHandle *handle);

/**
 * @brief Initialize an instance with an array of particle types
 * 
 * @param handle handle to populate
 * @param parts particle types to put in the list
 * @param numParts number of particle types
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_initP(struct tfParticleTypeListHandle *handle, struct tfParticleTypeHandle **parts, unsigned int numParts);

/**
 * @brief Initialize an instance with an array of particle type ids
 * 
 * @param handle handle to populate
 * @param parts particle type ids to put in the list
 * @param numParts number of particle type ids
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_initI(struct tfParticleTypeListHandle *handle, int *parts, unsigned int numParts);

/**
 * @brief Copy an instance
 * 
 * @param source list to copy
 * @param destination handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_copy(struct tfParticleTypeListHandle *source, struct tfParticleTypeListHandle *destination);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_destroy(struct tfParticleTypeListHandle *handle);

/**
 * @brief Get the particle type ids in the list
 * 
 * @param handle populated handle
 * @param parts particle type id array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getIds(struct tfParticleTypeListHandle *handle, int **parts);

/**
 * @brief Get the number of particle types
 * 
 * @param handle populated handle
 * @param numParts number of particle types
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getNumParts(struct tfParticleTypeListHandle *handle, unsigned int *numParts);

/**
 * @brief Free the memory associated with the list.
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_free(struct tfParticleTypeListHandle *handle);

/**
 * @brief Insert the given id into the list, returns the index of the item. 
 * 
 * @param handle populated handle
 * @param item id to insert
 * @param index index of the particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_insertI(struct tfParticleTypeListHandle *handle, int item, unsigned int *index);

/**
 * @brief Inserts the given particle type into the list, returns the index of the particle. 
 * 
 * @param handle populated handle
 * @param ptype particle type to insert
 * @param index index of the particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_insertP(struct tfParticleTypeListHandle *handle, struct tfParticleTypeHandle *ptype, unsigned int *index);

/**
 * @brief Looks for the item with the given id and deletes it from the list
 * 
 * @param handle populated handle
 * @param id id to remove
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_remove(struct tfParticleTypeListHandle *handle, int id);

/**
 * @brief inserts the contents of another list
 * 
 * @param handle populated handle
 * @param other another list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_extend(struct tfParticleTypeListHandle *handle, struct tfParticleTypeListHandle *other);

/**
 * @brief Test whether the list has an id
 * 
 * @param handle populated handle
 * @param pid id to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_hasId(struct tfParticleTypeListHandle *handle, int pid, bool *result);

/**
 * @brief Test whether the list has a particle type
 * 
 * @param handle populated handle
 * @param ptype type to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_hasType(struct tfParticleTypeListHandle *handle, struct tfParticleTypeHandle *ptype, bool *result);

/**
 * @brief Test whether the list has a particle
 * 
 * @param handle populated handle
 * @param part particle to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_hasPart(struct tfParticleTypeListHandle *handle, struct tfParticleHandleHandle *part, bool *result);

/**
 * @brief looks for the item at the given index and returns it if found, otherwise returns NULL
 * 
 * @param handle populated handle
 * @param i index of lookup
 * @param item returned item if found
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_item(struct tfParticleTypeListHandle *handle, unsigned int i, struct tfParticleTypeHandle *item);

/**
 * @brief Initialize an instance populated with all current particles
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getAll(struct tfParticleTypeListHandle *handle);

/**
 * @brief Get the virial tensor of the particles
 * 
 * @param handle populated handle
 * @param virial 9-element allocated array, virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getVirial(struct tfParticleTypeListHandle *handle, tfFloatP_t *virial);

/**
 * @brief Get the radius of gyration of the particles
 * 
 * @param handle populated handle
 * @param rog radius of gyration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getRadiusOfGyration(struct tfParticleTypeListHandle *handle, tfFloatP_t *rog);

/**
 * @brief Get the center of mass of the particles
 * 
 * @param handle populated handle
 * @param com 3-element allocated array, center of mass
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getCenterOfMass(struct tfParticleTypeListHandle *handle, tfFloatP_t **com);

/**
 * @brief Get the centroid of the particles
 * 
 * @param handle populated handle
 * @param cent 3-element allocated array, centroid
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getCentroid(struct tfParticleTypeListHandle *handle, tfFloatP_t **cent);

/**
 * @brief Get the moment of inertia of the particles
 * 
 * @param handle populated handle
 * @param moi 9-element allocated array, moment of inertia
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getMomentOfInertia(struct tfParticleTypeListHandle *handle, tfFloatP_t **moi);

/**
 * @brief Get the particle positions
 * 
 * @param handle populated handle
 * @param positions array of 3-element arrays, positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getPositions(struct tfParticleTypeListHandle *handle, tfFloatP_t **positions);

/**
 * @brief Get the particle velocities
 * 
 * @param handle populated handle
 * @param velocities array of 3-element arrays, velocities; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getVelocities(struct tfParticleTypeListHandle *handle, tfFloatP_t **velocities);

/**
 * @brief Get the forces acting on the particles
 * 
 * @param handle populated handle
 * @param forces array of 3-element arrays, forces; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getForces(struct tfParticleTypeListHandle *handle, tfFloatP_t **forces);

/**
 * @brief Get the spherical coordinates of each particle for a coordinate system at the center of the universe
 * 
 * @param handle populated handle
 * @param coordinates array of 3-element arrays, spherical positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_sphericalPositions(struct tfParticleTypeListHandle *handle, tfFloatP_t **coordinates);

/**
 * @brief Get the spherical coordinates of each particle for a coordinate system with a specified origin
 * 
 * @param handle populated handle
 * @param origin optional origin of coordinates; default is center of universe
 * @param coordinates array of 3-element arrays, spherical positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_sphericalPositionsO(struct tfParticleTypeListHandle *handle, tfFloatP_t *origin, tfFloatP_t **coordinates);

/**
 * @brief Get whether the list owns its data
 * 
 * @param handle populated handle
 * @param result flag signifying whether the list owns its data
*/
CAPI_FUNC(HRESULT) tfParticleTypeList_getOwnsData(struct tfParticleTypeListHandle *handle, bool* result);

/**
 * @brief Set whether the list owns its data
 * 
 * @param handle populated handle
 * @param flag flag signifying whether the list owns its data
*/
CAPI_FUNC(HRESULT) tfParticleTypeList_setOwnsData(struct tfParticleTypeListHandle *handle, bool flag);

/**
 * @brief Get whether the list is mutable
 * 
 * @param handle populated handle
 * @param result flag signifying whether the list is mutable
*/
CAPI_FUNC(HRESULT) tfParticleTypeList_getMutable(struct tfParticleTypeListHandle *handle, bool* result);

/**
 * @brief Set whether the list is mutable
 * 
 * @param handle populated handle
 * @param flag flag signifying whether the list is mutable
*/
CAPI_FUNC(HRESULT) tfParticleTypeList_setMutable(struct tfParticleTypeListHandle *handle, bool flag);

/**
 * @brief Get a particle list populated with particles of all current particle types
 * 
 * @param handle populated handle
 * @param plist handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_getParticles(struct tfParticleTypeListHandle *handle, struct tfParticleListHandle *plist);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_toString(struct tfParticleTypeListHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfParticleTypeList_fromString(struct tfParticleTypeListHandle *handle, const char *str);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Iterates over all parts, does a verify
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticle_Verify();

/**
 * @brief Get a registered particle type by type name
 * 
 * @param handle handle to populate
 * @param name name of particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_FindFromName(struct tfParticleTypeHandle *handle, const char* name);

/**
 * @brief Get a registered particle type by type id
 * 
 * @param handle handle to populate
 * @param pid id of type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfParticleType_getFromId(struct tfParticleTypeHandle *handle, unsigned int pid);

/**
 * @brief Get an array of available particle type colors
 * 
 * @return unsigned int *
 */
CAPI_FUNC(unsigned int*) tfParticle_Colors();

#endif // _WRAPS_C_TFCPARTICLE_H_