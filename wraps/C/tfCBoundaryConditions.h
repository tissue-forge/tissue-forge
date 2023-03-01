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
 * @file tfCBoundaryConditions.h
 * 
 */

#ifndef _WRAPS_C_TFCBOUNDARYCONDITIONS_H_
#define _WRAPS_C_TFCBOUNDARYCONDITIONS_H_

#include "tf_port_c.h"

#include "tfCParticle.h"
#include "tfCPotential.h"

// Handles

/**
 * @brief Enums for kind of boundary conditions along directions
 * 
 */
struct CAPI_EXPORT tfBoundaryConditionSpaceKindHandle {
    unsigned int SPACE_PERIODIC_X;
    unsigned int SPACE_PERIODIC_Y;
    unsigned int SPACE_PERIODIC_Z;
    unsigned int SPACE_PERIODIC_FULL;
    unsigned int SPACE_FREESLIP_X;
    unsigned int SPACE_FREESLIP_Y;
    unsigned int SPACE_FREESLIP_Z;
    unsigned int SPACE_FREESLIP_FULL;
};

/**
 * @brief Enums for kind of individual boundary condition
 * 
 */
struct CAPI_EXPORT tfBoundaryConditionKindHandle {
    unsigned int BOUNDARY_PERIODIC;
    unsigned int BOUNDARY_FREESLIP;
    unsigned int BOUNDARY_RESETTING;
};

/**
 * @brief Handle to a @ref BoundaryCondition instance
 * 
 */
struct CAPI_EXPORT tfBoundaryConditionHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref BoundaryConditions instance
 * 
 */
struct CAPI_EXPORT tfBoundaryConditionsHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref BoundaryConditionsArgsContainer instance
 * 
 */
struct CAPI_EXPORT tfBoundaryConditionsArgsContainerHandle {
    void *tfObj;
};


////////////////////////////////
// BoundaryConditionSpaceKind //
////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return HRESULT S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionSpaceKind_init(struct tfBoundaryConditionSpaceKindHandle *handle);


///////////////////////////
// BoundaryConditionKind //
///////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionKind_init(struct tfBoundaryConditionKindHandle *handle);


///////////////////////
// BoundaryCondition //
///////////////////////


/**
 * @brief Get the id of a boundary condition
 * 
 * @param handle populated handle
 * @param bid boundary condition id
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_getId(struct tfBoundaryConditionHandle *handle, int *bid);

/**
 * @brief Get the velocity of a boundary condition
 * 
 * @param handle populated handle
 * @param velocity 3-element allocated array of velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_getVelocity(struct tfBoundaryConditionHandle *handle, tfFloatP_t **velocity);

/**
 * @brief Set the velocity of a boundary condition
 * 
 * @param handle populated handle
 * @param velocity 3-element allocated array of velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_setVelocity(struct tfBoundaryConditionHandle *handle, tfFloatP_t *velocity);

/**
 * @brief Get the restore coefficient of a boundary condition
 * 
 * @param handle populated handle
 * @param restore restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_getRestore(struct tfBoundaryConditionHandle *handle, tfFloatP_t *restore);

/**
 * @brief Set the restore coefficient of a boundary condition
 * 
 * @param handle populated handle
 * @param restore restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_setRestore(struct tfBoundaryConditionHandle *handle, tfFloatP_t restore);

/**
 * @brief Get the normal of a boundary condition
 * 
 * @param handle populated handle
 * @param normal 3-element allocated array of normal
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_getNormal(struct tfBoundaryConditionHandle *handle, tfFloatP_t **normal);

/**
 * @brief Get the equivalent radius of a boundary condition
 * 
 * @param handle populated handle
 * @param radius equivalent radius
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_getRadius(struct tfBoundaryConditionHandle *handle, tfFloatP_t *radius);

/**
 * @brief Set the equivalent radius of a boundary condition
 * 
 * @param handle populated handle
 * @param radius equivalent radius
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_setRadius(struct tfBoundaryConditionHandle *handle, tfFloatP_t radius);

/**
 * @brief Set the potential of a boundary condition for a particle type
 * 
 * @param handle populated handle
 * @param partHandle particle type
 * @param potHandle boundary potential
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryCondition_setPotential(struct tfBoundaryConditionHandle *handle, struct tfParticleTypeHandle *partHandle, struct tfPotentialHandle *potHandle);


////////////////////////
// BoundaryConditions //
////////////////////////


/**
 * @brief Get the top boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditions_getTop(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle);

/**
 * @brief Get the bottom boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditions_getBottom(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle);

/**
 * @brief Get the left boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditions_getLeft(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle);

/**
 * @brief Get the right boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditions_getRight(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle);

/**
 * @brief Get the front boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditions_getFront(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle);

/**
 * @brief Get the back boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditions_getBack(struct tfBoundaryConditionsHandle *handle, struct tfBoundaryConditionHandle *bchandle);

/**
 * @brief Set the potential on all boundaries for a particle type
 * 
 * @param handle populated handle
 * @param partHandle particle type
 * @param potHandle potential
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditions_setPotential(struct tfBoundaryConditionsHandle *handle, struct tfParticleTypeHandle *partHandle, struct tfPotentialHandle *potHandle);


/////////////////////////////////////
// BoundaryConditionsArgsContainer //
/////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_init(struct tfBoundaryConditionsArgsContainerHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_destroy(struct tfBoundaryConditionsArgsContainerHandle *handle);

/**
 * @brief Test whether a value type is applied to all boundaries
 * 
 * @param handle populated handle
 * @param has flag for whether a value type is applied to all boundaries
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_hasValueAll(struct tfBoundaryConditionsArgsContainerHandle *handle, bool *has);

/**
 * @brief Get the boundary type value on all boundaries
 * 
 * @param handle populated handle
 * @param _bcValue boundary type value on all boundaries
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_getValueAll(struct tfBoundaryConditionsArgsContainerHandle *handle, unsigned int *_bcValue);

/**
 * @brief Set the boundary type value on all boundaries
 * 
 * @param handle populated handle
 * @param _bcValue boundary type value on all boundaries
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_setValueAll(struct tfBoundaryConditionsArgsContainerHandle *handle, unsigned int _bcValue);

/**
 * @brief Test whether a boundary has a boundary type value
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param has flag signifying whether a boundary has a boundary type value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_hasValue(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has);

/**
 * @brief Get the boundary type value of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param value boundary type value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_getValue(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, unsigned int *value);

/**
 * @brief Set the boundary type value of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param value boundary type value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_setValue(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, unsigned int value);

/**
 * @brief Test whether a boundary has a velocity
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param has flag signifying whether a boundary has a velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_hasVelocity(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has);

/**
 * @brief Get the velocity of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param velocity boundary velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_getVelocity(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, tfFloatP_t **velocity);

/**
 * @brief Set the velocity of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param velocity boundary velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_setVelocity(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, tfFloatP_t *velocity);

/**
 * @brief Test whether a boundary has a restore coefficient
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param has flag signifying whether a boundary has a restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_hasRestore(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has);

/**
 * @brief Get the restore coefficient of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param restore restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_getRestore(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, tfFloatP_t *restore);

/**
 * @brief Set the restore coefficient of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param restore restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBoundaryConditionsArgsContainer_setRestore(struct tfBoundaryConditionsArgsContainerHandle *handle, const char *name, tfFloatP_t restore);

#endif // _WRAPS_C_TFCBOUNDARYCONDITIONS_H_