/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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
 * @file tfCBond.h
 * 
 */

#ifndef _WRAPS_C_TFCBOND_H_
#define _WRAPS_C_TFCBOND_H_

#include "tf_port_c.h"

#include "tfCParticle.h"
#include "tfCPotential.h"

// Handles

/**
 * @brief Handle to a @ref BondHandle instance
 * 
 */
struct CAPI_EXPORT tfBondHandleHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref AngleHandle instance
 * 
 */
struct CAPI_EXPORT tfAngleHandleHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref DihedralHandle instance
 * 
 */
struct CAPI_EXPORT tfDihedralHandleHandle {
    void *tfObj;
};


////////////////
// BondHandle //
////////////////


/**
 * @brief Construct a new bond handle from an existing bond id
 * 
 * @param handle handle to populate
 * @param id id of existing bond
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_init(struct tfBondHandleHandle *handle, unsigned int id);

/**
 * @brief Construct a new bond handle and underlying bond. 
 * 
 * @param handle handle to populate
 * @param potential bond potential
 * @param i ith particle
 * @param j jth particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_create(
    struct tfBondHandleHandle *handle, 
    struct tfPotentialHandle *potential,
    struct tfParticleHandleHandle *i, 
    struct tfParticleHandleHandle *j
);

/**
 * @brief Get the id. Returns -1 if the underlying bond is invalid
 * 
 * @param handle populated handle
 * @param id bond id
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getId(struct tfBondHandleHandle *handle, int *id);

/**
 * @brief Get a summary string of the bond
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBondHandle_str(struct tfBondHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Check the validity of the handle
 * 
 * @param handle populated handle
 * @param flag true if OK
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_check(struct tfBondHandleHandle *handle, bool *flag);

/**
 * @brief Destroy the bond. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_destroy(struct tfBondHandleHandle *handle);

/**
 * @brief Tests whether this bond decays
 * 
 * @param handle populated handle
 * @param flag true when the bond should decay
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_decays(struct tfBondHandleHandle *handle, bool *flag);

/**
 * @brief Test whether the bond has an id
 * 
 * @param handle populated handle
 * @param pid id to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBondHandle_hasPartId(struct tfBondHandleHandle *handle, int pid, bool *result);

/**
 * @brief Test whether the bond has a particle
 * 
 * @param handle populated handle
 * @param part particle to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBondHandle_hasPart(struct tfBondHandleHandle *handle, struct tfParticleHandleHandle *part, bool *result);

/**
 * @brief Get the current energy of the bond
 * 
 * @param handle populated handle
 * @param value current energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getEnergy(struct tfBondHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Get the ids of the particles of the bond
 * 
 * @param handle populated handle
 * @param parti ith particle; -1 if no particle
 * @param partj jth particle; -1 if no particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getParts(struct tfBondHandleHandle *handle, int *parti, int *partj);

/**
 * @brief Get a list of the particles of the bond
 * 
 * @param handle populated handle
 * @param plist handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBondHandle_getPartList(struct tfBondHandleHandle *handle, struct tfParticleListHandle *plist);

/**
 * @brief Get the potential of the bond
 * 
 * @param handle populated handle
 * @param potential bond potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getPotential(struct tfBondHandleHandle *handle, struct tfPotentialHandle *potential);

/**
 * @brief Get the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getDissociationEnergy(struct tfBondHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_setDissociationEnergy(struct tfBondHandleHandle *handle, tfFloatP_t value);

/**
 * @brief Get the half life
 * 
 * @param handle populated handle
 * @param value half life; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getHalfLife(struct tfBondHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the half life
 * 
 * @param handle populated handle
 * @param value half life
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_setHalfLife(struct tfBondHandleHandle *handle, tfFloatP_t value);

/**
 * @brief Set whether a bond is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_setActive(struct tfBondHandleHandle *handle, bool flag);

/**
 * @brief Get the bond style
 * 
 * @param handle populated handle
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getStyle(struct tfBondHandleHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Set the bond style
 * 
 * @param handle populated handle
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_setStyle(struct tfBondHandleHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Get the default bond style
 * 
 * @param style bond style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getStyleDef(struct tfRenderingStyleHandle *style);

/**
 * @brief Get the age of the bond
 * 
 * @param handle populated handle
 * @param value bond age
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getAge(struct tfBondHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBondHandle_toString(struct tfBondHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfBondHandle_fromString(struct tfBondHandleHandle *handle, const char *str);

/** Test whether lhs < rhs */
CAPI_FUNC(HRESULT) tfBondHandle_lt(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result);

/** Test whether lhs > rhs */
CAPI_FUNC(HRESULT) tfBondHandle_gt(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result);

/** Test whether lhs <= rhs */
CAPI_FUNC(HRESULT) tfBondHandle_le(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result);

/** Test whether lhs >= rhs */
CAPI_FUNC(HRESULT) tfBondHandle_ge(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result);

/** Test whether lhs == rhs */
CAPI_FUNC(HRESULT) tfBondHandle_eq(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result);

/** Test whether lhs != rhs */
CAPI_FUNC(HRESULT) tfBondHandle_ne(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result);


/////////////////
// AngleHandle //
/////////////////


/**
 * @brief Construct a new angle handle from an existing angle id
 * 
 * @param handle handle to populate
 * @param id id of existing angle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_init(struct tfAngleHandleHandle *handle, unsigned int id);

/**
 * @brief Construct a new angle handle and underlying angle. 
 * 
 * @param handle handle to populate
 * @param potential angle potential
 * @param i ith particle
 * @param j jth particle
 * @param k kth particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_create(
    struct tfAngleHandleHandle *handle, 
    struct tfPotentialHandle *potential,
    struct tfParticleHandleHandle *i, 
    struct tfParticleHandleHandle *j, 
    struct tfParticleHandleHandle *k
);

/**
 * @brief Get a summary string of the angle
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfAngleHandle_str(struct tfAngleHandleHandle *handle, char **str, unsigned int *numChars);


/**
 * @brief Check the validity of the handle
 * 
 * @param handle populated handle
 * @param flag true if OK
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_check(struct tfAngleHandleHandle *handle, bool *flag);

/**
 * @brief Destroy the angle. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_destroy(struct tfAngleHandleHandle *handle);

/**
 * @brief Tests whether this angle decays
 * 
 * @param handle populated handle
 * @param flag true when the angle should decay
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_decays(struct tfAngleHandleHandle *handle, bool *flag);

/**
 * @brief Test whether the bond has an id
 * 
 * @param handle populated handle
 * @param pid id to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfAngleHandle_hasPartId(struct tfAngleHandleHandle *handle, int pid, bool *result);

/**
 * @brief Test whether the bond has a particle
 * 
 * @param handle populated handle
 * @param part particle to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfAngleHandle_hasPart(struct tfAngleHandleHandle *handle, struct tfParticleHandleHandle *part, bool *result);

/**
 * @brief Get the current energy of the angle
 * 
 * @param handle populated handle
 * @param value current energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getEnergy(struct tfAngleHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Get the ids of the particles of the angle
 * 
 * @param handle populated handle
 * @param parti ith particle; -1 if no particle
 * @param partj jth particle; -1 if no particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getParts(struct tfAngleHandleHandle *handle, int *parti, int *partj);

/**
 * @brief Get a list of the particles of the angle
 * 
 * @param handle populated handle
 * @param plist handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getPartList(struct tfAngleHandleHandle *handle, struct tfParticleListHandle *plist);

/**
 * @brief Get the potential of the angle
 * 
 * @param handle populated handle
 * @param potential angle potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getPotential(struct tfAngleHandleHandle *handle, struct tfPotentialHandle *potential);

/**
 * @brief Get the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getDissociationEnergy(struct tfAngleHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_setDissociationEnergy(struct tfAngleHandleHandle *handle, tfFloatP_t value);

/**
 * @brief Get the half life
 * 
 * @param handle populated handle
 * @param value half life; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getHalfLife(struct tfAngleHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the half life
 * 
 * @param handle populated handle
 * @param value half life
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_setHalfLife(struct tfAngleHandleHandle *handle, tfFloatP_t value);

/**
 * @brief Set whether a angle is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_setActive(struct tfAngleHandleHandle *handle, bool flag);

/**
 * @brief Get the angle style
 * 
 * @param handle populated handle
 * @param style angle style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getStyle(struct tfAngleHandleHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Set the angle style
 * 
 * @param handle populated handle
 * @param style angle style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_setStyle(struct tfAngleHandleHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Get the angle style
 * 
 * @param style angle style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getStyleDef(struct tfRenderingStyleHandle *style);

/**
 * @brief Get the age of the angle
 * 
 * @param handle populated handle
 * @param value angle age
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getAge(struct tfAngleHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfAngleHandle_toString(struct tfAngleHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfAngleHandle_fromString(struct tfAngleHandleHandle *handle, const char *str);

/** Test whether lhs < rhs */
CAPI_FUNC(HRESULT) tfAngleHandle_lt(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result);

/** Test whether lhs > rhs */
CAPI_FUNC(HRESULT) tfAngleHandle_gt(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result);

/** Test whether lhs <= rhs */
CAPI_FUNC(HRESULT) tfAngleHandle_le(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result);

/** Test whether lhs >= rhs */
CAPI_FUNC(HRESULT) tfAngleHandle_ge(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result);

/** Test whether lhs == rhs */
CAPI_FUNC(HRESULT) tfAngleHandle_eq(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result);

/** Test whether lhs != rhs */
CAPI_FUNC(HRESULT) tfAngleHandle_ne(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result);


////////////////////
// DihedralHandle //
////////////////////


/**
 * @brief Construct a new dihedral handle from an existing dihedral id
 * 
 * @param handle handle to populate
 * @param id id of existing dihedral
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_init(struct tfDihedralHandleHandle *handle, unsigned int id);

/**
 * @brief Construct a new dihedral handle and underlying dihedral. 
 * 
 * @param handle handle to populate
 * @param potential dihedral potential
 * @param i ith particle
 * @param j jth particle
 * @param k kth particle
 * @param l lth particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_create(
    struct tfDihedralHandleHandle *handle, 
    struct tfPotentialHandle *potential,
    struct tfParticleHandleHandle *i, 
    struct tfParticleHandleHandle *j, 
    struct tfParticleHandleHandle *k, 
    struct tfParticleHandleHandle *l
);

/**
 * @brief Get a summary string of the dihedral
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_str(struct tfDihedralHandleHandle *handle, char **str, unsigned int *numChars);


/**
 * @brief Check the validity of the handle
 * 
 * @param handle populated handle
 * @param flag true if OK
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_check(struct tfDihedralHandleHandle *handle, bool *flag);

/**
 * @brief Destroy the angle. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_destroy(struct tfDihedralHandleHandle *handle);

/**
 * @brief Tests whether this dihedral decays
 * 
 * @param handle populated handle
 * @param flag true when the dihedral should decay
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_decays(struct tfDihedralHandleHandle *handle, bool *flag);

/**
 * @brief Test whether the bond has an id
 * 
 * @param handle populated handle
 * @param pid id to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_hasPartId(struct tfDihedralHandleHandle *handle, int pid, bool *result);

/**
 * @brief Test whether the bond has a particle
 * 
 * @param handle populated handle
 * @param part particle to test
 * @param result result of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_hasPart(struct tfDihedralHandleHandle *handle, struct tfParticleHandleHandle *part, bool *result);

/**
 * @brief Get the current energy of the dihedral
 * 
 * @param handle populated handle
 * @param value current energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getEnergy(struct tfDihedralHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Get the ids of the particles of the dihedral
 * 
 * @param handle populated handle
 * @param parti ith particle; -1 if no particle
 * @param partj jth particle; -1 if no particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getParts(struct tfDihedralHandleHandle *handle, int *parti, int *partj);

/**
 * @brief Get a list of the particles of the dihedral
 * 
 * @param handle populated handle
 * @param plist handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getPartList(struct tfDihedralHandleHandle *handle, struct tfParticleListHandle *plist);

/**
 * @brief Get the potential of the dihedral
 * 
 * @param handle populated handle
 * @param potential dihedral potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getPotential(struct tfDihedralHandleHandle *handle, struct tfPotentialHandle *potential);

/**
 * @brief Get the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getDissociationEnergy(struct tfDihedralHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the dissociation energy
 * 
 * @param handle populated handle
 * @param value dissociation energy
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_setDissociationEnergy(struct tfDihedralHandleHandle *handle, tfFloatP_t value);

/**
 * @brief Get the half life
 * 
 * @param handle populated handle
 * @param value half life; -1 if not defined
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getHalfLife(struct tfDihedralHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the half life
 * 
 * @param handle populated handle
 * @param value half life
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_setHalfLife(struct tfDihedralHandleHandle *handle, tfFloatP_t value);

/**
 * @brief Set whether a dihedral is active
 * 
 * @param handle populated handle
 * @param flag true when active
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_setActive(struct tfDihedralHandleHandle *handle, bool flag);

/**
 * @brief Get the dihedral style
 * 
 * @param handle populated handle
 * @param style dihedral style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getStyle(struct tfDihedralHandleHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Set the dihedral style
 * 
 * @param handle populated handle
 * @param style dihedral style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_setStyle(struct tfDihedralHandleHandle *handle, struct tfRenderingStyleHandle *style);

/**
 * @brief Get the dihedral style
 * 
 * @param style dihedral style
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getStyleDef(struct tfRenderingStyleHandle *style);

/**
 * @brief Get the age of the dihedral
 * 
 * @param handle populated handle
 * @param value dihedral age
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getAge(struct tfDihedralHandleHandle *handle, tfFloatP_t *value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_toString(struct tfDihedralHandleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_fromString(struct tfDihedralHandleHandle *handle, const char *str);

/** Test whether lhs < rhs */
CAPI_FUNC(HRESULT) tfDihedralHandle_lt(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result);

/** Test whether lhs > rhs */
CAPI_FUNC(HRESULT) tfDihedralHandle_gt(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result);

/** Test whether lhs <= rhs */
CAPI_FUNC(HRESULT) tfDihedralHandle_le(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result);

/** Test whether lhs >= rhs */
CAPI_FUNC(HRESULT) tfDihedralHandle_ge(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result);

/** Test whether lhs == rhs */
CAPI_FUNC(HRESULT) tfDihedralHandle_eq(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result);

/** Test whether lhs != rhs */
CAPI_FUNC(HRESULT) tfDihedralHandle_ne(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Gets all bonds in the universe
 * 
 * @param handles
 * @param numBonds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBondHandle_getAll(struct tfBondHandleHandle **handles, unsigned int *numBonds);

/**
 * @brief Apply bonds to a list of particles. 
 * 
 * @param pot the potential of the created bonds
 * @param parts list of particles
 * @param cutoff cutoff distance of particles that are bonded
 * @param ppairsA first elements of type pairs that are bonded
 * @param ppairsB second elements of type pairs that are bonded
 * @param numTypePairs number of type pairs
 * @param half_life bond half life, optional
 * @param bond_energy bond energy, optional
 * @param bonds created bonds
 * @param numBonds number of created bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBond_pairwise(
    struct tfPotentialHandle *pot, 
    struct tfParticleListHandle *parts, 
    tfFloatP_t cutoff, 
    struct tfParticleTypeHandle *ppairsA, 
    struct tfParticleTypeHandle *ppairsB, 
    unsigned int numTypePairs, 
    tfFloatP_t *half_life, 
    tfFloatP_t *bond_energy, 
    struct tfBondHandleHandle **bonds, 
    unsigned int *numBonds
);

/**
 * @brief Find all the bonds that interact with the given particle id
 * 
 * @param pid particle id
 * @param bids bond ids
 * @param numIds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBond_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds);

/**
 * @brief Deletes all bonds in the universe. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfBond_destroyAll();

/**
 * @brief Gets all bonds in the universe
 * 
 * @param handles
 * @param numBonds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngleHandle_getAll(struct tfAngleHandleHandle **handles, unsigned int *numBonds);

/**
 * @brief Find all the bonds that interact with the given particle id
 * 
 * @param pid particle id
 * @param bids bond ids
 * @param numIds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngle_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds);

/**
 * @brief Deletes all bonds in the universe. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfAngle_destroyAll();

/**
 * @brief Gets all bonds in the universe
 * 
 * @param handles
 * @param numBonds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedralHandle_getAll(struct tfDihedralHandleHandle **handles, unsigned int *numBonds);

/**
 * @brief Find all the bonds that interact with the given particle id
 * 
 * @param pid particle id
 * @param bids bond ids
 * @param numIds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedral_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds);

/**
 * @brief Deletes all bonds in the universe. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfDihedral_destroyAll();

#endif // _WRAPS_C_TFCBOND_H_