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
 * @file tfCSpecies.h
 * 
 */

#ifndef _WRAPS_C_TFCSPECIES_H_
#define _WRAPS_C_TFCSPECIES_H_

#include "tf_port_c.h"

#include "tfCStateVector.h"
#include "tfCParticle.h"

// Handles

/**
 * @brief Handle to a @ref state::Species instance
 * 
 */
struct CAPI_EXPORT tfStateSpeciesHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref state::SpeciesList instance
 * 
 */
struct CAPI_EXPORT tfStateSpeciesListHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref state::SpeciesValue instance
 * 
 */
struct CAPI_EXPORT tfStateSpeciesValueHandle {
    void *tfObj;
};


////////////////////
// state::Species //
////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_init(struct tfStateSpeciesHandle *handle);

/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param s string constructor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_initS(struct tfStateSpeciesHandle *handle, const char *s);

/**
 * @brief Copy an instance
 * 
 * @param source populated handle
 * @param destination handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_copy(struct tfStateSpeciesHandle *source, struct tfStateSpeciesHandle *destination);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_destroy(struct tfStateSpeciesHandle *handle);

/**
 * @brief Get the species id
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getId(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species id
 * 
 * @param handle populated handle
 * @param sid id string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setId(struct tfStateSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the species name
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getName(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species name
 * 
 * @param handle populated handle
 * @param name name string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setName(struct tfStateSpeciesHandle *handle, const char *name);

/**
 * @brief Get the species type
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array character
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getSpeciesType(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species type
 * 
 * @param handle populated handle
 * @param sid type string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setSpeciesType(struct tfStateSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the species compartment
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getCompartment(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species compartment
 * 
 * @param handle populated handle
 * @param sid compartment string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setCompartment(struct tfStateSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the initial amount
 * 
 * @param handle populated handle
 * @param value initial amount
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getInitialAmount(struct tfStateSpeciesHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the initial amount
 * 
 * @param handle populated handle
 * @param value initial amount
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setInitialAmount(struct tfStateSpeciesHandle *handle, tfFloatP_t value);

/**
 * @brief Get the initial concentration
 * 
 * @param handle populated handle
 * @param value initial concentration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getInitialConcentration(struct tfStateSpeciesHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the initial concentration
 * 
 * @param handle populated handle
 * @param value initial concentration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setInitialConcentration(struct tfStateSpeciesHandle *handle, tfFloatP_t value);

/**
 * @brief Get the substance units
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getSubstanceUnits(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the substance units
 * 
 * @param handle populated handle
 * @param sid substance units string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setSubstanceUnits(struct tfStateSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the spatial size units
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getSpatialSizeUnits(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the spatial size units
 * 
 * @param handle populated handle
 * @param sid spatial size units string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setSpatialSizeUnits(struct tfStateSpeciesHandle *handle, const char *sid);

/**
 * @brief Get the units
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getUnits(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the units
 * 
 * @param handle populated handle
 * @param sname units string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setUnits(struct tfStateSpeciesHandle *handle, const char *sname);

/**
 * @brief Get whether a species has only substance units
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species has only substance units
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getHasOnlySubstanceUnits(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Set whether a species has only substance units
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species has only substance units
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setHasOnlySubstanceUnits(struct tfStateSpeciesHandle *handle, bool value);

/**
 * @brief Get whether a species has a boundary condition
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species has a boundary condition
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getBoundaryCondition(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Set whether a species has a boundary condition
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species has a boundary condition
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setBoundaryCondition(struct tfStateSpeciesHandle *handle, bool value);

/**
 * @brief Get the species charge
 * 
 * @param handle populated handle
 * @param charge species charge
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getCharge(struct tfStateSpeciesHandle *handle, int *charge);

/**
 * @brief Set the species charge
 * 
 * @param handle populated handle
 * @param value species charge
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setCharge(struct tfStateSpeciesHandle *handle, int value);

/**
 * @brief Get whether a species is constnat
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species is constnat
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getConstant(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Set whether a species is constnat
 * 
 * @param handle populated handle
 * @param value flag signifying whether a species is constnat
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setConstant(struct tfStateSpeciesHandle *handle, int value);

/**
 * @brief Get the species conversion factor
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_getConversionFactor(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Set the species conversion factor
 * 
 * @param handle populated handle
 * @param sid species conversion factor string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_setConversionFactor(struct tfStateSpeciesHandle *handle, const char *sid);

/**
 * @brief Test whether the species id is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetId(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species name is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetName(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species type is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetSpeciesType(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species compartment is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetCompartment(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species initial amount is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetInitialAmount(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species initial concentration is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetInitialConcentration(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species substance units are set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetSubstanceUnits(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species spatial size units are set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetSpatialSizeUnits(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species units are set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetUnits(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species charge is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetCharge(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species conversion factor is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetConversionFactor(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species is constant
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetConstant(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether the species boundary condition is set
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetBoundaryCondition(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Test whether a species has only substance units
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_isSetHasOnlySubstanceUnits(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Unset the species id
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetId(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species name
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetName(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species constant flag
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetConstant(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species type
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetSpeciesType(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species initial amount
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetInitialAmount(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species initial concentration
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetInitialConcentration(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species substance units
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetSubstanceUnits(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species spatial size units
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetSpatialSizeUnits(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species units
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetUnits(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species charge
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetCharge(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species conversion factor
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetConversionFactor(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species compartment
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetCompartment(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species boundary condition
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetBoundaryCondition(struct tfStateSpeciesHandle *handle);

/**
 * @brief Unset the species has only substance units flag
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_unsetHasOnlySubstanceUnits(struct tfStateSpeciesHandle *handle);

/**
 * @brief Test whether a species has required attributes
 * 
 * @param handle populated handle
 * @param value outcome of test
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_hasRequiredAttributes(struct tfStateSpeciesHandle *handle, bool *value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_toString(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpecies_fromString(struct tfStateSpeciesHandle *handle, const char *str);


////////////////////////
// state::SpeciesList //
////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_init(struct tfStateSpeciesListHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_destroy(struct tfStateSpeciesListHandle *handle);

/**
 * @brief Get a string representation
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_getStr(struct tfStateSpeciesListHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Get the index of a species name
 * 
 * @param handle populated handle
 * @param s species name
 * @param i species index
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_indexOf(struct tfStateSpeciesListHandle *handle, const char *s, unsigned int *i);

/**
 * @brief Get the size of a list
 * 
 * @param handle populated handle
 * @param size size of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_getSize(struct tfStateSpeciesListHandle *handle, unsigned int *size);

/**
 * @brief Get a species by index
 * 
 * @param handle populated handle
 * @param index index of the species
 * @param species species at the given index
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_getItem(struct tfStateSpeciesListHandle *handle, unsigned int index, struct tfStateSpeciesHandle *species);

/**
 * @brief Get a species by name
 * 
 * @param handle populated handle
 * @param s name of species
 * @param species species at the given index
 * @return state::Species* 
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_getItemS(struct tfStateSpeciesListHandle *handle, const char *s, struct tfStateSpeciesHandle *species);

/**
 * @brief Insert a species
 * 
 * @param handle populated handle
 * @param species species to insert
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_insert(struct tfStateSpeciesListHandle *handle, struct tfStateSpeciesHandle *species);

/**
 * @brief Insert a species by name
 * 
 * @param handle populated handle
 * @param s name of the species to insert
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_insertS(struct tfStateSpeciesListHandle *handle, const char *s);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_toString(struct tfStateSpeciesListHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesList_fromString(struct tfStateSpeciesListHandle *handle, const char *str);


/////////////////////////
// state::SpeciesValue //
/////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param value species value
 * @param state_vector state vector of species
 * @param index species index in state vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_init(struct tfStateSpeciesValueHandle *handle, tfFloatP_t value, struct tfStateStateVectorHandle *state_vector, unsigned int index);

/**
 * @brief Get the value
 * 
 * @param handle populated handle 
 * @param value species value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_getValue(struct tfStateSpeciesValueHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the value
 * 
 * @param handle populated handle 
 * @param value species value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_setValue(struct tfStateSpeciesValueHandle *handle, tfFloatP_t value);

/**
 * @brief Get the state vector
 * 
 * @param handle populated handle 
 * @param state_vector state vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_getStateVector(struct tfStateSpeciesValueHandle *handle, struct tfStateStateVectorHandle *state_vector);

/**
 * @brief Get the species index
 * 
 * @param handle populated handle 
 * @param index species index
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_getIndex(struct tfStateSpeciesValueHandle *handle, unsigned int *index);

/**
 * @brief Get the species
 * 
 * @param handle populated handle 
 * @param species handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_getSpecies(struct tfStateSpeciesValueHandle *handle, struct tfStateSpeciesHandle *species);

/**
 * @brief Test whether the species has a boundary condition
 * 
 * @param handle populated handle 
 * @param value flag signifying whether the species has a boundary condition
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_getBoundaryCondition(struct tfStateSpeciesValueHandle *handle, bool *value);

/**
 * @brief Set whether the species has a boundary condition
 * 
 * @param handle populated handle 
 * @param value flag signifying whether the species has a boundary condition
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_setBoundaryCondition(struct tfStateSpeciesValueHandle *handle, bool value);

/**
 * @brief Get the species initial amount
 * 
 * @param handle populated handle 
 * @param value initial amount
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_getInitialAmount(struct tfStateSpeciesValueHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the species initial amount
 * 
 * @param handle populated handle 
 * @param value initial amount
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_setInitialAmount(struct tfStateSpeciesValueHandle *handle, tfFloatP_t value);

/**
 * @brief Get the species initial concentration
 * 
 * @param handle populated handle 
 * @param value initial concentration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_getInitialConcentration(struct tfStateSpeciesValueHandle *handle, tfFloatP_t *value);

/**
 * @brief Set the species initial concentration
 * 
 * @param handle populated handle 
 * @param value initial concentration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_setInitialConcentration(struct tfStateSpeciesValueHandle *handle, tfFloatP_t value);

/**
 * @brief Test whether the species is constant
 * 
 * @param handle populated handle 
 * @param value flag signifying whether the species is constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_getConstant(struct tfStateSpeciesValueHandle *handle, bool *value);

/**
 * @brief Set whether the species is constant
 * 
 * @param handle populated handle
 * @param value flag signifying whether the species is constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_setConstant(struct tfStateSpeciesValueHandle *handle, int value);

/**
 * @brief Secrete this species into a neighborhood. 
 * 
 * @param handle populated handle
 * @param amount amount to secrete. 
 * @param to list of particles to secrete to. 
 * @param secreted amount actually secreted, accounting for availability and other subtleties. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_secreteL(struct tfStateSpeciesValueHandle *handle, tfFloatP_t amount, struct tfParticleListHandle *to, tfFloatP_t *secreted);

/**
 * @brief Secrete this species into a neighborhood. 
 * 
 * @param handle populated handle
 * @param amount amount to secrete. 
 * @param distance neighborhood distance. 
 * @param secreted amount actually secreted, accounting for availability and other subtleties. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesValue_secreteD(struct tfStateSpeciesValueHandle *handle, tfFloatP_t amount, tfFloatP_t distance, tfFloatP_t *secreted);

#endif // _WRAPS_C_TFCSPECIES_H_