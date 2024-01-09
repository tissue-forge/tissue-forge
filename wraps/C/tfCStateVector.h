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
 * @file tfCStateVector.h
 * 
 */

#ifndef _WRAPS_C_TFCSTATEVECTOR_H_
#define _WRAPS_C_TFCSTATEVECTOR_H_

#include "tf_port_c.h"

// Handles

/**
 * @brief Handle to a @ref state::StateVector instance
 * 
 */
struct CAPI_EXPORT tfStateStateVectorHandle {
    void *tfObj;
};


////////////////////////
// state::StateVector //
////////////////////////


/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateStateVector_destroy(struct tfStateStateVectorHandle *handle);

/**
 * @brief Get size of vector
 * 
 * @param handle populated handle
 * @param size vector size
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateStateVector_getSize(struct tfStateStateVectorHandle *handle, unsigned int *size);

/**
 * @brief Get the species of the state vector
 * 
 * @param handle populated handle
 * @param slist species list
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfStateStateVector_getSpecies(struct tfStateStateVectorHandle *handle, struct tfStateSpeciesListHandle *slist);

/**
 * @brief reset the species values based on the values specified in the species.
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateStateVector_reset(struct tfStateStateVectorHandle *handle);

/**
 * @brief Get a summary string of the state vector
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateStateVector_getStr(struct tfStateStateVectorHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Get the value of an item
 * 
 * @param handle populated handle
 * @param i index of item
 * @param value value of item
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateStateVector_getItem(struct tfStateStateVectorHandle *handle, int i, tfFloatP_t *value);

/**
 * @brief Set the value of an item
 * 
 * @param handle populated handle
 * @param i index of item
 * @param value value of item
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateStateVector_setItem(struct tfStateStateVectorHandle *handle, int i, tfFloatP_t value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateStateVector_toString(struct tfStateStateVectorHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateStateVector_fromString(struct tfStateStateVectorHandle *handle, const char *str);

#endif // _WRAPS_C_TFCSTATEVECTOR_H_