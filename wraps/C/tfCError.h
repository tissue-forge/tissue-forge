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
 * @file tfCError.h
 * 
 */

#ifndef _WRAPS_C_TFCERROR_H_
#define _WRAPS_C_TFCERROR_H_

#include "tf_port_c.h"

// Handles

/**
 * @brief Handle to a @ref Error instance
 * 
 */
struct CAPI_EXPORT tfErrorHandle {
    void *tfObj;
};


///////////
// Error //
///////////


/**
 * @brief Get the error code
 * 
 * @param handle populated handle
 * @param err error code
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfError_getErr(struct tfErrorHandle *handle, HRESULT *err);

/**
 * @brief Get the error message
 * 
 * @param handle populated handle
 * @param msg error message
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfError_getMsg(struct tfErrorHandle *handle, char **msg, unsigned int *numChars);

/**
 * @brief Get the originating line number
 * 
 * @param handle populated handle
 * @param lineno originating line number
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfError_getLineno(struct tfErrorHandle *handle, int *lineno);

/**
 * @brief Get the originating file name
 * 
 * @param handle populated handle
 * @param fname originating file name
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfError_getFname(struct tfErrorHandle *handle, char **fname, unsigned int *numChars);

/**
 * @brief Get the originating function name
 * 
 * @param handle populated handle
 * @param func originating function name
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfError_getFunc(struct tfErrorHandle *handle, char **func, unsigned int *numChars);


//////////////////////
// Module functions //
//////////////////////

/**
 * Set the error indicator. If there is a previous error indicator, then the previous indicator is moved down the stack. 
 */
CAPI_FUNC(HRESULT) tfErrSet(HRESULT code, const char* msg, int line, const char* file, const char* func);

/**
 * Check whether there is an error indicator. 
 */
CAPI_FUNC(bool) tfErrOccurred();

/**
 * Clear the error indicator. If the error indicator is not set, there is no effect.
 */
CAPI_FUNC(void) tfErrClear();

/**
 * Get a string representation of an error.
 */
CAPI_FUNC(HRESULT) tfErrStr(struct tfErrorHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Get all error indicators.
 * 
 * @param handles error indicators
 * @param numErrors number of error indicators
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfErrGetAll(struct tfErrorHandle ***handles, unsigned int *numErrors);

/**
 * Get the first error
 */
CAPI_FUNC(HRESULT) tfErrGetFirst(struct tfErrorHandle **handle);

/**
 * Clear the first error
 */
CAPI_FUNC(void) tfErrClearFirst();

/**
 * Get and clear the first error
 */
CAPI_FUNC(HRESULT) tfErrPopFirst(struct tfErrorHandle **handle);


#endif // _WRAPS_C_TFCERROR_H_