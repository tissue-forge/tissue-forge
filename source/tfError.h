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
 * @file tfError.h
 * 
 */

#ifndef _SOURCE_TFERROR_H_
#define _SOURCE_TFERROR_H_

#include <tf_port.h>

#include <exception>
#include <string>
#include <vector>


namespace TissueForge {


    struct CAPI_EXPORT Error {
        /** Error code */
        HRESULT err;

        /** Error message */
        std::string msg;

        /** Originating line number */
        int lineno;

        /** Originating file name */
        std::string fname;

        /** Originating function name */
        std::string func;
    };

    /** Called on every occurrence of an error */
    typedef void (*ErrorCallback)(const Error &err);

    /**
     * @brief Register an error callback
     * 
     * @param cb callback to register
     * @return id of callback in registry
     */
    CPPAPI_FUNC(const unsigned int) addErrorCallback(ErrorCallback &cb);

    /**
     * @brief Remove an error callback from the registry
     * 
     * @param cb_id id of registered callback
     */
    CPPAPI_FUNC(HRESULT) removeErrorCallback(const unsigned int &cb_id);

    /**
     * @brief Remove all error callbacks from the registry
     */
    CPPAPI_FUNC(HRESULT) clearErrorCallbacks();

    #define tf_error(code, msg) errSet(code, msg, __LINE__, __FILE__, TF_FUNCTION)

    #define tf_exp(e) expSet(e, "", __LINE__, __FILE__, TF_FUNCTION)

    #define TF_RETURN_EXP(e) expSet(e, "", __LINE__, __FILE__, TF_FUNCTION); return NULL

    /**
     * Set the error indicator. If there is a previous error indicator, then the previous indicator is moved down the stack. 
     */
    CPPAPI_FUNC(HRESULT) errSet(HRESULT code, const char* msg, int line, const char* file, const char* func);

    /**
     * Set the error indicator. If there is a previous error indicator, then the previous indicator is moved down the stack. 
     */
    CPPAPI_FUNC(HRESULT) expSet(const std::exception&, const char* msg, int line, const char* file, const char* func);

    /**
     * Check whether there is an error indicator. 
     */
    CPPAPI_FUNC(bool) errOccurred();

    /**
     * Clear the error indicators. If no error indicator is set, there is no effect.
     */
    CPPAPI_FUNC(void) errClear();

    /**
     * Get a string representation of an error.
     */
    CPPAPI_FUNC(std::string) errStr(const Error &err);

    /**
     * Get all error indicators
     */
    CPPAPI_FUNC(std::vector<Error>) errGetAll();

    /**
     * Get the first error
     */
    CPPAPI_FUNC(Error) errGetFirst();

    /**
     * Clear the first error
     */
    CPPAPI_FUNC(void) errClearFirst();

    /**
     * Get and clear the first error
     */
    CPPAPI_FUNC(Error) errPopFirst();

};

inline std::ostream& operator<<(std::ostream& os, const TissueForge::Error &err)
{
    os << std::string("Code: ");
    os << std::to_string(err.err);
    os << std::string(", Msg: ");
    os << err.msg;
    os << std::string(", File: ");
    os << err.fname;
    os << std::string(", Line: ");
    os << std::to_string(err.lineno);
    os << std::string(", Function: ");
    os << err.func;
    return os;
}

#endif // _SOURCE_TFERROR_H_