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

#ifndef _SOURCE_TFERROR_H_
#define _SOURCE_TFERROR_H_

#include <tf_port.h>

#include <exception>


namespace TissueForge {


    struct CAPI_EXPORT Error {
        HRESULT err;
        const char* msg;
        int lineno;
        const char* fname;
        const char* func;
    };

    #define tf_error(code, msg) errSet(code, msg, __LINE__, __FILE__, TF_FUNCTION)

    #define tf_exp(e) expSet(e, "", __LINE__, __FILE__, TF_FUNCTION)

    #define TF_RETURN_EXP(e) expSet(e, "", __LINE__, __FILE__, TF_FUNCTION); return NULL

    CPPAPI_FUNC(HRESULT) errSet(HRESULT code, const char* msg, int line, const char* file, const char* func);

    CPPAPI_FUNC(HRESULT) expSet(const std::exception&, const char* msg, int line, const char* file, const char* func);

    CPPAPI_FUNC(Error*) errOccurred();

    /**
     * Clear the error indicator. If the error indicator is not set, there is no effect.
     */
    CPPAPI_FUNC(void) errClear();

};

#endif // _SOURCE_TFERROR_H_