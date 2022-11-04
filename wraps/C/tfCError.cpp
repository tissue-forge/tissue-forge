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

#include "tfCError.h"

#include "TissueForge_c_private.h"

#include <tfError.h>


using namespace TissueForge;


namespace TissueForge { 


    Error *castC(struct tfErrorHandle *handle) {
        return castC<Error, tfErrorHandle>(handle);
    }

}

#define TFC_ERROR_GET(handle, varname) \
    Error *varname = TissueForge::castC<Error, tfErrorHandle>(handle); \
    TFC_PTRCHECK(varname);



///////////
// Error //
///////////

HRESULT tfError_getErr(struct tfErrorHandle *handle, HRESULT *err) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(err);
    TFC_ERROR_GET(handle, _err);
    *err = _err->err;
    return S_OK;
}

HRESULT tfError_getMsg(struct tfErrorHandle *handle, char **msg, unsigned int *numChars) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(msg);
    TFC_PTRCHECK(numChars);
    TFC_ERROR_GET(handle, _err);
    return capi::str2Char(_err->msg, msg, numChars);
}

HRESULT tfError_getLineno(struct tfErrorHandle *handle, int *lineno) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(lineno);
    TFC_ERROR_GET(handle, _err);
    *lineno = _err->lineno;
    return S_OK;
}

HRESULT tfError_getFname(struct tfErrorHandle *handle, char **fname, unsigned int *numChars) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(fname);
    TFC_PTRCHECK(numChars);
    TFC_ERROR_GET(handle, _err);
    return capi::str2Char(_err->fname, fname, numChars);
}

HRESULT tfError_getFunc(struct tfErrorHandle *handle, char **func, unsigned int *numChars) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(func);
    TFC_PTRCHECK(numChars);
    TFC_ERROR_GET(handle, _err);
    return capi::str2Char(_err->func, func, numChars);
}



//////////////////////
// Module functions //
//////////////////////

HRESULT tfErrSet(HRESULT code, const char* msg, int line, const char* file, const char* func) {
    return errSet(code, msg, line, file, func);
}

bool tfErrOccurred(struct tfErrorHandle *handle) {
    Error *err = errOccurred();
    if(!err) 
        return false;

    TFC_PTRCHECK(handle);
    handle->tfObj = (void*)err;
    return true;
}

void tfErrClear() {
    errClear();
}

HRESULT tfErrStr(struct tfErrorHandle *handle, char **str, unsigned int *numChars) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    TFC_ERROR_GET(handle, _err);
    return capi::str2Char(errStr(*_err), str, numChars);
}
