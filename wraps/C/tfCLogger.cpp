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

#include "tfCLogger.h"

#include "TissueForge_c_private.h"

#include <tfLogger.h>


using namespace TissueForge;


//////////////
// LogLevel //
//////////////


HRESULT tfLogLevel_init(struct tfLogLevelHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->LOG_CURRENT = LOG_CURRENT;
    handle->LOG_FATAL = LOG_FATAL;
    handle->LOG_CRITICAL = LOG_CRITICAL;
    handle->LOG_ERROR = LOG_ERROR;
    handle->LOG_WARNING = LOG_WARNING;
    handle->LOG_NOTICE = LOG_NOTICE;
    handle->LOG_INFORMATION = LOG_INFORMATION;
    handle->LOG_DEBUG = LOG_DEBUG;
    handle->LOG_TRACE = LOG_TRACE;
    return S_OK;
}


//////////////
// LogEvent //
//////////////


HRESULT tfLogEventHandle_init(struct tfLogEventHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->LOG_OUTPUTSTREAM_CHANGED = LOG_OUTPUTSTREAM_CHANGED;
    handle->LOG_LEVEL_CHANGED = LOG_LEVEL_CHANGED;
    handle->LOG_CALLBACK_SET = LOG_CALLBACK_SET;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfLogger_setLevel(unsigned int level) {
    Logger::setLevel(level);
    return S_OK;
}

HRESULT tfLogger_getLevel(unsigned int *level) {
    TFC_PTRCHECK(level);
    *level = Logger::getLevel();
    return S_OK;
}

HRESULT tfLogger_enableFileLogging(const char *fileName, unsigned int level) {
    Logger::enableFileLogging(fileName, level);
    return S_OK;
}

HRESULT tfLogger_disableFileLogging() {
    Logger::disableFileLogging();
    return S_OK;
}

HRESULT tfLogger_getFileName(char **str, unsigned int *numChars) {
    return TissueForge::capi::str2Char(Logger::getFileName(), str, numChars);
}

HRESULT tfLogger_log(unsigned int level, const char *msg) {
    Logger::log((LogLevel)level, msg);
    return S_OK;
}
