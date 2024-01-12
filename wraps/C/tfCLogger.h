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
 * @file tfCLogger.h
 * 
 */

#ifndef _WRAPS_C_TFCLOGGER_H_
#define _WRAPS_C_TFCLOGGER_H_

#include "tf_port_c.h"

// Handles

struct CAPI_EXPORT tfLogLevelHandle {
    unsigned int LOG_CURRENT;
    unsigned int LOG_FATAL;
    unsigned int LOG_CRITICAL;
    unsigned int LOG_ERROR;
    unsigned int LOG_WARNING;
    unsigned int LOG_NOTICE;
    unsigned int LOG_INFORMATION;
    unsigned int LOG_DEBUG;
    unsigned int LOG_TRACE;
};

struct CAPI_EXPORT tfLogEventHandle {
    unsigned int LOG_OUTPUTSTREAM_CHANGED;
    unsigned int LOG_LEVEL_CHANGED;
    unsigned int LOG_CALLBACK_SET;
};


//////////////
// LogLevel //
//////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfLogLevel_init(struct tfLogLevelHandle *handle);


//////////////
// LogEvent //
//////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfLogEventHandle_init(struct tfLogEventHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Set the Level objectsets the logging level to one a value from Logger::Level
 * 
 * @param level logging level
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfLogger_setLevel(unsigned int level);

/**
 * @brief Get the Level objectget the current logging level.
 * 
 * @param level logging level
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfLogger_getLevel(unsigned int *level);

/**
 * @brief turns on file logging to the given file as the given level.
 * 
 * @param fileName path to log file
 * @param level logging level
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfLogger_enableFileLogging(const char *fileName, unsigned int level);

/**
 * @brief turns off file logging, but has no effect on console logging.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfLogger_disableFileLogging();

/**
 * @brief Get the File Name objectget the name of the currently used log file.
 * 
 * @param str string array of file name
 * @param numChars number of characters in string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfLogger_getFileName(char **str, unsigned int *numChars);

/**
 * @brief logs a message to the log.
 * 
 * @param level logging level
 * @param msg log message
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfLogger_log(unsigned int level, const char *msg);

#endif // _WRAPS_C_TFCLOGGER_H_