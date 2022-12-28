/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCMESHLOGGER_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCMESHLOGGER_H_

#include <tf_port_c.h>

#include <tfCLogger.h>

// Handles


/**
 * @brief Handle to a @ref models::vertex::MeshLogEventType instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverMeshLogEventTypeHandle {
    unsigned int None;
    unsigned int Create;
    unsigned int Destroy;
    unsigned int Operation;
};


//////////////////////
// MeshLogEventType //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogEventType_init(struct tfVertexSolverMeshLogEventTypeHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Clear the log
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogger_clear();

/**
 * @brief Add a log event to the log
 * 
 * @param type log event type
 * @param ids object ids
 * @param numIds number of object ids
 * @param typeLabels type labels
 * @param numTypeLabels number of type labels
 * @param name name of event
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogger_log(
    unsigned int type, 
    int *ids, 
    unsigned int numIds, 
    unsigned int *typeLabels, 
    unsigned int numTypeLabels, 
    const char *name
);

/**
 * @brief Get the number of log events
 * 
 * @param numEvents number of log events
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogger_getNumEvents(unsigned int *numEvents);

/**
 * @brief Get a log event
 * 
 * @param idx event index
 * @param type log event type
 * @param ids object ids
 * @param numIds number of object ids
 * @param typeLabels type labels
 * @param numTypeLabels number of type labels
 * @param name name of event
 * @param numChars number of chars in name
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogger_getEvent(
    unsigned int idx, 
    unsigned int *type, 
    int **ids, 
    unsigned int *numIds, 
    unsigned int **typeLabels, 
    unsigned int *numTypeLabels, 
    char **name, 
    unsigned int *numChars
);

/**
 * @brief Get whether the logger is fowarding log events to the main Tissue Forge logger
 * 
 * @param forward flag indicating whether log events are forwarded
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogger_getForwardLogging(bool *forward);

/**
 * @brief Set whether to foward log events to the main Tissue Forge logger
 * 
 * @param forward flag indicating whether log events are forwarded
 * @return CAPI_FUNC(HRESULT) 
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogger_setForwardLogging(bool forward);

/**
 * @brief Get the current log level
 * 
 * @param level current log level
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogger_getLogLevel(unsigned int *level);

/**
 * @brief Set the current log level
 * 
 * @param level current log level
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshLogger_setLogLevel(unsigned int level);


#endif // _WRAPS_C_VERTEX_SOLVER_TFCMESHLOGGER_H_