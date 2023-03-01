/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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

#include "tfCMeshLogger.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tfMeshLogger.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


//////////////////////
// MeshLogEventType //
//////////////////////


HRESULT tfVertexSolverMeshLogEventType_init(struct tfVertexSolverMeshLogEventTypeHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->None = MeshLogEventType::None;
    handle->Create = MeshLogEventType::Create;
    handle->Destroy = MeshLogEventType::Destroy;
    handle->Operation = MeshLogEventType::Operation;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolverMeshLogger_clear() {
    return MeshLogger::clear();
}

HRESULT tfVertexSolverMeshLogger_log(
    unsigned int type, 
    int *ids, 
    unsigned int numIds, 
    unsigned int *typeLabels, 
    unsigned int numTypeLabels, 
    const char *name
) {
    TFC_PTRCHECK(ids);
    TFC_PTRCHECK(typeLabels);
    TFC_PTRCHECK(name);
    std::vector<int> _ids;
    for(size_t i = 0; i < numIds; i++) 
        _ids.push_back(ids[i]);
    std::vector<MeshObjTypeLabel> _typeLabels;
    for(size_t i = 0; i < numTypeLabels; i++) 
        _typeLabels.push_back((MeshObjTypeLabel)typeLabels[i]);
    MeshLogEvent e;
    e.type = (MeshLogEventType)type;
    e.objIDs = _ids;
    e.objTypes = _typeLabels;
    e.name = name;
    return MeshLogger::log(e);
}

HRESULT tfVertexSolverMeshLogger_getNumEvents(unsigned int *numEvents) {
    TFC_PTRCHECK(numEvents);
    *numEvents = MeshLogger::events().size();
    return S_OK;
}

HRESULT tfVertexSolverMeshLogger_getEvent(
    unsigned int idx, 
    unsigned int *type, 
    int **ids, 
    unsigned int *numIds, 
    unsigned int **typeLabels, 
    unsigned int *numTypeLabels, 
    char **name, 
    unsigned int *numChars
) {
    TFC_PTRCHECK(type);
    TFC_PTRCHECK(ids);
    TFC_PTRCHECK(numIds);
    TFC_PTRCHECK(typeLabels);
    TFC_PTRCHECK(numTypeLabels);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(numChars);

    auto events = MeshLogger::events();
    if(idx >= events.size()) 
        return E_FAIL;
    auto e = events[idx];

    *type = e.type;
    *numIds = e.objIDs.size();
    if(*numIds > 0) {
        *ids = (int*)malloc(sizeof(int) * e.objIDs.size());
        memcpy(&(*ids)[0], &e.objIDs.data()[0], sizeof(int) * e.objIDs.size());
    }
    *numTypeLabels = e.objTypes.size();
    if(*numTypeLabels > 0) {
        *typeLabels = (unsigned int*)malloc(sizeof(unsigned int) * e.objTypes.size());
        memcpy(&(*typeLabels)[0], &e.objTypes.data()[0], sizeof(unsigned int) * e.objTypes.size());
    }
    return TissueForge::capi::str2Char(e.name, name, numChars);
}

HRESULT tfVertexSolverMeshLogger_getForwardLogging(bool *forward) {
    TFC_PTRCHECK(forward);
    *forward = MeshLogger::getForwardLogging();
    return S_OK;
}

HRESULT tfVertexSolverMeshLogger_setForwardLogging(bool forward) {
    return MeshLogger::setForwardLogging(forward);
}

HRESULT tfVertexSolverMeshLogger_getLogLevel(unsigned int *level) {
    TFC_PTRCHECK(level);
    *level = MeshLogger::getLogLevel();
    return S_OK;
}

HRESULT tfVertexSolverMeshLogger_setLogLevel(unsigned int level) {
    return MeshLogger::setLogLevel((LogLevel)level);
}
