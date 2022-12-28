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

#include "tfCVertexSolverFIO.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tfVertexSolverFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex::io;


HRESULT tfVertexSolverFIOHasImport(bool *result) {
    TFC_PTRCHECK(result);
    return S_OK;
}

HRESULT tfVertexSolverFIOMapImportVertexId(unsigned int fid, unsigned int *mapId) {
    TFC_PTRCHECK(VertexSolverFIOModule::importSummary);
    TFC_PTRCHECK(mapId);

    auto itr = VertexSolverFIOModule::importSummary->vertexIdMap.find(fid);
    if(itr == VertexSolverFIOModule::importSummary->vertexIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}

HRESULT tfVertexSolverFIOMapImportSurfaceId(unsigned int fid, unsigned int *mapId) {
    TFC_PTRCHECK(VertexSolverFIOModule::importSummary);
    TFC_PTRCHECK(mapId);

    auto itr = VertexSolverFIOModule::importSummary->surfaceIdMap.find(fid);
    if(itr == VertexSolverFIOModule::importSummary->surfaceIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}

HRESULT tfVertexSolverFIOMapImportSurfaceTypeId(unsigned int fid, unsigned int *mapId) {
    TFC_PTRCHECK(VertexSolverFIOModule::importSummary);
    TFC_PTRCHECK(mapId);

    auto itr = VertexSolverFIOModule::importSummary->surfaceTypeIdMap.find(fid);
    if(itr == VertexSolverFIOModule::importSummary->surfaceTypeIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}

HRESULT tfVertexSolverFIOMapImportBodyId(unsigned int fid, unsigned int *mapId) {
    TFC_PTRCHECK(VertexSolverFIOModule::importSummary);
    TFC_PTRCHECK(mapId);

    auto itr = VertexSolverFIOModule::importSummary->bodyIdMap.find(fid);
    if(itr == VertexSolverFIOModule::importSummary->bodyIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}

HRESULT tfVertexSolverFIOMapImportBodyTypeId(unsigned int fid, unsigned int *mapId) {
    TFC_PTRCHECK(VertexSolverFIOModule::importSummary);
    TFC_PTRCHECK(mapId);

    auto itr = VertexSolverFIOModule::importSummary->bodyTypeIdMap.find(fid);
    if(itr == VertexSolverFIOModule::importSummary->bodyTypeIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}
