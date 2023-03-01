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

#include "tfCMeshQuality.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tfMeshQuality.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    MeshQuality *castC(struct tfVertexSolverMeshQualityHandle *handle) {
        return castC<MeshQuality, tfVertexSolverMeshQualityHandle>(handle);
    }

}

#define TFC_MESHQUALITY_GET(handle, name) \
    MeshQuality *name = TissueForge::castC<MeshQuality, tfVertexSolverMeshQualityHandle>(handle); \
    TFC_PTRCHECK(name);


HRESULT tfVertexSolverMeshQuality_init(
    struct tfVertexSolverMeshQualityHandle *handle, 
    tfFloatP_t vertexMergeDistCf, 
    tfFloatP_t surfaceDemoteAreaCf, 
    tfFloatP_t bodyDemoteVolumeCf, 
    tfFloatP_t _edgeSplitDistCf
) {
    TFC_PTRCHECK(handle);
    MeshQuality *tfObj = new MeshQuality(vertexMergeDistCf, surfaceDemoteAreaCf, bodyDemoteVolumeCf, _edgeSplitDistCf);
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfVertexSolverMeshQuality_destroy(struct tfVertexSolverMeshQualityHandle *handle) {
    return TissueForge::capi::destroyHandle<MeshQuality, tfVertexSolverMeshQualityHandle>(handle);
}

HRESULT tfVertexSolverMeshQuality_toString(
    struct tfVertexSolverMeshQualityHandle *handle, 
    char **str, 
    unsigned int *numChar
) {
    TFC_MESHQUALITY_GET(handle, quality);
    // 
    return S_OK;
}

/**
 * @brief Perform quality operations work
 * 
 * @param handle populated handle
 */
HRESULT tfVertexSolverMeshQuality_doQuality(struct tfVertexSolverMeshQualityHandle *handle) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->doQuality();
}

HRESULT tfVertexSolverMeshQuality_working(struct tfVertexSolverMeshQualityHandle *handle, bool *working) {
    TFC_MESHQUALITY_GET(handle, quality);
    *working = quality->working();
    return S_OK;
}

HRESULT tfVertexSolverMeshQuality_getVertexMergeDistance(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t *val) {
    TFC_MESHQUALITY_GET(handle, quality);
    *val = quality->getVertexMergeDistance();
    return S_OK;
}

HRESULT tfVertexSolverMeshQuality_setVertexMergeDistance(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t val) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->setVertexMergeDistance(val);
}

HRESULT tfVertexSolverMeshQuality_getSurfaceDemoteArea(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t *val) {
    TFC_MESHQUALITY_GET(handle, quality);
    *val = quality->getSurfaceDemoteArea();
    return S_OK;
}

HRESULT tfVertexSolverMeshQuality_setSurfaceDemoteArea(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t val) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->setSurfaceDemoteArea(val);
}

HRESULT tfVertexSolverMeshQuality_getBodyDemoteVolume(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t *val) {
    TFC_MESHQUALITY_GET(handle, quality);
    *val = quality->getBodyDemoteVolume();
    return S_OK;
}

HRESULT tfVertexSolverMeshQuality_setBodyDemoteVolume(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t val) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->setBodyDemoteVolume(val);
}

HRESULT tfVertexSolverMeshQuality_getEdgeSplitDist(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t *val) {
    TFC_MESHQUALITY_GET(handle, quality);
    *val = quality->getEdgeSplitDist();
    return S_OK;
}

HRESULT tfVertexSolverMeshQuality_setEdgeSplitDist(struct tfVertexSolverMeshQualityHandle *handle, tfFloatP_t val) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->setEdgeSplitDist(val);
}

HRESULT tfVertexSolverMeshQuality_getCollision2D(struct tfVertexSolverMeshQualityHandle *handle, bool *collision2D) {
    TFC_MESHQUALITY_GET(handle, quality);
    *collision2D = quality->getCollision2D();
    return S_OK;
}

HRESULT tfVertexSolverMeshQuality_setCollision2D(struct tfVertexSolverMeshQualityHandle *handle, bool collision2D) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->setCollision2D(collision2D);
}

HRESULT tfVertexSolverMeshQuality_excludeVertex(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->excludeVertex(id);
}

HRESULT tfVertexSolverMeshQuality_excludeSurface(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->excludeSurface(id);
}

HRESULT tfVertexSolverMeshQuality_excludeBody(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->excludeBody(id);
}

HRESULT tfVertexSolverMeshQuality_includeVertex(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->includeVertex(id);
}

HRESULT tfVertexSolverMeshQuality_includeSurface(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->includeSurface(id);
}

HRESULT tfVertexSolverMeshQuality_includeBody(struct tfVertexSolverMeshQualityHandle *handle, const unsigned int &id) {
    TFC_MESHQUALITY_GET(handle, quality);
    return quality->includeBody(id);
}
