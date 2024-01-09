/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego and Tien Comlekoglu
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

#include "tfC_mesh_create.h"

#include "tfCSurface.h"
#include "tfCBody.h"

#include <TissueForge_c_private.h>

#include <models/vertex/solver/tf_mesh_create.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    static SurfaceHandle *castC(struct tfVertexSolverSurfaceHandleHandle *handle) {
        return castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle);
    }

    static SurfaceType *castC(struct tfVertexSolverSurfaceTypeHandle *handle) {
        return castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle);
    }

    static BodyHandle *castC(struct tfVertexSolverBodyHandleHandle *handle) {
        return castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle);
    }

    static BodyType *castC(struct tfVertexSolverBodyTypeHandle *handle) {
        return castC<BodyType, tfVertexSolverBodyTypeHandle>(handle);
    }

}

#define TFC_MESHCREATE_GETSURFACEHANDLE(handle, name) \
    SurfaceHandle *name = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHCREATE_GETSURFACETYPE(handle, name) \
    SurfaceType *name = TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHCREATE_GETBODYHANDLE(handle, name) \
    BodyHandle *name = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHCREATE_GETBODYTYPE(handle, name) \
    BodyType *name = TissueForge::castC<BodyType, tfVertexSolverBodyTypeHandle>(handle); \
    TFC_PTRCHECK(name);


static HRESULT tfVertexSolverCreate_returnSurfaceArrayRegular(
    const std::vector<std::vector<SurfaceHandle> > &_result, 
    tfVertexSolverSurfaceHandleHandle **result
) {
    size_t num_1 = _result.size();
    if(num_1 == 0) 
        return E_FAIL;
    size_t num_2 = _result.front().size();
    if(num_2 == 0) 
        return E_FAIL;

    *result = (tfVertexSolverSurfaceHandleHandle*)malloc(sizeof(tfVertexSolverSurfaceHandleHandle) * num_1 * num_2);
    for(size_t i = 0; i < num_1; i++) {
        tfVertexSolverSurfaceHandleHandle *result_i = &(*result)[num_2 * i];
        const std::vector<SurfaceHandle> &_result_i = _result[i];
        for(size_t j = 0; j < num_2; j++) 
            if(tfVertexSolverSurfaceHandle_init(&result_i[j], _result_i[j].id) != S_OK) 
                return E_FAIL;
    }
    return S_OK;
}

static HRESULT tfVertexSolverCreate_returnBodyArrayRegular(
    const std::vector<std::vector<std::vector<BodyHandle> > > &_result, 
    tfVertexSolverBodyHandleHandle **result
) {
    size_t num_1 = _result.size();
    if(num_1 == 0) 
        return E_FAIL;
    size_t num_2 = _result.front().size();
    if(num_2 == 0) 
        return E_FAIL;
    size_t num_3 = _result.front().front().size();
    if(num_3 == 0) 
        return E_FAIL;

    *result = (tfVertexSolverBodyHandleHandle*)malloc(sizeof(tfVertexSolverBodyHandleHandle) * num_1 * num_2 * num_3);
    for(size_t i = 0; i < num_1; i++) {
        tfVertexSolverBodyHandleHandle *result_i = &(*result)[num_2 * num_3 * i];
        const std::vector<std::vector<BodyHandle> > &_result_i = _result[i];
        for(size_t j = 0; j < num_2; j++) {
        tfVertexSolverBodyHandleHandle *result_j = &result_i[num_3 * j];
            const std::vector<BodyHandle> &_result_j = _result_i[j];
            for(size_t k = 0; k < num_3; k++) 
                if(tfVertexSolverBodyHandle_init(&result_j[k], _result_j[k]) != S_OK) 
                    return E_FAIL;
        }
    }

    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolverCreateQuadMesh(
    struct tfVertexSolverSurfaceTypeHandle *stype,
    tfFloatP_t *startPos, 
    unsigned int num_1, 
    unsigned int num_2, 
    tfFloatP_t len_1,
    tfFloatP_t len_2,
    const char *ax_1, 
    const char *ax_2, 
    struct tfVertexSolverSurfaceHandleHandle **result
) {
    TFC_MESHCREATE_GETSURFACETYPE(stype, _stype);
    FVector3 _startPos = FVector3::from(startPos);
    std::vector<std::vector<SurfaceHandle> > _result = createQuadMesh(
        _stype, _startPos, num_1, num_2, len_1, len_2, ax_1, ax_2
    );
    if(_result.size() == 0) 
        return E_FAIL;
    return tfVertexSolverCreate_returnSurfaceArrayRegular(_result, result);
}

HRESULT tfVertexSolverCreatePLPDMesh(
    struct tfVertexSolverBodyTypeHandle *btype, 
    struct tfVertexSolverSurfaceTypeHandle *stype,
    tfFloatP_t *startPos, 
    unsigned int num_1, 
    unsigned int num_2, 
    unsigned int num_3, 
    tfFloatP_t len_1,
    tfFloatP_t len_2,
    tfFloatP_t len_3,
    const char *ax_1, 
    const char *ax_2, 
    struct tfVertexSolverBodyHandleHandle **result
) {
    TFC_MESHCREATE_GETBODYTYPE(btype, _btype);
    TFC_MESHCREATE_GETSURFACETYPE(stype, _stype);
    FVector3 _startPos = FVector3::from(startPos);
    std::vector<std::vector<std::vector<BodyHandle> > > _result = createPLPDMesh(
        _btype, _stype, _startPos, num_1, num_2, num_3, len_1, len_2, len_3, ax_1, ax_2
    );
    if(_result.size() == 0) 
        return E_FAIL;
    return tfVertexSolverCreate_returnBodyArrayRegular(_result, result);
}

HRESULT tfVertexSolverCreateHex2DMesh(
    struct tfVertexSolverSurfaceTypeHandle *stype, 
    tfFloatP_t *startPos, 
    unsigned int num_1, 
    unsigned int num_2, 
    tfFloatP_t hexRad,
    const char *ax_1, 
    const char *ax_2, 
    struct tfVertexSolverSurfaceHandleHandle **result
) {
    TFC_MESHCREATE_GETSURFACETYPE(stype, _stype);
    FVector3 _startPos = FVector3::from(startPos);
    std::vector<std::vector<SurfaceHandle> > _result = createHex2DMesh(
        _stype, _startPos, num_1, num_2, hexRad, ax_1, ax_2
    );
    if(_result.size() == 0) 
        return E_FAIL;
    return tfVertexSolverCreate_returnSurfaceArrayRegular(_result, result);
}

HRESULT tfVertexSolverCreateHex3DMesh(
    struct tfVertexSolverBodyTypeHandle *btype, 
    struct tfVertexSolverSurfaceTypeHandle *stype,
    tfFloatP_t *startPos, 
    unsigned int num_1, 
    unsigned int num_2, 
    unsigned int num_3, 
    tfFloatP_t hexRad,
    tfFloatP_t hex_height,
    const char *ax_1, 
    const char *ax_2, 
    struct tfVertexSolverBodyHandleHandle **result
) {
    TFC_MESHCREATE_GETBODYTYPE(btype, _btype);
    TFC_MESHCREATE_GETSURFACETYPE(stype, _stype);
    FVector3 _startPos = FVector3::from(startPos);
    std::vector<std::vector<std::vector<BodyHandle> > > _result = createHex3DMesh(
        _btype, _stype, _startPos, num_1, num_2, num_3, hexRad, hex_height, ax_1, ax_2
    );
    if(_result.size() == 0) 
        return E_FAIL;
    return tfVertexSolverCreate_returnBodyArrayRegular(_result, result);
}
