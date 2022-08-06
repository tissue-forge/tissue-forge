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

#include "tfC_cuda.h"

#include "TissueForge_c_private.h"

#include <cuda/tfSimulatorConfig.h>
#include <tfSimulator.h>


using namespace TissueForge;



//////////////////
// Module casts //
//////////////////


namespace TissueForge { 
    

    cuda::EngineConfig *castC(struct tfCudaEngineConfigHandle *handle) {
        return castC<cuda::EngineConfig, tfCudaEngineConfigHandle>(handle);
    }

    cuda::BondConfig *castC(struct tfCudaBondConfigHandle *handle) {
        return castC<cuda::BondConfig, tfCudaBondConfigHandle>(handle);
    }

    cuda::AngleConfig *castC(struct tfCudaAngleConfigHandle *handle) {
        return castC<cuda::AngleConfig, tfCudaAngleConfigHandle>(handle);
    }

    cuda::SimulatorConfig *castC(struct tfCudaSimulatorConfigHandle *handle) {
        return castC<cuda::SimulatorConfig, tfCudaSimulatorConfigHandle>(handle);
    }

}

#define TFC_CUDAENGINECONFIGHANDLE_GET(handle, varname) \
    cuda::EngineConfig *varname = TissueForge::castC<cuda::EngineConfig, tfCudaEngineConfigHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_CUDABONDCONFIGHANDLE_GET(handle, varname) \
    cuda::BondConfig *varname = TissueForge::castC<cuda::BondConfig, tfCudaBondConfigHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_CUDAANGLECONFIGHANDLE_GET(handle, varname) \
    cuda::AngleConfig *varname = TissueForge::castC<cuda::AngleConfig, tfCudaAngleConfigHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_CUDASIMULATORCONFIGHANDLE_GET(handle, varname) \
    cuda::SimulatorConfig *varname = TissueForge::castC<cuda::SimulatorConfig, tfCudaSimulatorConfigHandle>(handle); \
    TFC_PTRCHECK(varname);


////////////////////////
// cuda::EngineConfig //
////////////////////////


HRESULT tfCudaEngineConfig_onDevice(struct tfCudaEngineConfigHandle *handle, bool *onDevice) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    TFC_PTRCHECK(onDevice);
    *onDevice = engcuda->onDevice();
    return S_OK;
}

HRESULT tfCudaEngineConfig_getDevice(struct tfCudaEngineConfigHandle *handle, int *deviceId) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    TFC_PTRCHECK(deviceId);
    *deviceId = engcuda->getDevice();
    return S_OK;
}

HRESULT tfCudaEngineConfig_setDevice(struct tfCudaEngineConfigHandle *handle, unsigned int deviceId) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->setDevice(deviceId);
}

HRESULT tfCudaEngineConfig_clearDevice(struct tfCudaEngineConfigHandle *handle) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->clearDevice();
}

HRESULT tfCudaEngineConfig_toDevice(struct tfCudaEngineConfigHandle *handle) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->toDevice();
}

HRESULT tfCudaEngineConfig_fromDevice(struct tfCudaEngineConfigHandle *handle) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->fromDevice();
}

HRESULT tfCudaEngineConfig_setBlocks(struct tfCudaEngineConfigHandle *handle, unsigned int numBlocks) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->setBlocks(numBlocks);
}

HRESULT tfCudaEngineConfig_setThreads(struct tfCudaEngineConfigHandle *handle, unsigned int numThreads) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->setThreads(numThreads);
}

HRESULT tfCudaEngineConfig_refreshPotentials(struct tfCudaEngineConfigHandle *handle) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->refreshPotentials();
}

HRESULT tfCudaEngineConfig_refreshFluxes(struct tfCudaEngineConfigHandle *handle) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->refreshFluxes();
}

HRESULT tfCudaEngineConfig_refreshBoundaryConditions(struct tfCudaEngineConfigHandle *handle) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->refreshBoundaryConditions();
}

HRESULT tfCudaEngineConfig_refresh(struct tfCudaEngineConfigHandle *handle) {
    TFC_CUDAENGINECONFIGHANDLE_GET(handle, engcuda);
    return engcuda->refresh();
}


//////////////////////
// cuda::BondConfig //
//////////////////////


HRESULT tfCudaBondConfig_onDevice(struct tfCudaBondConfigHandle *handle, bool *onDevice) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    TFC_PTRCHECK(onDevice);
    *onDevice = bondcuda->onDevice();
    return S_OK;
}

HRESULT tfCudaBondConfig_getDevice(struct tfCudaBondConfigHandle *handle, int *deviceId) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    TFC_PTRCHECK(deviceId);
    *deviceId = bondcuda->getDevice();
    return S_OK;
}

HRESULT tfCudaBondConfig_setDevice(struct tfCudaBondConfigHandle *handle, unsigned deviceId) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setDevice(deviceId);
}

HRESULT tfCudaBondConfig_toDevice(struct tfCudaBondConfigHandle *handle) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->toDevice();
}

HRESULT tfCudaBondConfig_fromDevice(struct tfCudaBondConfigHandle *handle) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->fromDevice();
}

HRESULT tfCudaBondConfig_setBlocks(struct tfCudaBondConfigHandle *handle, unsigned int numBlocks) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setBlocks(numBlocks);
}

HRESULT tfCudaBondConfig_setThreads(struct tfCudaBondConfigHandle *handle, unsigned int numThreads) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setThreads(numThreads);
}

HRESULT tfCudaBondConfig_refreshBond(struct tfCudaBondConfigHandle *handle, struct tfBondHandleHandle *bh) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    TFC_PTRCHECK(bh); TFC_PTRCHECK(bh->tfObj);
    return bondcuda->refreshBond((BondHandle*)bh->tfObj);
}

HRESULT tfCudaBondConfig_refreshBonds(struct tfCudaBondConfigHandle *handle, struct tfBondHandleHandle **bonds, unsigned int numBonds) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    TFC_PTRCHECK(bonds);
    std::vector<BondHandle*> _bonds;
    tfBondHandleHandle *bh;
    for(unsigned int i = 0; i < numBonds; i++) {
        bh = bonds[i];
        TFC_PTRCHECK(bh); TFC_PTRCHECK(bh->tfObj);
        _bonds.push_back((BondHandle*)bh->tfObj);
    }
    return bondcuda->refreshBonds(_bonds);
}

HRESULT tfCudaBondConfig_refresh(struct tfCudaBondConfigHandle *handle) {
    TFC_CUDABONDCONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->refresh();
}


///////////////////////
// cuda::AngleConfig //
///////////////////////


HRESULT tfCudaAngleConfig_onDevice(struct tfCudaAngleConfigHandle *handle, bool *onDevice) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    TFC_PTRCHECK(onDevice);
    *onDevice = bondcuda->onDevice();
    return S_OK;
}

HRESULT tfCudaAngleConfig_getDevice(struct tfCudaAngleConfigHandle *handle, int *deviceId) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    TFC_PTRCHECK(deviceId);
    *deviceId = bondcuda->getDevice();
    return S_OK;
}

HRESULT tfCudaAngleConfig_toDevice(struct tfCudaAngleConfigHandle *handle) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->toDevice();
}

HRESULT tfCudaAngleConfig_fromDevice(struct tfCudaAngleConfigHandle *handle) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->fromDevice();
}

HRESULT tfCudaAngleConfig_setBlocks(struct tfCudaAngleConfigHandle *handle, unsigned int numBlocks) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setBlocks(numBlocks);
}

HRESULT tfCudaAngleConfig_setThreads(struct tfCudaAngleConfigHandle *handle, unsigned int numThreads) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->setThreads(numThreads);
}

HRESULT tfCudaAngleConfig_refreshAngle(struct tfCudaAngleConfigHandle *handle, struct tfAngleHandleHandle *bh) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    TFC_PTRCHECK(bh); TFC_PTRCHECK(bh->tfObj);
    return bondcuda->refreshAngle((AngleHandle*)bh->tfObj);
}

HRESULT tfCudaAngleConfig_refreshAngles(struct tfCudaAngleConfigHandle *handle, struct tfAngleHandleHandle **angles, unsigned int numAngles) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    TFC_PTRCHECK(angles);
    std::vector<AngleHandle*> _angles;
    tfAngleHandleHandle *ah;
    for(unsigned int i = 0; i < numAngles; i++) {
        ah = angles[i];
        TFC_PTRCHECK(ah); TFC_PTRCHECK(ah->tfObj);
        _angles.push_back((AngleHandle*)ah->tfObj);
    }
    return bondcuda->refreshAngles(_angles);
}

HRESULT tfCudaAngleConfig_refresh(struct tfCudaAngleConfigHandle *handle) {
    TFC_CUDAANGLECONFIGHANDLE_GET(handle, bondcuda);
    return bondcuda->refresh();
}


///////////////////////////
// cuda::SimulatorConfig //
///////////////////////////


HRESULT tfSimulator_getCUDAConfig(struct tfCudaSimulatorConfigHandle *handle) {
    TFC_PTRCHECK(handle);
    cuda::SimulatorConfig *simcuda = Simulator::getCUDAConfig();
    TFC_PTRCHECK(simcuda);
    handle->tfObj = (void*)simcuda;
    return S_OK;
}

HRESULT tfSimulatorCUDAConfig_getEngine(struct tfCudaSimulatorConfigHandle *handle, struct tfCudaEngineConfigHandle *itf) {
    TFC_CUDASIMULATORCONFIGHANDLE_GET(handle, simcuda);
    TFC_PTRCHECK(itf);
    itf->tfObj = (void*)(&simcuda->engine);
    return S_OK;
}

HRESULT tfSimulatorCUDAConfig_getBonds(struct tfCudaSimulatorConfigHandle *handle, struct tfCudaBondConfigHandle *itf) {
    TFC_CUDASIMULATORCONFIGHANDLE_GET(handle, simcuda);
    TFC_PTRCHECK(itf);
    itf->tfObj = (void*)(&simcuda->bonds);
    return S_OK;
}

HRESULT tfSimulatorCUDAConfig_getAngles(struct tfCudaSimulatorConfigHandle *handle, struct tfCudaAngleConfigHandle *itf) {
    TFC_CUDASIMULATORCONFIGHANDLE_GET(handle, simcuda);
    TFC_PTRCHECK(itf);
    itf->tfObj = (void*)(&simcuda->angles);
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfCudaArchs(char **str, unsigned int *numChars) {
    return TissueForge::capi::str2Char(TF_CUDA_ARCHS, str, numChars);
}
