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

#include "tfEngineConfig.h"

#include <tfEngine.h>
#include <tfLogger.h>


using namespace TissueForge;


cuda::EngineConfig::EngineConfig() : 
    on_device{false}
{}

bool cuda::EngineConfig::onDevice() {
    return this->on_device;
}

int cuda::EngineConfig::getDevice() {
    if(_Engine.nr_devices == 0) return -1;
    return _Engine.devices[0];
}

HRESULT cuda::EngineConfig::setDevice(int deviceId) {
    if(this->onDevice()) tf_error(E_FAIL, "Engine already on device");

    if(engine_cuda_setdevice(&_Engine, deviceId) < 0)
        return E_FAIL;

    return S_OK;
}

HRESULT cuda::EngineConfig::clearDevice() {
    if(this->onDevice()) tf_error(E_FAIL, "Engine on device");

    if(engine_cuda_cleardevices(&_Engine) < 0)
        return E_FAIL;

    return S_OK;
}

HRESULT cuda::EngineConfig::toDevice() {
    if(this->onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting send to device when already sent. Ignoring.";
        return S_OK;
    }

    if(_Engine.nr_devices == 0) this->setDevice();
    if(engine_toCUDA(&_Engine) < 0) {
        TF_Log(LOG_CRITICAL) << "Attempting send to device failed (" << engine_err << ").";
        return E_FAIL;
    }

    TF_Log(LOG_INFORMATION) << "Successfully sent engine to device";
    this->on_device = true;

    return S_OK;
}

HRESULT cuda::EngineConfig::fromDevice() {
    if(!this->onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting pull from device when not sent. Ignoring.";
        return S_OK;
    }

    if(engine_fromCUDA(&_Engine) < 0) {
        TF_Log(LOG_CRITICAL) << "Attempting pull from device failed (" << engine_err << ").";
        return E_FAIL;
    }

    TF_Log(LOG_INFORMATION) << "Successfully pulled engine from device";
    this->on_device = false;

    return S_OK;
}

HRESULT cuda::EngineConfig::setBlocks(unsigned int numBlocks, int deviceId) {
    if(this->onDevice()) {
        tf_error(E_FAIL, "Engine already on device.");
    }
    
    if(_Engine.nr_devices == 0) this->setDevice();
    if(deviceId < 0) deviceId = _Engine.devices[0];
    
    if(engine_cuda_setblocks(&_Engine, deviceId, numBlocks) < 0)
        return E_FAIL;

    return S_OK;
}

HRESULT cuda::EngineConfig::setThreads(unsigned int numThreads, int deviceId) {
    if(this->onDevice()) {
        tf_error(E_FAIL, "Engine already on device.");
    }

    if(_Engine.nr_devices == 0) this->setDevice();
    if(deviceId < 0) deviceId = _Engine.devices[0];

    if(engine_cuda_setthreads(&_Engine, deviceId, numThreads) < 0)
        return E_FAIL;

    return S_OK;
}

HRESULT cuda::EngineConfig::refreshPotentials() {
    if(!this->onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh potentials when not on device. Ignoring.";
        return S_OK;
    }

    if(engine_cuda_refresh_pots(&_Engine) < 0) {
        TF_Log(LOG_CRITICAL) << "Attempting to refresh potentials failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT cuda::EngineConfig::refreshFluxes() {
    if(!this->onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh fluxes when not on device. Ignoring.";
        return S_OK;
    }

    if(engine_cuda_refresh_fluxes(&_Engine) < 0) {
        TF_Log(LOG_CRITICAL) << "Attempting to refresh fluxes failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT cuda::EngineConfig::refreshBoundaryConditions() {
    if(!this->onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh boundary conditions when not on device. Ignoring.";
        return S_OK;
    }

    if(engine_cuda_boundary_conditions_refresh(&_Engine) < 0) {
        TF_Log(LOG_CRITICAL) << "Attempting to refresh boundary conditions failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT cuda::EngineConfig::refresh() {
    if(!this->onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh engine when not on device. Ignoring.";
        return S_OK;
    }

    if(engine_cuda_refresh(&_Engine) < 0) {
        TF_Log(LOG_CRITICAL) << "Attempting to refresh engine failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}
