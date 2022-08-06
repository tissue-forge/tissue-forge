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

#include "tfBondConfig.h"

#include <tfEngine.h>
#include <tfLogger.h>


using namespace TissueForge;


bool cuda::BondConfig::onDevice() {
    return _Engine.bonds_cuda;
}

int cuda::BondConfig::getDevice() {
    return cuda::Bond_getDevice();
}

HRESULT cuda::BondConfig::setDevice(int deviceId) {
    if(cuda::Bond_setDevice(&_Engine, deviceId) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT cuda::BondConfig::toDevice() {
    if(cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting send to device when already sent. Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_toDevice(&_Engine) < 0) { 
        TF_Log(LOG_CRITICAL) << "Attempting send to device failed (" << engine_err << ").";
        return E_FAIL;
    }

    TF_Log(LOG_INFORMATION) << "Successfully sent bonds to device";

    return S_OK;
}

HRESULT cuda::BondConfig::fromDevice() {
    if(!cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting pull from device when not sent. Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_fromDevice(&_Engine) < 0) { 
        TF_Log(LOG_CRITICAL) << "Attempting pull from device failed (" << engine_err << ").";
        return E_FAIL;
    }

    TF_Log(LOG_INFORMATION) << "Successfully pulled bonds from device";
    
    return S_OK;
}

HRESULT cuda::BondConfig::setBlocks(unsigned int numBlocks) {
    if(cuda::BondConfig::onDevice()) 
        tf_error(E_FAIL, "Bonds already on device.");

    if(cuda::Bond_setBlocks(numBlocks) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT cuda::BondConfig::setThreads(unsigned int numThreads) {
    if(cuda::BondConfig::onDevice()) 
        tf_error(E_FAIL, "Bonds already on device.");

    if(cuda::Bond_setThreads(numThreads) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT cuda::BondConfig::refreshBond(BondHandle *bh) {
    if(!cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh bonds when not on device. Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_refreshBond(&_Engine, bh) < 0) { 
        TF_Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT cuda::BondConfig::refreshBonds(std::vector<BondHandle*> bonds) {
    if(!cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh bonds when not on device. Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_refreshBonds(&_Engine, bonds.data(), bonds.size()) < 0) { 
        TF_Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT cuda::BondConfig::refresh() {
    if(!cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh bonds when not on device. Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_refresh(&_Engine) < 0) { 
        TF_Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}
