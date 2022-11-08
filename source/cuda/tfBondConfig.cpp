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

#include <tfError.h>
#include <tfEngine.h>
#include <tfLogger.h>
#include <tf_cuda.h>


using namespace TissueForge;


#define error(id)   tf_error(E_FAIL, tfcuda_err_msg[id])


bool cuda::BondConfig::onDevice() {
    return _Engine.bonds_cuda;
}

int cuda::BondConfig::getDevice() {
    return cuda::Bond_getDevice();
}

HRESULT cuda::BondConfig::setDevice(int deviceId) {
    if(cuda::Bond_setDevice(&_Engine, deviceId) != S_OK) 
        return error(TFCUDAERR_setdevice);
    return S_OK;
}

HRESULT cuda::BondConfig::toDevice() {
    if(cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_ondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_toDevice(&_Engine) != S_OK) 
        return error(TFCUDAERR_send);

    TF_Log(LOG_INFORMATION) << "Successfully sent bonds to device";

    return S_OK;
}

HRESULT cuda::BondConfig::fromDevice() {
    if(!cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_notondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_fromDevice(&_Engine) != S_OK) 
        return error(TFCUDAERR_pull);

    TF_Log(LOG_INFORMATION) << "Successfully pulled bonds from device";
    
    return S_OK;
}

HRESULT cuda::BondConfig::setBlocks(unsigned int numBlocks) {
    if(cuda::BondConfig::onDevice()) 
        return error(TFCUDAERR_ondevice);

    if(cuda::Bond_setBlocks(numBlocks) != S_OK) 
        return error(TFCUDAERR_setblocks);
    return S_OK;
}

HRESULT cuda::BondConfig::setThreads(unsigned int numThreads) {
    if(cuda::BondConfig::onDevice()) 
        return error(TFCUDAERR_ondevice);

    if(cuda::Bond_setThreads(numThreads) != S_OK) 
        return error(TFCUDAERR_setthreads);
    return S_OK;
}

HRESULT cuda::BondConfig::refreshBond(BondHandle *bh) {
    if(!cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_notondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_refreshBond(&_Engine, bh) != S_OK) 
        return error(TFCUDAERR_refresh);

    return S_OK;
}

HRESULT cuda::BondConfig::refreshBonds(std::vector<BondHandle*> bonds) {
    if(!cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_notondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_refreshBonds(&_Engine, bonds.data(), bonds.size()) != S_OK) 
        return error(TFCUDAERR_refresh);

    return S_OK;
}

HRESULT cuda::BondConfig::refresh() {
    if(!cuda::BondConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_notondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Bond_refresh(&_Engine) != S_OK) 
        return error(TFCUDAERR_refresh);

    return S_OK;
}
