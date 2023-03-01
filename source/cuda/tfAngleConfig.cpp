/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#include "tfAngleConfig.h"

#include <tfError.h>
#include <tfEngine.h>
#include <tfLogger.h>
#include <tf_cuda.h>


using namespace TissueForge;


#define error(id)   tf_error(E_FAIL, tfcuda_err_msg[id])


bool cuda::AngleConfig::onDevice() {
    return _Engine.angles_cuda;
}

int cuda::AngleConfig::getDevice() {
    return cuda::Angle_getDevice();
}

HRESULT cuda::AngleConfig::toDevice() {
    if(cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_ondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_toDevice(&_Engine) != S_OK) 
        return error(TFCUDAERR_send);

    TF_Log(LOG_INFORMATION) << "Successfully sent angles to device";

    return S_OK;
}

HRESULT cuda::AngleConfig::fromDevice() {
    if(!cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_notondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_fromDevice(&_Engine) != S_OK) 
        return error(TFCUDAERR_pull);

    TF_Log(LOG_INFORMATION) << "Successfully pulled angles from device";
    
    return S_OK;
}

HRESULT cuda::AngleConfig::setBlocks(unsigned int numBlocks) {
    if(cuda::AngleConfig::onDevice()) 
        return error(TFCUDAERR_ondevice);

    if(cuda::Angle_setBlocks(numBlocks) != S_OK) 
        return error(TFCUDAERR_setblocks);
    return S_OK;
}

HRESULT cuda::AngleConfig::setThreads(unsigned int numThreads) {
    if(cuda::AngleConfig::onDevice()) 
        return error(TFCUDAERR_ondevice);

    if(cuda::Angle_setThreads(numThreads) != S_OK) 
        return error(TFCUDAERR_setthreads);
    return S_OK;
}

HRESULT cuda::AngleConfig::refreshAngle(AngleHandle *bh) {
    if(!cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_notondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_refreshAngle(&_Engine, bh) != S_OK) 
        return error(TFCUDAERR_refresh);

    return S_OK;
}

HRESULT cuda::AngleConfig::refreshAngles(std::vector<AngleHandle*> angles) {
    if(!cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_notondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_refreshAngles(&_Engine, angles.data(), angles.size()) != S_OK) 
        return error(TFCUDAERR_refresh);

    return S_OK;
}

HRESULT cuda::AngleConfig::refresh() {
    if(!cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << tfcuda_err_msg[TFCUDAERR_notondevice] << " Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_refresh(&_Engine) != S_OK) 
        return error(TFCUDAERR_refresh);

    return S_OK;
}
