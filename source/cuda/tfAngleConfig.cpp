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

#include "tfAngleConfig.h"

#include <tfEngine.h>
#include <tfLogger.h>


using namespace TissueForge;


bool cuda::AngleConfig::onDevice() {
    return _Engine.angles_cuda;
}

int cuda::AngleConfig::getDevice() {
    return cuda::Angle_getDevice();
}

HRESULT cuda::AngleConfig::toDevice() {
    if(cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting send to device when already sent. Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_toDevice(&_Engine) < 0) { 
        TF_Log(LOG_CRITICAL) << "Attempting send to device failed (" << engine_err << ").";
        return E_FAIL;
    }

    TF_Log(LOG_INFORMATION) << "Successfully sent angles to device";

    return S_OK;
}

HRESULT cuda::AngleConfig::fromDevice() {
    if(!cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting pull from device when not sent. Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_fromDevice(&_Engine) < 0) { 
        TF_Log(LOG_CRITICAL) << "Attempting pull from device failed (" << engine_err << ").";
        return E_FAIL;
    }

    TF_Log(LOG_INFORMATION) << "Successfully pulled angles from device";
    
    return S_OK;
}

HRESULT cuda::AngleConfig::setBlocks(unsigned int numBlocks) {
    if(cuda::AngleConfig::onDevice()) 
        tf_error(E_FAIL, "Angles already on device.");

    if(cuda::Angle_setBlocks(numBlocks) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT cuda::AngleConfig::setThreads(unsigned int numThreads) {
    if(cuda::AngleConfig::onDevice()) 
        tf_error(E_FAIL, "Angles already on device.");

    if(cuda::Angle_setThreads(numThreads) < 0) 
        return E_FAIL;
    return S_OK;
}

HRESULT cuda::AngleConfig::refreshAngle(AngleHandle *bh) {
    if(!cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh angles when not on device. Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_refreshAngle(&_Engine, bh) < 0) { 
        TF_Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT cuda::AngleConfig::refreshAngles(std::vector<AngleHandle*> angles) {
    if(!cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh angles when not on device. Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_refreshAngles(&_Engine, angles.data(), angles.size()) < 0) { 
        TF_Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}

HRESULT cuda::AngleConfig::refresh() {
    if(!cuda::AngleConfig::onDevice()) {
        TF_Log(LOG_DEBUG) << "Attempting to refresh angles when not on device. Ignoring.";
        return S_OK;
    }

    if(cuda::Angle_refresh(&_Engine) < 0) { 
        TF_Log(LOG_CRITICAL) << "Refresh failed (" << engine_err << ").";
        return E_FAIL;
    }

    return S_OK;
}
