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

/**
 * @file tfC_cuda.h
 * 
 */

#ifndef _WRAPS_C_TFC_CUDA_H_
#define _WRAPS_C_TFC_CUDA_H_

#include "tf_port_c.h"

#include "tfCBond.h"

// Handles

/**
 * @brief Handle to a @ref cuda::EngineConfig instance
 * 
 */
struct CAPI_EXPORT tfCudaEngineConfigHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref cuda::BondConfig instance
 * 
 */
struct CAPI_EXPORT tfCudaBondConfigHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref cuda::AngleConfig instance
 * 
 */
struct CAPI_EXPORT tfCudaAngleConfigHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref cuda::SimulatorConfig instance
 * 
 */
struct CAPI_EXPORT tfCudaSimulatorConfigHandle {
    void *tfObj;
};


////////////////////////
// cuda::EngineConfig //
////////////////////////


/**
 * @brief Check whether the engine is currently on a device. 
 * 
 * @param handle populated handle
 * @param onDevice true if currently on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_onDevice(struct tfCudaEngineConfigHandle *handle, bool *onDevice);

/**
 * @brief Get the id of the device running the engine. 
 * 
 * @param handle populated handle
 * @param deviceId device id; -1 if engine is not on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_getDevice(struct tfCudaEngineConfigHandle *handle, int *deviceId);

/**
 * @brief Set the id of the device for running the engine. 
 * 
 * Fails if engine is currently on a device. 
 * 
 * @param handle populated handle
 * @param deviceId device id
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_setDevice(struct tfCudaEngineConfigHandle *handle, unsigned int deviceId);

/**
 * @brief Clear configured device for the engine. 
 * 
 * Fails if engine is currently on a device. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_clearDevice(struct tfCudaEngineConfigHandle *handle);

/**
 * @brief Send engine to device. If engine is already on device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_toDevice(struct tfCudaEngineConfigHandle *handle);

/**
 * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_fromDevice(struct tfCudaEngineConfigHandle *handle);

/**
 * @brief Set the number of blocks of the CUDA configuration for the current CUDA device. 
 * 
 * Throws an error if called when the engine is already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numBlocks number of blocks
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_setBlocks(struct tfCudaEngineConfigHandle *handle, unsigned int numBlocks);

/**
 * @brief Set the number of threads of the CUDA configuration for the current CUDA device. 
 * 
 * Throws an error if called when the engine is already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numThreads number of threads
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_setThreads(struct tfCudaEngineConfigHandle *handle, unsigned int numThreads);

/**
 * @brief Update potentials on a CUDA device. 
 * 
 * Useful for notifying the device that a potential has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_refreshPotentials(struct tfCudaEngineConfigHandle *handle);

/**
 * @brief Update fluxes on a CUDA device. 
 * 
 * Useful for notifying the device that a flux has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_refreshFluxes(struct tfCudaEngineConfigHandle *handle);

/**
 * @brief Update boundary conditions on a CUDA device. 
 * 
 * Useful for notifying the device that a boundary condition has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_refreshBoundaryConditions(struct tfCudaEngineConfigHandle *handle);

/**
 * @brief Update the image of the engine on a CUDA device. 
 * 
 * Necessary to notify the device of changes to engine data that 
 * are not automatically handled by Tissue Forge. Refer to documentation 
 * of specific functions and members for which Tissue Forge 
 * automatically handles. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaEngineConfig_refresh(struct tfCudaEngineConfigHandle *handle);


//////////////////////
// cuda::BondConfig //
//////////////////////


/**
 * @brief Check whether the bonds are currently on a device. 
 * 
 * @param handle populated handle
 * @param onDevice true if currently on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_onDevice(struct tfCudaBondConfigHandle *handle, bool *onDevice);

/**
 * @brief Get the id of the device designated for running bonds. 
 * 
 * @param handle populated handle
 * @param deviceId device id; -1 if engine is not on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_getDevice(struct tfCudaBondConfigHandle *handle, int *deviceId);

/**
 * @brief Set the id of the device for running bonds. 
 * 
 * Can be safely called while bonds are currently on a device. 
 * 
 * @param handle populated handle
 * @param deviceId device id
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_setDevice(struct tfCudaBondConfigHandle *handle, unsigned int deviceId);

/**
 * @brief Send bonds to device. If bonds are already on device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_toDevice(struct tfCudaBondConfigHandle *handle);

/**
 * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_fromDevice(struct tfCudaBondConfigHandle *handle);

/**
 * @brief Set the number of blocks of the CUDA configuration for a CUDA device. 
 * 
 * Throws an error if called when the bonds are already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numBlocks number of blocks
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_setBlocks(struct tfCudaBondConfigHandle *handle, unsigned int numBlocks);

/**
 * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
 * 
 * Throws an error if called when bonds are already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numThreads number of threads
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_setThreads(struct tfCudaBondConfigHandle *handle, unsigned int numThreads);

/**
 * @brief Update a bond on a CUDA device. 
 * 
 * Useful for notifying the device that a bond has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @param bh bond to update
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_refreshBond(struct tfCudaBondConfigHandle *handle, struct tfBondHandleHandle *bh);

/**
 * @brief Update bonds on a CUDA device. 
 * 
 * Useful for notifying the device that bonds have changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @param bonds bonds to update
 * @param numBonds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_refreshBonds(struct tfCudaBondConfigHandle *handle, struct tfBondHandleHandle **bonds, unsigned int numBonds);

/**
 * @brief Update all bonds on a CUDA device. 
 * 
 * Useful for notifying the device that bonds have changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaBondConfig_refresh(struct tfCudaBondConfigHandle *handle);


///////////////////////
// cuda::AngleConfig //
///////////////////////


/**
 * @brief Check whether the angles are currently on a device. 
 * 
 * @param handle populated handle
 * @param onDevice true if currently on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_onDevice(struct tfCudaAngleConfigHandle *handle, bool *onDevice);

/**
 * @brief Get the id of the device designated for running angles. 
 * 
 * @param handle populated handle
 * @param deviceId device id; -1 if engine is not on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_getDevice(struct tfCudaAngleConfigHandle *handle, int *deviceId);

/**
 * @brief Send angles to device. If angles are already on device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_toDevice(struct tfCudaAngleConfigHandle *handle);

/**
 * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_fromDevice(struct tfCudaAngleConfigHandle *handle);

/**
 * @brief Set the number of blocks of the CUDA configuration for a CUDA device. 
 * 
 * Throws an error if called when the angles are already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numBlocks number of blocks
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_setBlocks(struct tfCudaAngleConfigHandle *handle, unsigned int numBlocks);

/**
 * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
 * 
 * Throws an error if called when angles are already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numThreads number of threads
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_setThreads(struct tfCudaAngleConfigHandle *handle, unsigned int numThreads);

/**
 * @brief Update a angle on a CUDA device. 
 * 
 * Useful for notifying the device that a angle has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @param bh angle to update
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_refreshAngle(struct tfCudaAngleConfigHandle *handle, struct tfAngleHandleHandle *bh);

/**
 * @brief Update angles on a CUDA device. 
 * 
 * Useful for notifying the device that angles have changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @param angles angles to update
 * @param numAngles number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_refreshAngles(struct tfCudaAngleConfigHandle *handle, struct tfAngleHandleHandle **angles, unsigned int numAngles);

/**
 * @brief Update all angles on a CUDA device. 
 * 
 * Useful for notifying the device that angles have changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaAngleConfig_refresh(struct tfCudaAngleConfigHandle *handle);


///////////////////////////
// cuda::SimulatorConfig //
///////////////////////////


/**
 * @brief Get simulator CUDA runtime interface
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSimulator_getCUDAConfig(struct tfCudaSimulatorConfigHandle *handle);

/**
 * @brief Get the engine CUDA runtime control interface
 * 
 * @param handle populated handle
 * @param itf control interface
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSimulatorCUDAConfig_getEngine(struct tfCudaSimulatorConfigHandle *handle, struct tfCudaEngineConfigHandle *itf);

/**
 * @brief Get the bond CUDA runtime control interface
 * 
 * @param handle populated handle
 * @param itf control interface
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSimulatorCUDAConfig_getBonds(struct tfCudaSimulatorConfigHandle *handle, struct tfCudaBondConfigHandle *itf);

/**
 * @brief Get the angle CUDA runtime control interface
 * 
 * @param handle populated handle
 * @param itf control interface
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSimulatorCUDAConfig_getAngles(struct tfCudaSimulatorConfigHandle *handle, struct tfCudaAngleConfigHandle *itf);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the supported CUDA architectures of this installation
 * 
 * @param str architectures
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfCudaArchs(char **str, unsigned int *numChars);


#endif // _WRAPS_C_TFC_CUDA_H_