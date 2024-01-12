/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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

#ifndef _SOURCE_CUDA_TFENGINECONFIG_H_
#define _SOURCE_CUDA_TFENGINECONFIG_H_

#include <tf_cuda.h>


namespace TissueForge::cuda { 


    /**
     * @brief CUDA runtime control interface for Tissue Forge engine. 
     * 
     * This object provides control for configuring engine calculations 
     * on CUDA devices. Associated calculations include nonbonded particle 
     * interactions and sorting, fluxes and space partitioning. 
     * 
     * At any time during a simulation, supported engine calculations 
     * can be sent to a particular CUDA device, or brought back to the 
     * CPU when deployed on a CUDA device. CUDA dynamic parallelism can 
     * also be specified before deploying engine calculations to a CUDA device. 
     * Future Tissue Forge versions will support deployment on multiple devices. 
     * 
     */
    struct CAPI_EXPORT EngineConfig {
        
        EngineConfig();
        ~EngineConfig() {}

        /**
         * @brief Check whether the engine is currently on a device. 
         * 
         * @return true 
         * @return false 
         */
        bool onDevice();

        /**
         * @brief Get the id of the device running the engine. 
         * 
         * Returns -1 if engine is not on a device. 
         * 
         * @return int 
         */
        int getDevice();

        /**
         * @brief Set the id of the device for running the engine. 
         * 
         * Fails if engine is currently on a device. 
         * 
         * @param deviceId 
         * @return HRESULT 
         */
        HRESULT setDevice(int deviceId=0);

        /**
         * @brief Clear configured device for the engine. 
         * 
         * Fails if engine is currently on a device. 
         * 
         * @return HRESULT 
         */
        HRESULT clearDevice();

        /**
         * @brief Send engine to device. If engine is already on device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        HRESULT toDevice();

        /**
         * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        HRESULT fromDevice();

        /**
         * @brief Set the number of blocks of the CUDA configuration for a CUDA device. 
         * 
         * Throws an error if called when the engine is already deployed to a CUDA device. 
         * 
         * @param numBlocks number of blocks
         * @param deviceId device ID (optional)
         * @return HRESULT 
         */
        HRESULT setBlocks(unsigned int numBlocks, int deviceId=-1);
        
        /**
         * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
         * 
         * Throws an error if called when the engine is already deployed to a CUDA device. 
         * 
         * @param numThreads number of threads
         * @param deviceId device ID (optional)
         * @return HRESULT 
         */
        HRESULT setThreads(unsigned int numThreads, int deviceId=-1);

        /**
         * @brief Update potentials on a CUDA device. 
         * 
         * Useful for notifying the device that a potential has changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        HRESULT refreshPotentials();

        /**
         * @brief Update fluxes on a CUDA device. 
         * 
         * Useful for notifying the device that a flux has changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        HRESULT refreshFluxes();

        /**
         * @brief Update boundary conditions on a CUDA device. 
         * 
         * Useful for notifying the device that a boundary condition has changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        HRESULT refreshBoundaryConditions();

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
         * @return HRESULT 
         */
        HRESULT refresh();

    private:
        bool on_device;
    };

};

#endif // _SOURCE_CUDA_TFENGINECONFIG_H_