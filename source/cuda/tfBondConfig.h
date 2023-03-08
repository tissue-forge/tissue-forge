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

#ifndef _SOURCE_CUDA_TFBONDCONFIG_H_
#define _SOURCE_CUDA_TFBONDCONFIG_H_

#include <tfBond_cuda.h>


namespace TissueForge::cuda { 


    /**
     * @brief CUDA runtime control interface for Tissue Forge bonds. 
     * 
     * This object provides control for configuring bond calculations 
     * on CUDA devices. 
     * 
     * At any time during a simulation, supported bond calculations 
     * can be sent to a particular CUDA device, or brought back to the 
     * CPU when deployed on a CUDA device. CUDA dynamic parallelism can 
     * also be specified before deploying bond calculations to a CUDA device. 
     * Future Tissue Forge versions will support deployment on multiple devices. 
     * 
     */
    struct CAPI_EXPORT BondConfig {
        
        /**
         * @brief Check whether the bonds are currently on a device. 
         * 
         * @return true 
         * @return false 
         */
        static bool onDevice();

        /**
         * @brief Get the id of the device designated for running bonds. 
         * 
         * @return int 
         */
        static int getDevice();

        /**
         * @brief Set the id of the device for running bonds. 
         * 
         * Can be safely called while bonds are currently on a device. 
         * 
         * @param deviceId 
         * @return HRESULT 
         */
        static HRESULT setDevice(int deviceId=0);

        /**
         * @brief Send bonds to device. If bonds are already on device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        static HRESULT toDevice();

        /**
         * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        static HRESULT fromDevice();

        /**
         * @brief Set the number of blocks of the CUDA configuration for a CUDA device. 
         * 
         * Throws an error if called when the bonds are already deployed to a CUDA device. 
         * 
         * @param numBlocks number of blocks
         * @return HRESULT 
         */
        static HRESULT setBlocks(unsigned int numBlocks);
        
        /**
         * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
         * 
         * Throws an error if called when bonds are already deployed to a CUDA device. 
         * 
         * @param numThreads number of threads
         * @return HRESULT 
         */
        static HRESULT setThreads(unsigned int numThreads);

        /**
         * @brief Update a bond on a CUDA device. 
         * 
         * Useful for notifying the device that a bond has changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @param bh bond to update
         * @return HRESULT 
         */
        static HRESULT refreshBond(BondHandle *bh);

        /**
         * @brief Update bonds on a CUDA device. 
         * 
         * Useful for notifying the device that bonds have changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @param bonds bonds to update
         * @return HRESULT 
         */
        static HRESULT refreshBonds(std::vector<BondHandle*> bonds);

        /**
         * @brief Update all bonds on a CUDA device. 
         * 
         * Useful for notifying the device that bonds have changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        static HRESULT refresh();

    };

};

#endif // _SOURCE_CUDA_TFBONDCONFIG_H_