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

#ifndef _SOURCE_CUDA_TFANGLECONFIG_H_
#define _SOURCE_CUDA_TFANGLECONFIG_H_

#include <tfAngle_cuda.h>


namespace TissueForge::cuda { 


    /**
     * @brief CUDA runtime control interface for Tissue Forge angles. 
     * 
     * This object provides control for configuring angle calculations 
     * on CUDA devices. 
     * 
     * At any time during a simulation, supported angle calculations 
     * can be sent to a particular CUDA device, or brought back to the 
     * CPU when deployed on a CUDA device. CUDA dynamic parallelism can 
     * also be specified before deploying angle calculations to a CUDA device. 
     * Future Tissue Forge versions will support deployment on multiple devices. 
     * 
     */
    struct CAPI_EXPORT AngleConfig {
        
        /**
         * @brief Check whether the angles are currently on a device. 
         * 
         * @return true 
         * @return false 
         */
        static bool onDevice();

        /**
         * @brief Get the id of the device designated for running angles. 
         * 
         * @return int 
         */
        static int getDevice();

        /**
         * @brief Send angles to device. If angles are already on device, then the call is ignored. 
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
         * Throws an error if called when the angles are already deployed to a CUDA device. 
         * 
         * @param numBlocks number of blocks
         * @return HRESULT 
         */
        static HRESULT setBlocks(unsigned int numBlocks);
        
        /**
         * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
         * 
         * Throws an error if called when angles are already deployed to a CUDA device. 
         * 
         * @param numThreads number of threads
         * @return HRESULT 
         */
        static HRESULT setThreads(unsigned int numThreads);

        /**
         * @brief Update a angle on a CUDA device. 
         * 
         * Useful for notifying the device that a angle has changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @param bh angle to update
         * @return HRESULT 
         */
        static HRESULT refreshAngle(AngleHandle *bh);

        /**
         * @brief Update angles on a CUDA device. 
         * 
         * Useful for notifying the device that angles have changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @param angles angles to update
         * @return HRESULT 
         */
        static HRESULT refreshAngles(std::vector<AngleHandle*> angles);

        /**
         * @brief Update all angles on a CUDA device. 
         * 
         * Useful for notifying the device that angles have changed. 
         * 
         * If engine is not on a device, then the call is ignored. 
         * 
         * @return HRESULT 
         */
        static HRESULT refresh();

    };

};

#endif // _SOURCE_CUDA_TFANGLECONFIG_H_