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

#ifndef _SOURCE_CUDA_TFSIMULATORCONFIG_H_
#define _SOURCE_CUDA_TFSIMULATORCONFIG_H_

#include <tf_cuda.h>

#include "tfEngineConfig.h"
#include "tfBondConfig.h"
#include "tfAngleConfig.h"


namespace TissueForge::cuda { 


    /**
     * @brief CUDA runtime control interface for Simulator. 
     * 
     * This object aggregates all CUDA runtime control interfaces relevant 
     * to a Tissue Forge simulation. 
     * 
     */
    struct CAPI_EXPORT SimulatorConfig {
        /** Tissue Forge engine CUDA runtime control interface */
        EngineConfig engine;

        /** Tissue Forge bonds CUDA runtime control interface */
        BondConfig bonds;

        /** Tissue Forge angles CUDA runtime control interface */
        AngleConfig angles;
    };

};

#endif // _SOURCE_CUDA_TFSIMULATORCONFIG_H_