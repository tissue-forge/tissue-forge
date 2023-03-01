/*******************************************************************************
 * This file is part of mdcore.
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

#ifndef _MDCORE_SOURCE_TFBOUNDARYCONDITIONS_CUDA_H_
#define _MDCORE_SOURCE_TFBOUNDARYCONDITIONS_CUDA_H_

#include <tfBoundaryConditions.h>

#include "tfPotential_cuda.h"


namespace TissueForge::cuda {


    struct BoundaryCondition {

        float3 normal;

        float3 velocity;

        float radius;

        float pad;

        __host__ __device__ 
        BoundaryCondition() {}

        __host__ 
        BoundaryCondition(const TissueForge::BoundaryCondition &_bc);

        __host__ 
        void finalize();
    };


    struct BoundaryConditions {
        
        // Left, right, front, back, bottom, top
        BoundaryCondition *bcs, *bcs_h;

        __host__ __device__ 
        BoundaryConditions() {}
        
        __host__ 
        BoundaryConditions(const TissueForge::BoundaryConditions &_bcs);

    };

};

#endif // _MDCORE_SOURCE_TFBOUNDARYCONDITIONS_CUDA_H_