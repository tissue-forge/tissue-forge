/*******************************************************************************
 * This file is part of mdcore.
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

// TODO: improve error handling in BoundaryConditions_cuda

#include "tfBoundaryConditions_cuda.h"

#include <tfEngine.h>


using namespace TissueForge;


// BoundaryCondition


__host__ 
cuda::BoundaryCondition::BoundaryCondition(const TissueForge::BoundaryCondition &_bc) {
    this->normal = make_float3(_bc.normal[0], _bc.normal[1], _bc.normal[2]);
    this->velocity = make_float3(_bc.velocity[0], _bc.velocity[1], _bc.velocity[2]);
    this->radius = _bc.radius;
}


// BoundaryConditions


__host__ 
cuda::BoundaryConditions::BoundaryConditions(const TissueForge::BoundaryConditions &_bcs) {
    size_t size_bcs = sizeof(cuda::BoundaryCondition) * 6;

    if(cudaMalloc(&this->bcs, size_bcs) != cudaSuccess) {
        printf("Boundary conditions allocation failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        return;
    }

    this->bcs_h = (cuda::BoundaryCondition*)malloc(size_bcs);

    this->bcs_h[0] = cuda::BoundaryCondition(_bcs.left);
    this->bcs_h[1] = cuda::BoundaryCondition(_bcs.right);
    this->bcs_h[2] = cuda::BoundaryCondition(_bcs.front);
    this->bcs_h[3] = cuda::BoundaryCondition(_bcs.back);
    this->bcs_h[4] = cuda::BoundaryCondition(_bcs.bottom);
    this->bcs_h[5] = cuda::BoundaryCondition(_bcs.top);

    if(cudaMemcpy(this->bcs, this->bcs_h, size_bcs, cudaMemcpyHostToDevice) != cudaSuccess)
        printf("Boundary conditions copy H2D failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
}
