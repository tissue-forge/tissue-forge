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

#ifndef _MDCORE_SOURCE_TFPARTICLE_CUDA_H_
#define _MDCORE_SOURCE_TFPARTICLE_CUDA_H_

#include <tfParticle.h>

#include <cuda_runtime.h>


namespace TissueForge::cuda {


    // A wrap of Particle
    struct Particle {
        float4 x;
        // v[0], v[1], v[2], radius
        float4 v;

        // id, typeId, clusterId, flags
        int4 w;

        __host__ __device__ 
        Particle()  :
            w{-1, -1, -1, PARTICLE_NONE}
        {}

        __host__ __device__ 
        Particle(TissueForge::Particle *p) : 
            x{p->x[0], p->x[1], p->x[2], p->x[3]}, 
            v{p->v[0], p->v[1], p->v[2], p->radius}, 
            w{p->id, p->typeId, p->clusterId, p->flags}
        {}

        __host__ 
        Particle(TissueForge::Particle *p, int nr_states) : 
            x{p->x[0], p->x[1], p->x[2], p->x[3]}, 
            v{p->v[0], p->v[1], p->v[2], p->radius}, 
            w{p->id, p->typeId, p->clusterId, p->flags}
        {}
    };

};

#endif // _MDCORE_SOURCE_TFPARTICLE_CUDA_H_