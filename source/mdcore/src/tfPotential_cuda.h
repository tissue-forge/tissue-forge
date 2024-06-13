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

#ifndef _MDCORE_SOURCE_TFPOTENTIAL_CUDA_H_
#define _MDCORE_SOURCE_TFPOTENTIAL_CUDA_H_

#include <tfPotential.h>
#include <tfDPDPotential.h>

#include "tfParticle_cuda.h"


namespace TissueForge::cuda {


    /**
     * @brief Loads a potential onto a CUDA device
     * 
     * @param p The potential
     * 
     * @return The loaded potential, or NULL if failed
     */
    TissueForge::Potential toCUDADevice(const TissueForge::Potential &p);


    __host__ __device__ 
    void cudaFree(TissueForge::Potential *p);


    struct PotentialData {
        /** Flags. */
        uint32_t flags;

        // a, b, r0_plusone
        float3 w;

        /** coordinate offset */
        float3 offset;

        /** Coefficients for the interval transform. */
        float4 alpha;

        /** Nr of intervals. */
        int n;

        /** The coefficients. */
        float *c;

        __host__ 
        PotentialData() : flags{POTENTIAL_NONE} {}

        __host__ 
        PotentialData(TissueForge::Potential *p);

        __host__ 
        void finalize();
    };

    struct DPDPotentialData {
        /** Flags. */
        uint32_t flags;

        // a, b
        float2 w;

        // DPD coefficients alpha, gamma, sigma
        float3 dpd_cfs;

        __host__ 
        DPDPotentialData() : flags{POTENTIAL_NONE} {}

        __host__ 
        DPDPotentialData(DPDPotential *p);
    };

    // A wrap of Potential
    struct Potential {
        // Number of underlying potentials
        int nr_pots;

        // Number of dpd potentials
        int nr_dpds;

        // Data of all underlying potentials, excluding dpd
        PotentialData *data_pots;

        // Data of all underlying dpd potentials
        DPDPotentialData *data_dpds;

        __host__ __device__ 
        Potential() : 
            nr_pots{0}, 
            nr_dpds{0}
        {}

        __host__ 
        Potential(TissueForge::Potential *p);
        
        __host__ 
        void finalize() {
            if(this->nr_pots == 0 && this->nr_dpds) 
                return;

            for(int i = 0; i < this->nr_pots; i++) 
                this->data_pots[i].finalize();

            this->nr_pots = 0;
            this->nr_dpds = 0;
        }
    };

};

#endif // _MDCORE_SOURCE_TFPOTENTIAL_CUDA_H_