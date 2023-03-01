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

#ifndef _MDCORE_SOURCE_TFFLUX_CUDA_H_
#define _MDCORE_SOURCE_TFFLUX_CUDA_H_

#include <cuda_runtime.h>

#include <mdcore_config.h>
#include <tfFlux.h>


namespace TissueForge::cuda {


    struct FluxTypeIdPair {
        int16_t a;
        int16_t b;

        FluxTypeIdPair(TissueForge::TypeIdPair tip) : a{tip.a}, b{tip.b} {}
    };


    // A wrap of Flux
    struct Flux {
        int32_t size;
        int8_t *kinds;
        FluxTypeIdPair *type_ids;
        int32_t *indices_a;
        int32_t *indices_b;
        float *coef;
        float *decay_coef;
        float *target;

        __host__ 
        Flux(TissueForge::Flux f);

        __device__ 
        void finalize();
    };


    // A wrap of Fluxes
    struct Fluxes {
        int32_t size;
        Flux *fluxes;

        __host__ 
        Fluxes(TissueForge::Fluxes *f);

        __device__ 
        void finalize();
    };


    __device__ 
    void Flux_getFluxes(int **fxind_cuda, Fluxes **fluxes_cuda);

    __device__ 
    void Flux_getNrFluxes(unsigned int *nr_fluxes);

    __device__ 
    void Flux_getNrStates(unsigned int *nr_states);

};

#endif // _MDCORE_SOURCE_TFFLUX_CUDA_H_