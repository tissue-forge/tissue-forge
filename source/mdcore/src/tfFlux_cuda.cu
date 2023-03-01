/*******************************************************************************
 * This file is part of mdcore.
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

#include "tfFlux_cuda.h"

#include <tf_errs.h>
#include <tfError.h>
#include <tfEngine.h>

#include <cuda.h>


using namespace TissueForge;


// Diagonal entries and flux index lookup table
__constant__ int *cuda_fxind;

// The fluxes
__constant__ struct cuda::Fluxes *cuda_fluxes = NULL;
__constant__ int cuda_nr_fluxes = 0;

#define error(id)               (tf_error(E_FAIL, errs_err_msg[id]))
#define cuda_error()            (tf_error(E_FAIL, cudaGetErrorString(cudaGetLastError())))
#define cuda_safe_call(f)       { if(f != cudaSuccess) return cuda_error(); }


#define TF_FLUXCUDA_CUDAMALLOC(member, type, member_name)                                                       \
    if(cudaMalloc(&member, this->size * sizeof(type)) != cudaSuccess) {                                         \
        printf("Flux allocation failed (%s): %s\n", member_name, cudaGetErrorString(cudaPeekAtLastError()));    \
        return;                                                                                                 \
    } 

template <typename T>
__host__ 
cudaError_t FluxCUDA_toDevice(struct cuda::Flux *f, T *memPtr, T *fluxPtr) {
    T *tmpPtr = (T*)malloc(f->size * sizeof(T));
    for(int i = 0; i < f->size; i++) tmpPtr[i] = fluxPtr[i];
    cudaMemcpy(memPtr, tmpPtr, f->size * sizeof(T), cudaMemcpyHostToDevice);
    free(tmpPtr);
    return cudaPeekAtLastError();
}

#define TF_FLUXCUDA_MEMCPYH2D(member, flux_member, member_name)                                                 \
    if(FluxCUDA_toDevice(this, member, flux_member) != cudaSuccess) {                                           \
        printf("Flux member copy failed (%s): %s\n", member_name, cudaGetErrorString(cudaPeekAtLastError()));   \
        return;                                                                                                 \
    }


#define TF_FLUXCUDA_CUDAFREE(member, member_name)                                                               \
    if(cudaFree(member) != cudaSuccess) {                                                                       \
        printf("Flux member free failed (%s): %s\n", member_name, cudaGetErrorString(cudaPeekAtLastError()));   \
        return;                                                                                                 \
    }


// cuda::Flux


__host__ 
cuda::Flux::Flux(TissueForge::Flux f) : 
    size{f.size}
{
    TF_FLUXCUDA_CUDAMALLOC(this->kinds,      int8_t,                 "kinds");
    TF_FLUXCUDA_CUDAMALLOC(this->type_ids,   cuda::FluxTypeIdPair,   "type_ids");
    TF_FLUXCUDA_CUDAMALLOC(this->indices_a,  int32_t,                "indices_a");
    TF_FLUXCUDA_CUDAMALLOC(this->indices_b,  int32_t,                "indices_b");
    TF_FLUXCUDA_CUDAMALLOC(this->coef,       float,                  "coef");
    TF_FLUXCUDA_CUDAMALLOC(this->decay_coef, float,                  "decay_coef");
    TF_FLUXCUDA_CUDAMALLOC(this->target,     float,                  "target");

    TF_FLUXCUDA_MEMCPYH2D(this->kinds,       f.kinds,        "kinds");
    TF_FLUXCUDA_MEMCPYH2D(this->indices_a,   f.indices_a,    "indices_a");
    TF_FLUXCUDA_MEMCPYH2D(this->indices_b,   f.indices_b,    "indices_b");
    TF_FLUXCUDA_MEMCPYH2D(this->coef,        f.coef,         "coef");
    TF_FLUXCUDA_MEMCPYH2D(this->decay_coef,  f.decay_coef,   "decay_coef");
    TF_FLUXCUDA_MEMCPYH2D(this->target,      f.target,       "target");

    cuda::FluxTypeIdPair *_type_ids = (cuda::FluxTypeIdPair*)malloc(sizeof(cuda::FluxTypeIdPair) * this->size);
    for(int i = 0; i < this->size; i++) _type_ids[i] = cuda::FluxTypeIdPair(f.type_ids[i]);
    cudaMemcpy(this->type_ids, _type_ids, sizeof(cuda::FluxTypeIdPair) * this->size, cudaMemcpyHostToDevice);
    free(_type_ids);
}

__device__ 
void cuda::Flux::finalize() {
    TF_FLUXCUDA_CUDAFREE(this->kinds,        "kinds");
    TF_FLUXCUDA_CUDAFREE(this->indices_a,    "indices_a");
    TF_FLUXCUDA_CUDAFREE(this->indices_b,    "indices_b");
    TF_FLUXCUDA_CUDAFREE(this->coef,         "coef");
    TF_FLUXCUDA_CUDAFREE(this->decay_coef,   "decay_coef");
    TF_FLUXCUDA_CUDAFREE(this->target,       "target");
}


// cuda::Fluxes


__host__ 
cuda::Fluxes::Fluxes(TissueForge::Fluxes *f) : 
    size{f->size}
{
    if(cudaMalloc(&this->fluxes, sizeof(cuda::Flux) * this->size) != cudaSuccess) {
        printf("Fluxes allocation failed (fluxes): %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        return;
    }

    cuda::Flux *_fluxes = (cuda::Flux *)malloc(sizeof(cuda::Flux) * this->size);
    for(int i = 0; i < this->size; i++) _fluxes[i] = cuda::Flux(f->fluxes[i]);

    if(cudaMemcpy(this->fluxes, _fluxes, sizeof(cuda::Flux) * this->size, cudaMemcpyHostToDevice) != cudaSuccess) 
        printf("Fluxes member free failed (fluxes): %s\n", cudaGetErrorString(cudaPeekAtLastError()));

    free(_fluxes);
}

__device__ 
void cuda::Fluxes::finalize() {
    for(int i = 0; i < this->size; i++)
        this->fluxes[i].finalize();
    
    if(cudaFree(this->fluxes) != cudaSuccess)
        printf("Fluxes member free failed (fluxes): %s\n", cudaGetErrorString(cudaPeekAtLastError()));
}

__device__ 
void cuda::Flux_getFluxes(int **fxind_cuda, cuda::Fluxes **fluxes_cuda) {
    *fxind_cuda = cuda_fxind;
    *fluxes_cuda = cuda_fluxes;
}

__device__ 
void cuda::Flux_getNrFluxes(unsigned int *nr_fluxes) {
    *nr_fluxes = cuda_nr_fluxes;
}

__device__ 
void cuda::Flux_getNrStates(unsigned int *nr_states) {
    *nr_states = cuda_nr_fluxes - 1;
}

extern "C" HRESULT cuda::engine_cuda_load_fluxes(struct engine *e) {

    int i, j, nr_fluxes;
    int did;
    int nr_devices = e->nr_devices;
    int *fxind = (int*)malloc(sizeof(int) * e->max_type * e->max_type);
    struct TissueForge::Fluxes **fluxes = (TissueForge::Fluxes**)malloc(sizeof(Fluxes*) * e->nr_types * (e->nr_types + 1) / 2 + 1);
    
    // Start by identifying the unique fluxes in the engine
    nr_fluxes = 1;
    for(i = 0 ; i < e->max_type * e->max_type ; i++) {
    
        /* Skip if there is no flux or no parts of this type. */
        if(e->fluxes[i] == NULL)
            continue;

        /* Check this flux against previous fluxes. */
        for(j = 0 ; j < nr_fluxes && e->fluxes[i] != fluxes[j] ; j++);
        if(j < nr_fluxes)
            continue;

        /* Store this flux and the number of coefficient entries it has. */
        fluxes[nr_fluxes] = e->fluxes[i];
        nr_fluxes += 1;
    
    }

    /* Pack the flux matrix. */
    for(i = 0 ; i < e->max_type * e->max_type ; i++) {
        if(e->fluxes[i] == NULL) {
            fxind[i] = 0;
        }
        else {
            for(j = 0 ; j < nr_fluxes && fluxes[j] != e->fluxes[i] ; j++);
            fxind[i] = j;
        }
    }

    // Pack the fluxes
    cuda::Fluxes *fluxes_cuda = (cuda::Fluxes*)malloc(sizeof(cuda::Flux) * nr_fluxes);
    for(i = 1; i < nr_fluxes; i++) {
        fluxes_cuda[i] = cuda::Fluxes(fluxes[i]);
    }
    
    /* Store find and other stuff as constant. */
    for(did = 0; did < nr_devices; did++) {
        cuda_safe_call(cudaSetDevice(e->devices[did]));
        cuda_safe_call(cudaMalloc(&e->fxind_cuda[did], sizeof(int) * e->max_type * e->max_type));
        cuda_safe_call(cudaMemcpy(e->fxind_cuda[did], fxind, sizeof(int) * e->max_type * e->max_type, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpyToSymbol(cuda_fxind, &e->fxind_cuda[did], sizeof(void *), 0, cudaMemcpyHostToDevice));
    }
    free(fxind);

    // Store the fluxes
    for(did = 0; did < nr_devices; did++) {
        cuda_safe_call(cudaSetDevice(e->devices[did]));
        cuda_safe_call(cudaMalloc(&e->fluxes_cuda[did], sizeof(cuda::Fluxes) * nr_fluxes));
        cuda_safe_call(cudaMemcpy(e->fluxes_cuda[did], fluxes_cuda, sizeof(cuda::Fluxes) * nr_fluxes, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpyToSymbol(cuda_fluxes, &e->fluxes_cuda[did], sizeof(void*), 0, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpyToSymbol(cuda_nr_fluxes, &nr_fluxes, sizeof(int), 0, cudaMemcpyHostToDevice));
    }
    free(fluxes);
    free(fluxes_cuda);

    int nr_states_current = e->nr_fluxes_cuda - 1;
    int nr_states_next = nr_fluxes - 1;

    if(nr_states_current > 0 && nr_states_next == 0 && engine_cuda_finalize_particle_states(e) != S_OK) 
        return error(MDCERR_cuda);

    e->nr_fluxes_cuda = nr_fluxes;

    if(nr_states_next > 0 && nr_states_current == 0 && engine_cuda_allocate_particle_states(e) != S_OK) 
        return error(MDCERR_cuda);
    else if(nr_states_current != nr_states_next && engine_cuda_refresh_particle_states(e) != S_OK) 
        return error(MDCERR_cuda);

    // Allocate the flux buffer
    if(nr_states_next > 0) {
        for(did = 0; did < nr_devices; did++) {
            cuda_safe_call(cudaSetDevice(e->devices[did]));
            cuda_safe_call(cudaMalloc(&e->fluxes_next_cuda[did], sizeof(float) * nr_states_next * e->s.size_parts));
        }
    }

    return S_OK;
}


__global__ 
void engine_cuda_unload_fluxes_device(int nr_fluxes) {
    if(nr_fluxes < 1)
        return;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(tid == 0) 
        tid += stride;

    while(tid < nr_fluxes) {
        cuda_fluxes[tid].finalize();

        tid += stride;
    }
}

extern "C" HRESULT cuda::engine_cuda_unload_fluxes(struct engine *e) {

    int nr_states = e->nr_fluxes_cuda - 1;

    for(int did = 0; did < e->nr_devices; did++) {

        cuda_safe_call(cudaSetDevice(e->devices[did]));

        // Free the fluxes.
        
        cuda_safe_call(cudaFree(e->fxind_cuda[did]));
        
        engine_cuda_unload_fluxes_device<<<8, 512>>>(nr_states);

        cuda_safe_call(cudaPeekAtLastError());
        
        cuda_safe_call(cudaFree((cuda::Fluxes*)e->fluxes_cuda[did]));

        cuda_safe_call(cudaFree(e->fluxes_next_cuda[did]));

    }

    e->nr_fluxes_cuda = 0;

    return S_OK;
}

extern "C" HRESULT cuda::engine_cuda_refresh_fluxes(struct engine *e) {
    
    if(engine_cuda_unload_fluxes(e) != S_OK)
        return error(MDCERR_cuda);

    if(engine_cuda_load_fluxes(e) != S_OK)
        return error(MDCERR_cuda);

    for(int did = 0; did < e->nr_devices; did++) {

        cuda_safe_call(cudaSetDevice(e->devices[did]));

        cuda_safe_call(cudaDeviceSynchronize());

    }

    return S_OK;
}
