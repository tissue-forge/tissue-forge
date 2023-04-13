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

#include "tfAngle_cuda.h"
#include "tfBond_cuda.h"

#include <tfEngine.h>
#include <tf_errs.h>
#include <tfError.h>
#include "tfRunner_cuda.h"
#include <tf_cuda.h>
#include <tfTaskScheduler.h>

#include "cutil_math.h"


using namespace TissueForge;


__constant__ float3 cuda_bonds_cell_edge_lens;
__constant__ float cuda_bonds_dt;
__constant__ int3 *cuda_bonds_cloc;
static unsigned int cuda_bonds_dev_constants = 0;

// Partice list
__constant__ float4* cuda_bonds_parts_pos;
__constant__ int* cuda_bonds_parts_flags;
__constant__ bool cuda_bonds_parts_from_engine;
__constant__ unsigned int *cuda_bonds_parts_from_engine_key;
static unsigned int *cuda_bonds_parts_from_engine_key_arr;
static size_t cuda_bonds_parts_from_engine_key_size = 0;
static float4* cuda_bonds_parts_pos_arr;
static int* cuda_bonds_parts_flags_arr;
static unsigned int cuda_bonds_parts_counter = 0;
static unsigned int cuda_bonds_parts_checkouts = 0;

// Random number generators
__device__ curandState *cuda_rand_unif;
void *rand_unif_cuda;

// Counter tracking number of random number generator checkouts
static unsigned int cuda_rand_unif_init = 0;

// Random number generator seed
static int cuda_rand_unif_seed;

// Number of available random states
static unsigned int cuda_rand_unif_states = 0;

// ID of device for this module
static int cuda_bonds_device = 0;

// Shared object counter
static unsigned int cuda_bonds_shared_arrs = 0;

// Shared streams
static cudaStream_t *cuda_bonds_stream, cuda_bonds_stream1, cuda_bonds_stream2;
static unsigned int cuda_bonds_front_stream = 1;

// Shared device arrays
static float *cuda_bonds_potenergies_arr, *cuda_bonds_potenergies_arr1, *cuda_bonds_potenergies_arr2;
static bool *cuda_bonds_todestroy_arr, *cuda_bonds_todestroy_arr1, *cuda_bonds_todestroy_arr2;

// Shared host arrays
static float *cuda_bonds_potenergies_local, *cuda_bonds_potenergies_local1, *cuda_bonds_potenergies_local2;
static bool *cuda_bonds_todestroy_local, *cuda_bonds_todestroy_local1, *cuda_bonds_todestroy_local2;

// Bond arrays
__device__ cuda::Bond *cuda_bonds;
static cuda::Bond *cuda_bonds_device_arr;
static struct BondCUDAData *cuda_bonds_bonds_arr, *cuda_bonds_bonds_arr1, *cuda_bonds_bonds_arr2;
static struct BondCUDAData *cuda_bonds_bonds_local, *cuda_bonds_bonds_local1, *cuda_bonds_bonds_local2;
static float *cuda_bonds_forces_arr, *cuda_bonds_forces_arr1, *cuda_bonds_forces_arr2;
static float *cuda_bonds_forces_local, *cuda_bonds_forces_local1, *cuda_bonds_forces_local2;
static int cuda_bonds_size = 0;

// Bond dynamic parallelism specs
static unsigned int cuda_bonds_nr_threads = 0;
static unsigned int cuda_bonds_nr_blocks = 0;

// Angle arrays
__device__ cuda::Angle *cuda_angles;
static cuda::Angle *cuda_angles_device_arr;
static struct AngleCUDAData *cuda_angles_angles_arr, *cuda_angles_angles_arr1, *cuda_angles_angles_arr2;
static struct AngleCUDAData *cuda_angles_angles_local, *cuda_angles_angles_local1, *cuda_angles_angles_local2;
static float *cuda_angles_forces_arr, *cuda_angles_forces_arr1, *cuda_angles_forces_arr2;
static float *cuda_angles_forces_local, *cuda_angles_forces_local1, *cuda_angles_forces_local2;
static int cuda_angles_size = 0;

// Angle dynamic parallism specs
static unsigned int cuda_angles_nr_threads = 0;
static unsigned int cuda_angles_nr_blocks = 0;

#define cuda_bonds_nrparts_chunk    102400
#define cuda_bonds_nrparts_incr     100

#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))
#define cuda_error()			(tf_error(E_FAIL, cudaGetErrorString(cudaGetLastError())))


template <typename T> __global__ 
void engine_cuda_memcpy(T *dst, T *src, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    int nr_elems = size / sizeof(T);
    for(int i = tid; i < nr_elems; i += stride) 
        dst[i] = src[i];
}

int cuda_bonds_module_dyn_specs(int *nr_blocks, int *nr_threads) {
    if(cuda_bonds_nr_blocks > 0) {
        *nr_blocks = cuda_bonds_nr_blocks;
        *nr_threads = cuda_bonds_nr_threads;
    }
    else if(cuda_angles_nr_blocks > 0) {
        *nr_blocks = cuda_angles_nr_blocks;
        *nr_threads = cuda_angles_nr_threads;
    }
    else {
        *nr_blocks = *nr_threads = 1;
    }
    return S_OK;
}

int cuda_bonds_device_constants_init(engine *e) {
    cuda_bonds_dev_constants++;

    if(cuda_bonds_dev_constants > 1) 
        return S_OK;

    int *buffi, i;
    int3 *cloc, *cloc_d;
    float3 cell_edge_lens_cuda = make_float3(e->s.h[0], e->s.h[1], e->s.h[2]);

    if(cudaMemcpyToSymbol(cuda_bonds_cell_edge_lens, &cell_edge_lens_cuda, sizeof(float3), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        return cuda_error();

    if(cudaMemcpyToSymbol(cuda_bonds_dt, &e->dt, sizeof(float), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        return cuda_error();

    /* Copy the cell locations to the device. */
    if((cloc = (int3 *)malloc(sizeof(int3) * e->s.nr_cells)) == NULL)
        return error(MDCERR_malloc);
    for(i = 0 ; i < e->s.nr_cells ; i++) {
        buffi = &e->s.cells[i].loc[0];
        cloc[i] = make_int3(buffi[0], buffi[1], buffi[2]);
    }

    if(cudaMalloc(&cloc_d, sizeof(int3) * e->s.nr_cells) != cudaSuccess)
        return cuda_error();
    if(cudaMemcpy(cloc_d, cloc, sizeof(int3) * e->s.nr_cells, cudaMemcpyHostToDevice) != cudaSuccess)
        return cuda_error();
    if(cudaMemcpyToSymbol(cuda_bonds_cloc, &cloc_d, sizeof(void *), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        return cuda_error();

    free(cloc);

    return S_OK;
}

int cuda_bonds_parts_hold() {
    cuda_bonds_parts_checkouts++;
    return S_OK;
}

int cuda_bonds_parts_release() {
    cuda_bonds_parts_checkouts--;
    return S_OK;
}

__global__ 
void cuda_bonds_load_from_engine(int4* part_datai, int nr_parts) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i < nr_parts; i += stride) {
        int4 pdatai = part_datai[i];
        cuda_bonds_parts_from_engine_key[pdatai.x] = i;
        cuda_bonds_parts_flags[i] = pdatai.z;
    }
}

int cuda_bonds_parts_load(engine *e) { 
    cuda_bonds_parts_counter += 2;

    if(cuda_bonds_parts_counter > 2) 
        return S_OK;

    int i;

    bool using_engine = e->flags & engine_flag_cuda;
    if(cudaMemcpyToSymbol(cuda_bonds_parts_from_engine, &using_engine, sizeof(bool), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    if(using_engine) { 
        if(e->s.size_parts != cuda_bonds_parts_from_engine_key_size) {
            if(cuda_bonds_parts_from_engine_key_size > 0) 
                if(cudaFree(cuda_bonds_parts_from_engine_key_arr) != cudaSuccess) 
                    return cuda_error();

            cuda_bonds_parts_from_engine_key_size = e->s.size_parts;
            if(cudaMalloc(&cuda_bonds_parts_from_engine_key_arr, e->s.size_parts * sizeof(unsigned int)) != cudaSuccess) 
                return cuda_error();

            if(cudaMemcpyToSymbol(cuda_bonds_parts_from_engine_key, &cuda_bonds_parts_from_engine_key_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
                return cuda_error();
        }

        for(i = 0; i < e->nr_devices; i++) 
            if(e->devices[i] == cuda_bonds_device) { 
                cuda_bonds_parts_pos_arr = (float4*)e->parts_pos_cuda[i];
                if(cudaMemcpyToSymbol(cuda_bonds_parts_pos, &cuda_bonds_parts_pos_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
                    return cuda_error();
                
                int nr_blocks, nr_threads;
                cuda_bonds_module_dyn_specs(&nr_blocks, &nr_threads);
                cuda_bonds_load_from_engine<<<nr_blocks, nr_threads>>>((int4*)e->parts_datai_cuda[i], e->s.nr_parts);
                if(cudaPeekAtLastError() != cudaSuccess)
                    return cuda_error();
                
                return S_OK;
            }

        return MDCERR_cuda;

    } 
    else {
        float3* bonds_parts_pos = (float3*)malloc(e->s.size_parts * sizeof(float3));
        float* bonds_parts_radius = (float*)malloc(e->s.size_parts * sizeof(float));
        int* bonds_parts_flags = (int*)malloc(e->s.size_parts * sizeof(int));
        for(i = 0; i < e->s.size_parts; i++) {
            Particle* part = e->s.partlist[i];
            if(part) {
                bonds_parts_pos[part->id] = {part->x[0], part->x[1], part->x[2]};
                bonds_parts_radius[part->id] = part->radius;
                bonds_parts_flags[part->id] = part->flags;
            }
        }
        if(cudaMalloc(&cuda_bonds_parts_pos_arr, e->s.size_parts * sizeof(float4)) != cudaSuccess) 
            return cuda_error();
        if(cudaMemcpy(cuda_bonds_parts_pos_arr, bonds_parts_pos, e->s.size_parts * sizeof(float4), cudaMemcpyHostToDevice) != cudaSuccess) 
            return cuda_error();
        if(cudaMemcpyToSymbol(cuda_bonds_parts_pos, &cuda_bonds_parts_pos_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
            return cuda_error();
        if(cudaMalloc(&cuda_bonds_parts_flags_arr, e->s.size_parts * sizeof(int)) != cudaSuccess) 
            return cuda_error();
        if(cudaMemcpy(cuda_bonds_parts_flags_arr, bonds_parts_flags, e->s.size_parts * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) 
            return cuda_error();
        if(cudaMemcpyToSymbol(cuda_bonds_parts_flags, &cuda_bonds_parts_flags_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
            return cuda_error();

        free(bonds_parts_pos);
        free(bonds_parts_radius);
        free(bonds_parts_flags);
    }

    return S_OK;
}

int cuda_bonds_parts_unload(engine *e) {
    cuda_bonds_parts_counter--;

    if(cuda_bonds_parts_counter != cuda_bonds_parts_checkouts) 
        return S_OK;

    cuda_bonds_parts_counter = 0;

    if(e->flags & engine_flag_cuda) {
        return S_OK;
    } 
    else {
        if(cudaFree(cuda_bonds_parts_pos_arr) != cudaSuccess) 
            return cuda_error();
        if(cudaFree(cuda_bonds_parts_flags_arr) != cudaSuccess) 
            return cuda_error();
    }

    return S_OK;
}

__global__ void cuda_init_rand_unif_device(curandState *rand_unif, int nr_rands, unsigned long long seed) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(tid < nr_rands) {
        curand_init(seed, tid, 0, &rand_unif[tid]);
        tid += stride;
    }
}

extern "C" int engine_cuda_rand_unif_init(struct engine *e, int nr_rands) {

    cuda_rand_unif_init++;
    
    if(cuda_rand_unif_init > 1 && cuda_rand_unif_states > nr_rands) 
        return S_OK;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess)
        return cuda_error();

    if(cudaMalloc(&rand_unif_cuda, sizeof(curandState) * nr_rands) != cudaSuccess)
        return cuda_error();

    int nr_blocks, nr_threads;
    cuda_bonds_module_dyn_specs(&nr_blocks, &nr_threads);
    cuda_init_rand_unif_device<<<nr_blocks, nr_threads>>>((curandState *)rand_unif_cuda, nr_rands, cuda_rand_unif_seed);
    if(cudaPeekAtLastError() != cudaSuccess)
        return cuda_error();

    if(cudaMemcpyToSymbol(cuda_rand_unif, &rand_unif_cuda, sizeof(void *), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        return cuda_error();

    cuda_rand_unif_states = nr_rands;

    return S_OK;

}

int engine_cuda_rand_unif_finalize(struct engine *e) {

    cuda_rand_unif_init--;

    if(cuda_rand_unif_init > 0) 
        return S_OK;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess)
        return cuda_error();

    if(cudaFree(rand_unif_cuda) != cudaSuccess)
        return cuda_error();

    cuda_rand_unif_states = 0;
    
    return S_OK;

}

/**
 * @brief Sets the random seed for the CUDA uniform number generators. 
 * 
 * @param e The #engine
 * @param seed The seed
 * @param onDevice A flag specifying whether the engine is current on the device
 */
extern "C" int engine_cuda_rand_unif_setSeed(struct engine *e, unsigned int seed, bool onDevice) { 

    unsigned int nr_states = cuda_rand_unif_states;
    unsigned int nr_init = cuda_rand_unif_init;

    if(onDevice) 
        for(int i = 0; i < nr_init; i++) 
            if(engine_cuda_rand_unif_finalize(e) != S_OK)
                return cuda_error();

    cuda_rand_unif_seed = seed;

    if(onDevice) {
        for(int i = 0; i < nr_init; i++) 
            if(engine_cuda_rand_unif_init(e, nr_states) != S_OK)
                return cuda_error();

        if(cudaSetDevice(cuda_bonds_device) != cudaSuccess)
            return cuda_error();

        if(cudaDeviceSynchronize() != cudaSuccess)
            return cuda_error();
    }

    return S_OK;

}

__device__ 
void engine_cuda_rand_uniform(float *result) { 
    *result = curand_uniform(&cuda_rand_unif[threadIdx.x + blockIdx.x * blockDim.x]);
}

__global__ 
void engine_cuda_rand_unifs_device(int nr_rands, float *result) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    auto s = &cuda_rand_unif[threadID];

    for(int tid = 0; tid < nr_rands; tid += stride) {
        result[tid] = curand_uniform(s);
    }
}

int engine_cuda_rand_unifs(int nr_rands, float *result, cudaStream_t stream) {
    int nr_blocks = std::min(cuda_bonds_nr_blocks, (unsigned int)std::ceil((float)nr_rands / cuda_bonds_nr_threads));

    engine_cuda_rand_unifs_device<<<nr_blocks, cuda_bonds_nr_threads, 0, stream>>>(nr_rands, result);
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

int cuda_bonds_shared_arrays_malloc() {
    cuda_bonds_shared_arrs++;

    if(cuda_bonds_shared_arrs > 1) 
        return S_OK;

    if(cudaStreamCreate(&cuda_bonds_stream1) != cudaSuccess) 
        return cuda_error();

    if(cudaStreamCreate(&cuda_bonds_stream2) != cudaSuccess) 
        return cuda_error();

    size_t size_potenergies = cuda_bonds_nrparts_chunk * sizeof(float);
    size_t size_todestroy = cuda_bonds_nrparts_chunk * sizeof(bool);

    if(cudaMalloc(&cuda_bonds_potenergies_arr1, size_potenergies) != cudaSuccess) 
        return cuda_error();
    if(cudaMalloc(&cuda_bonds_potenergies_arr2, size_potenergies) != cudaSuccess) 
        return cuda_error();
    if((cuda_bonds_potenergies_local1 = (float*)malloc(size_potenergies)) == NULL) 
        return MDCERR_malloc;
    if((cuda_bonds_potenergies_local2 = (float*)malloc(size_potenergies)) == NULL) 
        return MDCERR_malloc;

    if(cudaMalloc(&cuda_bonds_todestroy_arr1, size_todestroy) != cudaSuccess) 
        return cuda_error();
    if(cudaMalloc(&cuda_bonds_todestroy_arr2, size_todestroy) != cudaSuccess) 
        return cuda_error();
    if((cuda_bonds_todestroy_local1 = (bool*)malloc(size_todestroy)) == NULL) 
        return MDCERR_malloc;
    if((cuda_bonds_todestroy_local2 = (bool*)malloc(size_todestroy)) == NULL)
        return MDCERR_malloc;

    return S_OK;
}

int cuda_bonds_shared_arrays_free() { 
    cuda_bonds_shared_arrs--;

    if(cuda_bonds_shared_arrs > 0) 
        return S_OK;

    if(cudaStreamDestroy(cuda_bonds_stream1) != cudaSuccess) 
        return cuda_error();

    if(cudaStreamDestroy(cuda_bonds_stream2) != cudaSuccess) 
        return cuda_error();

    if(cudaFree(cuda_bonds_potenergies_arr1) != cudaSuccess) 
        return cuda_error();
    if(cudaFree(cuda_bonds_potenergies_arr2) != cudaSuccess) 
        return cuda_error();
    free(cuda_bonds_potenergies_local1);
    free(cuda_bonds_potenergies_local2);

    if(cudaFree(cuda_bonds_todestroy_arr1) != cudaSuccess) 
        return cuda_error();
    if(cudaFree(cuda_bonds_todestroy_arr2) != cudaSuccess) 
        return cuda_error();
    free(cuda_bonds_todestroy_local1);
    free(cuda_bonds_todestroy_local2);

    return S_OK;
}

int engine_bond_flip_shared() {
    if(cuda_bonds_front_stream == 1) { 
        cuda_bonds_stream            = &cuda_bonds_stream2;

        cuda_bonds_potenergies_arr   = cuda_bonds_potenergies_arr2;
        cuda_bonds_todestroy_arr     = cuda_bonds_todestroy_arr2;

        cuda_bonds_potenergies_local = cuda_bonds_potenergies_local2;
        cuda_bonds_todestroy_local   = cuda_bonds_todestroy_local2;

        cuda_bonds_front_stream      = 2;
    }
    else {
        cuda_bonds_stream            = &cuda_bonds_stream1;

        cuda_bonds_potenergies_arr   = cuda_bonds_potenergies_arr1;
        cuda_bonds_todestroy_arr     = cuda_bonds_todestroy_arr1;

        cuda_bonds_potenergies_local = cuda_bonds_potenergies_local1;
        cuda_bonds_todestroy_local   = cuda_bonds_todestroy_local1;

        cuda_bonds_front_stream      = 1;
    }
    return S_OK;
}


// cuda::Bond


__host__ 
cuda::Bond::Bond() {
    this->flags = ~BOND_ACTIVE;
}

__host__ 
cuda::Bond::Bond(TissueForge::Bond *b) : 
    flags{b->flags}, 
    dissociation_energy{(float)b->dissociation_energy}, 
    half_life{(float)b->half_life}, 
    pids{b->i, b->j}
{
    this->p = cuda::toCUDADevice(*b->potential);
}

__host__ 
void cuda::Bond::finalize() {
    if(!(this->flags & BOND_ACTIVE)) 
        return;

    this->flags = ~BOND_ACTIVE;
    cuda::cudaFree(&this->p);
}


struct BondCUDAData {
    uint32_t id;
    int2 cid;

    BondCUDAData(Bond *b) : 
        id{b->id}, 
        cid{
            _Engine.s.celllist[_Engine.s.partlist[b->i]->id]->id, 
            _Engine.s.celllist[_Engine.s.partlist[b->j]->id]->id
        }
    {}
};


int cuda_bonds_bonds_initialize(engine *e, Bond *bonds, int N) {
    size_t size_bonds = N * sizeof(cuda::Bond);

    if(cudaMalloc(&cuda_bonds_device_arr, size_bonds) != cudaSuccess) 
        return cuda_error();

    cuda::Bond *bonds_cuda = (cuda::Bond*)malloc(size_bonds);

    int nr_runners = e->nr_runners;
    auto func = [&bonds, &bonds_cuda, N, nr_runners](size_t tid) {
        for(int i = tid; i < N; i += nr_runners) 
            bonds_cuda[i] = cuda::Bond(&bonds[i]);
    };
    parallel_for(nr_runners, func);

    if(cudaMemcpy(cuda_bonds_device_arr, bonds_cuda, size_bonds, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyToSymbol(cuda_bonds, &cuda_bonds_device_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    free(bonds_cuda);

    cuda_bonds_size = N;

    return S_OK;
}

int cuda_bonds_bonds_finalize(engine *e) { 
    size_t size_bonds = cuda_bonds_size * sizeof(cuda::Bond);

    cuda::Bond *bonds_cuda = (cuda::Bond*)malloc(size_bonds);
    if(cudaMemcpy(bonds_cuda, cuda_bonds_device_arr, size_bonds, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error();

    int nr_runners = e->nr_runners;
    int N = cuda_bonds_size;
    auto func = [&bonds_cuda, N, nr_runners](size_t tid) {
        for(int i = tid; i < N; i += nr_runners) 
            bonds_cuda[i].finalize();
    };
    parallel_for(nr_runners, func);

    free(bonds_cuda);

    if(cudaFree(cuda_bonds_device_arr) != cudaSuccess) 
        return cuda_error();
    
    cuda_bonds_size = 0;

    return S_OK;
}

int cuda_bonds_bonds_extend() {
    int Ni = cuda_bonds_size;
    int Nf = Ni + cuda_bonds_nrparts_incr;
    
    dim3 nr_threads(cuda_bonds_nr_threads, 1, 1);
    dim3 nr_blocks(std::max((unsigned)1, Ni / nr_threads.x), 1, 1);

    cuda::Bond *cuda_bonds_device_arr_new;
    if(cudaMalloc(&cuda_bonds_device_arr_new, Nf * sizeof(cuda::Bond)) != cudaSuccess) 
        return cuda_error();
    
    engine_cuda_memcpy<cuda::Bond><<<nr_blocks, nr_threads>>>(cuda_bonds_device_arr_new, cuda_bonds_device_arr, Ni * sizeof(cuda::Bond));
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    if(cudaFree(cuda_bonds_device_arr) != cudaSuccess) 
        return cuda_error();
    if(cudaMalloc(&cuda_bonds_device_arr, Nf * sizeof(cuda::Bond)) != cudaSuccess) 
        return cuda_error();

    engine_cuda_memcpy<cuda::Bond><<<nr_blocks, nr_threads>>>(cuda_bonds_device_arr, cuda_bonds_device_arr_new, Ni * sizeof(cuda::Bond));
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();
    if(cudaFree(cuda_bonds_device_arr_new) != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyToSymbol(cuda_bonds, &cuda_bonds_device_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    cuda_bonds_size = Nf;

    return S_OK;
}

__global__ 
void engine_cuda_set_bond_device(cuda::Bond b, unsigned int bid) {
    if(threadIdx.x > 0 || blockIdx.x > 0) 
        return;

    cuda_bonds[bid] = b;
}

int cuda::engine_cuda_add_bond(TissueForge::Bond *b) {
    cuda::Bond bc(b);
    auto bid = b->id;

    if(bid >= cuda_bonds_size) 
        if(cuda_bonds_bonds_extend() != S_OK) 
            return error(MDCERR_cuda);

    engine_cuda_set_bond_device<<<1, 1>>>(bc, bid);
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

__global__ 
void engine_cuda_set_bonds_device(cuda::Bond *bonds, unsigned int *bids, int nr_bonds) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = tid; i < nr_bonds; i += stride) 
        cuda_bonds[i] = bonds[bids[i]];
}

int cuda::engine_cuda_add_bonds(TissueForge::Bond *bonds, int nr_bonds) {
    TissueForge::Bond *b;
    uint32_t bidmax = 0;

    cuda::Bond *bcs = (cuda::Bond*)malloc(nr_bonds * sizeof(cuda::Bond));
    cuda::Bond *bcs_d;
    if(cudaMalloc(&bcs_d, nr_bonds * sizeof(cuda::Bond)) != cudaSuccess) 
        return cuda_error();

    unsigned int *bids = (unsigned int*)malloc(nr_bonds * sizeof(unsigned int));
    unsigned int *bids_d;
    if(cudaMalloc(&bids_d, nr_bonds * sizeof(unsigned int)) != cudaSuccess) 
        return cuda_error();

    for(int i = 0; i < nr_bonds; i++) {
        b = &bonds[i];
        bidmax = std::max(bidmax, b->id);
        bcs[i] = cuda::Bond(b);
        bids[i] = b->id;
    }

    if(cudaMemcpyAsync(bcs_d, bcs, nr_bonds * sizeof(cuda::Bond) , cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();
    if(cudaMemcpyAsync(bids_d, bids, nr_bonds * sizeof(unsigned int), cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    while(bidmax >= cuda_bonds_size)
        if(cuda_bonds_bonds_extend() != S_OK) 
            return error(MDCERR_cuda);

    dim3 nr_threads(cuda_bonds_nr_threads, 1, 1);
    dim3 nr_blocks(std::max((unsigned)1, nr_bonds / nr_threads.x), 1, 1);

    if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    engine_cuda_set_bonds_device<<<nr_blocks, nr_threads>>>(bcs_d, bids_d, nr_bonds);
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    free(bcs);
    free(bids);
    if(::cudaFree(bcs_d) != cudaSuccess) 
        return cuda_error();
    if(::cudaFree(bids_d) != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

int cuda::engine_cuda_finalize_bond(int bind) { 
    size_t size_bonds = cuda_bonds_size * sizeof(cuda::Bond);

    cuda::Bond *bonds_cuda = (cuda::Bond*)malloc(cuda_bonds_size * sizeof(cuda::Bond));
    if(cudaMemcpy(bonds_cuda, cuda_bonds_device_arr, size_bonds, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error();

    bonds_cuda[bind].finalize();

    if(cudaMemcpy(cuda_bonds_device_arr, bonds_cuda, size_bonds, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    free(bonds_cuda);

    return S_OK;
}

int cuda::engine_cuda_finalize_bonds(engine *e, int *binds, int nr_bonds) { 
    size_t size_bonds = cuda_bonds_size * sizeof(cuda::Bond);

    cuda::Bond *bonds_cuda = (cuda::Bond*)malloc(cuda_bonds_size * sizeof(cuda::Bond));
    if(cudaMemcpy(bonds_cuda, cuda_bonds_device_arr, size_bonds, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error();

    int nr_runners = e->nr_runners;
    auto func = [&bonds_cuda, &binds, nr_bonds, nr_runners](size_t tid) {
        for(int i = tid; i < nr_bonds; i += nr_runners) 
            bonds_cuda[binds[i]].finalize();
    };
    parallel_for(nr_runners, func);

    if(cudaMemcpy(cuda_bonds_device_arr, bonds_cuda, size_bonds, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    free(bonds_cuda);

    return S_OK;
}

int cuda::engine_cuda_finalize_bonds_all(engine *e) {
    size_t size_bonds = cuda_bonds_size * sizeof(cuda::Bond);

    cuda::Bond *bonds_cuda = (cuda::Bond*)malloc(cuda_bonds_size * sizeof(cuda::Bond));
    if(cudaMemcpy(bonds_cuda, cuda_bonds_device_arr, size_bonds, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error();

    int nr_bonds = cuda_bonds_size;
    int nr_runners = e->nr_runners;
    auto func = [&bonds_cuda, nr_bonds, nr_runners](size_t tid) {
        for(int i = tid; i < nr_bonds; i += nr_runners) 
            bonds_cuda[i].finalize();
    };
    parallel_for(nr_runners, func);

    if(cudaMemcpy(cuda_bonds_device_arr, bonds_cuda, size_bonds, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    free(bonds_cuda);

    return S_OK;
}

int engine_cuda_refresh_bond(TissueForge::Bond *b) {
    if(cuda::engine_cuda_finalize_bond(b->id) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda::engine_cuda_add_bond(b) != S_OK) 
        return error(MDCERR_cuda);

    return S_OK;
}

int engine_cuda_refresh_bonds(engine *e, TissueForge::Bond *bonds, int nr_bonds) { 
    int *binds = (int*)malloc(nr_bonds * sizeof(int));

    for(int i = 0; i < nr_bonds; i++) 
        binds[i] = bonds[i].id;

    if(cuda::engine_cuda_finalize_bonds(e, binds, nr_bonds) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda::engine_cuda_add_bonds(bonds, nr_bonds) != S_OK) 
        return error(MDCERR_cuda);

    free(binds);

    return S_OK;
}

int cuda_bonds_arrays_malloc() {
    size_t size_bonds = cuda_bonds_nrparts_chunk * sizeof(BondCUDAData);
    size_t size_forces = 3 * cuda_bonds_nrparts_chunk * sizeof(float);

    if(cudaMalloc(&cuda_bonds_forces_arr1, size_forces) != cudaSuccess) 
        return cuda_error();
    if(cudaMalloc(&cuda_bonds_forces_arr2, size_forces) != cudaSuccess) 
        return cuda_error();
    if((cuda_bonds_forces_local1 = (float*)malloc(size_forces)) == NULL) 
        return MDCERR_malloc;
    if((cuda_bonds_forces_local2 = (float*)malloc(size_forces)) == NULL) 
        return MDCERR_malloc;
    
    if(cudaMalloc(&cuda_bonds_bonds_arr1, size_bonds) != cudaSuccess) 
        return cuda_error();
    if(cudaMalloc(&cuda_bonds_bonds_arr2, size_bonds) != cudaSuccess) 
        return cuda_error();
    if((cuda_bonds_bonds_local1 = (BondCUDAData*)malloc(size_bonds)) == NULL) 
        return MDCERR_malloc;
    if((cuda_bonds_bonds_local2 = (BondCUDAData*)malloc(size_bonds)) == NULL) 
        return MDCERR_malloc;

    if(cuda_bonds_shared_arrays_malloc() != S_OK) 
        return error(MDCERR_cuda);

    return S_OK;
}

int cuda_bonds_arrays_free() {
    if(cudaFree(cuda_bonds_bonds_arr1) != cudaSuccess) 
        return cuda_error();
    if(cudaFree(cuda_bonds_bonds_arr2) != cudaSuccess) 
        return cuda_error();
    free(cuda_bonds_bonds_local1);
    free(cuda_bonds_bonds_local2);

    if(cudaFree(cuda_bonds_forces_arr1) != cudaSuccess) 
        return cuda_error();
    if(cudaFree(cuda_bonds_forces_arr2) != cudaSuccess) 
        return cuda_error();
    free(cuda_bonds_forces_local1);
    free(cuda_bonds_forces_local2);

    if(cuda_bonds_shared_arrays_free() != S_OK) 
        return error(MDCERR_cuda);

    return S_OK;
}

int cuda::Bond_setBlocks(const unsigned int &nr_blocks) {
    cuda_bonds_nr_blocks = nr_blocks;
    return S_OK;
}

int cuda::Bond_setThreads(const unsigned int &nr_threads) {
    cuda_bonds_nr_threads = nr_threads;
    return S_OK;
}

__device__ 
void bond_eval_single_cuda(TissueForge::Potential *pot, float ri, float rj, float _r2, float *dx, float *force, float *epot_out) {
    float r, ee, eff;
    int k;

    if(pot->kind == POTENTIAL_KIND_COMBINATION && pot->flags & POTENTIAL_SUM) {
        if(pot->pca != NULL) bond_eval_single_cuda(pot->pca, ri, rj, _r2, dx, force, epot_out);
        if(pot->pcb != NULL) bond_eval_single_cuda(pot->pcb, ri, rj, _r2, dx, force, epot_out);
        return;
    }

    /* Get r for the right type. */
    r = _r2 * rsqrtf(_r2);
    
    if(pot->flags & POTENTIAL_SCALED) {
        r = r / (ri + rj);
    }
    else if(pot->flags & POTENTIAL_SHIFTED) {
        r = r - (ri + rj) + pot->r0_plusone;
    }

    if(r > pot->b) 
        return;

    cuda::potential_eval_cuda(pot, fmax(r * r, pot->a * pot->a), &ee, &eff);

    // Update the forces
    for (k = 0; k < 3; k++) {
        force[k] -= eff * dx[k];
    }

    // Tabulate the energy
    *epot_out += ee;
}


__global__ 
void bond_eval_cuda(BondCUDAData *bonds, int nr_bonds, float *forces, float *epot_out, bool *toDestroy) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, k;
    cuda::Bond *b;
    BondCUDAData *bu;
    float dx[3], r2, fix[3], epot = 0.f;
    int shift[3];
    float rn;
    int3 cix, cjx;
    __shared__ unsigned int *parts_key;

    if(threadIdx.x == 0) {
        if(cuda_bonds_parts_from_engine) 
            parts_key = cuda_bonds_parts_from_engine_key;
        else 
            parts_key = NULL;
    }

    for(i = tid; i < nr_bonds; i += stride) {
        bu = &bonds[i];
        b = &cuda_bonds[bu->id];

        int pidi = parts_key == NULL ? b->pids.x : parts_key[b->pids.x];
        int pidj = parts_key == NULL ? b->pids.y : parts_key[b->pids.y];

        if(!(b->flags & BOND_ACTIVE) || (cuda_bonds_parts_flags[pidi] & PARTICLE_GHOST && cuda_bonds_parts_flags[pidj] & PARTICLE_GHOST)) {
            forces[3 * i    ] = 0.f;
            forces[3 * i + 1] = 0.f;
            forces[3 * i + 2] = 0.f;
            epot_out[i] = 0.f;
            continue;
        }

        // Test for decay
        if(!isinf(b->half_life) && b->half_life > 0.f) { 
            engine_cuda_rand_uniform(&rn);
            if(1.0 - powf(2.0, -cuda_bonds_dt / b->half_life) > rn) { 
                toDestroy[i] = true;
                forces[3 * i    ] = 0.f;
                forces[3 * i + 1] = 0.f;
                forces[3 * i + 2] = 0.f;
                epot_out[i] = 0.f;
                continue;
            }
        }

        // Get the distance between the particles
        cix = cuda_bonds_cloc[bu->cid.x];
        cjx = cuda_bonds_cloc[bu->cid.y];
        shift[0] = cix.x - cjx.x;
        shift[1] = cix.y - cjx.y;
        shift[2] = cix.z - cjx.z;
        for(k = 0; k < 3; k++) {
            if(shift[k] > 1) shift[k] = -1;
            else if(shift[k] < -1) shift[k] = 1;
        }
        float4 pix = cuda_bonds_parts_pos[pidi];
        float4 pjx = cuda_bonds_parts_pos[pidj];
        pix.x += cuda_bonds_cell_edge_lens.x * shift[0];
        pix.y += cuda_bonds_cell_edge_lens.y * shift[1];
        pix.z += cuda_bonds_cell_edge_lens.z * shift[2];

        r2 = 0.f;
        dx[0] = pix.x - pjx.x; r2 += dx[0]*dx[0];
        dx[1] = pix.y - pjx.y; r2 += dx[1]*dx[1];
        dx[2] = pix.z - pjx.z; r2 += dx[2]*dx[2];

        memset(fix, 0.f, 3 * sizeof(float));

        bond_eval_single_cuda(&b->p, pix.w, pjx.w, r2, dx, fix, &epot);
        forces[3 * i    ] = fix[0];
        forces[3 * i + 1] = fix[1];
        forces[3 * i + 2] = fix[2];
        epot_out[i] = epot;

        // Test for dissociation
        toDestroy[i] = epot >= b->dissociation_energy;
    }
}

int engine_bond_flip_stream() {
    if(cuda_bonds_front_stream == 1) { 
        cuda_bonds_bonds_arr         = cuda_bonds_bonds_arr2;
        cuda_bonds_bonds_local       = cuda_bonds_bonds_local2;
        cuda_bonds_forces_arr        = cuda_bonds_forces_arr2;
        cuda_bonds_forces_local      = cuda_bonds_forces_local2;
    }
    else {
        cuda_bonds_bonds_arr         = cuda_bonds_bonds_arr1;
        cuda_bonds_bonds_local       = cuda_bonds_bonds_local1;
        cuda_bonds_forces_arr        = cuda_bonds_forces_arr1;
        cuda_bonds_forces_local      = cuda_bonds_forces_local1;
    }

    return engine_bond_flip_shared();
}

int engine_bond_cuda_load_bond_chunk(Bond *bonds, int loc, int N) { 
    size_t size_bonds = N * sizeof(BondCUDAData);
    size_t size_potenergies = N * sizeof(float);
    size_t size_forces = 3 * size_potenergies;
    size_t size_todestroy = N * sizeof(bool);
    Bond *buff = &bonds[loc];

    int nr_runners = _Engine.nr_runners;
    auto bl = cuda_bonds_bonds_local;
    auto func = [&bl, &buff, N, nr_runners](size_t tid) -> void {
        for(int j = tid; j < N; j += nr_runners) 
            bl[j] = BondCUDAData(&buff[j]);
    };
    parallel_for(nr_runners, func);

    if(cudaMemcpyAsync(cuda_bonds_bonds_arr, cuda_bonds_bonds_local, size_bonds, cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    dim3 nr_threads(std::min((int)cuda_bonds_nr_threads, N), 1, 1);
    dim3 nr_blocks(std::min(cuda_bonds_nr_blocks, N / nr_threads.x), 1, 1);

    bond_eval_cuda<<<nr_blocks, nr_threads, 0, *cuda_bonds_stream>>>(
        cuda_bonds_bonds_arr, N, cuda_bonds_forces_arr, cuda_bonds_potenergies_arr, cuda_bonds_todestroy_arr
    );
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyAsync(cuda_bonds_forces_local, cuda_bonds_forces_arr, size_forces, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyAsync(cuda_bonds_potenergies_local, cuda_bonds_potenergies_arr, size_potenergies, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyAsync(cuda_bonds_todestroy_local, cuda_bonds_todestroy_arr, size_todestroy, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

int engine_bond_cuda_unload_bond_chunk(Bond *bonds, int loc, int N, struct engine *e, float *epot_out) { 
    int i, k;
    float epot = 0.f;
    float *bufff, ee;
    Bond *buffb = &bonds[loc];

    for(i = 0; i < N; i++) {
        auto b = &buffb[i];
        
        ee = cuda_bonds_potenergies_local[i];
        epot += ee;
        b->potential_energy += ee;
        if(cuda_bonds_todestroy_local[i]) {
            Bond_Destroy(b);
            continue;
        }

        bufff = &cuda_bonds_forces_local[3 * i];
        auto pi = e->s.partlist[b->i];
        auto pj = e->s.partlist[b->j];
        for(k = 0; k < 3; k++) {
            pi->f[k] += bufff[k];
            pj->f[k] -= bufff[k];
        }
    }
    
    // Store the potential energy.
    *epot_out += epot;

    return S_OK;
}

int cuda::engine_bond_eval_cuda(struct TissueForge::Bond *bonds, int N, struct engine *e, float *epot_out) {
    int i, n;
    float epot = 0.0;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error();

    if(cuda_bonds_parts_load(e) != S_OK) 
        return error(MDCERR_cuda);

    engine_bond_flip_stream();
    
    if(N < cuda_bonds_nrparts_chunk) {
        if(engine_bond_cuda_load_bond_chunk(bonds, 0, N) != S_OK) 
            return error(MDCERR_cuda);
        if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
            return cuda_error();
        if(engine_bond_cuda_unload_bond_chunk(bonds, 0, N, e, epot_out) != S_OK) 
            return error(MDCERR_cuda);
        if(cuda_bonds_parts_unload(e) != S_OK) 
            return error(MDCERR_cuda);
        return S_OK;
    }

    n = cuda_bonds_nrparts_chunk;
    if(engine_bond_cuda_load_bond_chunk(bonds, 0, n) != S_OK) 
        return error(MDCERR_cuda);
    for(i = cuda_bonds_nrparts_chunk; i < N; i += cuda_bonds_nrparts_chunk) {
        if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
            return cuda_error();
        if(engine_bond_cuda_unload_bond_chunk(bonds, i - cuda_bonds_nrparts_chunk, n, e, &epot) != S_OK) 
            return error(MDCERR_cuda);

        engine_bond_flip_stream();

        n = std::min(N - i, cuda_bonds_nrparts_chunk);
        if(engine_bond_cuda_load_bond_chunk(bonds, i, n) != S_OK) 
            return error(MDCERR_cuda);
    }
    if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();
    if(engine_bond_cuda_unload_bond_chunk(bonds, N - n, n, e, &epot) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_bonds_parts_unload(e) != S_OK) 
        return error(MDCERR_cuda);

    *epot_out += epot;
    
    return S_OK;
}

int engine_bond_cuda_initialize(engine *e) {
    if(e->bonds_cuda) 
        return S_OK;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error();

    cuda_bonds_parts_hold();

    if(cuda_bonds_device_constants_init(e) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_bonds_nr_blocks == 0) {
        cuda_bonds_nr_blocks = cuda::maxBlockDimX(0);
    }

    if(cuda_bonds_nr_threads == 0) {
        cuda_bonds_nr_threads = cuda::maxThreadsPerBlock(0);
    }

    int nr_rands = cuda_bonds_nr_threads * cuda_bonds_nr_blocks;
    if(engine_cuda_rand_unif_init(e, nr_rands) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_bonds_arrays_malloc() != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_bonds_bonds_initialize(e, e->bonds, e->nr_bonds) != S_OK) 
        return error(MDCERR_cuda);

    e->bonds_cuda = true;
    
    return S_OK;
}

int engine_bond_cuda_finalize(engine *e) {
    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error();

    cuda_bonds_parts_release();

    if(cuda_bonds_bonds_finalize(e) != S_OK) 
        return error(MDCERR_cuda);

    if(engine_cuda_rand_unif_finalize(e) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_bonds_arrays_free() != S_OK) 
        return error(MDCERR_cuda);

    e->bonds_cuda = false;

    return S_OK;
}

int engine_bond_cuda_refresh(engine *e) {
    if(engine_bond_cuda_finalize(e) != S_OK) 
        return error(MDCERR_cuda);

    if(engine_bond_cuda_initialize(e) != S_OK) 
        return error(MDCERR_cuda);

    if(cudaDeviceSynchronize() != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

int cuda::Bond_getDevice() {
    return cuda_bonds_device;
}

int cuda::Bond_setDevice(engine *e, const int &deviceId) {
    bool refreshing = e->bonds_cuda;

    if(refreshing) 
        if(engine_bond_cuda_finalize(e) != S_OK) 
            return error(MDCERR_cuda);

    cuda_bonds_device = deviceId;

    if(refreshing) 
        if(engine_bond_cuda_initialize(e) != S_OK) 
            return error(MDCERR_cuda);

    return S_OK;
}

int cuda::Bond_toDevice(engine *e) {
    if(e->bonds_cuda) 
        return S_OK;
    
    return engine_bond_cuda_initialize(e);
}

int cuda::Bond_fromDevice(engine *e) {
    if(!e->bonds_cuda) 
        return S_OK;
    
    return engine_bond_cuda_finalize(e);
}

int cuda::Bond_refresh(engine *e) {
    if(!e->bonds_cuda) 
        return S_OK;
    
    return engine_bond_cuda_refresh(e);
}

int cuda::Bond_refreshBond(engine *e, BondHandle *b) {
    if(e->bonds_cuda) 
        return S_OK;

    if(b == NULL) 
        return MDCERR_null;

    return engine_cuda_refresh_bond(b->get());
}

int cuda::Bond_refreshBonds(engine *e, BondHandle **bonds, int nr_bonds) {
    if(e->bonds_cuda) 
        return S_OK;

    TissueForge::Bond *bs = (TissueForge::Bond*)malloc(nr_bonds * sizeof(TissueForge::Bond));
    BondHandle *bh;
    for(int i = 0; i < nr_bonds; i++) { 
        bh = bonds[i];
        if(bh == NULL) 
            return MDCERR_null;
        bs[i] = *(bh->get());
    }
    
    if(engine_cuda_refresh_bonds(e, bs, nr_bonds) != S_OK) 
        return error(MDCERR_cuda);

    free(bs);

    return S_OK;
}


// cuda::Angle


__host__ 
cuda::Angle::Angle() {
    this->flags = ~ANGLE_ACTIVE;
}

__host__ 
cuda::Angle::Angle(TissueForge::Angle *a) : 
    flags{a->flags}, 
    dissociation_energy{(float)a->dissociation_energy}, 
    half_life{(float)a->half_life}, 
    pids{a->i, a->j, a->k}
{
    this->p = cuda::toCUDADevice(*a->potential);
}

__host__ 
void cuda::Angle::finalize() {
    if(!(this->flags & ANGLE_ACTIVE)) 
        return;

    this->flags = ~ANGLE_ACTIVE;
    cuda::cudaFree(&this->p);
}

struct AngleCUDAData {
    uint32_t id;
    int3 cid;

    AngleCUDAData(Angle *a, uint32_t aid) : 
        id{aid}, 
        cid{
            _Engine.s.celllist[_Engine.s.partlist[a->i]->id]->id, 
            _Engine.s.celllist[_Engine.s.partlist[a->j]->id]->id, 
            _Engine.s.celllist[_Engine.s.partlist[a->k]->id]->id
        }
    {}
};

int cuda_angles_arrays_malloc() {
    size_t size_angles = cuda_bonds_nrparts_chunk * sizeof(AngleCUDAData);
    size_t size_forces = 6 * cuda_bonds_nrparts_chunk * sizeof(float);

    if(cudaMalloc(&cuda_angles_forces_arr1, size_forces) != cudaSuccess) 
        return cuda_error();
    if(cudaMalloc(&cuda_angles_forces_arr2, size_forces) != cudaSuccess) 
        return cuda_error();
    if((cuda_angles_forces_local1 = (float*)malloc(size_forces)) == NULL) 
        return MDCERR_malloc;
    if((cuda_angles_forces_local2 = (float*)malloc(size_forces)) == NULL) 
        return MDCERR_malloc;
    
    if(cudaMalloc(&cuda_angles_angles_arr1, size_angles) != cudaSuccess) 
        return cuda_error();
    if(cudaMalloc(&cuda_angles_angles_arr2, size_angles) != cudaSuccess) 
        return cuda_error();
    if((cuda_angles_angles_local1 = (AngleCUDAData*)malloc(size_angles)) == NULL) 
        return MDCERR_malloc;
    if((cuda_angles_angles_local2 = (AngleCUDAData*)malloc(size_angles)) == NULL) 
        return MDCERR_malloc;

    if(cuda_bonds_shared_arrays_malloc() != S_OK) 
        return error(MDCERR_cuda);

    return S_OK;
}

int cuda_angles_arrays_free() {
    if(cudaFree(cuda_angles_angles_arr1) != cudaSuccess) 
        return cuda_error();
    if(cudaFree(cuda_angles_angles_arr2) != cudaSuccess) 
        return cuda_error();
    free(cuda_angles_angles_local1);
    free(cuda_angles_angles_local2);

    if(cudaFree(cuda_angles_forces_arr1) != cudaSuccess) 
        return cuda_error();
    if(cudaFree(cuda_angles_forces_arr2) != cudaSuccess) 
        return cuda_error();
    free(cuda_angles_forces_local1);
    free(cuda_angles_forces_local2);

    if(cuda_bonds_shared_arrays_free() != S_OK) 
        return error(MDCERR_cuda);

    return S_OK;
}

int cuda_angles_angles_initialize(engine *e, Angle *angles, int N) {
    size_t size_angles = N * sizeof(cuda::Angle);

    if(cudaMalloc(&cuda_angles_device_arr, size_angles) != cudaSuccess) 
        return cuda_error();

    cuda::Angle *angles_cuda = (cuda::Angle*)malloc(size_angles);

    int nr_runners = e->nr_runners;
    auto func = [&angles, &angles_cuda, N, nr_runners](size_t tid) {
        for(int i = tid; i < N; i += nr_runners) 
            angles_cuda[i] = cuda::Angle(&angles[i]);
    };
    parallel_for(nr_runners, func);

    if(cudaMemcpy(cuda_angles_device_arr, angles_cuda, size_angles, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyToSymbol(cuda_angles, &cuda_angles_device_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    free(angles_cuda);

    cuda_angles_size = N;

    return S_OK;
}

int cuda_angles_angles_finalize(engine *e) {
    size_t size_angles = cuda_angles_size * sizeof(cuda::Angle);

    cuda::Angle *angles_cuda = (cuda::Angle*)malloc(size_angles);
    if(cudaMemcpy(angles_cuda, cuda_angles_device_arr, size_angles, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error();

    int nr_runners = e->nr_runners;
    int N = cuda_angles_size;
    auto func = [&angles_cuda, N, nr_runners](size_t tid) {
        for(int i = tid; i < N; i += nr_runners) 
            angles_cuda[i].finalize();
    };
    parallel_for(nr_runners, func);

    free(angles_cuda);

    if(cudaFree(cuda_angles_device_arr) != cudaSuccess) 
        return cuda_error();
    
    cuda_angles_size = 0;

    return S_OK;
}

int cuda_angles_angles_extend() {
    int Ni = cuda_angles_size;
    int Nf = Ni + cuda_bonds_nrparts_incr;
    
    dim3 nr_threads(cuda_angles_nr_threads, 1, 1);
    dim3 nr_blocks(std::max((unsigned)1, Ni / nr_threads.x), 1, 1);

    cuda::Angle *cuda_angles_device_arr_new;
    if(cudaMalloc(&cuda_angles_device_arr_new, Nf * sizeof(cuda::Angle)) != cudaSuccess) 
        return cuda_error();
    
    engine_cuda_memcpy<cuda::Angle><<<nr_blocks, nr_threads>>>(cuda_angles_device_arr_new, cuda_angles_device_arr, Ni * sizeof(cuda::Angle));
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    if(cudaFree(cuda_angles_device_arr) != cudaSuccess) 
        return cuda_error();
    if(cudaMalloc(&cuda_angles_device_arr, Nf * sizeof(cuda::Angle)) != cudaSuccess) 
        return cuda_error();

    engine_cuda_memcpy<cuda::Angle><<<nr_blocks, nr_threads>>>(cuda_angles_device_arr, cuda_angles_device_arr_new, Ni * sizeof(cuda::Angle));
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();
    if(cudaFree(cuda_angles_device_arr_new) != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyToSymbol(cuda_angles, &cuda_angles_device_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    cuda_angles_size = Nf;

    return S_OK;
}

int engine_angle_cuda_initialize(engine *e) {
    if(e->angles_cuda) 
        return S_OK;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error();

    cuda_bonds_parts_hold();

    if(cuda_bonds_device_constants_init(e) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_angles_nr_blocks == 0) {
        cuda_angles_nr_blocks = cuda::maxBlockDimX(0);
    }

    if(cuda_angles_nr_threads == 0) {
        cuda_angles_nr_threads = cuda::maxThreadsPerBlock(0);
    }

    int nr_rands = cuda_angles_nr_threads * cuda_angles_nr_blocks;
    if(engine_cuda_rand_unif_init(e, nr_rands) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_angles_arrays_malloc() != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_angles_angles_initialize(e, e->angles, e->nr_angles) != S_OK) 
        return error(MDCERR_cuda);

    e->angles_cuda = true;
    
    return S_OK;
}

int engine_angle_cuda_finalize(engine *e) {
    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error();

    cuda_bonds_parts_release();

    if(cuda_angles_angles_finalize(e) != S_OK) 
        return error(MDCERR_cuda);

    if(engine_cuda_rand_unif_finalize(e) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_angles_arrays_free() != S_OK) 
        return error(MDCERR_cuda);

    e->angles_cuda = false;

    return S_OK;
}

__global__ 
void engine_cuda_set_angle_device(cuda::Angle a, unsigned int aid) {
    if(threadIdx.x > 0 || blockIdx.x > 0) 
        return;

    cuda_angles[aid] = a;
}

int cuda::engine_cuda_add_angle(AngleHandle *ah) {
    cuda::Angle ac(ah->get());
    auto aid = ah->id;

    if(aid >= cuda_angles_size) 
        if(cuda_angles_angles_extend() != S_OK) 
            return error(MDCERR_cuda);

    engine_cuda_set_angle_device<<<1, 1>>>(ac, aid);
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

int cuda::engine_cuda_finalize_angle(int aind) {
    size_t size_angles = cuda_angles_size * sizeof(cuda::Angle);

    cuda::Angle *angles_cuda = (cuda::Angle*)malloc(cuda_angles_size * sizeof(cuda::Angle));
    if(cudaMemcpy(angles_cuda, cuda_angles_device_arr, size_angles, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error();

    angles_cuda[aind].finalize();

    if(cudaMemcpy(cuda_angles_device_arr, angles_cuda, size_angles, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    free(angles_cuda);

    return S_OK;
}

int engine_cuda_refresh_angle(AngleHandle *ah) {
    if(cuda::engine_cuda_finalize_angle(ah->id) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda::engine_cuda_add_angle(ah) != S_OK) 
        return error(MDCERR_cuda);

    return S_OK;
}

__global__ 
void engine_cuda_set_angles_device(cuda::Angle *angles, unsigned int *aids, int nr_angles) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = tid; i < nr_angles; i += stride) 
        cuda_angles[i] = angles[aids[i]];
}

int engine_cuda_add_angles(AngleHandle *angles, int nr_angles) {
    AngleHandle *a;
    uint32_t aidmax = 0;

    cuda::Angle *acs = (cuda::Angle*)malloc(nr_angles * sizeof(cuda::Angle));
    cuda::Angle *acs_d;
    if(cudaMalloc(&acs_d, nr_angles * sizeof(cuda::Angle)) != cudaSuccess) 
        return cuda_error();

    unsigned int *aids = (unsigned int*)malloc(nr_angles * sizeof(unsigned int));
    unsigned int *aids_d;
    if(cudaMalloc(&aids_d, nr_angles * sizeof(unsigned int)) != cudaSuccess) 
        return cuda_error();

    for(int i = 0; i < nr_angles; i++) {
        a = &angles[i];
        aidmax = std::max(aidmax, (uint32_t)a->id);
        acs[i] = cuda::Angle(a->get());
        aids[i] = a->id;
    }

    if(cudaMemcpyAsync(acs_d, acs, nr_angles * sizeof(cuda::Angle), cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();
    if(cudaMemcpyAsync(aids_d, aids, nr_angles * sizeof(unsigned int), cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    while(aidmax >= cuda_angles_size)
        if(cuda_angles_angles_extend() != S_OK) 
            return error(MDCERR_cuda);

    dim3 nr_threads(cuda_angles_nr_threads, 1, 1);
    dim3 nr_blocks(std::max((unsigned)1, nr_angles / nr_threads.x), 1, 1);

    if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    engine_cuda_set_angles_device<<<nr_blocks, nr_threads>>>(acs_d, aids_d, nr_angles);
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    free(acs);
    free(aids);
    if(cudaFree(acs_d) != cudaSuccess) 
        return cuda_error();
    if(cudaFree(aids_d) != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

int cuda::engine_cuda_finalize_angles(engine *e, int *ainds, int nr_angles) {
    size_t size_angles = cuda_angles_size * sizeof(cuda::Angle);

    cuda::Angle *angles_cuda = (cuda::Angle*)malloc(cuda_angles_size * sizeof(cuda::Angle));
    if(cudaMemcpy(angles_cuda, cuda_angles_device_arr, size_angles, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error();

    int nr_runners = e->nr_runners;
    auto func = [&angles_cuda, &ainds, nr_angles, nr_runners](size_t tid) {
        for(int i = tid; i < nr_angles; i += nr_runners) 
            angles_cuda[ainds[i]].finalize();
    };
    parallel_for(nr_runners, func);

    if(cudaMemcpy(cuda_angles_device_arr, angles_cuda, size_angles, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    free(angles_cuda);

    return S_OK;
}

int engine_cuda_refresh_angles(engine *e, AngleHandle *angles, int nr_angles) { 
    int *ainds = (int*)malloc(nr_angles * sizeof(int));

    for(int i = 0; i < nr_angles; i++) 
        ainds[i] = angles[i].id;

    if(cuda::engine_cuda_finalize_angles(e, ainds, nr_angles) != S_OK) 
        return error(MDCERR_cuda);

    if(engine_cuda_add_angles(angles, nr_angles) != S_OK) 
        return error(MDCERR_cuda);

    free(ainds);

    return S_OK;
}

int cuda::engine_cuda_finalize_angles_all(engine *e) {
    size_t size_angles = cuda_angles_size * sizeof(cuda::Angle);

    cuda::Angle *angles_cuda = (cuda::Angle*)malloc(cuda_angles_size * sizeof(cuda::Angle));
    if(cudaMemcpy(angles_cuda, cuda_angles_device_arr, size_angles, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error();

    int nr_angles = cuda_angles_size;
    int nr_runners = e->nr_runners;
    auto func = [&angles_cuda, nr_angles, nr_runners](size_t tid) {
        for(int i = tid; i < nr_angles; i += nr_runners) 
            angles_cuda[i].finalize();
    };
    parallel_for(nr_runners, func);

    if(cudaMemcpy(cuda_angles_device_arr, angles_cuda, size_angles, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error();

    free(angles_cuda);

    return S_OK;
}

int engine_angle_flip_stream() {
    if(cuda_bonds_front_stream == 1) { 
        cuda_angles_angles_arr         = cuda_angles_angles_arr2;
        cuda_angles_angles_local       = cuda_angles_angles_local2;
        cuda_angles_forces_arr         = cuda_angles_forces_arr2;
        cuda_angles_forces_local       = cuda_angles_forces_local2;
    }
    else {
        cuda_angles_angles_arr         = cuda_angles_angles_arr1;
        cuda_angles_angles_local       = cuda_angles_angles_local1;
        cuda_angles_forces_arr         = cuda_angles_forces_arr1;
        cuda_angles_forces_local       = cuda_angles_forces_local1;
    }

    return engine_bond_flip_shared();
}

__device__ 
void angle_eval_single_cuda(TissueForge::Potential *pot, float _ctheta, float *dxi, float *dxk, float *force_i, float *force_k, float *epot_out) {
    float ee, eff;
    int k;

    if(pot->kind == POTENTIAL_KIND_COMBINATION && pot->flags & POTENTIAL_SUM) {
        if(pot->pca != NULL) angle_eval_single_cuda(pot->pca, _ctheta, dxi, dxk, force_i, force_k, epot_out);
        if(pot->pcb != NULL) angle_eval_single_cuda(pot->pcb, _ctheta, dxi, dxk, force_i, force_k, epot_out);
        return;
    }

    if(_ctheta > pot->b) 
        return;
    
    cuda::potential_eval_r_cuda(pot, fmax(_ctheta, pot->a), &ee, &eff);

    // Update the forces
    for (k = 0; k < 3; k++) {
        force_i[k] -= eff * dxi[k];
        force_k[k] -= eff * dxk[k];
    }

    // Tabulate the energy
    *epot_out += ee;
}

__global__ 
void angle_eval_cuda(AngleCUDAData *angles, int nr_angles, float *forces, float *epot_out, bool *toDestroy) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, k;
    cuda::Angle *a;
    AngleCUDAData *au;
    float xi[3], xj[3], xk[3], rji[3], rjk[3];
    float dxi[3], dxk[3], ctheta, fix[3], fkx[3], epot = 0.f;
    float dprod, inji, injk;
    float3 x, vik;
    int3 cix, cjx, ckx;
    __shared__ unsigned int *parts_key;

    if(threadIdx.x == 0) {
        if(cuda_bonds_parts_from_engine) 
            parts_key = cuda_bonds_parts_from_engine_key;
        else 
            parts_key = NULL;
    }
    
    __syncthreads();

    for(i = tid; i < nr_angles; i += stride) {
        au = &angles[i];
        a = &cuda_angles[au->id];

        int pidi = parts_key == NULL ? a->pids.x : parts_key[a->pids.x];
        int pidj = parts_key == NULL ? a->pids.y : parts_key[a->pids.y];
        int pidk = parts_key == NULL ? a->pids.z : parts_key[a->pids.z];

        if(!(a->flags & ANGLE_ACTIVE) || (cuda_bonds_parts_flags[pidi] & PARTICLE_GHOST && cuda_bonds_parts_flags[pidj] & PARTICLE_GHOST && cuda_bonds_parts_flags[pidk] & PARTICLE_GHOST)) {
            forces[6 * i    ] = 0.f;
            forces[6 * i + 1] = 0.f;
            forces[6 * i + 2] = 0.f;
            forces[6 * i + 3] = 0.f;
            forces[6 * i + 4] = 0.f;
            forces[6 * i + 5] = 0.f;
            epot_out[i] = 0.f;
            continue;
        }

        // Test for decay
        if(!isinf(a->half_life) && a->half_life > 0.f) { 
            engine_cuda_rand_uniform(&dprod);
            if(1.0 - powf(2.0, -cuda_bonds_dt / a->half_life) > dprod) { 
                toDestroy[i] = true;
                forces[6 * i    ] = 0.f;
                forces[6 * i + 1] = 0.f;
                forces[6 * i + 2] = 0.f;
                forces[6 * i + 3] = 0.f;
                forces[6 * i + 4] = 0.f;
                forces[6 * i + 5] = 0.f;
                epot_out[i] = 0.f;
                continue;
            }
        }

        // Get the distance and angle between the particles and angle rays relative to pj's cell. */

        cix = cuda_bonds_cloc[au->cid.x];
        cjx = cuda_bonds_cloc[au->cid.y];
        ckx = cuda_bonds_cloc[au->cid.z];

        float4 pix = cuda_bonds_parts_pos[pidi];
        float4 pjx = cuda_bonds_parts_pos[pidj];
        float4 pkx = cuda_bonds_parts_pos[pidk];

        xj[0] = pjx.x;
        xj[1] = pjx.y;
        xj[2] = pjx.z;

        k = cix.x - cjx.x;
        if(k > 1) k = -1;
        else if(k < -1) k = 1;
        xi[0] = pix.x + k * cuda_bonds_cell_edge_lens.x;

        k = cix.y - cjx.y;
        if(k > 1) k = -1;
        else if(k < -1) k = 1;
        xi[1] = pix.y + k * cuda_bonds_cell_edge_lens.y;

        k = cix.z - cjx.z;
        if(k > 1) k = -1;
        else if(k < -1) k = 1;
        xi[2] = pix.z + k * cuda_bonds_cell_edge_lens.z;

        k = ckx.x - cjx.x;
        if(k > 1) k = -1;
        else if(k < -1) k = 1;
        xk[0] = pkx.x + k * cuda_bonds_cell_edge_lens.x;

        k = ckx.y - cjx.y;
        if(k > 1) k = -1;
        else if(k < -1) k = 1;
        xk[1] = pkx.y + k * cuda_bonds_cell_edge_lens.y;

        k = ckx.z - cjx.z;
        if(k > 1) k = -1;
        else if(k < -1) k = 1;
        xk[2] = pkx.z + k * cuda_bonds_cell_edge_lens.z;

        dprod = 0.f;
        inji = 0.f;
        injk = 0.f;
        for(k = 0; k < 3; k++) {
            rji[k] = xi[k] - xj[k];
            rjk[k] = xk[k] - xj[k];
            dprod += rji[k] * rjk[k];
            inji += rji[k] * rji[k];
            injk += rjk[k] * rjk[k];
        }

        inji = rsqrtf(inji);
        injk = rsqrtf(injk);
        
        /* Compute the cosine. */
        ctheta = fmax(-1.f, fmin(1.f, dprod * inji * injk));

        // Set the derivatives.
        // particles could be perpenducular, then plan is undefined, so
        // choose a random orientation plane
        if(ctheta == 0 || ctheta == -1) {
            engine_cuda_rand_uniform(&dprod);
            x.x = 2.f * dprod - 1.f;
            engine_cuda_rand_uniform(&dprod);
            x.y = 2.f * dprod - 1.f;
            engine_cuda_rand_uniform(&dprod);
            x.z = 2.f * dprod - 1.f;
            
            // vector between outer particles
            vik = make_float3(xi[0] - xk[0], xi[1] - xk[1], xi[2] - xk[2]);
            
            // make it orthogonal to rji
            dprod = x.x * vik.x + x.y * vik.y + x.z * vik.z;
            x.x -= dprod * vik.x;
            x.y -= dprod * vik.y;
            x.z -= dprod * vik.z;
            
            // normalize it.
            dprod = sqrtf(x.x * x.x + x.y * x.y + x.z * x.z);
            dxi[0] = dxk[0] = x.x / dprod;
            dxi[1] = dxk[1] = x.y / dprod;
            dxi[2] = dxk[2] = x.z / dprod;
        } 
        else {
            for(k = 0; k < 3 ; k++) {
                dxi[k] = (rjk[k] * injk - ctheta * rji[k] * inji) * inji;
                dxk[k] = (rji[k] * inji - ctheta * rjk[k] * injk) * injk;
            }
        }

        fix[0] = fix[1] = fix[2] = fkx[0] = fkx[1] = fkx[2] = 0.f;
        angle_eval_single_cuda(&a->p, ctheta, dxi, dxk, fix, fkx, &epot);

        forces[6 * i    ] = fix[0];
        forces[6 * i + 1] = fix[1];
        forces[6 * i + 2] = fix[2];
        forces[6 * i + 3] = fkx[0];
        forces[6 * i + 4] = fkx[1];
        forces[6 * i + 5] = fkx[2];
        epot_out[i] = epot;

        // Test for dissociation
        toDestroy[i] = epot >= a->dissociation_energy;
    }
}

int engine_angle_cuda_load_angle_chunk(Angle *angles, int loc, int N) { 
    size_t size_angles = N * sizeof(AngleCUDAData);
    size_t size_potenergies = N * sizeof(float);
    size_t size_forces = 6 * size_potenergies;
    size_t size_todestroy = N * sizeof(bool);
    Angle *buff = &angles[loc];

    int nr_runners = _Engine.nr_runners;
    auto al = cuda_angles_angles_local;
    auto func = [&al, &buff, loc, N, nr_runners](size_t tid) -> void {
        for(int j = tid; j < N; j += nr_runners) 
            al[j] = AngleCUDAData(&buff[j], loc + j);
    };
    parallel_for(nr_runners, func);

    if(cudaMemcpyAsync(cuda_angles_angles_arr, cuda_angles_angles_local, size_angles, cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    dim3 nr_threads(std::min((int)cuda_angles_nr_threads, N), 1, 1);
    dim3 nr_blocks(std::min(cuda_angles_nr_blocks, N / nr_threads.x), 1, 1);

    angle_eval_cuda<<<nr_blocks, nr_threads, 0, *cuda_bonds_stream>>>(
        cuda_angles_angles_arr, N, cuda_angles_forces_arr, cuda_bonds_potenergies_arr, cuda_bonds_todestroy_arr
    );
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyAsync(cuda_angles_forces_local, cuda_angles_forces_arr, size_forces, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyAsync(cuda_bonds_potenergies_local, cuda_bonds_potenergies_arr, size_potenergies, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    if(cudaMemcpyAsync(cuda_bonds_todestroy_local, cuda_bonds_todestroy_arr, size_todestroy, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

int engine_angle_cuda_unload_angle_chunk(Angle *angles, int loc, int N, struct engine *e, float *epot_out) { 
    int i, k;
    float epot = 0.f;
    float *bufff, ee, fi, fk;
    Angle *buffa = &angles[loc];

    for(i = 0; i < N; i++) {
        auto a = &buffa[i];
        if(!(a->flags & ANGLE_ACTIVE)) 
            continue;
        
        ee = cuda_bonds_potenergies_local[i];
        epot += ee;
        a->potential_energy += ee;
        if(cuda_bonds_todestroy_local[i]) {
            Angle_Destroy(a);
            continue;
        }

        bufff = &cuda_angles_forces_local[6 * i];
        auto pi = e->s.partlist[a->i];
        auto pj = e->s.partlist[a->j];
        auto pk = e->s.partlist[a->k];
        for(k = 0; k < 3; k++) {
            fi = bufff[k];
            fk = bufff[k + 3];
            pi->f[k] += fi;
            pk->f[k] += fk;
            pj->f[k] -= fi + fk;
        }
    }
    
    // Store the potential energy.
    *epot_out += epot;

    return S_OK;
}

int cuda::engine_angle_eval_cuda(struct TissueForge::Angle *angles, int N, struct engine *e, float *epot_out) {
    int i, n;
    float epot = 0.0;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error();

    if(cuda_bonds_parts_load(e) != S_OK) 
        return error(MDCERR_cuda);

    engine_angle_flip_stream();
    
    if(N < cuda_bonds_nrparts_chunk) {
        if(engine_angle_cuda_load_angle_chunk(angles, 0, N) != S_OK) 
            return error(MDCERR_cuda);
        if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
            return cuda_error();
        if(engine_angle_cuda_unload_angle_chunk(angles, 0, N, e, epot_out) != S_OK) 
            return error(MDCERR_cuda);
        if(cuda_bonds_parts_unload(e) != S_OK) 
            return error(MDCERR_cuda);
        return S_OK;
    }

    n = cuda_bonds_nrparts_chunk;
    if(engine_angle_cuda_load_angle_chunk(angles, 0, n) != S_OK) 
        return error(MDCERR_cuda);
    for(i = cuda_bonds_nrparts_chunk; i < N; i += cuda_bonds_nrparts_chunk) {
        if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
            return cuda_error();
        if(engine_angle_cuda_unload_angle_chunk(angles, i - cuda_bonds_nrparts_chunk, n, e, &epot) != S_OK) 
            return error(MDCERR_cuda);

        engine_angle_flip_stream();

        n = std::min(N - i, cuda_bonds_nrparts_chunk);
        if(engine_angle_cuda_load_angle_chunk(angles, i, n) != S_OK) 
            return error(MDCERR_cuda);
    }
    if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
        return cuda_error();
    if(engine_angle_cuda_unload_angle_chunk(angles, N - n, n, e, &epot) != S_OK) 
        return error(MDCERR_cuda);

    if(cuda_bonds_parts_unload(e) != S_OK) 
        return error(MDCERR_cuda);

    *epot_out += epot;
    
    return S_OK;
}

int cuda::Angle_setThreads(const unsigned int &nr_threads) {
    cuda_angles_nr_threads = nr_threads;
    return S_OK;
}

int cuda::Angle_setBlocks(const unsigned int &nr_blocks) {
    cuda_angles_nr_blocks = nr_blocks;
    return S_OK;
}

int cuda::Angle_getDevice() {
    return cuda_bonds_device;
}

int cuda::Angle_toDevice(engine *e) {
    if(e->angles_cuda) 
        return S_OK;
    
    return engine_angle_cuda_initialize(e);
}

int cuda::Angle_fromDevice(engine *e) {
    if(!e->angles_cuda) 
        return S_OK;
    
    return engine_angle_cuda_finalize(e);
}

int cuda::Angle_refresh(engine *e) {
    if(engine_angle_cuda_finalize(e) != S_OK) 
        return error(MDCERR_cuda);

    if(engine_angle_cuda_initialize(e) != S_OK) 
        return error(MDCERR_cuda);

    if(cudaDeviceSynchronize() != cudaSuccess) 
        return cuda_error();

    return S_OK;
}

int cuda::Angle_refreshAngle(engine *e, AngleHandle *a) {
    if(e->angles_cuda) 
        return S_OK;

    if(a == NULL) 
        return MDCERR_null;

    return engine_cuda_refresh_angle(a);
}

int cuda::Angle_refreshAngles(engine *e, AngleHandle **angles, int nr_angles) {
    if(e->angles_cuda) 
        return S_OK;

    AngleHandle *as = (AngleHandle*)malloc(nr_angles * sizeof(AngleHandle));
    AngleHandle *ah;
    for(int i = 0; i < nr_angles; i++) { 
        ah = angles[i];
        if(ah == NULL) 
            return MDCERR_null;
        as[i] = *ah;
    }
    
    if(engine_cuda_refresh_angles(e, as, nr_angles) != S_OK) 
        return error(MDCERR_cuda);

    free(as);

    return S_OK;
}
