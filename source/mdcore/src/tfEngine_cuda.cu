/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>

/* Include conditional headers. */
#include <mdcore_config.h>
#ifdef HAVE_MPI
    #include <mpi.h>
#endif
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

/* Disable vectorization for the nvcc compiler's sake. */
#undef __SSE__
#undef __SSE2__
#undef __ALTIVEC__
#undef __AVX__

#include <cuda_runtime.h>

/* include local headers */
#include <cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfSpace_cell.h>
#include <tfTask.h>
#include <tfSpace.h>
#include <tfPotential.h>
#include <tfBond.h>
#include <tfRigid.h>
#include <tfAngle.h>
#include <tfDihedral.h>
#include <tfExclusion.h>
#include <tfEngine.h>
#include <tfRunner_cuda.h>
#include <tf_cuda.h>


using namespace TissueForge;


/* As of here there is only CUDA-related stuff. */
#ifdef HAVE_CUDA

/* the error macro. */
#define error(id)				(engine_err = errs_register(id, engine_err_msg[-(id)], __LINE__, __FUNCTION__, __FILE__))
#define cuda_error(id)			(engine_err = errs_register(id, cudaGetErrorString(cudaGetLastError()), __LINE__, __FUNCTION__, __FILE__))


/* The parts (non-texture access). */
extern __constant__ float4 *cuda_parts;


/**
 * @brief Set the number of threads of a CUDA device to use
 * 
 * @param id The CUDA device id
 * @param nr_threads The number of threads to use
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int cuda::engine_cuda_setthreads(struct engine *e, int id, int nr_threads) {
    if(id >= e->nr_devices)
        return engine_err_cuda;

    for(int i = 0; i < e->nr_devices; i++) {
        if(e->devices[i] == id) {
            e->nr_threads[i] = nr_threads;
            return engine_err_ok;
        }
    }
    return engine_err_cuda;
}

/**
 * @brief Set the number of blocks of a CUDA device to use
 * 
 * @param id The CUDA device id
 * @param nr_blocks The number of blocks to use
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int cuda::engine_cuda_setblocks(struct engine *e, int id, int nr_blocks) {
    if(id >= e->nr_devices)
        return engine_err_cuda;

    for(int i = 0; i < e->nr_devices; i++) {
        if(e->devices[i] == id) {
            e->nr_blocks[i] = nr_blocks;
            return engine_err_ok;
        }
    }
    return engine_err_cuda;
}

/**
 * @brief Set the ID of the CUDA device to use
 *
 * @param id The CUDA device ID.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int cuda::engine_cuda_setdevice(struct engine *e, int id) {

    /* Store the single device ID in the engine. */
    e->nr_devices = 1;
    e->devices[0] = id;

    /* Make sure the device works, init a stream on it. */
    if(cudaSetDevice(id) != cudaSuccess ||
         cudaStreamCreate((cudaStream_t *)&e->streams[0]) != cudaSuccess)
        return cuda_error(engine_err_cuda);
    else {
        // Do auto configuration
        engine_cuda_setthreads(e, id, cuda::maxThreadsPerBlock(id));
        engine_cuda_setblocks(e, id, cuda::maxBlockDimX(id));
        return engine_err_ok;
    }
        
}


/**
 * @brief Set the number of CUDA devices to use, as well as their IDs.
 *
 * @param id The CUDA device ID.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int cuda::engine_cuda_setdevices(struct engine *e, int nr_devices, int *ids) {

    int k;
    
    /* Sanity check. */
    if(nr_devices > engine_maxgpu)
        return error(engine_err_range);

    /* Store the single device ID in the engine. */
    e->nr_devices = nr_devices;
    for(k = 0 ; k < nr_devices ; k++) {
    
        /* Store the device ID. */
        e->devices[k] = ids[k];

        /* Make sure the device works, init a stream on it. */
        if(cudaSetDevice(ids[k]) != cudaSuccess || cudaStreamCreate((cudaStream_t *)&e->streams[k]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
            
        // Do auto configuration
        engine_cuda_setthreads(e, ids[k], cuda_defthreads);
        engine_cuda_setblocks(e, ids[k], cuda::maxBlockDimX(ids[k]));
        
        }
        
    /* That's it. */
    return engine_err_ok;
        
}

/**
 * @brief Clear CUDA devices. Engine must not be in CUDA run mode. 
 * 
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int cuda::engine_cuda_cleardevices(struct engine *e) {
    
    // Check inputs
    
    if(e == NULL)
        return error(engine_err_null);

    // Check state

    //  If nothing to do, then do nothing
    if(e->nr_devices == 0)
        return engine_err_ok;

	//  If already on device, then error
    if(e->flags & engine_flag_cuda)
		return engine_err_cuda;

    // Clear all set devices
    
    for(int i = 0; i < e->nr_devices; i++) {
        e->devices[i] = 0;
    }
    e->nr_devices = 0;

    return engine_err_ok;
}
    

/* End CUDA-related stuff. */
#endif
