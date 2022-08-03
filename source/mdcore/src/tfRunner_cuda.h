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
#ifndef _MDCORE_SOURCE_TFRUNNER_CUDA_H_
#define _MDCORE_SOURCE_TFRUNNER_CUDA_H_

#include <tfTask.h>

#include <curand_kernel.h>


/* Set the max number of parts for shared buffers. */
#define cuda_maxparts                       512
#define cuda_maxdiags                       352
#define cuda_ndiags                         (((cuda_maxdiags - 1) * cuda_maxdiags) / 2)
#define cuda_frame                          32
#define cuda_maxpots                        100
#define max_fingers                         1
#define cuda_defthreads                     128
#define cuda_memcpy_chunk                   6
#define cuda_sum_chunk                      3
#define cuda_maxqueues                      30


/* Some flags that control optional behaviour */
// #define TIMERS


/** Timers for the cuda parts. */
enum {
    tid_mutex = 0,
    tid_queue,
    tid_gettask,
    tid_memcpy,
    tid_update,
    tid_pack,
    tid_sort,
    tid_pair,
    tid_self,
    tid_potential,
    tid_potential4,
    tid_total,
    tid_count
    };
    

/* Timer functions. */
#ifdef TIMERS
    #define TIMER_TIC_ND if(threadIdx.x == 0) tic = clock();
    #define TIMER_TOC_ND(tid) toc = clock(); if(threadIdx.x == 0) atomicAdd(&cuda_timers[tid],(toc > tic) ? (toc - tic) :(toc + (0xffffffff - tic)));
    #define TIMER_TIC clock_t tic; if(threadIdx.x == 0) tic = clock();
    #define TIMER_TOC(tid) clock_t toc = clock(); if(threadIdx.x == 0) atomicAdd(&cuda_timers[tid],(toc > tic) ? (toc - tic) :(toc + (0xffffffff - tic)));
    #define TIMER_TIC2_ND if(threadIdx.x == 0) tic2 = clock();
    #define TIMER_TOC2_ND(tid) toc2 = clock(); if(threadIdx.x == 0) atomicAdd(&cuda_timers[tid],(toc2 > tic2) ? (toc2 - tic2) :(toc2 + (0xffffffff - tic2)));
    #define TIMER_TIC2 clock_t tic2; if(threadIdx.x == 0) tic2 = clock();
    #define TIMER_TOC2(tid) clock_t toc2 = clock(); if(threadIdx.x == 0) atomicAdd(&cuda_timers[tid],(toc2 > tic2) ? (toc2 - tic2) :(toc2 + (0xffffffff - tic2)));
#else
    #define TIMER_TIC_ND
    #define TIMER_TOC_ND(tid)
    #define TIMER_TIC
    #define TIMER_TOC(tid)
    #define TIMER_TIC2
    #define TIMER_TOC2(tid)
#endif


namespace TissueForge::cuda {


    /** Struct for a task queue. */
    struct queue_cuda {

        /* Indices to the first and last elements. */
        int first, last;
        
        /* Number of elements in this queue. */
        volatile int count;
        
        /* Number of elements in the recycled list. */
        volatile int rec_count;
        
        /* The queue data. */
        volatile int *data;
        
        /* The recycling list. */
        volatile int *rec_data;

    };


    /** Struct for each task. */
    struct task_cuda {

        /** Task type and subtype. */
        short int type, subtype;

        /** Wait counters. */
        volatile int wait;
        
        /** Task flags. */
        int flags;

        /** Indices of the cells involved. */
        int i, j;
        
        /** Nr of task that this task unlocks. */
        int nr_unlock;
        
        /** List of task that this task unlocks (dependencies). */
        int unlock[ task_max_unlock ];
        
    };

    /**
     * @brief Evaluates the given potential at the given point (interpolated).
     *
     * @param p The #potential to be evaluated.
     * @param r The radius at which it is to be evaluated.
     * @param e Pointer to a floating-point value in which to store the
     *      interaction energy.
     * @param f Pointer to a floating-point value in which to store the
     *      magnitude of the interaction force.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     */

    __device__ void potential_eval_r_cuda(struct TissueForge::Potential *p, FPTYPE r, FPTYPE *e, FPTYPE *f);    

    /** 
     * @brief Evaluates the given potential at the given point (interpolated).
     *
     * @param p The #potential to be evaluated.
     * @param r2 The radius at which it is to be evaluated, squared.
     * @param e Pointer to a floating-point value in which to store the
     *      interaction energy.
     * @param f Pointer to a floating-point value in which to store the
     *      magnitude of the interaction force divided by r.
     *
     * Note that for efficiency reasons, this function does not check if any
     * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
     * of the #potential @c p.
     */
    __device__ 
    void potential_eval_cuda(struct TissueForge::Potential *p, float r2, float *e, float *f);

};

#endif // _MDCORE_SOURCE_TFRUNNER_CUDA_H_