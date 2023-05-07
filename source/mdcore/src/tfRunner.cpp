/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* Include configuration header */
#include <mdcore_config.h>

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include <random>

/* Include some conditional headers. */
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef HAVE_SETAFFINITY
#include <sched.h>
#endif
#ifdef WITH_MPI
#include <mpi.h>
#endif

/* Include local headers */
#include <tf_cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfQueue.h>
#include <tfSpace_cell.h>
#include <tfTask.h>
#include <tfSpace.h>
#include <tfPotential.h>
#include <tfEngine.h>
#include <tfRunner.h>
#include <tfLogger.h>


using namespace TissueForge;


/* Global variables. */

/** Timers. */
ticks TissueForge::runner_timers[runner_timer_count];

/* the error macro. */
#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


void TissueForge::runner_sort_descending(unsigned int *parts, int N) {

    struct {
        short int lo, hi;
    } qstack[10];
    int qpos, i, j, lo, hi, pivot, imax;
    unsigned int temp;

    /* Sort parts in cell_i in decreasing order with quicksort */
    qstack[0].lo = 0; qstack[0].hi = N - 1; qpos = 0;
    while(qpos >= 0) {
        lo = qstack[qpos].lo; hi = qstack[qpos].hi;
        qpos -= 1;
        if(hi - lo < 15) {
            for(i = lo ; i < hi ; i++) {
                imax = i;
                for(j = i+1 ; j <= hi ; j++)
                    if((parts[j] & 0xffff) > (parts[imax] & 0xffff))
                        imax = j;
                if(imax != i) {
                    temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                }
            }
        }
        else {
            pivot = parts[(lo + hi) / 2 ] & 0xffff;
            i = lo; j = hi;
            while(i <= j) {
                while((parts[i] & 0xffff) > pivot) i++;
                while((parts[j] & 0xffff) < pivot) j--;
                if(i <= j) {
                    if(i < j) {
                        temp = parts[i]; parts[i] = parts[j]; parts[j] = temp;
                    }
                    i += 1; j -= 1;
                }
            }
            if(j >(lo + hi) / 2) {
                if(lo < j) {
                    qpos += 1;
                    qstack[qpos].lo = lo;
                    qstack[qpos].hi = j;
                }
                if(i < hi) {
                    qpos += 1;
                    qstack[qpos].lo = i;
                    qstack[qpos].hi = hi;
                }
            }
            else {
                if(i < hi) {
                    qpos += 1;
                    qstack[qpos].lo = i;
                    qstack[qpos].hi = hi;
                }
                if(lo < j) {
                    qpos += 1;
                    qstack[qpos].lo = lo;
                    qstack[qpos].hi = j;
                }
            }
        }
    }
}

void TissueForge::runner_sort_ascending(unsigned int *parts, int N) {

    struct {
        short int lo, hi;
    } qstack[10];
    int qpos, i, j, lo, hi, pivot, imax;
    unsigned int temp;

    /* Sort parts in cell_i in decreasing order with quicksort */
    qstack[0].lo = 0; qstack[0].hi = N - 1; qpos = 0;
    while(qpos >= 0) {
        lo = qstack[qpos].lo; hi = qstack[qpos].hi;
        qpos -= 1;
        if(hi - lo < 15) {
            for(i = lo ; i < hi ; i++) {
                imax = i;
                for(j = i+1 ; j <= hi ; j++)
                    if((parts[j] & 0xffff) < (parts[imax] & 0xffff))
                        imax = j;
                if(imax != i) {
                    temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                }
            }
        }
        else {
            pivot = parts[(lo + hi) / 2 ] & 0xffff;
            i = lo; j = hi;
            while(i <= j) {
                while((parts[i] & 0xffff) < pivot) i++;
                while((parts[j] & 0xffff) > pivot) j--;
                if(i <= j) {
                    if(i < j) {
                        temp = parts[i]; parts[i] = parts[j]; parts[j] = temp;
                    }
                    i += 1; j -= 1;
                }
            }
            if(j >(lo + hi) / 2) {
                if(lo < j) {
                    qpos += 1;
                    qstack[qpos].lo = lo;
                    qstack[qpos].hi = j;
                }
                if(i < hi) {
                    qpos += 1;
                    qstack[qpos].lo = i;
                    qstack[qpos].hi = hi;
                }
            }
            else {
                if(i < hi) {
                    qpos += 1;
                    qstack[qpos].lo = i;
                    qstack[qpos].hi = hi;
                }
                if(lo < j) {
                    qpos += 1;
                    qstack[qpos].lo = lo;
                    qstack[qpos].hi = j;
                }
            }
        }
    }
}


HRESULT TissueForge::runner_run(struct runner *r) {

    struct engine *e = r->e;
    struct space *s = &e->s;
    int k, err = 0, acc = 0, naq, qid, myqid = e->nr_queues * r->id / e->nr_runners;
    struct task *t = NULL;
    struct queue *myq = &e->queues[ myqid ], *queues[ e->nr_queues ];
    //unsigned int myseed = rand() + r->id;
    int count;

    std::default_random_engine randeng;

    /* give a hoot */
    TF_Log(LOG_INFORMATION) << "runner_run: runner " << r->id << " is up and running on queue " << myqid << " (tasks)";

    /* main loop, in which the runner should stay forever... */
    while(1) {

        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if(engine_barrier(e) != S_OK)
            return error(MDCERR_engine);

        /* Init the list of queues. */
        for(k = 0 ; k < e->nr_queues ; k++)
            queues[k] = &e->queues[k];
        naq = e->nr_queues - 1;
        queues[ myqid ] = queues[ naq ];

        /* while i can still get a pair... */
        /* printf("runner_run: runner %i paSSEd barrier, getting pairs...\n",r->id); */
        while(myq->next < myq->count || naq > 0) {

            /* Try to get a pair from my own queue. */
            TIMER_TIC
            if(myq->next == myq->count ||(t = queue_get(myq, r->id, 0)) == NULL) {

                /* Clean up the list of queues. */
                count = myq->count - myq->next;
                for(k = 0 ; k < naq ; k++) {
                    count += queues[k]->count - queues[k]->next;
                    if(queues[k]->next == queues[k]->count)
                        queues[k--] = queues[--naq];
                }

                /* If there are no queues left, go back to go, do not collect 200 FLOPs. */
                if(naq == 0)
                    continue;

                /* Otherwise, try to grab something from a random queue. */
                qid = randeng() % naq;
                if((t = queue_get(queues[qid], r->id, 1)) != NULL) {

                    /* Add this task to my own queue. */
                    if(!queue_insert(myq, t))
                        queue_insert(queues[qid], t);

                }

                /* If there are more queues than tasks, fall on sword. */
                if(t == NULL && count <= r->id)
                    break;

            }

            /* If I didn't get a task, try again, locking... */
            if(t == NULL) {

                /* Lock the mutex. */
                if(pthread_mutex_lock(&s->tasks_mutex) != 0)
                    return error(MDCERR_pthread);

                /* Try again to get a pair... */
                if(myq->next == myq->count ||(t = queue_get(myq, r->id, 0)) == NULL) {
                    count = myq->count - myq->next;
                    for(k = 0 ; k < naq ; k++) {
                        count += queues[k]->count - queues[k]->next;
                        if(queues[k]->next == queues[k]->count)
                            queues[k--] = queues[--naq];
                    }
                    if(naq != 0) {
                        qid = randeng() % naq;
                        if((t = queue_get(queues[qid], r->id, 1)) != NULL) {
                            if(!queue_insert(myq, t))
                                queue_insert(queues[qid], t);
                        }
                    }
                }

                /* If no pair, wait... */
                if(count > 0 && t == NULL)    
                    if(pthread_cond_wait(&s->tasks_avail, &s->tasks_mutex) != 0)
                        return error(MDCERR_pthread);

                /* Unlock the mutex. */
                if(pthread_mutex_unlock(&s->tasks_mutex) != 0)
                    return error(MDCERR_pthread);

                /* Skip back to the top of the queue if empty-handed. */
                if(t == NULL)
                    continue;

            }
            TIMER_TOC(runner_timer_queue);

            /* Check task type... */
            switch(t->type) {
                case task_type_sort:
                    TIMER_TIC_ND
                    if(s->verlet_rebuild && e->step_flux == 0)
                        if(runner_dosort(r, &s->cells[ t->i ], t->flags) != S_OK)
                            return error(MDCERR_runner);
                    s->cells_taboo[ t->i ] = 0;
                    TIMER_TOC(runner_timer_sort);
                    break;
                case task_type_self:
                    TIMER_TIC_ND
                    if(e->integrator_flags & INTEGRATOR_FLUX_SUBSTEP) {
                        if(runner_doself_fluxonly(r, &s->cells[ t->i ]) != S_OK)
                            return error(MDCERR_runner);
                    } 
                    else {
                        if(runner_doself(r, &s->cells[ t->i ]) != S_OK)
                            return error(MDCERR_runner);
                    }
                    s->cells_taboo[ t->i ] = 0;
                    TIMER_TOC(runner_timer_self);
                    break;
                case task_type_pair:
                    TIMER_TIC_ND
                    if(e->integrator_flags & INTEGRATOR_FLUX_SUBSTEP) {
                        if(runner_dopair_fluxonly(r, &s->cells[ t->i ], &s->cells[ t->j ], t->flags) != S_OK)
                            return error(MDCERR_runner);
                    } 
                    else {
                        if(runner_dopair(r, &s->cells[ t->i ], &s->cells[ t->j ], t->flags) != S_OK)
                            return error(MDCERR_runner);
                    }
                    s->cells_taboo[ t->i ] = 0;
                    s->cells_taboo[ t->j ] = 0;
                    TIMER_TOC(runner_timer_pair);
                    break;
                default:
                    return error(MDCERR_tasktype);
            }

            /* Unlock any dependent tasks. */
            for(k = 0 ; k < t->nr_unlock ; k++)
                __sync_fetch_and_sub(&t->unlock[k]->wait, 1);

            /* Bing! */
            if(pthread_mutex_lock(&s->tasks_mutex) != 0)
                return error(MDCERR_pthread);
            if(pthread_cond_broadcast(&s->tasks_avail) != 0)
                return error(MDCERR_pthread);
            if(pthread_mutex_unlock(&s->tasks_mutex) != 0)
                return error(MDCERR_pthread);

        }

        r->err = acc;

        /* did things go wrong? */
        /* printf("runner_run: runner %i done pairs.\n",r->id); fflush(stdout); */
        if(err < 0)
            return error(MDCERR_space);

        /* Bing! */
        if(pthread_mutex_lock(&s->tasks_mutex) != 0)
            return error(MDCERR_pthread);
        if(pthread_cond_broadcast(&s->tasks_avail) != 0)
            return error(MDCERR_pthread);
        if(pthread_mutex_unlock(&s->tasks_mutex) != 0)
            return error(MDCERR_pthread);
    }
}

HRESULT TissueForge::runner_init(struct runner *r, struct engine *e, int id) {

#if defined(HAVE_SETAFFINITY)
    cpu_set_t cpuset;
#endif

    /* make sure the inputs are ok */
    if(r == NULL || e == NULL)
        return error(MDCERR_null);

    /* remember who i'm working for */
    r->e = e;
    r->id = id;

    /* init the thread using tasks. */
    if(pthread_create(&r->thread, NULL, (void *(*)(void *))runner_run, r) != 0)
        return error(MDCERR_pthread);

    /* If we can, try to restrict this runner to a single CPU. */
#if defined(HAVE_SETAFFINITY)
    if(e->flags & engine_flag_affinity) {

        /* Set the cpu mask to zero | r->id. */
        CPU_ZERO(&cpuset);
        CPU_SET(r->id, &cpuset);

        /* Apply this mask to the runner's pthread. */
        if(pthread_setaffinity_np(r->thread, sizeof(cpu_set_t), &cpuset) != 0)
            return error(MDCERR_pthread);

    }
#endif

    /* all is well... */
    return S_OK;
}

/**
 * @brief The #runner's main routine (for Verlet lists).
 *
 * @param r Pointer to the #runner to run.
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having passed, checks first if the Verlet list should
 * be re-built and then proceeds to traverse the Verlet list cell-wise
 * and computes its interactions.
 */
HRESULT runner_run_verlet(struct runner *r) {

    int res, i, ci, j, cj, k, eff_size = 0, acc = 0;
    struct engine *e;
    struct space *s;
    struct celltuple *t;
    struct space_cell *c;
    FPTYPE shift[3], *eff = NULL;
    int count;

    /* check the inputs */
    if(r == NULL)
        return error(MDCERR_null);

    /* get a pointer on the engine. */
    e = r->e;
    s = &(e->s);

    /* give a hoot */
    TF_Log(LOG_INFORMATION) << "runner_run: runner" << r->id << "is up and running (Verlet)...";

    /* main loop, in which the runner should stay forever... */
    while(1) {

        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if(engine_barrier(e) != S_OK)
            return error(MDCERR_engine);

        /* Does the Verlet list need to be reconstructed? */
        if(s->verlet_rebuild) {

            /* Loop over tuples. */
            while(1) {

                /* Get a tuple. */
                if((res = space_gettuple(s, &t, 1)) < 0)
                    return r->err = MDCERR_space;

                /* If there were no tuples left, bail. */
                if(res < 1)
                    break;

                /* for each cell, prefetch the parts involved. */
                if(e->flags & engine_flag_prefetch)
                    for(i = 0 ; i < t->n ; i++) {
                        c = &(s->cells[t->cellid[i]]);
                        for(k = 0 ; k < c->count ; k++)
                            acc += c->parts[k].id;
                        }

                /* Loop over all pairs in this tuple. */
                for(i = 0 ; i < t->n ; i++) {

                    /* Get the cell ID. */
                    ci = t->cellid[i];

                    for(j = i ; j < t->n ; j++) {

                        /* Is this pair active? */
                        if(t->pairid[ space_pairind(i,j) ] < 0)
                            continue;

                        /* Get the cell ID. */
                        cj = t->cellid[j];

                        /* Compute the shift between ci and cj. */
                        for(k = 0 ; k < 3 ; k++) {
                            shift[k] = s->cells[cj].origin[k] - s->cells[ci].origin[k];
                            if(shift[k] * 2 > s->dim[k])
                                shift[k] -= s->dim[k];
                            else if(shift[k] * 2 < -s->dim[k])
                                shift[k] += s->dim[k];
                            }

                        /* Rebuild the Verlet entries for this cell pair. */
                        if(runner_verlet_fill(r, &(s->cells[ci]), &(s->cells[cj]), shift) != S_OK)
                            return error(MDCERR_runner);

                        /* release this pair */
                        if(space_releasepair(s, ci, cj) != S_OK)
                            return error(MDCERR_space);

                        }

                    }

                } /* loop over tuples. */

            /* did anything go wrong? */
            if(res < 0)
                return error(MDCERR_space);

            } /* reconstruct the Verlet list. */

        /* Otherwise, just run through the Verlet list. */
        else {

            /* Check if eff is large enough and re-allocate if needed. */
            if(eff_size < s->nr_parts) {

                /* Free old eff? */
                if(eff != NULL)
                    free(eff);

                /* Allocate new eff. */
                eff_size = s->nr_parts * 1.1;
                if((eff = (FPTYPE *)malloc(sizeof(FPTYPE) * eff_size * 4)) == NULL)
                    return error(MDCERR_malloc);

                }

            /* Reset the force vector. */
            memset(eff, 0, sizeof(FPTYPE) * s->nr_parts * 4);

            /* Re-set the potential energy. */
            r->epot = 0.0;

            /* While there are still chunks of the Verlet list out there... */
            while((count = space_getcell(s, &c)) > 0) {

                /* Dispatch the interactions to runner_verlet_eval. */
                runner_verlet_eval(r, c, eff);

                }

            /* did things go wrong? */
            if(count < 0)
                return error(MDCERR_space);

            /* Send the forces and energy back to the space. */
            if(space_verlet_force(s, eff, r->epot) < 0)
                return error(MDCERR_space);

            }

        r->err = acc;

        }

    /* end well... */
    return S_OK;

}
