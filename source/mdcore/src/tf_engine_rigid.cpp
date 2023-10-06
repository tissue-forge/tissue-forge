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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>

/* Include conditional headers. */
#include <mdcore_config.h>
#ifdef WITH_MPI
    #include <mpi.h>
#endif
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

/* include local headers */
#include <tf_cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfPotential.h>
#include <tfRunner.h>
#include <tfBond.h>
#include <tfRigid.h>
#include <tfAngle.h>
#include <tfDihedral.h>
#include <tfExclusion.h>
#include <tfEngine.h>


using namespace TissueForge;


#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


HRESULT TissueForge::engine_rigid_add(struct engine *e, int pid, int pjd, FPTYPE d) {

    struct rigid *dummy, *r;
    int ind, jnd, rid, rjd, k, j;

    /* Check inputs. */
    if(e == NULL)
        return error(MDCERR_null);
        
    /* If we don't have a part2rigid array, allocate and init one. */
    if(e->part2rigid == NULL) {
        if((e->part2rigid = (int *)malloc(sizeof(int *) * e->s.nr_parts)) == NULL)
            return error(MDCERR_malloc);
        for(k = 0 ; k < e->s.nr_parts ; k++)
            e->part2rigid[k] = -1;
    }
    
    /* Update the number of constraints (important for temp). */
    e->nr_constr += 1;
        
    /* Check if we already have a rigid constraint with either pid or pjd. */
    rid = e->part2rigid[pid]; rjd = e->part2rigid[pjd];
    if(rid < 0 && rjd < 0) {
    
        /* Do we need to grow the rigids array? */
        if(e->nr_rigids == e->rigids_size) {
            e->rigids_size  *= 1.414;
            if((dummy = (struct rigid *)malloc(sizeof(struct rigid) * e->rigids_size)) == NULL)
                return error(MDCERR_malloc);
            memcpy(dummy, e->rigids, sizeof(struct rigid) * e->nr_rigids);
            free(e->rigids);
            e->rigids = dummy;
            }

        /* Store this rigid. */
        r = &e->rigids[ e->nr_rigids ];
        r->nr_parts = 2;
        r->nr_constr = 1;
        r->parts[0] = pid;
        r->parts[1] = pjd;
        r->constr[0].i = 0;
        r->constr[0].j = 1;
        r->constr[0].d2 = d*d;
        e->part2rigid[pid] = e->nr_rigids;
        e->part2rigid[pjd] = e->nr_rigids;
        e->nr_rigids += 1;
    
    }
        
    /* Both particles are already in different groups. */
    else if(rid >= 0 && rjd >= 0 && rid != rjd) {
    
        /* Get a hold of both rigids. */
        r = &e->rigids[rid]; dummy = &e->rigids[rjd];
    
        /* Get indices for these parts in the respective rigids. */
        for(ind = 0 ; r->parts[ind] != pid ; ind++);
        for(jnd = 0 ; dummy->parts[jnd] != pjd ; jnd++);
                
        /* Merge the particles of rjd into rid. */
        for(j = 0 ; j < dummy->nr_parts ; j++) {
            r->parts[ r->nr_parts + j ] = dummy->parts[j];
            e->part2rigid[ dummy->parts[j] ] = rid;
        }
            
        /* Add the constraints from dummy to rid. */
        for(j = 0 ; j < dummy->nr_constr ; j++) {
            r->constr[ r->nr_constr + j ] = dummy->constr[ j ];
            r->constr[ r->nr_constr + j ].i += r->nr_parts;
            r->constr[ r->nr_constr + j ].j += r->nr_parts;
        }
            
        /* Adjust the number of parts and constr in rid. */
        r->nr_constr += dummy->nr_constr;
        r->nr_parts += dummy->nr_parts;
    
        /* Store the distance constraint. */
        r->constr[ r->nr_constr ].i = ind;
        r->constr[ r->nr_constr ].j = jnd;
        r->constr[ r->nr_constr ].d2 = d*d;
        r->nr_constr += 1;
        
        /* Remove the rigid rjd. */
        e->nr_rigids -= 1;
        if(rjd < e->nr_rigids) {
            e->rigids[ rjd ] = e->rigids[ e->nr_rigids ];
            for(j = 0 ; j < e->rigids[ rjd ].nr_parts ; j++)
                e->part2rigid[ e->rigids[rjd].parts[j] ] = rjd;
        }
        
    }
        
    /* Otherwise, one or both particles are in the same group. */
    else {
    
        /* Get a grip on the rigid. */
        if(rid < 0)
            rid = rjd;
        r = &e->rigids[rid];
        
        /* Try to get indices for these parts in the kth constraint. */
        ind = -1; jnd = -1;
        for(j = 0 ; j < r->nr_parts ; j++) {
            if(r->parts[j] == pid)
                ind = j;
            else if(r->parts[j] == pjd)
                jnd = j;
        }
                
        /* Do we need to store i or j? */
        if(ind < 0) {
            r->parts[ r->nr_parts ] = pid;
            ind = r->nr_parts;
            r->nr_parts += 1;
            e->part2rigid[pid] = rid;
        }
        else if(jnd < 0) {
            r->parts[ r->nr_parts ] = pjd;
            jnd = r->nr_parts;
            r->nr_parts += 1;
            e->part2rigid[pjd] = rid;
        }
            
        /* Store the distance constraint. */
        r->constr[ r->nr_constr ].i = ind;
        r->constr[ r->nr_constr ].j = jnd;
        r->constr[ r->nr_constr ].d2 = d*d;
        r->nr_constr += 1;
        
    }
        
    /* Fill the constraint shuffle matrix. */
    bzero(r->a, sizeof(FPTYPE) * r->nr_constr * r->nr_constr);
    for(k = 0 ; k < r->nr_constr ; k++)
        r->a[ k + k*r->nr_constr ] = 1.0f;
        
    /* Re-set the rigid local/semilocal counts. */
    e->rigids_local = e->rigids_semilocal = e->nr_rigids;
        
    /* It's the end of the world as we know it. */
    return S_OK;

}
 
HRESULT TissueForge::engine_rigid_unsort(struct engine *e) {

    int k, j;
    struct rigid temp;
    FPTYPE scale = ((FPTYPE)e->nr_rigids) / RAND_MAX;
    
    /* For each rigid... */
    for(k = 0 ; k < e->nr_rigids ; k++) {
    
        /* Get the id of another random rigid. */
        j = rand() * scale;
        
        /* Swap the jth and kth entry. */
        if(j != k) {
            temp = e->rigids[j];
            e->rigids[j] = e->rigids[k];
            e->rigids[k] = temp;
        }
            
    }
        
    /* Don't you forget about me... */
    return S_OK;

}

HRESULT TissueForge::engine_rigid_sort(struct engine *e) {

    struct space_cell **celllist;
    struct rigid temp;
    int nr_rigids = e->nr_rigids, nr_local, nr_ghosts, i, j, k;
    
    /* If not in parallel, then we've got nothing to do. */
    if(e->nr_nodes == 1) {
        e->rigids_semilocal = e->rigids_local = e->nr_rigids;
        return S_OK;
    }
    
    /* Get a handle on the celllist. */
    celllist = e->s.celllist;

    /* Split between local and completely non-local rigids. */
    i = 0; j = nr_rigids-1;
    while(i < j) {
        while(i < nr_rigids) {
            for(nr_ghosts = 0, k = 0 ; k < e->rigids[i].nr_parts && celllist[e->rigids[i].parts[k]] != NULL ; k++)
                if(celllist[e->rigids[i].parts[k]]->flags & cell_flag_ghost)
                    nr_ghosts += 1;
            if(k < e->rigids[i].nr_parts || nr_ghosts == e->rigids[i].nr_parts)
                break;
            i += 1;
        }
        while(j >= 0) {
            for(nr_ghosts = 0, k = 0 ; k < e->rigids[j].nr_parts && celllist[e->rigids[j].parts[k]] != NULL ; k++)
                if(celllist[e->rigids[j].parts[k]]->flags & cell_flag_ghost)
                    nr_ghosts += 1;
            if(k == e->rigids[j].nr_parts && nr_ghosts < e->rigids[j].nr_parts)
                break;
            j -= 1;
        }
        if(i < j) {
            temp = e->rigids[i];
            e->rigids[i] = e->rigids[j];
            e->rigids[j] = temp;
        }
    }
    nr_rigids = i;

    /* Split again between strictly local and semi-local (contains ghosts). */
    i = 0; j = nr_rigids-1;
    while(i < j) {
        while(i < nr_rigids) {
            for(k = 0 ; k < e->rigids[i].nr_parts && !(celllist[e->rigids[i].parts[k]]->flags & cell_flag_ghost) ; k++);
            if(k < e->rigids[i].nr_parts)
                break;
            i += 1;
        }
        while(j >= 0) {
            for(k = 0 ; k < e->rigids[j].nr_parts && !(celllist[e->rigids[j].parts[k]]->flags & cell_flag_ghost) ; k++);
            if(k == e->rigids[j].nr_parts)
                break;
            j -= 1;
        }
        if(i < j) {
            temp = e->rigids[i];
            e->rigids[i] = e->rigids[j];
            e->rigids[j] = temp;
        }
    }
    nr_local = i;


    /* Store the values in the engine. */
    e->rigids_local = nr_local;
    e->rigids_semilocal = nr_rigids;
        
    /* I'll be back... */
    return S_OK;

}

HRESULT TissueForge::engine_rigid_eval(struct engine *e) {

    int nr_local = e->rigids_local, nr_rigids = e->rigids_semilocal;
    #ifdef WITH_MPI
        ticks tic;
    #endif
    #ifdef HAVE_OPENMP
        // int k;
        int nr_threads, count = nr_rigids - nr_local;
        int finger_global = 0, finger, chunk;
    #endif
    
    /* Do we have asynchronous communication going on, e.g. are we waiting
       for ghosts? */
    if(e->nr_nodes > 1 && e->flags & engine_flag_async) {
    
        #ifdef HAVE_OPENMP

            /* Is it worth parallelizing? */
            // #pragma omp parallel private(k)
            #pragma omp parallel private(finger,count,chunk)
            if((nr_threads = omp_get_num_threads()) > 1) {
            
                /* Get a sensible chunk size. */
                chunk = nr_local / nr_threads;
                if(chunk > engine_rigids_chunk)
                    chunk = engine_rigids_chunk;
                
                /* Main loop. */
                while(finger_global < nr_local) {
                
                    /* Get a finger on the bonds list. */
                    #pragma omp critical
                    {
                        if(finger_global < nr_local) {
                            finger = finger_global;
                            count = chunk;
                            if(finger + count > nr_local)
                                count = nr_local - finger;
                            finger_global += count;
                            }
                        else
                            count = 0;
                    }
                
                    /* Compute the bonded interactions. */
                    if(count > 0) {
                        if(e->flags & engine_flag_shake)
                            rigid_eval_shake(&e->rigids[finger], count, e);
                        else
                            rigid_eval_pshake(&e->rigids[finger], count, e, (e->time < engine_pshake_steps));
                    }
                
                } /* main loop. */

            }

            /* Otherwise, evaluate directly. */
            else if(omp_get_thread_num() == 0) {
                if(e->flags & engine_flag_shake)
                    rigid_eval_shake(e->rigids, nr_local, e);
                else
                    rigid_eval_pshake(e->rigids, nr_local, e, (e->time < engine_pshake_steps));
            }
                
                
#ifdef WITH_MPI
            /* Wait for the async data to come in. */
            tic = getticks();
            if(engine_exchange_rigid_wait(e) < 0)
                return error(MDCERR_mpi);
            tic = getticks() - tic;
            e->timers[engine_timer_exchange1] += tic;
            e->timers[engine_timer_rigid] -= tic;
#endif
                
                
            /* Is it worth parallelizing? */
            // #pragma omp parallel private(k)
            #pragma omp parallel private(finger,count,chunk)
            if((nr_threads = omp_get_num_threads()) > 1) {

                /* Get a sensible chunk size. */
                chunk = (nr_rigids - nr_local) / nr_threads;
                if(chunk > engine_rigids_chunk)
                    chunk = engine_rigids_chunk;
                
                /* Main loop. */
                while(finger_global < nr_rigids) {
                
                    /* Get a finger on the bonds list. */
                    #pragma omp critical
                    {
                        if(finger_global < nr_rigids) {
                            finger = finger_global;
                            count = chunk;
                            if(finger + count > nr_rigids)
                                count = nr_rigids - finger;
                            finger_global += count;
                        }
                        else
                            count = 0;
                    }
                
                    /* Compute the bonded interactions. */
                    if(count > 0) {
                        if(e->flags & engine_flag_shake)
                            rigid_eval_shake(&e->rigids[finger], count, e);
                        else
                            rigid_eval_pshake(&e->rigids[finger], count, e, (e->time < engine_pshake_steps));
                        }
                
                } /* main loop. */

            }

            /* Otherwise, evaluate directly. */
            else if(omp_get_thread_num() == 0) {
                if(e->flags & engine_flag_shake)
                    rigid_eval_shake(&e->rigids[nr_local], nr_rigids-nr_local, e);
                else
                    rigid_eval_pshake(&e->rigids[nr_local], nr_rigids-nr_local, e, (e->time < engine_pshake_steps));
            }
                
        #else
        
            /* Shake local rigids. */
            if(e->flags & engine_flag_shake) {
                if(rigid_eval_shake(e->rigids, nr_local, e) != S_OK)
                    return error(MDCERR_rigid);
            }
            else {
                if(rigid_eval_pshake(e->rigids, nr_local, e, (e->time < engine_pshake_steps)) != S_OK)
                    return error(MDCERR_rigid);
            }
                
            /* Wait for exchange to come in. */
            #ifdef WITH_MPI
            //tic = getticks();
            if(e->flags & engine_flag_async)
                if(engine_exchange_wait(e) < 0)
                    return error(MDCERR_mpi);
            //tic = getticks() - tic;
            //e->timers[engine_timer_exchange1] += tic;
            //e->timers[engine_timer_verlet] -= tic;
            #endif
                
            /* Shake semi-local rigids. */
            if(e->flags & engine_flag_shake) {
                if(rigid_eval_shake(&(e->rigids[nr_local]), nr_rigids-nr_local, e) != S_OK)
                    return error(MDCERR_rigid);
            }
            else {
                if(rigid_eval_pshake(&(e->rigids[nr_local]), nr_rigids-nr_local, e, (e->time < engine_pshake_steps)) != S_OK)
                    return error(MDCERR_rigid);
            }
                
        #endif
    
    }
        
    /* No async, do it all at once. */
    else {
    
        #ifdef HAVE_OPENMP

            /* Is it worth parallelizing? */
            // #pragma omp parallel private(k)
            #pragma omp parallel private(finger,count,chunk)
            if((nr_threads = omp_get_num_threads()) > 1) {

                /* Get a sensible chunk size. */
                chunk = nr_rigids / nr_threads;
                if(chunk > engine_rigids_chunk)
                    chunk = engine_rigids_chunk;
                
                /* Main loop. */
                while(finger_global < nr_rigids) {
                
                    /* Get a finger on the bonds list. */
                    #pragma omp critical
                    {
                        if(finger_global < nr_rigids) {
                            finger = finger_global;
                            count = chunk;
                            if(finger + count > nr_rigids)
                                count = nr_rigids - finger;
                            finger_global += count;
                            }
                        else
                            count = 0;
                    }
                
                    /* Compute the bonded interactions. */
                    if(count > 0) {
                        if(e->flags & engine_flag_shake)
                            rigid_eval_shake(&e->rigids[finger], count, e);
                        else
                            rigid_eval_pshake(&e->rigids[finger], count, e, (e->time < engine_pshake_steps));
                    }
                
                } /* main loop. */

            }

            /* Otherwise, evaluate directly. */
            else if(omp_get_thread_num() == 0) {
                if(e->flags & engine_flag_shake)
                    rigid_eval_shake(e->rigids, nr_rigids, e);
                else
                    rigid_eval_pshake(e->rigids, nr_rigids, e, (e->time < engine_pshake_steps));
            }
                
        #else
        
            if(e->flags & engine_flag_shake) {
                if(rigid_eval_shake(e->rigids, nr_rigids, e) != S_OK)
                    return error(MDCERR_rigid);
            }
            else {
                if(rigid_eval_pshake(e->rigids, nr_rigids, e, (e->time < engine_pshake_steps)) != S_OK)
                    return error(MDCERR_rigid);
            }
                
        #endif
    
    }
        
    /* I'll be back... */
    return S_OK;

}
    

